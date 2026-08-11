import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import assert from "node:assert/strict";
import {
  advanceToNextTrick,
  applyAction,
  createInitialGame,
  getLegalActions,
  type Card,
  type GameAction,
  type GameState,
  type Suit
} from "@napoleon/game-core";

interface CanonicalSnapshot {
  phase: string;
  currentPlayerId: string;
  players: readonly { id: string; handIds: readonly string[] }[];
  currentTrick: readonly { playerId: string; cardId: string }[];
  completedTricks: readonly {
    trickNumber: number;
    winnerId: string;
    cards: readonly { playerId: string; cardId: string }[];
  }[];
  trumpSuit: string | null;
  contract: GameState["contract"];
  adjutant: GameState["adjutant"];
  bidding: GameState["bidding"];
  awardedPointCards: readonly { playerId: string; cardIds: readonly string[] }[];
  excludedCardIds: readonly string[];
  unusedCardIds: readonly string[];
  latestEvent: {
    type: "buried-cards-resolved";
    napoleonPlayerId: string;
    awardedPointCardIds: readonly string[];
    hiddenNonPointCardCount: number;
  } | null;
  result: GameState["result"];
  trickNumber: number;
  isTrickComplete: boolean;
  isGameOver: boolean;
}

function createSeededRandom(seed: number): () => number {
  let state = normalizeSeed(seed);

  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function normalizeSeed(seed: number): number {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error("seed must be an integer between 0 and 4294967295.");
  }

  return seed;
}

function createSeededGame(seed: number): GameState {
  return createInitialGame({ rng: createSeededRandom(seed) });
}

function toCanonicalSnapshot(state: GameState): CanonicalSnapshot {
  return {
    phase: state.phase,
    currentPlayerId: state.currentPlayerId,
    players: state.players.map((player) => ({
      id: player.id,
      handIds: player.hand.map((card) => card.id)
    })),
    currentTrick: state.currentTrick.map((played) => ({
      playerId: played.playerId,
      cardId: played.card.id
    })),
    completedTricks: state.completedTricks.map((trick) => ({
      trickNumber: trick.trickNumber,
      winnerId: trick.winnerId,
      cards: trick.cards.map((played) => ({
        playerId: played.playerId,
        cardId: played.card.id
      }))
    })),
    trumpSuit: state.trumpSuit,
    contract: state.contract,
    adjutant: state.adjutant,
    bidding: state.bidding,
    awardedPointCards: state.awardedPointCards.map((award) => ({
      playerId: award.playerId,
      cardIds: award.cards.map((card) => card.id)
    })),
    excludedCardIds: state.excludedCards.map((card) => card.id),
    unusedCardIds: state.unusedCards.map((card) => card.id),
    latestEvent:
      state.latestEvent === null
        ? null
        : {
            type: state.latestEvent.type,
            napoleonPlayerId: state.latestEvent.napoleonPlayerId,
            awardedPointCardIds: state.latestEvent.awardedPointCards.map((card) => card.id),
            hiddenNonPointCardCount: state.latestEvent.hiddenNonPointCardCount
          },
    result: state.result,
    trickNumber: state.trickNumber,
    isTrickComplete: state.isTrickComplete,
    isGameOver: state.isGameOver
  };
}

function applyScriptToTs(seed: number, script: readonly string[]): GameState {
  let state = createSeededGame(seed);

  for (const line of script) {
    const [type, ...parts] = line.split(" ");
    switch (type) {
      case "bid":
        state = applyAction(state, {
          type: "bid",
          playerId: parts[0],
          suit: parts[1] as Suit,
          targetPointCards: Number(parts[2])
        });
        break;
      case "pass":
        state = applyAction(state, { type: "pass", playerId: parts[0] });
        break;
      case "choose-adjutant":
        state = applyAction(state, {
          type: "choose-adjutant",
          playerId: parts[0],
          cardId: parts[1]
        });
        break;
      case "discard":
        state = applyAction(state, {
          type: "discard-cards",
          playerId: parts[0],
          cardIds: parts.slice(1)
        });
        break;
      case "play":
        state = applyAction(state, {
          type: "play-card",
          playerId: parts[0],
          cardId: parts[1]
        });
        break;
      case "next-trick":
        state = advanceToNextTrick(state);
        break;
      default:
        throw new Error(`Unknown script line: ${line}`);
    }
  }

  return state;
}

function writeDifferentialCase(name: string, seed: number, script: readonly string[]): string {
  const tsSnapshot = toCanonicalSnapshot(applyScriptToTs(seed, script));
  const slug = name.replaceAll(/[^a-zA-Z0-9]+/g, "-").replaceAll(/^-|-$/g, "").toLowerCase();
  const caseDir = resolve(".differential", slug);

  mkdirSync(caseDir, { recursive: true });
  writeFileSync(resolve(caseDir, "actions.txt"), `${script.join("\n")}\n`);
  writeFileSync(resolve(caseDir, "expected.json"), `${JSON.stringify(tsSnapshot)}\n`);
  writeFileSync(resolve(caseDir, "seed.txt"), `${seed}\n`);

  return slug;
}

function allPassBiddingScript(): string[] {
  return ["pass player-0", "pass player-1", "pass player-2", "pass player-3", "pass player-4"];
}

function makeReadyToPlayScript(seed: number): string[] {
  const script = allPassBiddingScript();
  let state = applyScriptToTs(seed, script);

  script.push(`choose-adjutant ${state.currentPlayerId} joker`);
  state = applyScriptToTs(seed, script);

  const napoleon = state.players.find((player) => player.id === state.currentPlayerId);
  if (napoleon === undefined) {
    throw new Error("Napoleon player not found.");
  }

  script.push(`discard ${state.currentPlayerId} ${napoleon.hand.slice(0, 3).map(cardId).join(" ")}`);
  return script;
}

function makeFirstLegalPlayScript(seed: number, maxPlayedCards: number): string[] {
  const script = makeReadyToPlayScript(seed);
  let state = applyScriptToTs(seed, script);
  let playedCards = 0;

  while (!state.isGameOver && playedCards < maxPlayedCards) {
    if (state.isTrickComplete) {
      script.push("next-trick");
      state = advanceToNextTrick(state);
      continue;
    }

    const action = getLegalActions(state, state.currentPlayerId).find(
      (candidate): candidate is Extract<GameAction, { type: "play-card" }> =>
        candidate.type === "play-card"
    );
    if (action === undefined) {
      throw new Error("Expected a legal play-card action.");
    }

    script.push(`play ${action.playerId} ${action.cardId}`);
    state = applyAction(state, action);
    playedCards += 1;
  }

  return script;
}

function cardId(card: Card): string {
  return card.id;
}

const cases = [
  writeDifferentialCase("initial seed 0", 0, []),
  writeDifferentialCase("initial seed 1", 1, []),
  writeDifferentialCase("initial max uint32 seed", 0xffffffff, []),
  writeDifferentialCase("exchange snapshot", 424242, makeReadyToPlayScript(424242)),
  writeDifferentialCase("playing snapshot", 98765, makeFirstLegalPlayScript(98765, 20)),
  writeDifferentialCase("terminal snapshot", 13579, makeFirstLegalPlayScript(13579, 50))
];

writeFileSync(resolve(".differential", "cases.txt"), `${cases.join("\n")}\n`);

for (const slug of cases) {
  assert.match(slug, /^[a-z0-9-]+$/);
}

console.log(`Wrote ${cases.length} TypeScript oracle differential cases.`);
