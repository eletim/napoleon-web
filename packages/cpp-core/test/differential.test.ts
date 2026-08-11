import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import assert from "node:assert/strict";
import {
  advanceToNextTrick,
  applyAction,
  createInitialGame,
  createPlayerView,
  getLegalActions,
  type Card,
  type GameAction,
  type GameState,
  type PlayerId,
  type Suit
} from "@napoleon/game-core";
import { selectPlayAction } from "@napoleon/ai";
import {
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_SCHEMA_VERSION,
  createPlayingModelInput,
  createRelativePlayerOrder,
  encodeBiddingHistoryFromPublicActions,
  encodePlayingObservation
} from "@napoleon/ai-observation";

interface CanonicalSnapshot {
  phase: string;
  currentPlayerId: string;
  players: readonly { id: string; handIds: readonly string[] }[];
  currentTrick: readonly { playerId: string; cardId: string }[];
  currentPlayerLegalActions: readonly CanonicalAction[];
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
  playingModelInput: CanonicalPlayingModelInput | null;
}

interface CanonicalPlayingModelInput {
  modelInputSchemaVersion: typeof MODEL_INPUT_SCHEMA_VERSION;
  modelInputFeatureCount: typeof MODEL_INPUT_FEATURE_COUNT;
  playerId: string;
  observation: ReturnType<typeof encodePlayingObservation>;
  legalPlayMask: readonly number[];
  modelInput: readonly number[];
}

interface PublicBiddingActionRecord {
  step: number;
  playerId: PlayerId;
  phase: "bidding";
  action:
    | { type: "bid"; playerId: PlayerId; suit: Suit; targetPointCards: number }
    | { type: "pass"; playerId: PlayerId };
}

type CanonicalAction =
  | { type: "bid"; playerId: string; suit: Suit; targetPointCards: number }
  | { type: "pass"; playerId: string }
  | { type: "choose-adjutant"; playerId: string; cardId: string }
  | { type: "discard-cards"; playerId: string; cardIds: readonly string[] }
  | { type: "play-card"; playerId: string; cardId: string };

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

function toCanonicalSnapshot(
  state: GameState,
  publicActionHistory: readonly PublicBiddingActionRecord[]
): CanonicalSnapshot {
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
    currentPlayerLegalActions: getLegalActions(state, state.currentPlayerId).map(toCanonicalAction),
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
    isGameOver: state.isGameOver,
    playingModelInput: toCanonicalPlayingModelInput(state, publicActionHistory)
  };
}

function toCanonicalPlayingModelInput(
  state: GameState,
  publicActionHistory: readonly PublicBiddingActionRecord[]
): CanonicalPlayingModelInput | null {
  if (state.phase !== "playing" || state.isTrickComplete) {
    return null;
  }

  const playerId = state.currentPlayerId;
  const absolutePlayerIds = state.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    publicActionHistory,
    relativePlayerIds
  );
  const observation = encodePlayingObservation(
    {
      playerId,
      view: createPlayerView(state, playerId),
      legalActions: getLegalActions(state, playerId),
      publicActionHistory
    },
    absolutePlayerIds,
    biddingHistory
  );
  const modelInput = createPlayingModelInput(observation);

  return {
    modelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    modelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
    playerId,
    observation,
    legalPlayMask: modelInput.legalPlayMask,
    modelInput: Array.from(modelInput.modelInput)
  };
}

function toCanonicalAction(action: GameAction): CanonicalAction {
  switch (action.type) {
    case "bid":
      return {
        type: action.type,
        playerId: action.playerId,
        suit: action.suit,
        targetPointCards: action.targetPointCards
      };
    case "pass":
      return {
        type: action.type,
        playerId: action.playerId
      };
    case "choose-adjutant":
      return {
        type: action.type,
        playerId: action.playerId,
        cardId: action.cardId
      };
    case "discard-cards":
      return {
        type: action.type,
        playerId: action.playerId,
        cardIds: action.cardIds
      };
    case "play-card":
      return {
        type: action.type,
        playerId: action.playerId,
        cardId: action.cardId
      };
  }
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

function publicBiddingHistoryFromScript(
  script: readonly string[]
): readonly PublicBiddingActionRecord[] {
  return script.flatMap((line, index): readonly PublicBiddingActionRecord[] => {
    const [type, ...parts] = line.split(" ");

    if (type === "bid") {
      return [{
        step: index + 1,
        playerId: parts[0],
        phase: "bidding",
        action: {
          type: "bid",
          playerId: parts[0],
          suit: parts[1] as Suit,
          targetPointCards: Number(parts[2])
        }
      }];
    }

    if (type === "pass") {
      return [{
        step: index + 1,
        playerId: parts[0],
        phase: "bidding",
        action: {
          type: "pass",
          playerId: parts[0]
        }
      }];
    }

    return [];
  });
}

function writeDifferentialCase(name: string, seed: number, script: readonly string[]): string {
  const tsState = applyScriptToTs(seed, script);
  const publicActionHistory = publicBiddingHistoryFromScript(script);
  const tsSnapshot = toCanonicalSnapshot(
    tsState,
    publicActionHistory
  );
  const slug = name.replaceAll(/[^a-zA-Z0-9]+/g, "-").replaceAll(/^-|-$/g, "").toLowerCase();
  const caseDir = resolve(".differential", slug);

  mkdirSync(caseDir, { recursive: true });
  writeFileSync(resolve(caseDir, "actions.txt"), `${script.join("\n")}\n`);
  writeFileSync(resolve(caseDir, "expected.json"), `${JSON.stringify(tsSnapshot)}\n`);
  writeFileSync(resolve(caseDir, "seed.txt"), `${seed}\n`);
  writeRuleBasedOracle(caseDir, tsState, script, seed);

  return slug;
}

function writeRuleBasedOracle(
  caseDir: string,
  state: GameState,
  script: readonly string[],
  seed: number
): void {
  const legalPlayActions = getLegalActions(state, state.currentPlayerId).filter(
    (action): action is Extract<GameAction, { type: "play-card" }> => action.type === "play-card"
  );

  if (state.phase !== "playing" || state.isTrickComplete || legalPlayActions.length === 0) {
    return;
  }

  const agentSeed = (seed ^ Math.imul(script.length + 1, 0x9e3779b9)) >>> 0;
  const legalActions = getLegalActions(state, state.currentPlayerId);
  const selected = selectPlayAction(
    state.currentPlayerId,
    createPlayerView(state, state.currentPlayerId),
    legalActions,
    createSeededRandom(agentSeed)
  );

  writeFileSync(resolve(caseDir, "rule_based_seed.txt"), `${agentSeed}\n`);
  writeFileSync(resolve(caseDir, "rule_based_expected.json"), `${JSON.stringify(selected)}\n`);
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

function makeRandomGameScript(seed: number, actionSeed: number): string[] {
  const script: string[] = [];
  let state = createSeededGame(seed);
  const rng = createSeededRandom(actionSeed);

  while (!state.isGameOver) {
    if (state.isTrickComplete) {
      script.push("next-trick");
      state = advanceToNextTrick(state);
      continue;
    }

    if (state.phase === "exchanging") {
      const napoleon = state.players.find((player) => player.id === state.currentPlayerId);

      if (napoleon === undefined) {
        throw new Error("Napoleon player not found.");
      }

      const cardIds = sampleDistinct(napoleon.hand.map(cardId), 3, rng);
      script.push(`discard ${state.currentPlayerId} ${cardIds.join(" ")}`);
      state = applyAction(state, {
        type: "discard-cards",
        playerId: state.currentPlayerId,
        cardIds
      });
      continue;
    }

    const actions = getLegalActions(state, state.currentPlayerId);

    if (actions.length === 0) {
      throw new Error(`Expected legal actions in phase ${state.phase}.`);
    }

    const action = actions[randomInt(actions.length, rng)];
    script.push(actionToLine(action));
    state = applyAction(state, action);
  }

  return script;
}

function sampleDistinct<T>(values: readonly T[], count: number, rng: () => number): T[] {
  const remaining = [...values];
  const result: T[] = [];

  for (let index = 0; index < count; index += 1) {
    const selected = randomInt(remaining.length, rng);
    result.push(remaining[selected]);
    remaining.splice(selected, 1);
  }

  return result;
}

function randomInt(exclusiveMax: number, rng: () => number): number {
  if (exclusiveMax <= 0) {
    throw new Error("exclusiveMax must be positive.");
  }

  return Math.floor(rng() * exclusiveMax);
}

function actionToLine(action: GameAction): string {
  switch (action.type) {
    case "bid":
      return `bid ${action.playerId} ${action.suit} ${action.targetPointCards}`;
    case "pass":
      return `pass ${action.playerId}`;
    case "choose-adjutant":
      return `choose-adjutant ${action.playerId} ${action.cardId}`;
    case "discard-cards":
      return `discard ${action.playerId} ${action.cardIds.join(" ")}`;
    case "play-card":
      return `play ${action.playerId} ${action.cardId}`;
  }
}

function writeRandomizedGameCases(index: number): string[] {
  const seed = 100000 + index * 7919;
  const actionSeed = 0x9e3779b9 ^ (index * 2654435761);
  const script = makeRandomGameScript(seed, actionSeed >>> 0);
  const checkpointLengths = new Set<number>();

  for (let length = 8; length < script.length; length += 8) {
    checkpointLengths.add(length);
  }
  checkpointLengths.add(script.length);

  return [...checkpointLengths].map((length) =>
    writeDifferentialCase(
      length === script.length
        ? `randomized full game ${index} terminal`
        : `randomized full game ${index} step ${length}`,
      seed,
      script.slice(0, length)
    )
  );
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
  writeDifferentialCase("terminal snapshot", 13579, makeFirstLegalPlayScript(13579, 50)),
  ...Array.from({ length: 32 }, (_, index) => writeRandomizedGameCases(index)).flat()
];

writeFileSync(resolve(".differential", "cases.txt"), `${cases.join("\n")}\n`);

for (const slug of cases) {
  assert.match(slug, /^[a-z0-9-]+$/);
}

console.log(`Wrote ${cases.length} TypeScript oracle differential cases.`);
