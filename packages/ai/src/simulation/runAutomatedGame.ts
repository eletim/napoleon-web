import {
  advanceToNextTrick,
  applyAction,
  createInitialGame,
  createPlayerView,
  getLegalActions
} from "@napoleon/game-core";
import type {
  Card,
  DiscardCardsAction,
  GameAction,
  GameState,
  PlayerId
} from "@napoleon/game-core";
import type { Agent, PlayerObservation } from "../types.js";
import { createSeededRandom, deriveSeed, normalizeSeed } from "./seededRandom.js";
import type {
  ActualCardState,
  ActualHands,
  AutomatedGameRecord,
  RunAutomatedGameOptions
} from "./types.js";

const defaultPlayerIds: readonly PlayerId[] = [
  "player-0",
  "player-1",
  "player-2",
  "player-3",
  "player-4"
];
const defaultMaxDecisionSteps = 1000;

export async function runAutomatedGame(
  options: RunAutomatedGameOptions
): Promise<AutomatedGameRecord> {
  const seed = normalizeSeed(options.seed);
  const playerIds = options.playerIds ?? defaultPlayerIds;
  let state = createInitialGame({
    playerIds,
    rng: createSeededRandom(deriveSeed(seed, "game"))
  });
  const agents = createAgents(options, seed, playerIds);
  const initialHands = captureHands(state);
  const initialActualState = captureActualCardState(state);
  const decisions: GameActionDecision[] = [];
  const maxDecisionSteps = options.maxDecisionSteps ?? defaultMaxDecisionSteps;

  while (!state.isGameOver) {
    if (state.isTrickComplete) {
      state = advanceToNextTrick(state);
      continue;
    }

    if (decisions.length >= maxDecisionSteps) {
      throw new Error(`Automated game exceeded ${maxDecisionSteps} decision steps.`);
    }

    const playerId = state.currentPlayerId;
    const agent = agents.get(playerId);

    if (agent === undefined) {
      throw new Error(`No automated agent was created for ${playerId}.`);
    }

    const view = createPlayerView(state, playerId);
    const legalActions = getAutomatedLegalActions(state, playerId);
    const observation: PlayerObservation = {
      playerId,
      view,
      legalActions
    };
    const action = await agent.selectAction(observation);

    if (!legalActions.some((legalAction) => actionsEqual(action, legalAction))) {
      throw new Error(
        [
          "Automated agent selected an illegal action.",
          `playerId=${playerId}`,
          `phase=${state.phase}`,
          `action=${JSON.stringify(action)}`,
          `legalActions=${JSON.stringify(legalActions)}`
        ].join(" ")
      );
    }

    decisions.push({
      step: decisions.length + 1,
      playerId,
      phase: state.phase,
      trickNumber: state.trickNumber,
      observation,
      legalActions,
      action,
      actualHands: captureHands(state),
      actualState: captureActualCardState(state),
      handCounts: captureHandCounts(state)
    });
    state = applyAction(state, action);
  }

  if (state.result === null) {
    throw new Error("Automated game finished without a result.");
  }

  return {
    schemaVersion: 1,
    seed,
    playerIds: [...playerIds],
    initialHands,
    initialActualState,
    decisions,
    result: state.result
  };
}

interface GameActionDecision {
  step: number;
  playerId: PlayerId;
  phase: GameState["phase"];
  trickNumber: number;
  observation: PlayerObservation;
  legalActions: readonly GameAction[];
  action: GameAction;
  actualHands: ActualHands;
  actualState: ActualCardState;
  handCounts: Readonly<Record<PlayerId, number>>;
}

function createAgents(
  options: RunAutomatedGameOptions,
  seed: number,
  playerIds: readonly PlayerId[]
): ReadonlyMap<PlayerId, Agent> {
  return new Map(
    playerIds.map((playerId, playerIndex) => [
      playerId,
      options.createAgent({
        playerId,
        playerIndex,
        rng: createSeededRandom(deriveSeed(seed, `agent:${playerId}`))
      })
    ])
  );
}

function getAutomatedLegalActions(state: GameState, playerId: PlayerId): readonly GameAction[] {
  if (state.phase !== "exchanging") {
    return getLegalActions(state, playerId);
  }

  const view = createPlayerView(state, playerId);
  const discardCount = view.exchangeRequirement?.discardCount;
  const self = view.players.find((player) => player.id === playerId);

  if (discardCount === undefined || self?.hand === undefined) {
    return [];
  }

  return getCardCombinations(self.hand, discardCount).map(
    (cards): DiscardCardsAction => ({
      type: "discard-cards",
      playerId,
      cardIds: cards.map((card) => card.id)
    })
  );
}

function getCardCombinations(cards: readonly Card[], count: number): readonly (readonly Card[])[] {
  if (count === 0) {
    return [[]];
  }

  if (cards.length < count) {
    return [];
  }

  const combinations: (readonly Card[])[] = [];

  for (let index = 0; index <= cards.length - count; index += 1) {
    const head = cards[index];
    const tails = getCardCombinations(cards.slice(index + 1), count - 1);

    for (const tail of tails) {
      combinations.push([head, ...tail]);
    }
  }

  return combinations;
}

function captureHands(state: GameState): ActualHands {
  return Object.fromEntries(
    state.players.map((player) => [player.id, player.hand.map((card) => card.id)])
  );
}

function captureHandCounts(state: GameState): Readonly<Record<PlayerId, number>> {
  return Object.fromEntries(
    state.players.map((player) => [player.id, player.hand.length])
  );
}

function captureActualCardState(state: GameState): ActualCardState {
  return {
    hands: captureHands(state),
    unusedCardIds: state.unusedCards.map((card) => card.id),
    excludedCardIds: state.excludedCards.map((card) => card.id),
    awardedPointCardIds: Object.fromEntries(
      state.players.map((player) => [
        player.id,
        state.awardedPointCards
          .filter((award) => award.playerId === player.id)
          .flatMap((award) => award.cards.map((card) => card.id))
      ])
    ),
    currentTrickCardIds: state.currentTrick.map((playedCard) => playedCard.card.id),
    completedTrickCardIds: state.completedTricks.flatMap((trick) =>
      trick.cards.map((playedCard) => playedCard.card.id)
    )
  };
}

function actionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }

  switch (left.type) {
    case "bid":
      return right.type === "bid" &&
        left.suit === right.suit &&
        left.targetPointCards === right.targetPointCards;
    case "pass":
      return right.type === "pass";
    case "choose-adjutant":
      return right.type === "choose-adjutant" && left.cardId === right.cardId;
    case "discard-cards":
      return right.type === "discard-cards" && sameCardIds(left.cardIds, right.cardIds);
    case "play-card":
      return right.type === "play-card" && left.cardId === right.cardId;
  }
}

function sameCardIds(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  const rightIds = new Set(right);
  return left.every((cardId) => rightIds.has(cardId));
}
