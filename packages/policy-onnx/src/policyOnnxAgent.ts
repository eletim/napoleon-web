import { RuleBasedAgent } from "@napoleon/ai";
import type { Agent, PlayerObservation } from "@napoleon/ai";
import type { GameAction, PlayerId } from "@napoleon/game-core";
import {
  createAdjutantModelInput,
  createBiddingModelInput,
  createExchangeModelInput,
  createPlayingModelInput,
  createRelativePlayerOrder,
  decodeAdjutantAction,
  decodeBiddingAction,
  encodeBiddingHistoryFromPublicActions,
  encodeAdjutantObservation,
  encodeBiddingObservation,
  encodeExchangeObservation,
  encodePlayingObservation,
  getCardId,
  validateEncodedExchangeAction
} from "@napoleon/ai-observation";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import { getNonPlayingPolicyOnnxSpec } from "./policySpecs.js";
import type { NonPlayingPolicyOnnxModel, PolicyOnnxModel } from "./policyOnnx.js";
import type { NonPlayingPolicyType } from "./types.js";

export interface PolicyOnnxAgentOptions {
  policy: PolicyOnnxModel;
  biddingPolicy?: NonPlayingPolicyOnnxModel;
  adjutantPolicy?: NonPlayingPolicyOnnxModel;
  exchangePolicy?: NonPlayingPolicyOnnxModel;
  rng?: () => number;
  playerIds?: readonly PlayerId[];
}

export interface PolicyOnnxBiddingInput {
  modelInput: Float32Array;
  legalBidMask: readonly number[];
}

export interface PolicyOnnxAdjutantInput {
  modelInput: Float32Array;
  legalAdjutantMask: readonly number[];
}

export interface PolicyOnnxExchangeInput {
  modelInput: Float32Array;
  legalDiscardCardMask: readonly number[];
}

export interface PolicyOnnxPlayInput {
  modelInput: Float32Array;
  legalPlayMask: readonly number[];
}

export class PolicyOnnxAgent implements Agent {
  private readonly policy: PolicyOnnxModel;
  private readonly biddingPolicy: NonPlayingPolicyOnnxModel | null;
  private readonly adjutantPolicy: NonPlayingPolicyOnnxModel | null;
  private readonly exchangePolicy: NonPlayingPolicyOnnxModel | null;
  private readonly ruleBasedAgent: RuleBasedAgent;
  private readonly playerIds: readonly PlayerId[] | null;

  constructor(options: PolicyOnnxAgentOptions) {
    this.policy = options.policy;
    this.biddingPolicy = options.biddingPolicy ?? null;
    this.adjutantPolicy = options.adjutantPolicy ?? null;
    this.exchangePolicy = options.exchangePolicy ?? null;
    assertNonPlayingPolicyType("biddingPolicy", this.biddingPolicy, "bidding");
    assertNonPlayingPolicyType("adjutantPolicy", this.adjutantPolicy, "adjutant");
    assertNonPlayingPolicyType("exchangePolicy", this.exchangePolicy, "exchange");
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
    this.playerIds = options.playerIds ?? null;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    switch (observation.view.phase) {
      case "bidding":
        return this.biddingPolicy === null
          ? this.ruleBasedAgent.selectAction(observation)
          : this.selectBiddingAction(observation, this.biddingPolicy);
      case "choosing-adjutant":
        return this.adjutantPolicy === null
          ? this.ruleBasedAgent.selectAction(observation)
          : this.selectAdjutantAction(observation, this.adjutantPolicy);
      case "exchanging":
        return this.exchangePolicy === null
          ? this.ruleBasedAgent.selectAction(observation)
          : this.selectExchangeAction(observation, this.exchangePolicy);
      case "playing":
        return this.selectPlayAction(observation);
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectBiddingAction(
    observation: PlayerObservation,
    biddingPolicy: NonPlayingPolicyOnnxModel
  ): Promise<GameAction> {
    const { modelInput, legalBidMask } = createPolicyOnnxBiddingInput(
      observation,
      this.playerIds ?? undefined
    );
    const selection = await biddingPolicy.selectBidding({ modelInput, legalBidMask });
    const selectedAction = decodeBiddingAction(selection.selectedIndex, observation.playerId);
    const legalAction = observation.legalActions.find((action) =>
      actionsEqual(action, selectedAction)
    );

    if (legalAction === undefined) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX bidding policy selected action index ${selection.selectedIndex} outside legal actions.`
      );
    }

    return legalAction;
  }

  private async selectAdjutantAction(
    observation: PlayerObservation,
    adjutantPolicy: NonPlayingPolicyOnnxModel
  ): Promise<GameAction> {
    const { modelInput, legalAdjutantMask } = createPolicyOnnxAdjutantInput(
      observation,
      this.playerIds ?? undefined
    );
    const selection = await adjutantPolicy.selectAdjutant({ modelInput, legalAdjutantMask });
    const selectedAction = decodeAdjutantAction(selection.selectedIndex, observation.playerId);
    const legalAction = observation.legalActions.find((action) =>
      actionsEqual(action, selectedAction)
    );

    if (legalAction === undefined) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX adjutant policy selected card index ${selection.selectedIndex} (${selectedAction.cardId}) outside legal actions.`
      );
    }

    return legalAction;
  }

  private async selectExchangeAction(
    observation: PlayerObservation,
    exchangePolicy: NonPlayingPolicyOnnxModel
  ): Promise<GameAction> {
    const { modelInput, legalDiscardCardMask } = createPolicyOnnxExchangeInput(
      observation,
      this.playerIds ?? undefined
    );
    const selection = await exchangePolicy.selectExchange({
      modelInput,
      legalDiscardMask: legalDiscardCardMask
    });
    const selectedCardIds = selection.selectedCardIndices.map((index) => getCardId(index));
    const selectedAction: GameAction = {
      type: "discard-cards",
      playerId: observation.playerId,
      cardIds: selectedCardIds
    };
    validateEncodedExchangeAction(
      {
        discardTargetMask: legalDiscardCardMask.map((_, index) =>
          selection.selectedCardIndices.includes(index) ? 1 : 0
        )
      },
      legalDiscardCardMask
    );

    return selectedAction;
  }

  private async selectPlayAction(observation: PlayerObservation): Promise<GameAction> {
    const { modelInput, legalPlayMask } = createPolicyOnnxPlayInput(
      observation,
      this.playerIds ?? undefined
    );
    const selection = await this.policy.selectLegalPlay({ modelInput, legalPlayMask });
    const selectedCardId = getCardId(selection.selectedCardIndex);
    const selectedAction = observation.legalActions.find(
      (action) => action.type === "play-card" && action.cardId === selectedCardId
    );

    if (selectedAction === undefined) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX policy selected card index ${selection.selectedCardIndex} (${selectedCardId}) outside legal actions.`
      );
    }

    return selectedAction;
  }
}

export function createPolicyOnnxBiddingInput(
  observation: PlayerObservation,
  playerIds?: readonly PlayerId[]
): PolicyOnnxBiddingInput {
  if (observation.view.phase !== "bidding") {
    throw new PolicyOnnxCompatibilityError(
      `createPolicyOnnxBiddingInput requires a bidding observation, got ${observation.view.phase}.`
    );
  }

  requirePublicActionHistory(observation);
  const absolutePlayerIds = getValidatedPlayerIds(observation, playerIds);
  const encodedObservation = encodeBiddingObservation(observation, absolutePlayerIds);

  return createBiddingModelInput(encodedObservation);
}

export function createPolicyOnnxAdjutantInput(
  observation: PlayerObservation,
  playerIds?: readonly PlayerId[]
): PolicyOnnxAdjutantInput {
  if (observation.view.phase !== "choosing-adjutant") {
    throw new PolicyOnnxCompatibilityError(
      `createPolicyOnnxAdjutantInput requires a choosing-adjutant observation, got ${observation.view.phase}.`
    );
  }

  const { absolutePlayerIds, biddingHistory } = createNonPlayingObservationContext(
    observation,
    playerIds
  );
  const encodedObservation = encodeAdjutantObservation(
    observation,
    absolutePlayerIds,
    biddingHistory
  );

  return createAdjutantModelInput(encodedObservation);
}

export function createPolicyOnnxExchangeInput(
  observation: PlayerObservation,
  playerIds?: readonly PlayerId[]
): PolicyOnnxExchangeInput {
  if (observation.view.phase !== "exchanging") {
    throw new PolicyOnnxCompatibilityError(
      `createPolicyOnnxExchangeInput requires an exchanging observation, got ${observation.view.phase}.`
    );
  }

  const { absolutePlayerIds, biddingHistory } = createNonPlayingObservationContext(
    observation,
    playerIds
  );
  const encodedObservation = encodeExchangeObservation(
    observation,
    absolutePlayerIds,
    biddingHistory
  );

  return createExchangeModelInput(encodedObservation);
}

export function createPolicyOnnxPlayInput(
  observation: PlayerObservation,
  playerIds?: readonly PlayerId[]
): PolicyOnnxPlayInput {
  if (observation.view.phase !== "playing") {
    throw new PolicyOnnxCompatibilityError(
      `createPolicyOnnxPlayInput requires a playing observation, got ${observation.view.phase}.`
    );
  }

  const { absolutePlayerIds, biddingHistory } = createNonPlayingObservationContext(
    observation,
    playerIds
  );
  const encodedObservation = encodePlayingObservation(
    observation,
    absolutePlayerIds,
    biddingHistory
  );

  return createPlayingModelInput(encodedObservation);
}

function createNonPlayingObservationContext(
  observation: PlayerObservation,
  playerIds?: readonly PlayerId[]
): {
  absolutePlayerIds: readonly PlayerId[];
  biddingHistory: ReturnType<typeof encodeBiddingHistoryFromPublicActions>;
} {
  const publicActionHistory = requirePublicActionHistory(observation);
  const absolutePlayerIds = getValidatedPlayerIds(observation, playerIds);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    publicActionHistory,
    relativePlayerIds
  );

  return { absolutePlayerIds, biddingHistory };
}

function getValidatedPlayerIds(
  observation: PlayerObservation,
  playerIds?: readonly PlayerId[]
): readonly PlayerId[] {
  const absolutePlayerIds = playerIds ?? observation.view.players.map((player) => player.id);
  validatePlayerIdsForObservation(absolutePlayerIds, observation);

  return absolutePlayerIds;
}

function requirePublicActionHistory(
  observation: PlayerObservation
): NonNullable<PlayerObservation["publicActionHistory"]> {
  const publicActionHistory = observation.publicActionHistory;
  if (publicActionHistory === undefined) {
    throw new PolicyOnnxCompatibilityError(
      "PolicyOnnxAgent requires PlayerObservation.publicActionHistory to build biddingHistory."
    );
  }

  return publicActionHistory;
}

function validatePlayerIdsForObservation(
  playerIds: readonly PlayerId[],
  observation: PlayerObservation
): void {
  const observedPlayerIds = observation.view.players.map((player) => player.id);

  if (playerIds.length !== observedPlayerIds.length) {
    throw new PolicyOnnxCompatibilityError(
      `PolicyOnnxAgent playerIds length mismatch: expected ${observedPlayerIds.length}, got ${playerIds.length}.`
    );
  }

  const observedSet = new Set(observedPlayerIds);
  const configuredSet = new Set(playerIds);
  if (
    observedSet.size !== observedPlayerIds.length ||
    configuredSet.size !== playerIds.length ||
    observedPlayerIds.some((playerId) => !configuredSet.has(playerId)) ||
    playerIds.some((playerId) => !observedSet.has(playerId))
  ) {
    throw new PolicyOnnxCompatibilityError(
      "PolicyOnnxAgent playerIds must contain exactly the same players as the observation."
    );
  }

  if (!playerIds.every((playerId, index) => playerId === observedPlayerIds[index])) {
    throw new PolicyOnnxCompatibilityError(
      "PolicyOnnxAgent playerIds must match the observation player order."
    );
  }
}

function assertNonPlayingPolicyType(
  optionName: string,
  policy: NonPlayingPolicyOnnxModel | null,
  expectedPolicyType: NonPlayingPolicyType
): void {
  if (policy === null) {
    return;
  }

  if (policy.policyType !== expectedPolicyType) {
    throw new PolicyOnnxCompatibilityError(
      `${optionName} policy type mismatch: expected ${expectedPolicyType}, got ${policy.policyType}.`
    );
  }

  const expectedArtifactType = getNonPlayingPolicyOnnxSpec(expectedPolicyType).artifactType;
  if (policy.metadata.artifactType !== expectedArtifactType) {
    throw new PolicyOnnxCompatibilityError(
      `${optionName} artifact type mismatch: expected ${expectedArtifactType}, got ${policy.metadata.artifactType}.`
    );
  }
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

  const sortedLeft = [...left].sort();
  const sortedRight = [...right].sort();

  return sortedLeft.every((cardId, index) => cardId === sortedRight[index]);
}
