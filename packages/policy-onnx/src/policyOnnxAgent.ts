import { RuleBasedAgent } from "@napoleon/ai";
import type { ActualCardState, Agent, PlayerObservation } from "@napoleon/ai";
import type { GameAction, PlayerId } from "@napoleon/game-core";
import {
  createAdjutantModelInput,
  createBiddingModelInput,
  createCompleteInfoPlayingModelInput,
  createExchangeModelInput,
  createPlayingModelInput,
  createRelativePlayerOrder,
  decodeAdjutantAction,
  decodeBiddingAction,
  encodeBiddingHistoryFromPublicActions,
  encodeAdjutantObservation,
  encodeBiddingObservation,
  encodeCompleteInfoPlayingObservation,
  encodeExchangeObservation,
  encodePlayingObservation,
  getCardId,
  validateEncodedExchangeAction
} from "@napoleon/ai-observation";
import type { EncodedExchangeObservation } from "@napoleon/ai-observation";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import {
  COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT,
  getNonPlayingPolicyOnnxSpec
} from "./policySpecs.js";
import { EXCHANGE_DECISION_MODE_SEQUENTIAL_CARD } from "./constants.js";
import type { NonPlayingPolicyOnnxModel, PolicyOnnxModel } from "./policyOnnx.js";
import type { NonPlayingPolicyType } from "./types.js";

export interface PolicyOnnxAgentOptions {
  policy: PolicyOnnxModel;
  biddingPolicy?: NonPlayingPolicyOnnxModel;
  adjutantPolicy?: NonPlayingPolicyOnnxModel;
  exchangePolicy?: NonPlayingPolicyOnnxModel;
  rng?: () => number;
  playerIds?: readonly PlayerId[];
  decisionMetrics?: PolicyOnnxAgentDecisionMetrics;
}

export interface PolicyOnnxAgentDecisionMetrics {
  biddingOnnxCallCount: number;
  adjutantOnnxCallCount: number;
  exchangeOnnxCallCount: number;
  playingOnnxCallCount: number;
  ruleBasedFallbackDecisionCount: number;
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

export interface PolicyOnnxCompleteInfoPlayInputContext {
  actualState: ActualCardState;
  playerIds?: readonly PlayerId[];
}

export class PolicyOnnxAgent implements Agent {
  private readonly policy: PolicyOnnxModel;
  private readonly biddingPolicy: NonPlayingPolicyOnnxModel | null;
  private readonly adjutantPolicy: NonPlayingPolicyOnnxModel | null;
  private readonly exchangePolicy: NonPlayingPolicyOnnxModel | null;
  private readonly ruleBasedAgent: RuleBasedAgent;
  private readonly playerIds: readonly PlayerId[] | null;
  private readonly decisionMetrics: PolicyOnnxAgentDecisionMetrics | null;

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
    this.decisionMetrics = options.decisionMetrics ?? null;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "bidding":
        if (this.biddingPolicy === null) {
          this.incrementRuleBasedFallbackDecisionCount();
          return this.ruleBasedAgent.selectAction(observation);
        }
        this.incrementOnnxCallCount("bidding");
        return this.selectBiddingAction(observation, this.biddingPolicy);
      case "choosing-adjutant":
        if (this.adjutantPolicy === null) {
          this.incrementRuleBasedFallbackDecisionCount();
          return this.ruleBasedAgent.selectAction(observation);
        }
        this.incrementOnnxCallCount("adjutant");
        return this.selectAdjutantAction(observation, this.adjutantPolicy);
      case "exchanging":
        if (this.exchangePolicy === null) {
          this.incrementRuleBasedFallbackDecisionCount();
          return this.ruleBasedAgent.selectAction(observation);
        }
        this.incrementOnnxCallCount("exchange");
        return this.selectExchangeAction(observation, this.exchangePolicy);
      case "playing":
        this.incrementOnnxCallCount("playing");
        return this.selectPlayAction(observation, context);
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private incrementOnnxCallCount(
    phase: "bidding" | "adjutant" | "exchange" | "playing"
  ): void {
    if (this.decisionMetrics === null) {
      return;
    }

    switch (phase) {
      case "bidding":
        this.decisionMetrics.biddingOnnxCallCount += 1;
        break;
      case "adjutant":
        this.decisionMetrics.adjutantOnnxCallCount += 1;
        break;
      case "exchange":
        this.decisionMetrics.exchangeOnnxCallCount += 1;
        break;
      case "playing":
        this.decisionMetrics.playingOnnxCallCount += 1;
        break;
    }
  }

  private incrementRuleBasedFallbackDecisionCount(): void {
    if (this.decisionMetrics !== null) {
      this.decisionMetrics.ruleBasedFallbackDecisionCount += 1;
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
    if (exchangePolicy.metadata.decisionMode === EXCHANGE_DECISION_MODE_SEQUENTIAL_CARD) {
      return this.selectSequentialExchangeAction(observation, exchangePolicy);
    }

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

  private async selectSequentialExchangeAction(
    observation: PlayerObservation,
    exchangePolicy: NonPlayingPolicyOnnxModel
  ): Promise<GameAction> {
    const selectedCardIndices: number[] = [];

    for (let exchangeStepIndex = 0; exchangeStepIndex < 3; exchangeStepIndex += 1) {
      const { modelInput, legalDiscardCardMask } = createPolicyOnnxSequentialExchangeInput(
        observation,
        selectedCardIndices,
        this.playerIds ?? undefined
      );
      const selection = await exchangePolicy.selectExchangeCard({
        modelInput,
        legalDiscardMask: legalDiscardCardMask
      });

      if (legalDiscardCardMask[selection.selectedIndex] !== 1) {
        throw new PolicyOnnxCompatibilityError(
          `ONNX sequential exchange policy selected illegal card index ${selection.selectedIndex}.`
        );
      }
      selectedCardIndices.push(selection.selectedIndex);
      if (exchangeStepIndex > 0) {
        this.incrementOnnxCallCount("exchange");
      }
    }

    const selectedCardIds = selectedCardIndices.map((index) => getCardId(index));
    const selectedAction: GameAction = {
      type: "discard-cards",
      playerId: observation.playerId,
      cardIds: selectedCardIds
    };
    const initialInput = createPolicyOnnxSequentialExchangeInput(
      observation,
      [],
      this.playerIds ?? undefined
    );
    validateEncodedExchangeAction(
      {
        discardTargetMask: initialInput.legalDiscardCardMask.map((_, index) =>
          selectedCardIndices.includes(index) ? 1 : 0
        )
      },
      initialInput.legalDiscardCardMask
    );

    return selectedAction;
  }

  private async selectPlayAction(
    observation: PlayerObservation,
    context: { actualState: ActualCardState; playerIds: readonly PlayerId[] } | undefined
  ): Promise<GameAction> {
    const { modelInput, legalPlayMask } =
      this.policy.metadata?.playingObservationVariant === COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT
        ? createPolicyOnnxCompleteInfoPlayInput(observation, {
            actualState: requireCompleteInfoContext(context),
            playerIds: this.playerIds ?? context?.playerIds
          })
        : createPolicyOnnxPlayInput(
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

export function createPolicyOnnxSequentialExchangeInput(
  observation: PlayerObservation,
  selectedCardIndices: readonly number[],
  playerIds?: readonly PlayerId[]
): PolicyOnnxExchangeInput {
  if (selectedCardIndices.length > 2) {
    throw new PolicyOnnxCompatibilityError("selectedCardIndices must contain at most 2 partial exchange cards.");
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
  const steppedObservation = createSequentialExchangeObservation(
    encodedObservation,
    selectedCardIndices
  );

  return createExchangeModelInput(steppedObservation);
}

function createSequentialExchangeObservation(
  observation: EncodedExchangeObservation,
  selectedCardIndices: readonly number[]
): EncodedExchangeObservation {
  const seen = new Set<number>();
  const partialDiscardMask = Array(observation.selfHandMask.length).fill(0);
  const legalDiscardCardMask = observation.selfHandMask.map((value) => value);

  for (const index of selectedCardIndices) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= observation.selfHandMask.length) {
      throw new PolicyOnnxCompatibilityError(`partial exchange card index is invalid: ${index}.`);
    }
    if (seen.has(index)) {
      throw new PolicyOnnxCompatibilityError(`partial exchange card index is duplicated: ${index}.`);
    }
    if (observation.selfHandMask[index] !== 1) {
      throw new PolicyOnnxCompatibilityError(`partial exchange card index is not in hand: ${index}.`);
    }
    seen.add(index);
    partialDiscardMask[index] = 1;
    legalDiscardCardMask[index] = 0;
  }

  return {
    ...observation,
    partialDiscardMask,
    legalDiscardCardMask,
    exchangeStepIndex: selectedCardIndices.length,
    remainingDiscardCount: 3 - selectedCardIndices.length
  };
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

export function createPolicyOnnxCompleteInfoPlayInput(
  observation: PlayerObservation,
  context: PolicyOnnxCompleteInfoPlayInputContext
): PolicyOnnxPlayInput {
  if (observation.view.phase !== "playing") {
    throw new PolicyOnnxCompatibilityError(
      `createPolicyOnnxCompleteInfoPlayInput requires a playing observation, got ${observation.view.phase}.`
    );
  }

  const absolutePlayerIds = getValidatedPlayerIds(observation, context.playerIds);
  const encodedObservation = encodeCompleteInfoPlayingObservation(
    observation,
    context.actualState,
    absolutePlayerIds
  );

  return createCompleteInfoPlayingModelInput(encodedObservation);
}

export function createPolicyOnnxAgentDecisionMetrics(): PolicyOnnxAgentDecisionMetrics {
  return {
    biddingOnnxCallCount: 0,
    adjutantOnnxCallCount: 0,
    exchangeOnnxCallCount: 0,
    playingOnnxCallCount: 0,
    ruleBasedFallbackDecisionCount: 0
  };
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

function requireCompleteInfoContext(
  context: { actualState: ActualCardState; playerIds: readonly PlayerId[] } | undefined
): ActualCardState {
  if (context === undefined) {
    throw new PolicyOnnxCompatibilityError(
      "Complete-info compact PolicyOnnxAgent requires runtime actual state."
    );
  }

  return context.actualState;
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
