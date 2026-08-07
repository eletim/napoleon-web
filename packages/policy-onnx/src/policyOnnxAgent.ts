import { RuleBasedAgent } from "@napoleon/ai";
import type { Agent, PlayerObservation } from "@napoleon/ai";
import type { GameAction, PlayerId } from "@napoleon/game-core";
import {
  createPlayingModelInput,
  createRelativePlayerOrder,
  encodeBiddingHistoryFromPublicActions,
  encodePlayingObservation,
  getCardId
} from "@napoleon/ai-observation";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import type { PolicyOnnxModel } from "./policyOnnx.js";

export interface PolicyOnnxAgentOptions {
  policy: PolicyOnnxModel;
  rng?: () => number;
  playerIds?: readonly PlayerId[];
}

export interface PolicyOnnxPlayInput {
  modelInput: Float32Array;
  legalPlayMask: readonly number[];
}

export class PolicyOnnxAgent implements Agent {
  private readonly policy: PolicyOnnxModel;
  private readonly ruleBasedAgent: RuleBasedAgent;
  private readonly playerIds: readonly PlayerId[] | null;

  constructor(options: PolicyOnnxAgentOptions) {
    this.policy = options.policy;
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
    this.playerIds = options.playerIds ?? null;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase !== "playing") {
      return this.ruleBasedAgent.selectAction(observation);
    }

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

export function createPolicyOnnxPlayInput(
  observation: PlayerObservation,
  playerIds?: readonly PlayerId[]
): PolicyOnnxPlayInput {
  if (observation.view.phase !== "playing") {
    throw new PolicyOnnxCompatibilityError(
      `createPolicyOnnxPlayInput requires a playing observation, got ${observation.view.phase}.`
    );
  }

  const publicActionHistory = observation.publicActionHistory;
  if (publicActionHistory === undefined) {
    throw new PolicyOnnxCompatibilityError(
      "PolicyOnnxAgent requires PlayerObservation.publicActionHistory to build biddingHistory."
    );
  }

  const absolutePlayerIds = playerIds ?? observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    publicActionHistory,
    null,
    relativePlayerIds
  );
  const encodedObservation = encodePlayingObservation(
    observation,
    absolutePlayerIds,
    biddingHistory
  );

  return createPlayingModelInput(encodedObservation);
}
