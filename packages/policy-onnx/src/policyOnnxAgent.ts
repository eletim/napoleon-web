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

    const publicActionHistory = observation.publicActionHistory;
    if (publicActionHistory === undefined) {
      throw new PolicyOnnxCompatibilityError(
        "PolicyOnnxAgent requires PlayerObservation.publicActionHistory to build biddingHistory."
      );
    }

    const playerIds = this.playerIds ?? observation.view.players.map((player) => player.id);
    const relativePlayerIds = createRelativePlayerOrder(playerIds, observation.playerId);
    const biddingHistory = encodeBiddingHistoryFromPublicActions(
      publicActionHistory,
      Number.POSITIVE_INFINITY,
      relativePlayerIds
    );
    const encodedObservation = encodePlayingObservation(
      observation,
      playerIds,
      biddingHistory
    );
    const { modelInput, legalPlayMask } = createPlayingModelInput(encodedObservation);
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
