import {
  ParameterizedNonPlayingAgent,
  RuleBasedAgent
} from "@napoleon/ai";
import type { Agent, PlayerObservation } from "@napoleon/ai";
import type { Card, GameAction } from "@napoleon/game-core";
import {
  CriticEvBiddingAgent,
  FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  PolicyOnnxAgent,
  T1NapoleonEvBiddingAgent,
  createT1NapoleonEvBiddingDiagnostics,
  getRepoManagedBiddingMarginPolicyBenchmark,
  getRepoManagedPlayingPolicyBenchmark,
  loadRepoManagedBiddingMarginPolicyBenchmark,
  loadRepoManagedPlayingActorBenchmark,
  loadRepoManagedPlayingCriticBenchmark
} from "@napoleon/policy-onnx";
import type {
  LoadedBiddingMarginPolicyBenchmark,
  LoadedPlayingActorBenchmark,
  LoadedPlayingCriticBenchmark,
  T1NapoleonEvBiddingDiagnostics
} from "@napoleon/policy-onnx";
import type {
  AiPolicyComposition,
  BiddingPolicyId,
  NonPlayingPolicyId,
  PlayingPolicyId,
  PublicAiPhaseCallDiagnostics,
  PublicPhasePolicyRegistry
} from "@napoleon/protocol";
import { AgentUnavailableError, UnknownPhasePolicyIdError } from "./agentErrors.js";
import {
  PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_ID,
  loadParameterizedPolicyArtifact
} from "./parameterizedPolicyArtifact.js";
import type { LoadedParameterizedPolicyArtifact } from "./parameterizedPolicyArtifact.js";

export const RULE_BASED_POLICY_ID = "rule-based" as const;

export interface PhasePolicyRegistryOptions {
  parameterizedArtifact?: LoadedParameterizedPolicyArtifact;
  loadPlayingPolicy?: () => Promise<LoadedPlayingActorBenchmark>;
  loadPlayingCritic?: () => Promise<LoadedPlayingCriticBenchmark>;
  loadBiddingPolicy?: () => Promise<LoadedBiddingMarginPolicyBenchmark>;
}

export interface PhasePolicyRegistry {
  initialize(): Promise<void>;
  describe(): PublicPhasePolicyRegistry;
  createDiagnostics(composition: AiPolicyComposition): PublicAiPhaseCallDiagnostics;
  createAgent(
    composition: AiPolicyComposition,
    diagnostics?: PublicAiPhaseCallDiagnostics,
    rng?: () => number
  ): Agent;
}

export function createPhasePolicyRegistry(
  options: PhasePolicyRegistryOptions = {}
): PhasePolicyRegistry {
  const parameterizedArtifact = options.parameterizedArtifact ?? loadParameterizedPolicyArtifact();
  let playing: LoadedPlayingActorBenchmark | undefined;
  let critic: LoadedPlayingCriticBenchmark | undefined;
  let bidding: LoadedBiddingMarginPolicyBenchmark | undefined;
  let playingError: unknown;
  let criticError: unknown;
  let biddingError: unknown;
  let initialization: Promise<void> | undefined;
  const initialize = () => {
    initialization ??= Promise.all([
      (options.loadPlayingPolicy?.() ??
        loadRepoManagedPlayingActorBenchmark(PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID, {
          inferenceDevice: "cpu"
        }))
        .then((loaded) => { playing = loaded; })
        .catch((error: unknown) => { playingError = error; }),
      (options.loadPlayingCritic?.() ??
        loadRepoManagedPlayingCriticBenchmark(PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID, {
          inferenceDevice: "cpu"
        }))
        .then((loaded) => { critic = loaded; })
        .catch((error: unknown) => { criticError = error; }),
      (options.loadBiddingPolicy?.() ??
        loadRepoManagedBiddingMarginPolicyBenchmark(FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID, {
          inferenceDevice: "cpu"
        }))
        .then((loaded) => { bidding = loaded; })
        .catch((error: unknown) => { biddingError = error; })
    ]).then(() => undefined);
    return initialization;
  };
  const requirePlaying = async () => playing ?? unavailable(
    PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
    playingError
  );
  const requireCritic = async () => critic ?? unavailable(
    FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
    criticError
  );
  const requireBidding = async () => bidding ?? unavailable(
    FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
    biddingError
  );
  const playingArtifact = getRepoManagedPlayingPolicyBenchmark(
    PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID
  );
  const biddingArtifact = getRepoManagedBiddingMarginPolicyBenchmark(
    FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID
  );

  return {
    initialize,
    describe: () => ({
      playing: [
        ruleBasedDescriptor(),
        {
          id: PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
          displayName: playingArtifact.displayName,
          isAvailable: playing !== undefined,
          artifactProvenance: {
            onnxSha256: playingArtifact.onnxSha256,
            metadataSha256: playingArtifact.metadataSha256,
            ...(playingArtifact.criticOnnxSha256 === undefined
              ? {}
              : { criticOnnxSha256: playingArtifact.criticOnnxSha256 })
          }
        }
      ],
      bidding: [
        ruleBasedDescriptor(),
        {
          id: FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
          displayName: biddingArtifact.displayName,
          isAvailable: bidding !== undefined && critic !== undefined,
          artifactProvenance: {
            onnxSha256: biddingArtifact.onnxSha256,
            metadataSha256: biddingArtifact.metadataSha256,
            ...(playingArtifact.criticOnnxSha256 === undefined
              ? {}
              : { playingCriticOnnxSha256: playingArtifact.criticOnnxSha256 }),
            ...(biddingArtifact.manifestSha256 === undefined
              ? {}
              : { manifestSha256: biddingArtifact.manifestSha256 })
          }
        }
      ],
      nonPlaying: [
        ruleBasedDescriptor(),
        {
          id: PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_ID,
          displayName: "Parameterized adjutant + exchange v1",
          isAvailable: true,
          artifactProvenance: {
            logicalArtifactSha256: parameterizedArtifact.provenance.logicalArtifactSha256,
            parameterSha256: parameterizedArtifact.provenance.parameterSha256,
            weightVectorSha256: parameterizedArtifact.provenance.weightVectorSha256,
            policyFileSha256: parameterizedArtifact.provenance.policyFileSha256,
            schemaFileSha256: parameterizedArtifact.provenance.schemaFileSha256,
            featureSchemaVersion: "1",
            adjutantWeightCount: "35",
            exchangeWeightCount: "60",
            optimizerIssue: parameterizedArtifact.provenance.optimizerIssue,
            verificationIssue: parameterizedArtifact.provenance.verificationIssue,
            verificationReportFileSha256:
              parameterizedArtifact.provenance.verificationReportFileSha256,
            verificationSeedManifestSha256:
              parameterizedArtifact.provenance.verificationSeedManifestSha256,
            verificationSeedManifestFileSha256:
              parameterizedArtifact.provenance.verificationSeedManifestFileSha256,
            biddingDependencySha256:
              parameterizedArtifact.provenance.biddingDependencySha256,
            playingDependencySha256:
              parameterizedArtifact.provenance.playingDependencySha256,
            playingCriticDependencySha256:
              parameterizedArtifact.provenance.playingCriticDependencySha256,
            evaluatorDependencySha256:
              parameterizedArtifact.provenance.evaluatorDependencySha256
          }
        }
      ]
    }),
    createDiagnostics: (composition) => createPhasePolicyDiagnostics(composition),
    createAgent: (
      composition,
      diagnostics = createPhasePolicyDiagnostics(composition),
      rng = Math.random
    ) => {
      validateComposition(composition);
      assertCompositionAvailable(composition, { playing, critic, bidding, playingError, criticError, biddingError });
      return new ComposedPhasePolicyAgent({
        composition,
        diagnostics,
        parameterizedArtifact,
        loadPlaying: requirePlaying,
        loadCritic: requireCritic,
        loadBidding: requireBidding,
        rng
      });
    }
  };
}

export function createPhasePolicyDiagnostics(
  composition: AiPolicyComposition
): PublicAiPhaseCallDiagnostics {
  return {
    composition: { ...composition },
    playingCalls: 0,
    biddingCalls: 0,
    adjutantCalls: 0,
    exchangeCalls: 0,
    fallbackCount: 0,
    illegalCount: 0
  };
}

export function validateComposition(composition: AiPolicyComposition): void {
  validatePolicyId("playing", composition.playing, [
    RULE_BASED_POLICY_ID,
    PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID
  ]);
  validatePolicyId("bidding", composition.bidding, [
    RULE_BASED_POLICY_ID,
    FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID
  ]);
  validatePolicyId("nonPlaying", composition.nonPlaying, [
    RULE_BASED_POLICY_ID,
    PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_ID
  ]);
}

class ComposedPhasePolicyAgent implements Agent {
  private readonly ruleBased: RuleBasedAgent;
  private readonly parameterized: ParameterizedNonPlayingAgent;
  private playing: PolicyOnnxAgent | undefined;
  private bidding: T1NapoleonEvBiddingAgent | undefined;
  private biddingDiagnostics: T1NapoleonEvBiddingDiagnostics | undefined;

  constructor(private readonly options: {
    composition: AiPolicyComposition;
    diagnostics: PublicAiPhaseCallDiagnostics;
    parameterizedArtifact: LoadedParameterizedPolicyArtifact;
    loadPlaying: () => Promise<LoadedPlayingActorBenchmark>;
    loadCritic: () => Promise<LoadedPlayingCriticBenchmark>;
    loadBidding: () => Promise<LoadedBiddingMarginPolicyBenchmark>;
    rng: () => number;
  }) {
    this.ruleBased = new RuleBasedAgent(options.rng);
    this.parameterized = new ParameterizedNonPlayingAgent(
      options.parameterizedArtifact.parameters
    );
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    let action: GameAction;
    switch (observation.view.phase) {
      case "bidding":
        this.options.diagnostics.biddingCalls += 1;
        action = this.options.composition.bidding === RULE_BASED_POLICY_ID
          ? await this.ruleBased.selectAction(observation)
          : await (await this.getBidding()).selectAction(observation);
        this.syncBiddingFallbackCount();
        break;
      case "choosing-adjutant":
        this.options.diagnostics.adjutantCalls += 1;
        action = this.options.composition.nonPlaying === RULE_BASED_POLICY_ID
          ? await this.ruleBased.selectAction(observation)
          : await this.parameterized.selectAction(observation);
        break;
      case "exchanging":
        this.options.diagnostics.exchangeCalls += 1;
        action = this.options.composition.nonPlaying === RULE_BASED_POLICY_ID
          ? await this.ruleBased.selectAction(observation)
          : await this.parameterized.selectAction(observation);
        break;
      case "playing":
        this.options.diagnostics.playingCalls += 1;
        action = this.options.composition.playing === RULE_BASED_POLICY_ID
          ? await this.ruleBased.selectAction(observation)
          : await (await this.getPlaying()).selectAction(observation);
        break;
      case "finished":
        return this.ruleBased.selectAction(observation);
    }
    if (!isLegalSelectedAction(observation, action)) {
      this.options.diagnostics.illegalCount += 1;
      throw new Error(
        `Phase policy selected an illegal ${action.type} action for ${observation.view.phase}.`
      );
    }
    return action;
  }

  private async getPlaying(): Promise<PolicyOnnxAgent> {
    if (this.playing === undefined) {
      try {
        const loaded = await this.options.loadPlaying();
        this.playing = new PolicyOnnxAgent({ policy: loaded.policy });
      } catch (error) {
        throw new AgentUnavailableError(
          PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
          `Playing policy ${PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID} could not be loaded: ${message(error)}`
        );
      }
    }
    return this.playing;
  }

  private async getBidding(): Promise<T1NapoleonEvBiddingAgent> {
    if (this.bidding === undefined) {
      try {
        const [critic, bidding] = await Promise.all([
          this.options.loadCritic(),
          this.options.loadBidding()
        ]);
        this.biddingDiagnostics = createT1NapoleonEvBiddingDiagnostics();
        this.bidding = new T1NapoleonEvBiddingAgent({
          marginModel: bidding.model,
          passEvAgent: new CriticEvBiddingAgent({ critic: critic.critic }),
          delegateAgent: this.ruleBased,
          diagnostics: this.biddingDiagnostics,
          fallbackOnInferenceError: false
        });
      } catch (error) {
        throw new AgentUnavailableError(
          FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
          `Bidding policy ${FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID} could not be loaded: ${message(error)}`
        );
      }
    }
    return this.bidding;
  }

  private syncBiddingFallbackCount(): void {
    if (this.biddingDiagnostics !== undefined) {
      this.options.diagnostics.fallbackCount = this.biddingDiagnostics.fallbackCount;
    }
  }
}

function isLegalSelectedAction(observation: PlayerObservation, action: GameAction): boolean {
  if (action.playerId !== observation.playerId) {
    return false;
  }
  if (observation.view.phase === "exchanging") {
    if (action.type !== "discard-cards") {
      return false;
    }
    const hand = observation.view.players.find(
      (player) => player.id === observation.playerId
    )?.hand;
    const required = observation.view.exchangeRequirement?.discardCount;
    return hand !== undefined &&
      required !== undefined &&
      action.cardIds.length === required &&
      new Set(action.cardIds).size === action.cardIds.length &&
      action.cardIds.every((cardId) => hand.some((card: Card) => card.id === cardId));
  }
  return observation.legalActions.some((legal) => actionsEqual(legal, action));
}

function actionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }
  switch (left.type) {
    case "pass":
      return true;
    case "bid":
      return right.type === "bid" &&
        left.suit === right.suit &&
        left.targetPointCards === right.targetPointCards;
    case "choose-adjutant":
      return right.type === "choose-adjutant" && left.cardId === right.cardId;
    case "play-card":
      return right.type === "play-card" && left.cardId === right.cardId;
    case "discard-cards":
      return right.type === "discard-cards" &&
        left.cardIds.length === right.cardIds.length &&
        left.cardIds.every((cardId, index) => cardId === right.cardIds[index]);
  }
}

function ruleBasedDescriptor() {
  return {
    id: RULE_BASED_POLICY_ID,
    displayName: "Rule-based",
    isAvailable: true,
    artifactProvenance: null
  } as const;
}

function validatePolicyId(
  axis: "playing" | "bidding" | "nonPlaying",
  policyId: PlayingPolicyId | BiddingPolicyId | NonPlayingPolicyId,
  allowed: readonly string[]
): void {
  if (!allowed.includes(policyId)) {
    throw new UnknownPhasePolicyIdError(axis, policyId);
  }
}

function assertCompositionAvailable(
  composition: AiPolicyComposition,
  status: {
    playing: LoadedPlayingActorBenchmark | undefined;
    critic: LoadedPlayingCriticBenchmark | undefined;
    bidding: LoadedBiddingMarginPolicyBenchmark | undefined;
    playingError: unknown;
    criticError: unknown;
    biddingError: unknown;
  }
): void {
  if (
    composition.playing === PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID &&
    status.playing === undefined
  ) {
    unavailable(PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID, status.playingError);
  }
  if (
    composition.bidding === FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID &&
    (status.bidding === undefined || status.critic === undefined)
  ) {
    unavailable(
      FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
      status.biddingError ?? status.criticError
    );
  }
}

function unavailable(policyId: string, error: unknown): never {
  throw new AgentUnavailableError(
    policyId,
    `Phase policy ${policyId} is unavailable` +
      (error === undefined ? ". Registry initialization has not completed." : `: ${message(error)}`)
  );
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
