import { RuleBasedAgent, createSeededRandom, deriveSeed } from "@napoleon/ai";
import type { PlayerObservation, PublicActionRecord } from "@napoleon/ai";
import {
  applyAction,
  createInitialGame,
  createPlayerView,
  getLegalActions
} from "@napoleon/game-core";
import type { Bid, GameAction, GameState, PlayerId } from "@napoleon/game-core";
import {
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  getRepoManagedPlayingPolicyBenchmark,
  validatePlayingPolicyArtifactReference
} from "./benchmarkArtifacts.js";
import { CriticEvBiddingAgent } from "./criticEvBiddingAgent.js";
import { loadPolicyCriticOnnxModel } from "./policyOnnx.js";
import type { PolicyOnnxInferenceDevice } from "./types.js";

interface InspectOptions {
  seeds: readonly number[];
  inferenceDevice: PolicyOnnxInferenceDevice;
}

interface BiddingTraceState {
  state: GameState;
  publicActionHistory: PublicActionRecord[];
}

const defaultSeeds = [193] as const;
const playerIds: readonly PlayerId[] = [
  "player-0",
  "player-1",
  "player-2",
  "player-3",
  "player-4"
];

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const artifact = getRepoManagedPlayingPolicyBenchmark(PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID);
  const criticOnnxPath = artifact.criticOnnxPath;
  const criticMetadataPath = artifact.criticMetadataPath;

  if (criticOnnxPath === undefined || criticMetadataPath === undefined) {
    throw new Error(`${artifact.id} does not include a critic ONNX artifact.`);
  }

  await validatePlayingPolicyArtifactReference(artifact);
  const critic = await loadPolicyCriticOnnxModel({
    onnxPath: criticOnnxPath,
    metadataPath: criticMetadataPath,
    inferenceDevice: options.inferenceDevice
  });
  const agent = new CriticEvBiddingAgent({
    critic,
    delegateAgent: new RuleBasedAgent(() => 0),
    playerIds
  });

  console.log("Critic EV bidding inspection");
  console.log(`artifact=${artifact.id}`);
  console.log(`criticOnnx=${criticOnnxPath}`);
  console.log(`criticMetadata=${criticMetadataPath}`);
  console.log(`requestedInferenceDevice=${critic.runtime.requestedInferenceDevice}`);
  console.log(`resolvedInferenceDevice=${critic.runtime.resolvedInferenceDevice}`);
  console.log(`executionProvider=${critic.runtime.executionProvider}`);

  for (const seed of options.seeds) {
    await inspectSeed(seed, agent);
  }
}

async function inspectSeed(seed: number, agent: CriticEvBiddingAgent): Promise<void> {
  const trace = createInitialTrace(seed);
  let turn = 1;

  console.log("");
  console.log(`seed=${seed}`);
  console.log("initialHands:");
  for (const player of trace.state.players) {
    console.log(`  ${player.id}: ${player.hand.map((card) => card.id).join(" ")}`);
  }

  while (trace.state.phase === "bidding") {
    const observation = createObservation(trace);
    const evaluations = await agent.evaluateLegalBiddingActions(observation);
    const selectedAction = await agent.selectAction(observation);
    const selectedBiddingAction = assertBiddingDecisionAction(selectedAction);

    console.log("");
    console.log(`biddingTurn=${turn} playerId=${observation.playerId}`);
    console.table(evaluations.map((evaluation) => ({
      action: formatAction(evaluation.action),
      contract: evaluation.contract === null ? "all-pass" : formatContract(evaluation.contract),
      role: evaluation.role,
      calledAdjutantCardId: evaluation.calledAdjutantCardId,
      criticValue: formatNumber(evaluation.criticValue),
      baseWinRateEquivalent: formatNumber(evaluation.baseWinRateEquivalent),
      effectiveNapoleonWinRate: formatNumber(evaluation.effectiveNapoleonWinRate),
      expectedValue: formatNumber(evaluation.expectedValue)
    })));
    console.log(`selectedAction=${formatAction(selectedBiddingAction)}`);

    trace.publicActionHistory.push({
      step: trace.publicActionHistory.length + 1,
      playerId: selectedBiddingAction.playerId,
      phase: "bidding",
      action: selectedBiddingAction
    });
    trace.state = applyAction(trace.state, selectedBiddingAction);
    turn += 1;
  }

  console.log("");
  console.log(`biddingComplete seed=${seed} turns=${turn - 1} nextPhase=${trace.state.phase}`);
  console.log(`highestBid=${trace.state.contract === null ? "none" : formatContract({
    playerId: trace.state.contract.napoleonPlayerId,
    suit: trace.state.contract.trumpSuit,
    targetPointCards: trace.state.contract.targetPointCards
  })}`);
}

function createInitialTrace(seed: number): BiddingTraceState {
  return {
    state: createInitialGame({
      playerIds,
      rng: createSeededRandom(deriveSeed(seed, "game"))
    }),
    publicActionHistory: []
  };
}

function createObservation(trace: BiddingTraceState): PlayerObservation {
  const playerId = trace.state.currentPlayerId;
  const view = createPlayerView(trace.state, playerId);

  return {
    playerId,
    view,
    legalActions: getLegalActions(trace.state, playerId),
    publicActionHistory: [...trace.publicActionHistory]
  };
}

function parseArgs(args: readonly string[]): InspectOptions {
  const seeds: number[] = [];
  let inferenceDevice: PolicyOnnxInferenceDevice = "cpu";

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    switch (arg) {
      case "--":
        break;
      case "--seed":
        seeds.push(parseSeedValue(requireValue(args, index, arg)));
        index += 1;
        break;
      case "--seeds":
        seeds.push(...requireValue(args, index, arg).split(",").map(parseSeedValue));
        index += 1;
        break;
      case "--inference-device":
        inferenceDevice = parseInferenceDevice(requireValue(args, index, arg));
        index += 1;
        break;
      case "--help":
      case "-h":
        printUsage();
        process.exit(0);
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return {
    seeds: seeds.length === 0 ? defaultSeeds : seeds,
    inferenceDevice
  };
}

function requireValue(args: readonly string[], index: number, option: string): string {
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`${option} requires a value.`);
  }
  return value;
}

function parseSeedValue(value: string): number {
  const seed = Number(value);
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error(`Invalid seed: ${value}`);
  }
  return seed;
}

function parseInferenceDevice(value: string): PolicyOnnxInferenceDevice {
  if (value === "cpu" || value === "auto" || value === "cuda") {
    return value;
  }
  throw new Error(`Invalid inference device: ${value}`);
}

function formatAction(action: Extract<GameAction, { type: "bid" | "pass" }>): string {
  if (action.type === "pass") {
    return `${action.playerId} pass`;
  }
  return `${action.playerId} bid ${action.suit}-${action.targetPointCards}`;
}

function assertBiddingDecisionAction(action: GameAction): Extract<GameAction, { type: "bid" | "pass" }> {
  if (action.type === "bid" || action.type === "pass") {
    return action;
  }
  throw new Error(`Expected a bidding decision action, got ${JSON.stringify(action)}.`);
}

function formatContract(contract: Bid): string {
  return `${contract.playerId} ${contract.suit}-${contract.targetPointCards}`;
}

function formatNumber(value: number): string {
  return value.toFixed(6);
}

function printUsage(): void {
  console.log([
    "Usage:",
    "  pnpm --filter @napoleon/policy-onnx inspect:critic-ev-bidding",
    "  pnpm --filter @napoleon/policy-onnx inspect:critic-ev-bidding -- --seed 193",
    "  pnpm --filter @napoleon/policy-onnx inspect:critic-ev-bidding -- --seeds 193,1000 --inference-device cpu",
    "",
    "Options:",
    "  --seed <uint32>                 Inspect one fixed-seed game state. Repeatable.",
    "  --seeds <uint32,uint32,...>     Inspect multiple fixed-seed game states.",
    "  --inference-device <cpu|auto|cuda>",
    "  --help"
  ].join("\n"));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
