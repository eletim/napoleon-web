import { existsSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { RuleBasedAgent, type Agent } from "@napoleon/ai";
import {
  NonPlayingPolicyOnnxModel,
  PolicyOnnxAgent,
  loadNonPlayingPolicyOnnxModel,
  loadPolicyOnnxModel,
  parseNonPlayingPolicyOnnxMetadata,
  parsePolicyOnnxMetadata,
  type NonPlayingPolicyType,
  type PolicyOnnxModel
} from "@napoleon/policy-onnx";
import type { PublicAgentDescriptor } from "@napoleon/protocol";
import {
  createLearnedPolicyEnvKey,
  createFullPolicyEnvKey,
  learnedPolicySlotNumbers,
  type LearnedPolicySlotNumber
} from "./agentEnv.js";

export const RULE_BASED_AGENT_ID = "rule-based";
export const PLAYING_POLICY_ONNX_AGENT_ID = "playing-policy-onnx";
export const PLAYING_POLICY_ONNX_AGENT_IDS = [
  ...learnedPolicySlotNumbers.map(createPlayingPolicyOnnxAgentId)
] as const;
export const FULL_POLICY_ONNX_AGENT_ID = "full-policy-onnx";
export const FULL_POLICY_ONNX_AGENT_IDS = [
  ...learnedPolicySlotNumbers.map(createFullPolicyOnnxAgentId)
] as const;

export class UnknownAgentIdError extends Error {
  constructor(readonly agentId: string) {
    super(`Unknown AI agent id: ${agentId}.`);
    this.name = "UnknownAgentIdError";
  }
}

export class AgentUnavailableError extends Error {
  constructor(readonly agentId: string, message: string) {
    super(message);
    this.name = "AgentUnavailableError";
  }
}

export interface PlayingPolicyOnnxPaths {
  onnxPath: string;
  metadataPath: string;
}

export interface PlayingPolicyOnnxAgentConfig extends PlayingPolicyOnnxPaths {
  id: string;
  displayName: string;
}

export interface NonPlayingPolicyOnnxPaths {
  onnxPath: string;
  metadataPath: string;
}

export interface FullPolicyOnnxAgentConfig {
  id: string;
  displayName: string;
  playing: PlayingPolicyOnnxPaths;
  bidding: NonPlayingPolicyOnnxPaths;
  adjutant: NonPlayingPolicyOnnxPaths;
  exchange: NonPlayingPolicyOnnxPaths;
}

export interface AgentRegistryOptions {
  playingPolicyOnnx?: PlayingPolicyOnnxPaths;
  playingPolicyOnnxAgents?: readonly PlayingPolicyOnnxAgentConfig[];
  loadPlayingPolicyOnnxModel?: (paths: PlayingPolicyOnnxPaths) => Promise<PolicyOnnxModel>;
  fullPolicyOnnxAgents?: readonly FullPolicyOnnxAgentConfig[];
  loadNonPlayingPolicyOnnxModel?: (
    paths: NonPlayingPolicyOnnxPaths
  ) => Promise<NonPlayingPolicyOnnxModel>;
}

export interface AgentRegistry {
  listAgents(): readonly PublicAgentDescriptor[];
  createAgent(agentId: string): Agent;
}

interface AgentDefinition {
  descriptor: PublicAgentDescriptor;
  createAgent(): Agent;
}

export function createAgentRegistry(options: AgentRegistryOptions = {}): AgentRegistry {
  const learnedPolicyConfigs =
    options.playingPolicyOnnxAgents ??
    (options.playingPolicyOnnx === undefined
      ? []
      : [
          {
            id: PLAYING_POLICY_ONNX_AGENT_ID,
            displayName: "Playing Policy ONNX",
            ...options.playingPolicyOnnx
          }
        ]);
  const loadPlayingPolicyOnnxModel =
    options.loadPlayingPolicyOnnxModel ?? loadPolicyOnnxModel;
  const loadNonPlayingOnnxModel =
    options.loadNonPlayingPolicyOnnxModel ?? loadNonPlayingPolicyOnnxModel;
  const definitions = new Map<string, AgentDefinition>([
    [
      RULE_BASED_AGENT_ID,
      {
        descriptor: {
          id: RULE_BASED_AGENT_ID,
          displayName: "Rule-based AI",
          isAvailable: true
        },
        createAgent: () => new RuleBasedAgent()
      }
    ]
  ]);

  for (const config of learnedPolicyConfigs) {
    let learnedPolicyPromise: Promise<PolicyOnnxModel> | undefined;
    const loadLearnedPolicy = () => {
      learnedPolicyPromise ??= loadPlayingPolicyOnnxModel(config).catch((error) => {
        learnedPolicyPromise = undefined;
        throw error;
      });

      return learnedPolicyPromise;
    };

    definitions.set(config.id, {
      descriptor: {
        id: config.id,
        displayName: config.displayName,
        isAvailable: true
      },
      createAgent: () => new LazyPolicyOnnxAgent(config.id, loadLearnedPolicy)
    });
  }

  for (const config of options.fullPolicyOnnxAgents ?? []) {
    let fullPolicyPromise: Promise<LoadedFullPolicyOnnxModels> | undefined;
    const loadFullPolicy = () => {
      fullPolicyPromise ??= loadFullPolicyOnnxModels(config, {
        loadPlayingPolicyOnnxModel,
        loadNonPlayingPolicyOnnxModel: loadNonPlayingOnnxModel
      }).catch((error) => {
        fullPolicyPromise = undefined;
        throw error;
      });

      return fullPolicyPromise;
    };

    definitions.set(config.id, {
      descriptor: {
        id: config.id,
        displayName: config.displayName,
        isAvailable: true
      },
      createAgent: () => new LazyFullPolicyOnnxAgent(config.id, loadFullPolicy)
    });
  }

  return {
    listAgents: () => [...definitions.values()].map((definition) => definition.descriptor),
    createAgent: (agentId: string) => {
      const definition = definitions.get(agentId);

      if (definition === undefined) {
        throw new UnknownAgentIdError(agentId);
      }

      return definition.createAgent();
    }
  };
}

export function createAgentRegistryFromEnvironment(
  env: NodeJS.ProcessEnv = process.env
): AgentRegistry {
  const fullPolicyOnnxAgents = readFullPolicyOnnxAgentConfigs(env).map((config) =>
    validateFullPolicyOnnxAgentConfigArtifacts(config)
  );

  return createAgentRegistry({
    playingPolicyOnnxAgents: readPlayingPolicyOnnxAgentConfigs(env),
    fullPolicyOnnxAgents
  });
}

export function readPlayingPolicyOnnxAgentConfigs(
  env: NodeJS.ProcessEnv,
  cwd = process.cwd()
): readonly PlayingPolicyOnnxAgentConfig[] {
  const workspaceRoot = findWorkspaceRoot(cwd);

  return learnedPolicySlotNumbers.flatMap((slotNumber) => {
    const displayNameVariable = createLearnedPolicyEnvKey(slotNumber, "DISPLAY_NAME");
    const onnxPathVariable = createLearnedPolicyEnvKey(slotNumber, "ONNX_PATH");
    const metadataPathVariable = createLearnedPolicyEnvKey(slotNumber, "METADATA_PATH");
    const displayName = env[displayNameVariable]?.trim() ?? "";

    if (displayName.length === 0) {
      return [];
    }

    const onnxPath = env[onnxPathVariable]?.trim() ?? "";
    const metadataPath = env[metadataPathVariable]?.trim() ?? "";
    const missingVariables = [
      onnxPath.length === 0 ? onnxPathVariable : undefined,
      metadataPath.length === 0 ? metadataPathVariable : undefined
    ].filter((variable): variable is string => variable !== undefined);

    if (missingVariables.length > 0) {
      throw new Error(
        `Incomplete learned ONNX agent configuration for slot ${slotNumber}: ` +
          `${missingVariables.join(", ")} must be set when ${displayNameVariable} is non-empty.`
      );
    }

    return [
      {
        id: createPlayingPolicyOnnxAgentId(slotNumber),
        displayName,
        onnxPath: resolveConfiguredPath(onnxPath, workspaceRoot),
        metadataPath: resolveConfiguredPath(metadataPath, workspaceRoot)
      }
    ];
  });
}

export function readFullPolicyOnnxAgentConfigs(
  env: NodeJS.ProcessEnv,
  cwd = process.cwd()
): readonly FullPolicyOnnxAgentConfig[] {
  const workspaceRoot = findWorkspaceRoot(cwd);

  return learnedPolicySlotNumbers.flatMap((slotNumber) => {
    const displayNameVariable = createFullPolicyEnvKey(slotNumber, "DISPLAY_NAME");
    const displayName = env[displayNameVariable]?.trim() ?? "";

    if (displayName.length === 0) {
      return [];
    }

    const phasePaths = {
      playing: readRequiredPhasePaths({
        env,
        workspaceRoot,
        slotNumber,
        displayNameVariable,
        onnxField: "PLAYING_ONNX_PATH",
        metadataField: "PLAYING_METADATA_PATH"
      }),
      bidding: readRequiredPhasePaths({
        env,
        workspaceRoot,
        slotNumber,
        displayNameVariable,
        onnxField: "BIDDING_ONNX_PATH",
        metadataField: "BIDDING_METADATA_PATH"
      }),
      adjutant: readRequiredPhasePaths({
        env,
        workspaceRoot,
        slotNumber,
        displayNameVariable,
        onnxField: "ADJUTANT_ONNX_PATH",
        metadataField: "ADJUTANT_METADATA_PATH"
      }),
      exchange: readRequiredPhasePaths({
        env,
        workspaceRoot,
        slotNumber,
        displayNameVariable,
        onnxField: "EXCHANGE_ONNX_PATH",
        metadataField: "EXCHANGE_METADATA_PATH"
      })
    };

    return [
      {
        id: createFullPolicyOnnxAgentId(slotNumber),
        displayName,
        ...phasePaths
      }
    ];
  });
}

export function validateFullPolicyOnnxAgentConfigArtifacts(
  config: FullPolicyOnnxAgentConfig
): FullPolicyOnnxAgentConfig {
  validateExistingFile(config.id, "playing ONNX", config.playing.onnxPath);
  validateExistingFile(config.id, "playing metadata", config.playing.metadataPath);
  validateExistingFile(config.id, "bidding ONNX", config.bidding.onnxPath);
  validateExistingFile(config.id, "bidding metadata", config.bidding.metadataPath);
  validateExistingFile(config.id, "adjutant ONNX", config.adjutant.onnxPath);
  validateExistingFile(config.id, "adjutant metadata", config.adjutant.metadataPath);
  validateExistingFile(config.id, "exchange ONNX", config.exchange.onnxPath);
  validateExistingFile(config.id, "exchange metadata", config.exchange.metadataPath);

  parsePhaseMetadata(config.id, "playing", config.playing.metadataPath, () =>
    parsePolicyOnnxMetadata(readFileSync(config.playing.metadataPath, "utf8"))
  );
  const bidding = parsePhaseMetadata(config.id, "bidding", config.bidding.metadataPath, () =>
    parseNonPlayingPolicyOnnxMetadata(readFileSync(config.bidding.metadataPath, "utf8"))
  );
  const adjutant = parsePhaseMetadata(config.id, "adjutant", config.adjutant.metadataPath, () =>
    parseNonPlayingPolicyOnnxMetadata(readFileSync(config.adjutant.metadataPath, "utf8"))
  );
  const exchange = parsePhaseMetadata(config.id, "exchange", config.exchange.metadataPath, () =>
    parseNonPlayingPolicyOnnxMetadata(readFileSync(config.exchange.metadataPath, "utf8"))
  );

  assertNonPlayingMetadata("bidding", bidding, "bidding");
  assertNonPlayingMetadata("adjutant", adjutant, "adjutant");
  assertNonPlayingMetadata("exchange", exchange, "exchange");
  if (exchange.decisionMode !== "sequential-card-v1") {
    throw new Error(
      `Full policy ONNX agent ${config.id} exchange policy decisionMode mismatch: ` +
        `expected sequential-card-v1, got ${exchange.decisionMode ?? "top3-set-v1"}.`
    );
  }

  return config;
}

function createPlayingPolicyOnnxAgentId(slotNumber: LearnedPolicySlotNumber): string {
  return slotNumber === 1 ? PLAYING_POLICY_ONNX_AGENT_ID : `playing-policy-onnx-${slotNumber}`;
}

function createFullPolicyOnnxAgentId(slotNumber: LearnedPolicySlotNumber): string {
  return slotNumber === 1 ? FULL_POLICY_ONNX_AGENT_ID : `full-policy-onnx-${slotNumber}`;
}

function readRequiredPhasePaths(options: {
  env: NodeJS.ProcessEnv;
  workspaceRoot: string;
  slotNumber: LearnedPolicySlotNumber;
  displayNameVariable: string;
  onnxField: Parameters<typeof createFullPolicyEnvKey>[1];
  metadataField: Parameters<typeof createFullPolicyEnvKey>[1];
}): NonPlayingPolicyOnnxPaths {
  const onnxPathVariable = createFullPolicyEnvKey(options.slotNumber, options.onnxField);
  const metadataPathVariable = createFullPolicyEnvKey(options.slotNumber, options.metadataField);
  const onnxPath = options.env[onnxPathVariable]?.trim() ?? "";
  const metadataPath = options.env[metadataPathVariable]?.trim() ?? "";
  const missingVariables = [
    onnxPath.length === 0 ? onnxPathVariable : undefined,
    metadataPath.length === 0 ? metadataPathVariable : undefined
  ].filter((variable): variable is string => variable !== undefined);

  if (missingVariables.length > 0) {
    throw new Error(
      `Incomplete full-policy ONNX agent configuration for slot ${options.slotNumber}: ` +
        `${missingVariables.join(", ")} must be set when ${options.displayNameVariable} is non-empty.`
    );
  }

  return {
    onnxPath: resolveConfiguredPath(onnxPath, options.workspaceRoot),
    metadataPath: resolveConfiguredPath(metadataPath, options.workspaceRoot)
  };
}

function validateExistingFile(agentId: string, label: string, path: string): void {
  if (!existsSync(path)) {
    throw new Error(`Full policy ONNX agent ${agentId} ${label} file does not exist: ${path}`);
  }
}

function parsePhaseMetadata<T>(
  agentId: string,
  phase: string,
  path: string,
  parse: () => T
): T {
  try {
    return parse();
  } catch (error) {
    throw new Error(
      `Full policy ONNX agent ${agentId} ${phase} metadata is invalid (${path}): ` +
        formatErrorMessage(error)
    );
  }
}

function assertNonPlayingMetadata(
  label: string,
  metadata: { policyType: NonPlayingPolicyType },
  expectedType: NonPlayingPolicyType
): void {
  if (metadata.policyType !== expectedType) {
    throw new Error(
      `Full policy ONNX agent ${label} metadata policy type mismatch: ` +
        `expected ${expectedType}, got ${metadata.policyType}.`
    );
  }
}

function resolveConfiguredPath(path: string, workspaceRoot: string): string {
  return isAbsolute(path) ? path : resolve(workspaceRoot, path);
}

function findWorkspaceRoot(cwd: string): string {
  let current = cwd;

  for (;;) {
    if (existsSync(join(current, "pnpm-workspace.yaml"))) {
      return current;
    }

    const parent = dirname(current);

    if (parent === current) {
      return cwd;
    }

    current = parent;
  }
}

class LazyPolicyOnnxAgent implements Agent {
  private delegate: PolicyOnnxAgent | undefined;

  constructor(
    private readonly agentId: string,
    private readonly loadPolicy: () => Promise<PolicyOnnxModel>
  ) {}

  async selectAction(input: Parameters<Agent["selectAction"]>[0]) {
    try {
      this.delegate ??= new PolicyOnnxAgent({
        policy: await this.loadPolicy()
      });
    } catch (error) {
      if (error instanceof AgentUnavailableError) {
        throw error;
      }

      throw new AgentUnavailableError(
        this.agentId,
        `Playing policy ONNX could not be loaded: ${formatErrorMessage(error)}`
      );
    }

    return this.delegate.selectAction(input);
  }
}

interface LoadedFullPolicyOnnxModels {
  playing: PolicyOnnxModel;
  bidding: NonPlayingPolicyOnnxModel;
  adjutant: NonPlayingPolicyOnnxModel;
  exchange: NonPlayingPolicyOnnxModel;
}

async function loadFullPolicyOnnxModels(
  config: FullPolicyOnnxAgentConfig,
  loaders: {
    loadPlayingPolicyOnnxModel: (paths: PlayingPolicyOnnxPaths) => Promise<PolicyOnnxModel>;
    loadNonPlayingPolicyOnnxModel: (
      paths: NonPlayingPolicyOnnxPaths
    ) => Promise<NonPlayingPolicyOnnxModel>;
  }
): Promise<LoadedFullPolicyOnnxModels> {
  const [playing, bidding, adjutant, exchange] = await Promise.all([
    loaders.loadPlayingPolicyOnnxModel(config.playing),
    loaders.loadNonPlayingPolicyOnnxModel(config.bidding),
    loaders.loadNonPlayingPolicyOnnxModel(config.adjutant),
    loaders.loadNonPlayingPolicyOnnxModel(config.exchange)
  ]);

  assertNonPlayingPolicy("bidding", bidding, "bidding");
  assertNonPlayingPolicy("adjutant", adjutant, "adjutant");
  assertNonPlayingPolicy("exchange", exchange, "exchange");
  if (exchange.metadata.decisionMode !== "sequential-card-v1") {
    throw new Error(
      `exchange policy decisionMode mismatch: expected sequential-card-v1, got ${exchange.metadata.decisionMode ?? "top3-set-v1"}.`
    );
  }

  return { playing, bidding, adjutant, exchange };
}

function assertNonPlayingPolicy(
  label: string,
  policy: NonPlayingPolicyOnnxModel,
  expectedType: NonPlayingPolicyType
): void {
  if (policy.policyType !== expectedType || policy.metadata.policyType !== expectedType) {
    throw new Error(
      `${label} policy type mismatch: expected ${expectedType}, got ${policy.metadata.policyType}.`
    );
  }
}

class LazyFullPolicyOnnxAgent implements Agent {
  private delegate: PolicyOnnxAgent | undefined;

  constructor(
    private readonly agentId: string,
    private readonly loadPolicies: () => Promise<LoadedFullPolicyOnnxModels>
  ) {}

  async selectAction(input: Parameters<Agent["selectAction"]>[0]) {
    try {
      if (this.delegate === undefined) {
        const policies = await this.loadPolicies();
        this.delegate = new PolicyOnnxAgent({
          policy: policies.playing,
          biddingPolicy: policies.bidding,
          adjutantPolicy: policies.adjutant,
          exchangePolicy: policies.exchange
        });
      }
    } catch (error) {
      if (error instanceof AgentUnavailableError) {
        throw error;
      }

      throw new AgentUnavailableError(
        this.agentId,
        `Full policy ONNX could not be loaded: ${formatErrorMessage(error)}`
      );
    }

    return this.delegate.selectAction(input);
  }
}

function formatErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
