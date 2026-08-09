import { RuleBasedAgent, type Agent } from "@napoleon/ai";
import {
  PolicyOnnxAgent,
  loadPolicyOnnxModel,
  type PolicyOnnxModel
} from "@napoleon/policy-onnx";
import type { PublicAgentDescriptor } from "@napoleon/protocol";
import {
  createLearnedPolicyEnvKey,
  learnedPolicySlotNumbers
} from "./agentEnv.js";

export const RULE_BASED_AGENT_ID = "rule-based";
export const PLAYING_POLICY_ONNX_AGENT_ID = "playing-policy-onnx";
export const PLAYING_POLICY_ONNX_AGENT_IDS = [
  PLAYING_POLICY_ONNX_AGENT_ID,
  "playing-policy-onnx-2",
  "playing-policy-onnx-3",
  "playing-policy-onnx-4",
  "playing-policy-onnx-5"
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

export interface AgentRegistryOptions {
  playingPolicyOnnx?: PlayingPolicyOnnxPaths;
  playingPolicyOnnxAgents?: readonly PlayingPolicyOnnxAgentConfig[];
  loadPlayingPolicyOnnxModel?: (paths: PlayingPolicyOnnxPaths) => Promise<PolicyOnnxModel>;
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
  return createAgentRegistry({
    playingPolicyOnnxAgents: readPlayingPolicyOnnxAgentConfigs(env)
  });
}

export function readPlayingPolicyOnnxAgentConfigs(
  env: NodeJS.ProcessEnv
): readonly PlayingPolicyOnnxAgentConfig[] {
  return learnedPolicySlotNumbers.flatMap((slotNumber, index) => {
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
        id: PLAYING_POLICY_ONNX_AGENT_IDS[index],
        displayName,
        onnxPath,
        metadataPath
      }
    ];
  });
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

function formatErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
