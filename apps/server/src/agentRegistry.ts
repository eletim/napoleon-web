import { RuleBasedAgent, type Agent } from "@napoleon/ai";
import {
  PolicyOnnxAgent,
  loadPolicyOnnxModel,
  type PolicyOnnxModel
} from "@napoleon/policy-onnx";
import type { PublicAgentDescriptor } from "@napoleon/protocol";

export const RULE_BASED_AGENT_ID = "rule-based";
export const PLAYING_POLICY_ONNX_AGENT_ID = "playing-policy-onnx";

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

export interface AgentRegistryOptions {
  playingPolicyOnnx?: PlayingPolicyOnnxPaths;
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
  const learnedPolicyPaths = options.playingPolicyOnnx;
  const loadPlayingPolicyOnnxModel =
    options.loadPlayingPolicyOnnxModel ?? loadPolicyOnnxModel;
  let learnedPolicyPromise: Promise<PolicyOnnxModel> | undefined;
  const loadLearnedPolicy = () => {
    if (learnedPolicyPaths === undefined) {
      throw new AgentUnavailableError(
        PLAYING_POLICY_ONNX_AGENT_ID,
        "Playing policy ONNX paths are not configured."
      );
    }

    learnedPolicyPromise ??= loadPlayingPolicyOnnxModel(learnedPolicyPaths).catch((error) => {
      learnedPolicyPromise = undefined;
      throw error;
    });

    return learnedPolicyPromise;
  };
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
    ],
    [
      PLAYING_POLICY_ONNX_AGENT_ID,
      {
        descriptor: {
          id: PLAYING_POLICY_ONNX_AGENT_ID,
          displayName: "Playing Policy ONNX",
          isAvailable: learnedPolicyPaths !== undefined
        },
        createAgent: () => {
          if (learnedPolicyPaths === undefined) {
            throw new AgentUnavailableError(
              PLAYING_POLICY_ONNX_AGENT_ID,
              "Playing policy ONNX paths are not configured."
            );
          }

          return new LazyPolicyOnnxAgent(loadLearnedPolicy);
        }
      }
    ]
  ]);

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
    playingPolicyOnnx: readPlayingPolicyOnnxPaths(env)
  });
}

function readPlayingPolicyOnnxPaths(
  env: NodeJS.ProcessEnv
): PlayingPolicyOnnxPaths | undefined {
  const onnxPath = env.NAPOLEON_POLICY_ONNX_PATH;
  const metadataPath = env.NAPOLEON_POLICY_METADATA_PATH;

  if (onnxPath === undefined || metadataPath === undefined) {
    return undefined;
  }

  return {
    onnxPath,
    metadataPath
  };
}

class LazyPolicyOnnxAgent implements Agent {
  private delegate: PolicyOnnxAgent | undefined;

  constructor(private readonly loadPolicy: () => Promise<PolicyOnnxModel>) {}

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
        PLAYING_POLICY_ONNX_AGENT_ID,
        `Playing policy ONNX could not be loaded: ${formatErrorMessage(error)}`
      );
    }

    return this.delegate.selectAction(input);
  }
}

function formatErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
