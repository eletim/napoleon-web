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

export class UnknownPhasePolicyIdError extends Error {
  constructor(readonly axis: "playing" | "bidding" | "nonPlaying", readonly policyId: string) {
    super(`Unknown ${axis} policy id: ${policyId}.`);
    this.name = "UnknownPhasePolicyIdError";
  }
}
