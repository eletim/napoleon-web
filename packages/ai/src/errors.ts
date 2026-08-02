import type { PlayerId } from "@napoleon/game-core";

export class NoLegalActionsError extends Error {
  constructor(playerId: PlayerId) {
    super(`No legal actions are available for ${playerId}.`);
    this.name = "NoLegalActionsError";
  }
}
