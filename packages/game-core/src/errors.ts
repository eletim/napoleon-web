export type GameRuleErrorCode =
  | "INVALID_PLAYER_COUNT"
  | "TRICK_COMPLETE"
  | "GAME_OVER"
  | "PLAYER_NOT_FOUND"
  | "NOT_PLAYERS_TURN"
  | "CARD_NOT_IN_HAND"
  | "TRICK_NOT_COMPLETE";

export class GameRuleError extends Error {
  readonly code: GameRuleErrorCode;

  constructor(code: GameRuleErrorCode, message: string) {
    super(message);
    this.name = "GameRuleError";
    this.code = code;
  }
}
