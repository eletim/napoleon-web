export type GameRuleErrorCode =
  | "INVALID_PLAYER_COUNT"
  | "TRICK_COMPLETE"
  | "GAME_OVER"
  | "PLAYER_NOT_FOUND"
  | "NOT_PLAYERS_TURN"
  | "CARD_NOT_IN_HAND"
  | "MUST_FOLLOW_SUIT"
  | "TRICK_NOT_COMPLETE"
  | "INVALID_ACTION_FOR_PHASE"
  | "INVALID_BID"
  | "BIDDING_NOT_AVAILABLE"
  | "BID_TOO_LOW"
  | "TRUMP_NOT_SET"
  | "NOT_NAPOLEON"
  | "INVALID_DISCARD_COUNT"
  | "DUPLICATE_CARD_ID"
  | "EXCHANGE_ALREADY_COMPLETED"
  | "INVALID_EXCHANGE_STATE"
  | "INVALID_ADJUTANT_CARD"
  | "ADJUTANT_ALREADY_CHOSEN"
  | "INVALID_ADJUTANT_STATE"
  | "INVALID_RESULT_STATE"
  | "POINT_CARD_COUNT_MISMATCH";

export class GameRuleError extends Error {
  readonly code: GameRuleErrorCode;

  constructor(code: GameRuleErrorCode, message: string) {
    super(message);
    this.name = "GameRuleError";
    this.code = code;
  }
}
