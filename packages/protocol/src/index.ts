import type { GameAction, PlayerView } from "@napoleon/game-core";

export interface CreateGameRequest {
  humanPlayerName?: string;
}

export interface CreateGameResponse {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

export type PublicGameState = PlayerView;

export interface SendActionRequest {
  action: GameAction;
}

export interface SendActionResponse {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

export interface GetGameResponse {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

export interface NextTrickResponse {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

export interface ApiError {
  error: {
    code: string;
    message: string;
  };
}
