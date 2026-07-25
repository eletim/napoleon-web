import type {
  CreateGameResponse,
  GetGameResponse,
  NextTrickResponse,
  SendActionResponse
} from "@napoleon/protocol";
import type { GameAction } from "@napoleon/game-core";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:3000";

export async function createGame(): Promise<CreateGameResponse> {
  return request<CreateGameResponse>("/api/games", {
    method: "POST",
    body: JSON.stringify({})
  });
}

export async function getGame(gameId: string): Promise<GetGameResponse> {
  return request<GetGameResponse>(`/api/games/${gameId}`);
}

export async function sendAction(
  gameId: string,
  action: GameAction
): Promise<SendActionResponse> {
  return request<SendActionResponse>(`/api/games/${gameId}/actions`, {
    method: "POST",
    body: JSON.stringify({ action })
  });
}

export async function nextTrick(gameId: string): Promise<NextTrickResponse> {
  return request<NextTrickResponse>(`/api/games/${gameId}/next-trick`, {
    method: "POST"
  });
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers =
    init.body === undefined
      ? init.headers
      : {
          "Content-Type": "application/json",
          ...init.headers
        };

  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers
  });

  if (!response.ok) {
    const fallback = `HTTP ${response.status}`;
    const message = await readErrorMessage(response, fallback);
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

async function readErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const body = (await response.json()) as unknown;

    if (
      typeof body === "object" &&
      body !== null &&
      "error" in body &&
      typeof body.error === "object" &&
      body.error !== null &&
      "message" in body.error &&
      typeof body.error.message === "string"
    ) {
      return body.error.message;
    }
  } catch {
    return fallback;
  }

  return fallback;
}
