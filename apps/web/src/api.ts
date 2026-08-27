import type {
  CreateGameRequest,
  CreateGameResponse,
  AiPolicyComposition,
  AiPreset,
  AiPresetId,
  GetAiPresetsResponse,
  GetAgentsResponse,
  GetGameResponse,
  NextTrickResponse,
  PublicGameAction,
  RunAutomatedSimulationResponse,
  SendActionResponse
} from "@napoleon/protocol";
import { resolveAppPath } from "./appPath";

export async function getAgents(): Promise<GetAgentsResponse> {
  return request<GetAgentsResponse>(resolveAppPath("api/agents"));
}

export async function getAiPresets(): Promise<GetAiPresetsResponse> {
  return request<GetAiPresetsResponse>(resolveAppPath("api/ai-presets"));
}

export async function updateAiPreset(
  presetId: AiPresetId,
  composition: AiPolicyComposition
): Promise<AiPreset> {
  return request<AiPreset>(resolveAppPath(`api/ai-presets/${presetId}`), {
    method: "PUT",
    body: JSON.stringify({ composition })
  });
}

export async function createGame(requestBody: CreateGameRequest = {}): Promise<CreateGameResponse> {
  return request<CreateGameResponse>(resolveAppPath("api/games"), {
    method: "POST",
    body: JSON.stringify(requestBody)
  });
}

export async function getGame(gameId: string): Promise<GetGameResponse> {
  return request<GetGameResponse>(resolveAppPath(`api/games/${gameId}`));
}

export async function sendAction(
  gameId: string,
  action: PublicGameAction
): Promise<SendActionResponse> {
  return request<SendActionResponse>(resolveAppPath(`api/games/${gameId}/actions`), {
    method: "POST",
    body: JSON.stringify({ action })
  });
}

export async function nextTrick(gameId: string): Promise<NextTrickResponse> {
  return request<NextTrickResponse>(resolveAppPath(`api/games/${gameId}/next-trick`), {
    method: "POST"
  });
}

export async function runAutomatedSimulation(
  seed: number
): Promise<RunAutomatedSimulationResponse> {
  return request<RunAutomatedSimulationResponse>(resolveAppPath("api/simulations"), {
    method: "POST",
    body: JSON.stringify({ seed })
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

  const response = await fetch(path, {
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
