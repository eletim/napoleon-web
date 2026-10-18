import { authHeaders } from "./auth";
import type {
  CreateGameRequest,
  CreateGameResponse,
  AiPolicyComposition,
  AiPreset,
  AiPresetId,
  AdvanceMatchResponse,
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
  return request<GetGameResponse>(gamePath(gameId));
}

export async function sendAction(
  gameId: string,
  action: PublicGameAction
): Promise<SendActionResponse> {
  return request<SendActionResponse>(gamePath(gameId, "/actions"), {
    method: "POST",
    body: JSON.stringify({ action })
  });
}

export async function nextTrick(gameId: string): Promise<NextTrickResponse> {
  return request<NextTrickResponse>(gamePath(gameId, "/next-trick"), {
    method: "POST"
  });
}

export async function advanceMatch(gameId: string): Promise<AdvanceMatchResponse> {
  return request<AdvanceMatchResponse>(gamePath(gameId, "/next-round"), {
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

  const authorization = await authHeaders();
  const ratedId = /\/rated-matches\/(rated-[^/]+)/.exec(path)?.[1];
  if (ratedId && init.method === "POST") authorization["If-Match"] = String(versions.get(ratedId) ?? -1);
  const response = await fetch(path, {
    ...init,
    headers: Object.keys(authorization).length === 0 ? headers : { ...headers, ...authorization }
  });

  if (!response.ok) {
    const fallback = `HTTP ${response.status}`;
    const message = await readErrorMessage(response, fallback);
    throw new Error(message);
  }

  const body = await response.json();
  if (body.gameId && body.competition) versions.set(body.gameId, Math.max(versions.get(body.gameId) ?? -1, body.competition.version));
  return body as T;
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

const versions = new Map<string, number>();
function gamePath(id: string, suffix = "") {
  return resolveAppPath(`api/${id.startsWith("rated-") ? "rated-matches" : "games"}/${id}${suffix}`);
}
export interface CpuEntry { id: string; displayName: string; rating: number; games: number; }
export interface QueueState { status: "idle" | "waiting" | "matched"; gameId?: string; waitedMs?: number; fillAfterMs?: number; }
export const getCpuOpponents = () => request<{ cpus: CpuEntry[] }>(resolveAppPath("api/cpus"));
export const getMyRating = () => request<CpuEntry>(resolveAppPath("api/ratings/me"));
export const getMatchmaking = () => request<QueueState>(resolveAppPath("api/matchmaking"));
export const joinMatchmaking = (cpuId: string) => request<QueueState>(resolveAppPath("api/matchmaking"), { method: "POST", body: JSON.stringify({ cpuId }) });
export const cancelMatchmaking = () => request<QueueState>(resolveAppPath("api/matchmaking"), { method: "DELETE" });
export const startCpuMatch = (cpuId: string) => request<CreateGameResponse>(resolveAppPath("api/matchmaking/cpu"), { method: "POST", body: JSON.stringify({ cpuId }) });
