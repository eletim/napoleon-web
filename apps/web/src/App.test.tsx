// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  CreateGameResponse,
  GetAiPresetsResponse,
  PublicGameState,
  PublicMatchPlayerFinalScore,
  PublicMatchState
} from "@napoleon/protocol";
import { App } from "./App";
import { createGame, getAiPresets } from "./api";

vi.mock("./api", () => ({
  advanceMatch: vi.fn(),
  createGame: vi.fn(),
  getAgents: vi.fn(),
  getAiPresets: vi.fn(),
  getGame: vi.fn(),
  nextTrick: vi.fn(),
  runAutomatedSimulation: vi.fn(),
  sendAction: vi.fn(),
  updateAiPreset: vi.fn()
}));

vi.mock("./TableSurface", () => ({
  TableSurface: ({ match }: { match?: PublicMatchState }) => (
    <section aria-label="ゲームテーブル">
      {match === undefined ? null : (
        <span className="production-match-round">第{match.currentRound}局</span>
      )}
    </section>
  )
}));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

afterEach(() => {
  vi.clearAllMocks();
});

describe("App match final result flow", () => {
  it("replaces a completed match with a fresh round-one session when starting a new match", async () => {
    vi.mocked(getAiPresets).mockResolvedValue(aiPresetResponse());
    vi.mocked(createGame)
      .mockResolvedValueOnce(completedSession())
      .mockResolvedValueOnce(freshSession());

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<App />);
    });

    await clickButton(container, "ゲーム開始");

    expect(container.querySelector(".match-final-results")).not.toBeNull();
    expect(container.textContent).toContain("全5局 終了");
    expect(container.querySelector(".match-round-indicator")).toBeNull();

    await clickButton(container, "新しい試合を始める");

    expect(createGame).toHaveBeenCalledTimes(2);
    expect(createGame).toHaveBeenNthCalledWith(2, { aiPresetId: "com-ai" });
    expect(container.querySelector(".match-final-results")).toBeNull();
    expect(container.textContent).not.toContain("全5局 終了");
    expect(container.querySelector(".app-shell-match-completed")).toBeNull();
    expect(container.querySelector(".match-round-indicator")).toBeNull();
    expect(container.querySelector(".match-score-details")).toBeNull();
    expect(container.querySelector(".production-match-round")?.textContent).toBe("第1局");
    expect(container.querySelector('[aria-label="ゲームテーブル"]')).not.toBeNull();

    await act(async () => root.unmount());
    container.remove();
  });
});

async function clickButton(container: HTMLElement, label: string): Promise<void> {
  const button = [...container.querySelectorAll("button")].find(
    (candidate) => candidate.textContent?.trim() === label
  );
  expect(button, `${label} button`).toBeDefined();

  await act(async () => {
    button?.click();
  });
}

function aiPresetResponse(): GetAiPresetsResponse {
  const policy = (id: string) => ({
    id,
    displayName: id,
    isAvailable: true,
    artifactProvenance: null
  });

  return {
    presets: [
      {
        id: "com-ai",
        displayName: "COM-AI",
        composition: {
          playing: "ppo-separated-v1000",
          bidding: "frozen-raise-v1",
          nonPlaying: "parameterized-adjutant-exchange-v1"
        }
      }
    ],
    policyRegistry: {
      playing: [policy("ppo-separated-v1000")],
      bidding: [policy("frozen-raise-v1")],
      nonPlaying: [policy("parameterized-adjutant-exchange-v1")]
    }
  };
}

function completedSession(): CreateGameResponse {
  const finalScores: readonly PublicMatchPlayerFinalScore[] = [
    finalScore("player-0", 1, 20, 20, 40, 40),
    finalScore("player-1", 2, 10, 10, 20, 20),
    finalScore("player-2", 3, 0, 0, 0, 0),
    finalScore("player-3", 4, -10, -10, -20, -20),
    finalScore("player-4", 5, -20, -20, -40, -40)
  ];

  return {
    gameId: "completed-game",
    playerId: "player-0",
    state: gameState(true),
    match: {
      currentRound: 5,
      roundCount: 5,
      completedRoundCount: 5,
      remainingRounds: 0,
      completed: true,
      players: finalScores,
      finalScores
    }
  };
}

function freshSession(): CreateGameResponse {
  return {
    gameId: "fresh-game",
    playerId: "player-0",
    state: gameState(false),
    match: progressMatch()
  };
}

function progressMatch(): PublicMatchState {
  return {
    currentRound: 1,
    roundCount: 5,
    completedRoundCount: 0,
    remainingRounds: 5,
    completed: false,
    players: playerIds.map((playerId) => ({ playerId, roundScores: [], rawMatchScore: 0 })),
    finalScores: null
  };
}

function gameState(completed: boolean): PublicGameState {
  return {
    self: {
      id: "player-0",
      handCount: 0,
      hand: [],
      capturedPointCards: []
    },
    opponents: playerIds.slice(1).map((id) => ({
      id,
      handCount: completed ? 0 : 10,
      capturedPointCards: []
    })),
    phase: completed ? "finished" : "bidding",
    trumpSuit: null,
    contract: null,
    specialCards: {
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: null,
      uraJackCardId: null
    },
    adjutant: null,
    latestEvent: null,
    result: completed
      ? {
          resultType: "all-pass",
          starterPlayerId: "player-0",
          payoffs: playerIds.map((playerId) => ({ playerId, payoff: 0 }))
        }
      : null,
    bidding: completed
      ? null
      : {
          starterPlayerId: "player-0",
          highestBid: null,
          consecutivePassCount: 0,
          history: []
        },
    exchange: null,
    adjutantChoice: null,
    currentPlayerId: "player-0",
    currentTrick: [],
    completedTrickCount: 0,
    trickNumber: 1,
    isTrickComplete: false,
    isGameOver: completed,
    legalActions: completed ? [] : [{ type: "pass" }]
  };
}

function finalScore(
  playerId: string,
  rank: number,
  rawMatchScore: number,
  uma: number,
  score: number,
  finalValue: number
): PublicMatchPlayerFinalScore {
  return {
    playerId,
    rank,
    roundScores: [0, 0, 0, 0, rawMatchScore],
    rawMatchScore,
    uma,
    score,
    finalValue
  };
}

const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;
