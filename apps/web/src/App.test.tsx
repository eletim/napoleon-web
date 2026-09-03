// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  AdvanceMatchResponse,
  CreateGameResponse,
  GetAiPresetsResponse,
  PublicGameState,
  PublicMatchPlayerFinalScore,
  PublicMatchState,
  PublicPlayedCard,
  PublicRank,
  PublicSuit
} from "@napoleon/protocol";
import { App } from "./App";
import { advanceMatch, createGame, getAiPresets, nextTrick } from "./api";

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
  TableSurface: ({ actionPanel, match }: { actionPanel?: React.ReactNode; match?: PublicMatchState }) => (
    <section aria-label="ゲームテーブル">
      {match === undefined ? null : (
        <span className="production-match-round">第{match.currentRound}局</span>
      )}
      <div className="production-table-background">卓背景</div>
      <a href="#round-details">結果リンク</a>
      <details>
        <summary>局の詳細</summary>
        <p id="round-details">詳細内容</p>
      </details>
      {actionPanel}
    </section>
  )
}));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

afterEach(() => {
  vi.clearAllMocks();
});

describe("App match final result flow", () => {
  it.each([1, 2, 3, 4])(
    "advances round %i from the non-interactive result background",
    async (currentRound) => {
      vi.mocked(getAiPresets).mockResolvedValue(aiPresetResponse());
      vi.mocked(createGame).mockResolvedValue(roundResultSession(currentRound));
      vi.mocked(advanceMatch).mockResolvedValue(freshSession(currentRound + 1));

      const { container, root } = await renderStartedApp();

      expect(container.textContent).toContain("次局へ");
      await clickElement(container.querySelector(".production-table-background"));

      expect(advanceMatch).toHaveBeenCalledTimes(1);
      expect(advanceMatch).toHaveBeenCalledWith("round-result-game");
      expect(container.querySelector(".production-match-round")?.textContent).toBe(
        `第${currentRound + 1}局`
      );

      await act(async () => root.unmount());
      container.remove();
    }
  );

  it("opens the final results from the fifth-round result background", async () => {
    vi.mocked(getAiPresets).mockResolvedValue(aiPresetResponse());
    vi.mocked(createGame).mockResolvedValue(roundResultSession(5));
    vi.mocked(advanceMatch).mockResolvedValue(completedSession());

    const { container, root } = await renderStartedApp();

    expect(container.textContent).toContain("試合結果へ");
    await clickElement(container.querySelector(".round-result-winner"));

    expect(advanceMatch).toHaveBeenCalledTimes(1);
    expect(container.querySelector(".match-final-results")).not.toBeNull();
    expect(container.textContent).toContain("全5局 終了");

    await act(async () => root.unmount());
    container.remove();
  });

  it("does not advance from interactive result-screen elements", async () => {
    vi.mocked(getAiPresets).mockResolvedValue(aiPresetResponse());
    vi.mocked(createGame).mockResolvedValue(roundResultSession(2));

    const { container, root } = await renderStartedApp();

    await clickElement(container.querySelector("summary"));
    await clickElement(container.querySelector("#round-details"));
    await clickElement(container.querySelector("a"));
    expect(advanceMatch).not.toHaveBeenCalled();

    await act(async () => root.unmount());
    container.remove();
  });

  it("uses one request for a bubbling button click and rapid background clicks", async () => {
    vi.mocked(getAiPresets).mockResolvedValue(aiPresetResponse());
    vi.mocked(createGame).mockResolvedValue(roundResultSession(3));
    let resolveAdvance: ((session: AdvanceMatchResponse) => void) | undefined;
    vi.mocked(advanceMatch).mockImplementation(
      () => new Promise((resolve) => {
        resolveAdvance = resolve;
      })
    );

    const { container, root } = await renderStartedApp();

    await act(async () => {
      const button = [...container.querySelectorAll("button")].find(
        (candidate) => candidate.textContent?.trim() === "次局へ"
      );
      button?.click();
      container.querySelector<HTMLElement>(".production-table-background")?.click();
      container.querySelector<HTMLElement>(".production-table-background")?.click();
    });

    expect(advanceMatch).toHaveBeenCalledTimes(1);

    await act(async () => resolveAdvance?.(freshSession(4)));
    await act(async () => root.unmount());
    container.remove();
  });

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
    expect(createGame).toHaveBeenNthCalledWith(2, {
      aiAgents: [
        { playerId: "player-1", policyComposition: aiPresetResponse().presets[0].composition },
        { playerId: "player-2", policyComposition: aiPresetResponse().presets[0].composition },
        { playerId: "player-3", policyComposition: aiPresetResponse().presets[0].composition },
        { playerId: "player-4", policyComposition: aiPresetResponse().presets[0].composition }
      ]
    });
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

describe("per-seat AI preset selection", () => {
  it("sends each of the four seats' independently selected AI preset when starting a game", async () => {
    vi.mocked(getAiPresets).mockResolvedValue(twoAiPresetResponse());
    vi.mocked(createGame).mockResolvedValue(freshSession());

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<App />);
    });

    // Mixed selection: RuleBase / AI / RuleBase / AI across the four seats,
    // with the same preset ("com-rule-base") reused on two different seats.
    await selectSeatPreset(container, "左側AI", "com-rule-base");
    await selectSeatPreset(container, "奥左AI", "com-ai");
    await selectSeatPreset(container, "奥右AI", "com-rule-base");
    await selectSeatPreset(container, "右側AI", "com-ai");

    await clickButton(container, "ゲーム開始");

    const { presets } = twoAiPresetResponse();
    const ruleBase = presets.find(({ id }) => id === "com-rule-base")?.composition;
    const ai = presets.find(({ id }) => id === "com-ai")?.composition;

    expect(createGame).toHaveBeenCalledWith({
      aiAgents: [
        { playerId: "player-1", policyComposition: ruleBase },
        { playerId: "player-2", policyComposition: ai },
        { playerId: "player-3", policyComposition: ruleBase },
        { playerId: "player-4", policyComposition: ai }
      ]
    });

    await act(async () => root.unmount());
    container.remove();
  });

  it("does not turn the human's own seat into a COM seat", async () => {
    vi.mocked(getAiPresets).mockResolvedValue(twoAiPresetResponse());
    vi.mocked(createGame).mockResolvedValue(freshSession());

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<App />);
    });

    expect(container.querySelector('[aria-label="自分の対戦AI"]')).toBeNull();
    expect(container.querySelectorAll(".agent-selector-grid select")).toHaveLength(4);

    await clickButton(container, "ゲーム開始");

    const [request] = vi.mocked(createGame).mock.calls[0] ?? [];
    expect(request?.aiAgents?.some((selection) => selection.playerId === "player-0")).toBe(false);

    await act(async () => root.unmount());
    container.remove();
  });

  it("still starts a game when all four seats keep the same default preset unset", async () => {
    vi.mocked(getAiPresets).mockResolvedValue(twoAiPresetResponse());
    vi.mocked(createGame).mockResolvedValue(freshSession());

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<App />);
    });

    await clickButton(container, "ゲーム開始");

    const defaultComposition = twoAiPresetResponse().presets
      .find(({ id }) => id === "com-ai")?.composition;

    expect(createGame).toHaveBeenCalledWith({
      aiAgents: [
        { playerId: "player-1", policyComposition: defaultComposition },
        { playerId: "player-2", policyComposition: defaultComposition },
        { playerId: "player-3", policyComposition: defaultComposition },
        { playerId: "player-4", policyComposition: defaultComposition }
      ]
    });

    await act(async () => root.unmount());
    container.remove();
  });

  it("lets every seat share the same explicitly selected preset", async () => {
    vi.mocked(getAiPresets).mockResolvedValue(twoAiPresetResponse());
    vi.mocked(createGame).mockResolvedValue(freshSession());

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<App />);
    });

    for (const seatLabel of ["左側AI", "奥左AI", "奥右AI", "右側AI"]) {
      await selectSeatPreset(container, seatLabel, "com-rule-base");
    }

    await clickButton(container, "ゲーム開始");

    const ruleBase = twoAiPresetResponse().presets
      .find(({ id }) => id === "com-rule-base")?.composition;

    expect(createGame).toHaveBeenCalledWith({
      aiAgents: [
        { playerId: "player-1", policyComposition: ruleBase },
        { playerId: "player-2", policyComposition: ruleBase },
        { playerId: "player-3", policyComposition: ruleBase },
        { playerId: "player-4", policyComposition: ruleBase }
      ]
    });

    await act(async () => root.unmount());
    container.remove();
  });
});

describe("automatic trick progression", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("advances to the next trick on its own once the trick animation finishes, without a next-trick button", async () => {
    vi.useFakeTimers();
    vi.mocked(getAiPresets).mockResolvedValue(aiPresetResponse());
    vi.mocked(createGame).mockResolvedValue({
      gameId: "trick-game",
      playerId: "player-0",
      state: trickCompleteState()
    });
    vi.mocked(nextTrick).mockResolvedValue({
      gameId: "trick-game",
      playerId: "player-0",
      state: trickCompleteState({
        currentTrick: [],
        isTrickComplete: false,
        trickNumber: 4
      })
    });

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<App />);
    });
    await act(async () => {
      const button = [...container.querySelectorAll("button")].find(
        (candidate) => candidate.textContent?.trim() === "ゲーム開始"
      );
      button?.click();
    });

    expect(container.textContent).not.toContain("次へ");
    expect(nextTrick).not.toHaveBeenCalled();

    // Let the show-then-collect animation run its course.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });

    expect(nextTrick).toHaveBeenCalledWith("trick-game");

    await act(async () => root.unmount());
    container.remove();
  });

  it("does not reveal the round-end screen until the final trick's animation finishes", async () => {
    vi.useFakeTimers();
    vi.mocked(getAiPresets).mockResolvedValue(aiPresetResponse());
    vi.mocked(createGame).mockResolvedValue({
      gameId: "final-trick-game",
      playerId: "player-0",
      state: trickCompleteState({
        isGameOver: true,
        phase: "finished",
        result: {
          resultType: "standard",
          winner: "napoleon-team",
          napoleonTeamPointCards: 14,
          alliancePointCards: 6,
          targetPointCards: 13,
          napoleonPlayerId: "player-1",
          adjutantPlayerId: null
        },
        trickNumber: 10
      })
    });

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<App />);
    });
    await act(async () => {
      const button = [...container.querySelectorAll("button")].find(
        (candidate) => candidate.textContent?.trim() === "ゲーム開始"
      );
      button?.click();
    });

    // The final card just arrived: the round-end result panel must not have
    // instantly replaced the (still animating) trick view.
    expect(container.querySelector('[aria-label="ゲーム結果"]')).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });

    expect(container.querySelector('[aria-label="ゲーム結果"]')).not.toBeNull();

    await act(async () => root.unmount());
    container.remove();
  });
});

function playedCard(playerId: string, suit: PublicSuit, rank: PublicRank): PublicPlayedCard {
  return {
    playerId,
    card: {
      type: "standard",
      id: `${suit}-${rank}`,
      suit,
      rank
    }
  };
}

function trickCompleteState(overrides: Partial<PublicGameState> = {}): PublicGameState {
  return {
    self: { id: "player-0", handCount: 9, hand: [], capturedPointCards: [] },
    opponents: playerIds.slice(1).map((id) => ({ id, handCount: 9, capturedPointCards: [] })),
    phase: "playing",
    trumpSuit: "spades",
    contract: {
      napoleonPlayerId: "player-1",
      trumpSuit: "spades",
      targetPointCards: 13
    },
    specialCards: {
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: null,
      uraJackCardId: null
    },
    adjutant: { calledCardId: "hearts-A", revealedPlayerId: null },
    latestEvent: null,
    result: null,
    bidding: null,
    exchange: null,
    adjutantChoice: null,
    currentPlayerId: "player-0",
    currentTrick: [
      playedCard("player-1", "spades", "10"),
      playedCard("player-2", "spades", "9"),
      playedCard("player-3", "spades", "8"),
      playedCard("player-4", "spades", "7"),
      playedCard("player-0", "spades", "6")
    ],
    completedTrickCount: 2,
    trickNumber: 3,
    isTrickComplete: true,
    isGameOver: false,
    legalActions: [],
    ...overrides
  };
}

async function selectSeatPreset(
  container: HTMLElement,
  seatLabel: string,
  presetId: string
): Promise<void> {
  const select = container.querySelector<HTMLSelectElement>(
    `[aria-label="${seatLabel}の対戦AI"]`
  );
  expect(select, `${seatLabel} select`).not.toBeNull();

  await act(async () => {
    if (select !== null) {
      select.value = presetId;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
}

function twoAiPresetResponse(): GetAiPresetsResponse {
  const policy = (id: string) => ({
    id,
    displayName: id,
    isAvailable: true,
    artifactProvenance: null
  });

  return {
    presets: [
      {
        id: "com-rule-base",
        displayName: "COM-RuleBase",
        composition: {
          playing: "rule-based",
          bidding: "rule-based",
          nonPlaying: "rule-based"
        }
      },
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
      playing: [policy("rule-based"), policy("ppo-separated-v1000")],
      bidding: [policy("rule-based"), policy("frozen-raise-v1")],
      nonPlaying: [policy("rule-based"), policy("parameterized-adjutant-exchange-v1")]
    }
  };
}

async function renderStartedApp() {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);

  await act(async () => {
    root.render(<App />);
  });
  await clickButton(container, "ゲーム開始");

  return { container, root };
}

async function clickElement(element: Element | null): Promise<void> {
  expect(element).not.toBeNull();
  await act(async () => {
    (element as HTMLElement).click();
  });
}

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

function completedSession(): AdvanceMatchResponse {
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

function freshSession(currentRound = 1): AdvanceMatchResponse {
  return {
    gameId: "fresh-game",
    playerId: "player-0",
    state: gameState(false),
    match: progressMatch(currentRound)
  };
}

function roundResultSession(currentRound: number): CreateGameResponse {
  return {
    gameId: "round-result-game",
    playerId: "player-0",
    state: gameState(true),
    match: progressMatch(currentRound, true)
  };
}

function progressMatch(currentRound = 1, hasRoundResult = false): PublicMatchState {
  const completedRoundCount = hasRoundResult ? currentRound : currentRound - 1;
  return {
    currentRound,
    roundCount: 5,
    completedRoundCount,
    remainingRounds: 5 - completedRoundCount,
    completed: false,
    players: playerIds.map((playerId) => ({
      playerId,
      roundScores: Array.from({ length: completedRoundCount }, () => 0),
      rawMatchScore: 0
    })),
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
