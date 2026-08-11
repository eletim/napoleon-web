import { describe, expect, it } from "vitest";
import type { PublicGameState } from "@napoleon/protocol";
import {
  createGameStatusDisplay,
  createMessage,
  formatAdjutant,
  formatCardId,
  formatContract,
  formatPlayerLabel
} from "./displayText";
import { createTablePlayers } from "./tablePlayers";

describe("display text", () => {
  it("does not generate undecided status chips before a game starts", () => {
    expect(createGameStatusDisplay(undefined, [])).toEqual({
      primary: [],
      secondary: []
    });
  });

  it("shows only compact bidding phase with current turn in the accessible name", () => {
    const state = createPublicState({
      phase: "bidding",
      currentPlayerId: "player-2",
      trumpSuit: null,
      contract: null,
      adjutant: null,
      specialCards: {
        orumaCardId: "spades-A",
        yoromekiCardId: "hearts-Q",
        seiJackCardId: null,
        uraJackCardId: null
      }
    });
    const display = createGameStatusDisplay(state, createTablePlayers(state));

    expect(display.primary).toEqual([
      {
        label: "競り",
        ariaLabel: "競り。現在の手番は奥左AIです。",
        tone: "phase"
      }
    ]);
    expect(display.secondary).toEqual([]);
    expect(display.primary.some((chip) => chip.label === "契約")).toBe(false);
    expect(display.primary.some((chip) => chip.label === "切り札")).toBe(false);
  });

  it("shows exchange setup without adjutant status", () => {
    const state = createPublicState({
      phase: "exchanging",
      currentPlayerId: "player-4",
      contract: {
        napoleonPlayerId: "player-4",
        trumpSuit: "diamonds",
        targetPointCards: 14
      },
      trumpSuit: "diamonds"
    });
    const display = createGameStatusDisplay(state, createTablePlayers(state));

    expect(display.primary).toEqual([
      { label: "交換", ariaLabel: "埋札交換", tone: "phase" },
      {
        label: "♦",
        value: "14",
        ariaLabel: "契約: ナポレオンは右側AI、切り札は♦、契約は14枚です。",
        tone: "contract",
        suitColor: "red"
      }
    ]);
    expect(display.secondary).toEqual([]);
    expect(display.secondary.some((chip) => chip.label === "副")).toBe(false);
  });

  it("shows the chosen adjutant card during exchange without revealing its owner", () => {
    const state = createPublicState({
      phase: "exchanging",
      currentPlayerId: "player-4",
      contract: {
        napoleonPlayerId: "player-4",
        trumpSuit: "diamonds",
        targetPointCards: 14
      },
      trumpSuit: "diamonds",
      adjutant: {
        calledCardId: "joker",
        revealedPlayerId: null
      }
    });
    const display = createGameStatusDisplay(state, createTablePlayers(state));

    expect(display.primary).toEqual([
      { label: "交換", ariaLabel: "埋札交換", tone: "phase" },
      {
        label: "♦",
        value: "14",
        ariaLabel: "契約: ナポレオンは右側AI、切り札は♦、契約は14枚です。",
        tone: "contract",
        suitColor: "red"
      }
    ]);
    expect(display.secondary).toContainEqual({
      label: "副",
      value: "ジョーカー・?",
      ariaLabel: "副官札: ジョーカー / 未判明",
      tone: "role"
    });
  });

  it("shows adjutant choice setup without undecided adjutant clutter", () => {
    const state = createPublicState({
      phase: "choosing-adjutant",
      contract: {
        napoleonPlayerId: "player-4",
        trumpSuit: "diamonds",
        targetPointCards: 14
      },
      trumpSuit: "diamonds",
      adjutant: null
    });
    const display = createGameStatusDisplay(state, createTablePlayers(state));

    expect(display.primary).toEqual([
      { label: "副官", ariaLabel: "副官指定", tone: "phase" },
      {
        label: "♦",
        value: "14",
        ariaLabel: "契約: ナポレオンは右側AI、切り札は♦、契約は14枚です。",
        tone: "contract",
        suitColor: "red"
      }
    ]);
    expect(display.secondary.some((chip) => chip.label === "副")).toBe(false);
  });

  it("shows play status as compact trick, contract, and adjutant without special-card marks", () => {
    const state = createPublicState({
      phase: "playing",
      trickNumber: 4,
      currentPlayerId: "player-0",
      contract: {
        napoleonPlayerId: "player-4",
        trumpSuit: "diamonds",
        targetPointCards: 14
      },
      trumpSuit: "diamonds",
      adjutant: {
        calledCardId: "hearts-A",
        revealedPlayerId: null
      },
      specialCards: {
        orumaCardId: "spades-A",
        yoromekiCardId: "hearts-Q",
        seiJackCardId: "diamonds-J",
        uraJackCardId: "hearts-J"
      }
    });
    const display = createGameStatusDisplay(state, createTablePlayers(state));

    expect(display.primary).toEqual([
      {
        label: "T4",
        ariaLabel: "第4トリック。現在の手番は自分です。",
        tone: "phase"
      },
      {
        label: "♦",
        value: "14",
        ariaLabel: "契約: ナポレオンは右側AI、切り札は♦、契約は14枚です。",
        tone: "contract",
        suitColor: "red"
      }
    ]);
    expect(display.secondary).toEqual([
      {
        label: "副",
        value: "A♥・?",
        ariaLabel: "副官札: A♥ / 未判明",
        tone: "role"
      }
    ]);
  });

  it("uses revealed adjutant seat labels and omits undecided jack chips", () => {
    const state = createPublicState({
      phase: "playing",
      adjutant: {
        calledCardId: "hearts-A",
        revealedPlayerId: "player-2"
      },
      specialCards: {
        orumaCardId: "spades-A",
        yoromekiCardId: "hearts-Q",
        seiJackCardId: null,
        uraJackCardId: null
      }
    });
    const display = createGameStatusDisplay(state, createTablePlayers(state));

    expect(display.secondary).toContainEqual({
      label: "副",
      value: "A♥・奥左AI",
      ariaLabel: "副官札: A♥ / 奥左AI",
      tone: "role"
    });
    expect(display.secondary.some((chip) => chip.label === "オ")).toBe(false);
    expect(display.secondary.some((chip) => chip.label === "よ")).toBe(false);
    expect(display.secondary.some((chip) => chip.label === "正")).toBe(false);
    expect(display.secondary.some((chip) => chip.label === "裏")).toBe(false);
  });

  it("shows concise finished status without duplicating score details", () => {
    const state = createPublicState({
      phase: "finished",
      isGameOver: true,
      result: {
        winner: "napoleon-team",
        napoleonTeamPointCards: 15,
        alliancePointCards: 5,
        targetPointCards: 14,
        napoleonPlayerId: "player-4",
        adjutantPlayerId: "player-2"
      },
      contract: {
        napoleonPlayerId: "player-4",
        trumpSuit: "diamonds",
        targetPointCards: 14
      },
      trumpSuit: "diamonds"
    });
    const display = createGameStatusDisplay(state, createTablePlayers(state));

    expect(display.primary).toEqual([
      { label: "終了", ariaLabel: "ゲーム終了", tone: "phase" },
      { label: "N勝", ariaLabel: "勝者: ナポレオン陣営", tone: "result" },
      {
        label: "♦",
        value: "14",
        ariaLabel: "契約: ナポレオンは右側AI、切り札は♦、契約は14枚です。",
        tone: "contract",
        suitColor: "red"
      }
    ]);
    expect(display.primary.some((chip) => chip.label === "ナポレオン陣営")).toBe(false);
    expect(display.primary.some((chip) => chip.label === "連合軍")).toBe(false);
  });

  it("formats contract and revealed adjutant owner with seat labels", () => {
    const state = createPublicState({
      contract: {
        napoleonPlayerId: "player-4",
        trumpSuit: "diamonds",
        targetPointCards: 14
      },
      adjutant: {
        calledCardId: "hearts-A",
        revealedPlayerId: "player-2"
      }
    });
    const players = createTablePlayers(state);

    expect(formatContract(state, players)).toBe("右側AI ♦14");
    expect(formatAdjutant(state.adjutant, players)).toBe("A♥ / 奥左AI");
  });

  it("keeps unrevealed adjutant owner hidden", () => {
    const state = createPublicState({
      adjutant: {
        calledCardId: "joker",
        revealedPlayerId: null
      }
    });

    expect(formatAdjutant(state.adjutant, createTablePlayers(state))).toBe("ジョーカー / 未判明");
    expect(formatCardId("joker")).toBe("ジョーカー");
  });

  it("uses seat labels in phase messages", () => {
    const players = createTablePlayers(createPublicState());

    expect(
      createMessage(
        createPublicState({
          phase: "bidding",
          currentPlayerId: "player-2"
        }),
        "player-0",
        players
      )
    ).toBe("奥左AIの競り手番です。");
    expect(
      createMessage(
        createPublicState({
          phase: "exchanging",
          currentPlayerId: "player-4",
          exchange: {
            napoleonPlayerId: "player-4",
            requiredDiscardCount: 3
          }
        }),
        "player-0",
        players
      )
    ).toBe("右側AIが埋札交換中です。");
    expect(
      createMessage(
        createPublicState({
          phase: "choosing-adjutant",
          currentPlayerId: "player-4",
          adjutantChoice: {
            napoleonPlayerId: "player-4",
            jokerAllowed: true
          }
        }),
        "player-0",
        players
      )
    ).toBe("右側AIが副官を指定中です。");
    expect(
      createMessage(
        createPublicState({
          phase: "playing",
          currentPlayerId: "player-3"
        }),
        "player-0",
        players
      )
    ).toBe("奥右AIの番です。");
  });

  it("falls back to raw ids for unknown players", () => {
    expect(formatPlayerLabel("player-9", createTablePlayers(createPublicState()))).toBe("player-9");
  });
});

function createPublicState(overrides: Partial<PublicGameState> = {}): PublicGameState {
  return {
    self: {
      id: "player-0",
      handCount: 10,
      hand: [],
      capturedPointCards: []
    },
    opponents: [
      { id: "player-1", handCount: 10, capturedPointCards: [] },
      { id: "player-2", handCount: 10, capturedPointCards: [] },
      { id: "player-3", handCount: 10, capturedPointCards: [] },
      { id: "player-4", handCount: 10, capturedPointCards: [] }
    ],
    phase: "playing",
    trumpSuit: "spades",
    contract: {
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 13
    },
    specialCards: {
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: "spades-J",
      uraJackCardId: "clubs-J"
    },
    adjutant: null,
    latestEvent: null,
    result: null,
    bidding: null,
    exchange: null,
    adjutantChoice: null,
    currentPlayerId: "player-0",
    currentTrick: [],
    completedTrickCount: 0,
    trickNumber: 1,
    isTrickComplete: false,
    isGameOver: false,
    legalActions: [],
    ...overrides
  };
}
