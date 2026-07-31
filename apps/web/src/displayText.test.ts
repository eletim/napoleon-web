import { describe, expect, it } from "vitest";
import type { PublicGameState } from "@napoleon/protocol";
import {
  createMessage,
  formatAdjutant,
  formatContract,
  formatPlayerLabel
} from "./displayText";
import { createTablePlayers } from "./tablePlayers";

describe("display text", () => {
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
        calledCardId: "hearts-A",
        revealedPlayerId: null
      }
    });

    expect(formatAdjutant(state.adjutant, createTablePlayers(state))).toBe("A♥ / 未判明");
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
            standardCardsOnly: true
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
