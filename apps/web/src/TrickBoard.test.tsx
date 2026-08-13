import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { PublicPlayedCard, PublicRank, PublicSuit } from "@napoleon/protocol";
import { TrickBoard } from "./TrickBoard";
import type { TablePlayer } from "./tableTypes";

describe("TrickBoard", () => {
  it("highlights the current winning card when enabled", () => {
    const html = renderToStaticMarkup(
      <TrickBoard
        currentTrick={[
          played("player-1", "hearts", "A"),
          played("player-2", "spades", "2")
        ]}
        highlightWinningCard={true}
        players={players}
        trickNumber={2}
        trumpSuit="spades"
      />
    );

    expect(html).toContain("played-card-winning");
    expect(html).toContain("奥左AIが2♠を出しました。現在勝っています");
    expect(html).not.toContain("左側AIがA♥を出しました。現在勝っています");
  });

  it("does not highlight the current winning card when disabled", () => {
    const html = renderToStaticMarkup(
      <TrickBoard
        currentTrick={[
          played("player-1", "hearts", "A"),
          played("player-2", "spades", "2")
        ]}
        highlightWinningCard={false}
        players={players}
        trickNumber={2}
        trumpSuit="spades"
      />
    );

    expect(html).not.toContain("played-card-winning");
    expect(html).not.toContain("現在勝っています");
  });

  it("marks trick cards with entry and collection direction classes", () => {
    const html = renderToStaticMarkup(
      <TrickBoard
        collectingWinnerId="player-2"
        currentTrick={[
          played("player-1", "hearts", "A"),
          played("player-2", "spades", "2")
        ]}
        highlightWinningCard={true}
        isResultEmphasisActive={true}
        players={players}
        trickNumber={2}
        trumpSuit="spades"
      />
    );

    expect(html).toContain("trick-board-result");
    expect(html).toContain("trick-board-collecting-to-top-left");
    expect(html).toContain("played-card-from-left");
    expect(html).toContain("played-card-from-top-left");
    expect(html).toContain("played-card-collecting");
  });

  it("renders a mobile center summary for the contract and adjutant card", () => {
    const html = renderToStaticMarkup(
      <TrickBoard
        adjutant={{ calledCardId: "spades-A", revealedPlayerId: "player-2" }}
        contract={{
          napoleonPlayerId: "player-1",
          trumpSuit: "spades",
          targetPointCards: 13
        }}
        currentTrick={[
          played("player-1", "hearts", "A"),
          played("player-2", "spades", "2"),
          played("player-3", "clubs", "K"),
          played("player-4", "diamonds", "Q")
        ]}
        highlightWinningCard={false}
        players={players}
        trickNumber={2}
        trumpSuit="spades"
      />
    );

    expect(html).toContain("♠ 13");
    expect(html).toContain("副官 ♠A");
    expect(html).toContain("trick-mobile-status-summary");
    expect(html).toContain("trick-count-summary");
    expect(html).not.toContain("副官 ♠A・");
  });
});

const players: readonly TablePlayer[] = [
  createPlayer("player-1", "左側AI", "left"),
  createPlayer("player-2", "奥左AI", "top-left"),
  createPlayer("player-3", "奥右AI", "top-right"),
  createPlayer("player-4", "右側AI", "right"),
  createPlayer("player-0", "自分", "self")
];

function createPlayer(id: string, label: string, seat: TablePlayer["seat"]): TablePlayer {
  return {
    id,
    label,
    seat,
    handCount: 10,
    capturedPointCards: [],
    isSelf: seat === "self",
    biddingDeclaration: undefined
  };
}

function played(playerId: string, suit: PublicSuit, rank: PublicRank): PublicPlayedCard {
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
