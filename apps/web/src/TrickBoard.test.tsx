import { readFileSync } from "node:fs";
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

  it("renders only contract and adjutant details in the center summary", () => {
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

    expect(html).toContain("♠13");
    expect(html).toContain("副官 ♠A");
    expect(html).toContain("trick-status-summary");
    expect(html).not.toContain("4 / 5");
    expect(html).not.toContain("trick-count-summary");
    expect(html).not.toContain("副官 ♠A・");
  });

  it("marks red suit contract and adjutant summaries with the red text class", () => {
    const redHtml = renderToStaticMarkup(
      <TrickBoard
        adjutant={{ calledCardId: "diamonds-J", revealedPlayerId: "player-2" }}
        contract={{
          napoleonPlayerId: "player-1",
          trumpSuit: "hearts",
          targetPointCards: 13
        }}
        currentTrick={[]}
        highlightWinningCard={false}
        players={players}
        trickNumber={2}
        trumpSuit="hearts"
      />
    );
    const blackHtml = renderToStaticMarkup(
      <TrickBoard
        adjutant={{ calledCardId: "clubs-J", revealedPlayerId: "player-2" }}
        contract={{
          napoleonPlayerId: "player-1",
          trumpSuit: "spades",
          targetPointCards: 13
        }}
        currentTrick={[]}
        highlightWinningCard={false}
        players={players}
        trickNumber={2}
        trumpSuit="spades"
      />
    );

    expect(redHtml).toContain('class="trick-contract-summary red-text">♥13');
    expect(redHtml).toContain('副官 <span class="red-text">♦J</span>');
    expect(blackHtml).toContain('class="trick-contract-summary">♠13');
    expect(blackHtml).toContain("副官 ♣J");
    expect(blackHtml).not.toContain('class="trick-contract-summary red-text">♠13');
    expect(blackHtml).not.toContain('<span class="red-text">♣J</span>');
  });

  it("uses shared simple center summary styling across desktop and mobile", () => {
    const styles = readFileSync("src/styles.css", "utf8");
    const landscapeBlock = getMediaBlock(
      styles,
      "@media (max-width: 960px) and (max-height: 560px) and (orientation: landscape)"
    );

    expect(styles).not.toContain("trick-count-summary");
    expect(styles).not.toContain("trick-mobile-status-summary");
    expect(styles).toContain(".trick-status-summary");
    expect(styles).toContain("background: transparent;");
    expect(styles).toContain("border: 0;");
    expect(landscapeBlock).toContain(".app-shell-game-active .trick-status-summary");
    expect(landscapeBlock).toContain("font-size: 1.08rem;");
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

function getMediaBlock(styles: string, query: string): string {
  const start = styles.indexOf(query);

  if (start === -1) {
    throw new Error(`Media query not found: ${query}`);
  }

  const nextMedia = styles.indexOf("@media", start + query.length);

  return styles.slice(start, nextMedia === -1 ? undefined : nextMedia);
}
