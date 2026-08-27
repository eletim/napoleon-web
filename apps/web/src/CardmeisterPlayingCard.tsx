import { createElement, useEffect, type CSSProperties } from "react";
import { resolveAppPath } from "./appPath";
import { cardmeisterFourColorCsv } from "./cardSuitTheme";
import type { MockPlayingCard } from "./mockPlayingCardAdapter";

export function CardmeisterPlayingCard({
  card,
  className,
  style
}: {
  card: MockPlayingCard;
  className: string;
  style?: CSSProperties;
}) {
  return createElement("playing-card", {
    cid: cardmeisterCardId(card),
    className,
    rankcolor: cardmeisterFourColorCsv,
    style,
    suitcolor: cardmeisterFourColorCsv
  });
}

export function useCardmeisterScript() {
  useEffect(() => {
    if (document.querySelector("script[data-card-design-cardmeister]") !== null) {
      return;
    }

    const script = document.createElement("script");
    script.dataset.cardDesignCardmeister = "true";
    script.src = resolveAppPath("/vendor/card-design/cardmeister/elements.cardmeister.full.js");
    script.async = true;
    document.head.append(script);
  }, []);
}

export function cardmeisterCardId(card: MockPlayingCard): string {
  const rank = card.rank === "10" ? "T" : card.rank;
  const suit = {
    spades: "s",
    hearts: "h",
    diamonds: "d",
    clubs: "c"
  }[card.suit];

  return `${rank}${suit}`;
}
