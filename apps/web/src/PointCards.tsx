import type { PublicStandardCard } from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";

interface PointCardsProps {
  cards: readonly PublicStandardCard[];
}

export function PointCards({ cards }: PointCardsProps) {
  if (cards.length === 0) {
    return <span className="muted-text">なし</span>;
  }

  return (
    <>
      {cards.map((card) => (
        <span
          className={isRedSuit(card.suit) ? "mini-card red-text" : "mini-card black-text"}
          key={card.id}
        >
          {card.rank}
          {suitSymbols[card.suit]}
        </span>
      ))}
    </>
  );
}
