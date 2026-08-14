import type { PublicStandardCard } from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";

interface PointCardsProps {
  cards: readonly PublicStandardCard[];
  fixedSlotCount?: number;
}

export function PointCards({ cards, fixedSlotCount }: PointCardsProps) {
  const emptySlotCount =
    fixedSlotCount === undefined ? 0 : Math.max(0, fixedSlotCount - cards.length);

  if (cards.length === 0) {
    return (
      <>
        <span className="muted-text point-cards-empty-label">なし</span>
        {Array.from({ length: emptySlotCount }, (_, index) => (
          <span
            aria-hidden="true"
            className="point-card-empty-slot"
            key={`empty-point-card-slot-${index}`}
          />
        ))}
      </>
    );
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
      {Array.from({ length: emptySlotCount }, (_, index) => (
        <span
          aria-hidden="true"
          className="point-card-empty-slot"
          key={`empty-point-card-slot-${index}`}
        />
      ))}
    </>
  );
}
