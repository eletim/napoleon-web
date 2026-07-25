import type { Card } from "@napoleon/game-core";
import { isRedSuit, suitSymbols } from "./cardSymbols";

interface CardButtonProps {
  card: Card;
  disabled: boolean;
  onPlay: (card: Card) => void;
}

export function CardButton({ card, disabled, onPlay }: CardButtonProps) {
  return (
    <button
      className={`card ${isRedSuit(card.suit) ? "card-red" : "card-black"}`}
      disabled={disabled}
      onClick={() => onPlay(card)}
      type="button"
    >
      <span>{card.rank}</span>
      <span className="card-suit">{suitSymbols[card.suit]}</span>
    </button>
  );
}
