import type { PublicCard } from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";

interface CardButtonProps {
  card: PublicCard;
  disabled: boolean;
  selected?: boolean;
  onPlay: (card: PublicCard) => void;
}

export function CardButton({ card, disabled, selected = false, onPlay }: CardButtonProps) {
  const selectedClassName = selected ? " card-selected" : "";

  if (card.type === "joker") {
    return (
      <button
        className={`card card-joker${selectedClassName}`}
        disabled={disabled}
        onClick={() => onPlay(card)}
        type="button"
      >
        <span>JOKER</span>
      </button>
    );
  }

  return (
    <button
      className={`card ${isRedSuit(card.suit) ? "card-red" : "card-black"}${selectedClassName}`}
      disabled={disabled}
      onClick={() => onPlay(card)}
      type="button"
    >
      <span>{card.rank}</span>
      <span className="card-suit">{suitSymbols[card.suit]}</span>
    </button>
  );
}
