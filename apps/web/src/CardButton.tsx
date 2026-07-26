import type { PublicCard } from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";

interface CardButtonProps {
  card: PublicCard;
  disabled: boolean;
  interactionState?: "legal" | "blocked" | "selectable";
  selected?: boolean;
  onPlay: (card: PublicCard) => void;
}

export function CardButton({
  card,
  disabled,
  interactionState = "blocked",
  selected = false,
  onPlay
}: CardButtonProps) {
  const selectedClassName = selected ? " card-selected" : "";
  const interactionClassName = ` card-${interactionState}`;

  if (card.type === "joker") {
    return (
      <button
        aria-label="JOKER"
        className={`card card-joker${interactionClassName}${selectedClassName}`}
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
      aria-label={`${card.rank}${suitSymbols[card.suit]}`}
      className={`card ${
        isRedSuit(card.suit) ? "card-red" : "card-black"
      }${interactionClassName}${selectedClassName}`}
      disabled={disabled}
      onClick={() => onPlay(card)}
      type="button"
    >
      <span>{card.rank}</span>
      <span className="card-suit">{suitSymbols[card.suit]}</span>
    </button>
  );
}
