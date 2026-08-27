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
  const stateDescriptionId = `${card.id}-interaction`;
  const stateDescription = formatInteractionState(interactionState, selected);

  if (card.type === "joker") {
    return (
      <button
        aria-describedby={stateDescriptionId}
        aria-label="JOKER"
        className={`card card-joker${interactionClassName}${selectedClassName}`}
        disabled={disabled}
        onClick={() => onPlay(card)}
        type="button"
      >
        <span className="card-corner card-corner-top">JOKER</span>
        <span className="card-center-mark">Joker</span>
        <span className="card-corner card-corner-bottom">JOKER</span>
        <span className="visually-hidden" id={stateDescriptionId}>
          {stateDescription}
        </span>
      </button>
    );
  }

  return (
    <button
      aria-describedby={stateDescriptionId}
      aria-label={`${card.rank}${suitSymbols[card.suit]}`}
      className={`card ${
        isRedSuit(card.suit) ? "card-red" : "card-black"
      }${interactionClassName}${selectedClassName}`}
      disabled={disabled}
      onClick={() => onPlay(card)}
      type="button"
    >
      <span className="card-corner card-corner-top">
        {card.rank}
        {suitSymbols[card.suit]}
      </span>
      <span className="card-suit card-center-mark">{suitSymbols[card.suit]}</span>
      <span className="card-corner card-corner-bottom">
        {card.rank}
        {suitSymbols[card.suit]}
      </span>
      <span className="visually-hidden" id={stateDescriptionId}>
        {stateDescription}
      </span>
    </button>
  );
}

function formatInteractionState(
  interactionState: NonNullable<CardButtonProps["interactionState"]>,
  selected: boolean
): string {
  if (selected) {
    return "交換対象に選択済み";
  }

  switch (interactionState) {
    case "legal":
      return "出せるカード";
    case "selectable":
      return "交換候補にできるカード";
    case "blocked":
      return "現在は操作できないカード";
  }
}
