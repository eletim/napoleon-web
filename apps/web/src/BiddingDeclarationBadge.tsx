import type { BiddingDeclarationDisplay } from "./biddingDeclarations";
import { suitSymbols } from "./cardSymbols";

interface BiddingDeclarationBadgeProps {
  playerLabel: string;
  declaration: BiddingDeclarationDisplay | undefined;
}

export function BiddingDeclarationBadge({
  playerLabel,
  declaration
}: BiddingDeclarationBadgeProps) {
  if (declaration === undefined) {
    return null;
  }

  const className = [
    "latest-bid-declaration",
    `latest-bid-${declaration.type}`,
    declaration.type === "bid" ? `latest-bid-${declaration.color}` : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      aria-label={`${playerLabel}の最新競り宣言: ${declaration.label}`}
      className={className}
    >
      {declaration.type === "bid" ? (
        <strong className="latest-bid-value">
          <span className="latest-bid-suit">{suitSymbols[declaration.suit]}</span>
          <span className="latest-bid-target">{declaration.targetPointCards}</span>
        </strong>
      ) : (
        <strong>{declaration.label}</strong>
      )}
    </div>
  );
}
