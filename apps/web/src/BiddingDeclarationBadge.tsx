import type { BiddingDeclarationDisplay } from "./biddingDeclarations";

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
      <span>競り</span>
      <strong>{declaration.label}</strong>
    </div>
  );
}
