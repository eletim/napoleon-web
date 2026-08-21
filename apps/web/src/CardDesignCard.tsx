import type { CSSProperties } from "react";
import type { MockPlayingCard, MockPlayingCardRank, MockPlayingCardSuit } from "./mockPlayingCardAdapter";

export const cardDesignSuitSymbols = {
  spades: "♠",
  hearts: "♥",
  diamonds: "♦",
  clubs: "♣"
} as const satisfies Record<MockPlayingCardSuit, string>;

export const cardDesignSuitLabels = {
  spades: "Spades",
  hearts: "Hearts",
  diamonds: "Diamonds",
  clubs: "Clubs"
} as const satisfies Record<MockPlayingCardSuit, string>;

export const cardDesignSuitOrder = ["spades", "clubs", "hearts", "diamonds"] as const satisfies readonly MockPlayingCardSuit[];

export const cardDesignRankOrder = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] as const satisfies readonly MockPlayingCardRank[];

export const cardDesignComparisonRanks = ["A", "2", "5", "10", "J", "Q", "K"] as const satisfies readonly MockPlayingCardRank[];

export const cardDesignConfig = {
  aspectRatio: {
    width: 5,
    height: 7
  },
  colors: {
    spades: "#111827",
    hearts: "#dc2626",
    diamonds: "#2563eb",
    clubs: "#15803d"
  },
  layout: {
    leftIdentificationAreaRatio: 0.25,
    rankFontRatio: 0.18,
    suitSymbolRatio: 0.2,
    cornerPaddingRatio: 0.055,
    rankSuitGapRatio: 0.165,
    centerSymbolRatio: 0.38,
    centerRankRatio: 0.28,
    borderWidthRatio: 0.016,
    borderRadiusRatio: 0.065,
    fontWeight: 900
  },
  surfaces: {
    backgroundColor: "#fffdf6",
    identificationAreaColor: "#f7f0d7",
    borderColor: "#111827",
    guideColor: "rgb(17 24 39 / 20%)"
  },
  sizes: {
    normalWidth: 124,
    smallWidth: 70,
    overlapWidth: 96
  }
} as const;

export function createCardDesignDeck(
  ranks: readonly MockPlayingCardRank[] = cardDesignRankOrder
): readonly MockPlayingCard[] {
  return cardDesignSuitOrder.flatMap((suit) => ranks.map((rank) => ({ rank, suit })));
}

export function cardDesignCardHeight(width: number): number {
  return width * cardDesignConfig.aspectRatio.height / cardDesignConfig.aspectRatio.width;
}

export function cardDesignOverlapWidth(cardCount: number, cardWidth = cardDesignConfig.sizes.overlapWidth): number {
  if (cardCount <= 0) {
    return 0;
  }

  return cardWidth + (cardCount - 1) * cardWidth * cardDesignConfig.layout.leftIdentificationAreaRatio;
}

export function CardDesignPrototypeCard({
  card,
  className,
  width = cardDesignConfig.sizes.normalWidth
}: {
  card: MockPlayingCard;
  className?: string;
  width?: number;
}) {
  const config = cardDesignConfig;
  const suitColor = config.colors[card.suit];
  const symbol = cardDesignSuitSymbols[card.suit];
  const leftWidth = config.layout.leftIdentificationAreaRatio * 100;
  const cornerX = leftWidth / 2;
  const cornerPadding = config.layout.cornerPaddingRatio * 100;
  const rankY = cornerPadding + config.layout.rankFontRatio * 50;
  const suitY = rankY + config.layout.rankSuitGapRatio * 100;
  const centerX = leftWidth + (100 - leftWidth) / 2;
  const borderWidth = config.layout.borderWidthRatio * 100;
  const borderRadius = config.layout.borderRadiusRatio * 100;
  const height = cardDesignCardHeight(width);
  const isCourtCard = card.rank === "J" || card.rank === "Q" || card.rank === "K";
  const pipRows = createPipRows(card.rank);

  return (
    <article
      aria-label={`${card.rank}${symbol}`}
      className={["card-design-card", className].filter(Boolean).join(" ")}
      style={
        {
          "--card-design-card-height": `${height}px`,
          "--card-design-card-width": `${width}px`
        } as CSSProperties
      }
    >
      <svg
        aria-hidden="true"
        className="card-design-card-svg"
        focusable="false"
        preserveAspectRatio="none"
        viewBox="0 0 100 140"
      >
        <rect
          fill={config.surfaces.backgroundColor}
          height={140 - borderWidth * 2}
          rx={borderRadius}
          ry={borderRadius}
          stroke={config.surfaces.borderColor}
          strokeWidth={borderWidth}
          width={100 - borderWidth * 2}
          x={borderWidth}
          y={borderWidth}
        />
        <rect
          fill={config.surfaces.identificationAreaColor}
          height={140 - cornerPadding * 2}
          rx={borderRadius * 0.68}
          ry={borderRadius * 0.68}
          width={leftWidth - cornerPadding * 1.25}
          x={cornerPadding * 0.65}
          y={cornerPadding}
        />
        <line
          stroke={config.surfaces.guideColor}
          strokeDasharray="2 2"
          strokeWidth="0.65"
          x1={leftWidth}
          x2={leftWidth}
          y1={cornerPadding}
          y2={140 - cornerPadding}
        />
        <text
          className="card-design-rank"
          fill={suitColor}
          fontSize={100 * config.layout.rankFontRatio}
          fontWeight={config.layout.fontWeight}
          x={cornerX}
          y={rankY}
        >
          {card.rank}
        </text>
        <text
          className="card-design-suit"
          fill={suitColor}
          fontSize={100 * config.layout.suitSymbolRatio}
          fontWeight={config.layout.fontWeight}
          x={cornerX}
          y={suitY}
        >
          {symbol}
        </text>
        <text
          className="card-design-rank card-design-inverted"
          fill={suitColor}
          fontSize={100 * config.layout.rankFontRatio}
          fontWeight={config.layout.fontWeight}
          x={cornerX}
          y={140 - rankY}
        >
          {card.rank}
        </text>
        <text
          className="card-design-suit card-design-inverted"
          fill={suitColor}
          fontSize={100 * config.layout.suitSymbolRatio}
          fontWeight={config.layout.fontWeight}
          x={cornerX}
          y={140 - suitY}
        >
          {symbol}
        </text>
        {isCourtCard ? (
          <>
            <text
              className="card-design-center-rank"
              fill={suitColor}
              fontSize={100 * config.layout.centerRankRatio}
              fontWeight={config.layout.fontWeight}
              x={centerX}
              y="66"
            >
              {card.rank}
            </text>
            <text
              className="card-design-center-suit"
              fill={suitColor}
              fontSize={100 * config.layout.centerSymbolRatio * 0.82}
              fontWeight={config.layout.fontWeight}
              x={centerX}
              y="99"
            >
              {symbol}
            </text>
          </>
        ) : (
          <>
            <text
              className="card-design-center-suit"
              fill={suitColor}
              fontSize={100 * config.layout.centerSymbolRatio}
              fontWeight={config.layout.fontWeight}
              x={centerX}
              y="73"
            >
              {symbol}
            </text>
            {pipRows.map((row, rowIndex) =>
              row.map((pipX, pipIndex) => (
                <text
                  className="card-design-pip"
                  fill={suitColor}
                  fontSize="12"
                  fontWeight={config.layout.fontWeight}
                  key={`${rowIndex}-${pipIndex}`}
                  x={pipX}
                  y={112 + rowIndex * 13}
                >
                  {symbol}
                </text>
              ))
            )}
          </>
        )}
      </svg>
    </article>
  );
}

function createPipRows(rank: MockPlayingCardRank): readonly (readonly number[])[] {
  const pipCount = rank === "A" ? 1 : Number.parseInt(rank, 10);

  if (!Number.isFinite(pipCount)) {
    return [];
  }

  if (pipCount <= 1) {
    return [[62.5]];
  }

  if (pipCount <= 5) {
    return [Array.from({ length: pipCount }, (_, index) => 50 + index * 6.25)];
  }

  return [
    [50, 62.5, 75],
    [53, 62.5, 72]
  ];
}
