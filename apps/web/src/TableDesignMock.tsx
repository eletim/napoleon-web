import type { CSSProperties } from "react";
import "./TableDesignMock.css";

type Suit = "spades" | "hearts" | "diamonds" | "clubs";

interface MockCard {
  rank: string;
  suit: Suit;
  face?: "king" | "queen" | "jack";
}

interface Point {
  x: number;
  y: number;
}

interface Box extends Point {
  height: number;
  width: number;
}

interface SeatLayout {
  avatar: Point;
  hand: Point & { rotation: number };
  id: string;
  label: string;
  river: Point & { rotation: number };
  trick: Point & { rotation: number };
}

interface TableDesignMockLayout {
  action: Box;
  cardSizes: {
    river: { height: number; width: number };
    selfHand: { height: number; width: number };
    trick: { height: number; width: number };
  };
  center: Box;
  hud: Box;
  page: {
    background: string;
    height: number;
    width: number;
  };
  riverSize: {
    height: number;
    width: number;
  };
  riverGrid: {
    maxColumns: number;
    maxRows: number;
    rowGap: number;
    selfSideWidthRatio: number;
  };
  seats: readonly SeatLayout[];
}

// Source of Truth: https://github.com/eletim/napoleon-web/issues/308#issuecomment-5348323047
// Keep the screenshot-facing coordinates here so the mock can be tuned without
// hunting through individual elements.
export const tableDesignMockLayout: TableDesignMockLayout = {
  page: {
    width: 2200,
    height: 1830,
    background: "#1d1d1d"
  },
  hud: {
    x: 0,
    y: 0,
    width: 376,
    height: 286
  },
  center: {
    x: 1120,
    y: 890,
    width: 338,
    height: 303
  },
  cardSizes: {
    trick: { width: 132, height: 178 },
    river: { width: 78, height: 106 },
    selfHand: { width: 172, height: 228 }
  },
  riverSize: {
    width: 260,
    height: 210
  },
  riverGrid: {
    maxColumns: 5,
    maxRows: 4,
    rowGap: 24,
    selfSideWidthRatio: 0.62
  },
  action: {
    x: 1120,
    y: 1376,
    width: 224,
    height: 274
  },
  seats: [
    {
      id: "top-left",
      label: "北西",
      avatar: { x: 586, y: 132 },
      hand: { x: 672, y: 292, rotation: -19 },
      trick: { x: 1034, y: 713, rotation: -38 },
      river: { x: 836, y: 540, rotation: -37 }
    },
    {
      id: "top-right",
      label: "北東",
      avatar: { x: 1590, y: 144 },
      hand: { x: 1538, y: 294, rotation: 19 },
      trick: { x: 1298, y: 732, rotation: 31 },
      river: { x: 1376, y: 540, rotation: 37 }
    },
    {
      id: "right",
      label: "右席",
      avatar: { x: 2112, y: 1074 },
      hand: { x: 1942, y: 1080, rotation: 55 },
      trick: { x: 1307, y: 1016, rotation: 14 },
      river: { x: 1578, y: 1036, rotation: 12 }
    },
    {
      id: "self",
      label: "自分",
      avatar: { x: 808, y: 1492 },
      hand: { x: 1118, y: 1684, rotation: 0 },
      trick: { x: 1122, y: 1138, rotation: 0 },
      river: { x: 1120, y: 1110, rotation: 0 }
    },
    {
      id: "left",
      label: "左席",
      avatar: { x: 136, y: 1070 },
      hand: { x: 308, y: 1086, rotation: -54 },
      trick: { x: 902, y: 934, rotation: -12 },
      river: { x: 644, y: 1038, rotation: -12 }
    }
  ]
};

const selfCards: readonly MockCard[] = [
  { rank: "5", suit: "spades" },
  { rank: "7", suit: "spades" }
];

const trickCards: Record<string, MockCard> = {
  "top-left": { rank: "10", suit: "hearts" },
  "top-right": { rank: "Q", suit: "hearts", face: "queen" },
  right: { rank: "K", suit: "spades", face: "king" },
  left: { rank: "A", suit: "spades" }
};

const riverCards: Record<string, readonly MockCard[]> = {
  "top-left": [
    { rank: "10", suit: "clubs" },
    { rank: "Q", suit: "spades" }
  ],
  "top-right": [
    { rank: "K", suit: "hearts" },
    { rank: "10", suit: "diamonds" }
  ],
  right: [
    { rank: "A", suit: "clubs" },
    { rank: "K", suit: "clubs" },
    { rank: "Q", suit: "clubs" }
  ],
  left: [
    { rank: "A", suit: "hearts" },
    { rank: "K", suit: "spades" }
  ],
  self: [
    { rank: "A", suit: "hearts" },
    { rank: "10", suit: "spades" },
    { rank: "J", suit: "clubs" },
    { rank: "K", suit: "diamonds" },
    { rank: "A", suit: "diamonds" }
  ]
};

const suitMarks: Record<Suit, string> = {
  spades: "♠",
  hearts: "♥",
  diamonds: "♦",
  clubs: "♣"
};

const roleCells = [
  { label: "副", className: "role-cell role-cell-top-left" },
  { label: "兵", className: "role-cell role-cell-top-right" },
  { label: "ナポ", className: "role-cell role-cell-self" },
  { label: "副", className: "role-cell role-cell-left" },
  { label: "兵", className: "role-cell role-cell-right" }
];

export function TableDesignMock() {
  const layout = tableDesignMockLayout;

  return (
    <main
      aria-label="Issue 308 table design mock"
      className="table-design-mock-page"
      style={
        {
          "--mock-page-background": layout.page.background,
          "--mock-page-height": `${layout.page.height}px`,
          "--mock-page-width": `${layout.page.width}px`,
          "--mock-river-height": `${layout.riverSize.height}px`,
          "--mock-river-width": `${layout.riverSize.width}px`,
          "--mock-river-card-height": `${layout.cardSizes.river.height}px`,
          "--mock-river-card-width": `${layout.cardSizes.river.width}px`,
          "--mock-self-river-row-gap": `${layout.riverGrid.rowGap}px`,
          "--mock-self-river-width": `${selfRiverWidth(layout)}px`,
          "--mock-self-card-height": `${layout.cardSizes.selfHand.height}px`,
          "--mock-self-card-width": `${layout.cardSizes.selfHand.width}px`,
          "--mock-trick-card-height": `${layout.cardSizes.trick.height}px`,
          "--mock-trick-card-width": `${layout.cardSizes.trick.width}px`
        } as CSSProperties
      }
    >
      <div className="table-design-stage">
        <HudBox layout={layout.hud} />
        {layout.seats.map((seat) => (
          <SeatArtifacts key={seat.id} seat={seat} />
        ))}
        <RoleBoard layout={layout.center} />
        <SelfActionFocus layout={layout.action} />
      </div>
    </main>
  );
}

function HudBox({ layout }: { layout: Box }) {
  return (
    <aside
      aria-label="契約HUD"
      className="mock-hud"
      style={boxStyle(layout, "top-left")}
    >
      <div className="mock-hud-contract">
        <span className="mock-hud-suit">♠</span>
        <span>15</span>
      </div>
      <div className="mock-hud-adjutant">
        <span>副</span>
        <span className="mock-hud-suit">♠</span>
        <span>A</span>
      </div>
    </aside>
  );
}

function SeatArtifacts({ seat }: { seat: SeatLayout }) {
  return (
    <>
      <div
        aria-label={`${seat.label} プレイヤー`}
        className={`mock-avatar mock-avatar-${seat.id}`}
        style={pointStyle(seat.avatar)}
      >
        <span className="mock-avatar-head" />
        <span className="mock-avatar-body" />
      </div>

      {seat.id === "self" ? (
        <SelfHand seat={seat} />
      ) : (
        <CardBackFan seat={seat} />
      )}

      {seat.id === "self" ? (
        null
      ) : (
        <PlayingCard
          card={trickCards[seat.id]}
          className="mock-trick-card"
          style={pointWithRotationStyle(seat.trick)}
        />
      )}

      <PointRiver seat={seat} />
    </>
  );
}

function SelfHand({ seat }: { seat: SeatLayout }) {
  return (
    <div
      aria-label="自分の表向き手札"
      className="mock-self-hand"
      style={pointWithRotationStyle(seat.hand)}
    >
      {selfCards.map((card) => (
        <PlayingCard card={card} className="mock-self-hand-card" key={`${card.rank}-${card.suit}`} />
      ))}
    </div>
  );
}

function CardBackFan({ seat }: { seat: SeatLayout }) {
  return (
    <div
      aria-label={`${seat.label}の裏向き手札`}
      className={`mock-card-back-fan mock-card-back-fan-${seat.id}`}
      style={pointWithRotationStyle(seat.hand)}
    >
      {[0, 1, 2].map((index) => (
        <span className="mock-card-back" key={index} />
      ))}
    </div>
  );
}

function PointRiver({ seat }: { seat: SeatLayout }) {
  const cards = riverCards[seat.id] ?? [];
  const layout = tableDesignMockLayout;
  const riverPlacements =
    seat.id === "self" ? createRiverPlacements(cards.length, layout) : [];

  return (
    <section
      aria-label={`${seat.label}のポイント札の河`}
      className={`mock-point-river mock-point-river-${seat.id}`}
      style={riverStyle(seat, cards.length, layout)}
    >
      {cards.slice(0, layout.riverGrid.maxColumns * layout.riverGrid.maxRows).map((card, index) => (
        <PlayingCard
          card={card}
          className="mock-river-card"
          key={`${card.rank}-${card.suit}-${index}`}
          style={{
            "--mock-river-card-index": index,
            "--mock-river-card-rotation": `${riverPlacements[index]?.rotation ?? (index - 1) * 4}deg`,
            "--mock-river-card-x": `${riverPlacements[index]?.x ?? 18 + index * 34}px`,
            "--mock-river-card-y": `${riverPlacements[index]?.y ?? 32}px`
          } as CSSProperties}
        />
      ))}
    </section>
  );
}

function RoleBoard({ layout }: { layout: Box }) {
  return (
    <section
      aria-label="中央役職表示"
      className="mock-role-board"
      style={boxStyle(layout)}
    >
      <div className="mock-role-board-shape">
        {roleCells.map((role) => (
          <span className={role.className} key={role.className}>
            {role.label}
          </span>
        ))}
        <span className="mock-role-board-core" />
      </div>
    </section>
  );
}

function SelfActionFocus({ layout }: { layout: Box }) {
  return (
    <section
      aria-label="自席操作UI"
      className="mock-self-action"
      style={boxStyle(layout)}
    >
      <PlayingCard card={{ rank: "J", suit: "spades", face: "jack" }} className="mock-action-card" />
      <div className="mock-action-controls" aria-label="操作">
        <button type="button">出す</button>
        <button type="button">待機</button>
      </div>
    </section>
  );
}

function PlayingCard({
  card,
  className,
  style
}: {
  card: MockCard | undefined;
  className: string;
  style?: CSSProperties;
}) {
  if (card === undefined) {
    return null;
  }

  const mark = suitMarks[card.suit];
  const isRed = card.suit === "hearts" || card.suit === "diamonds";
  const faceText = card.face === undefined ? mark : faceGlyph(card.face);

  return (
    <article
      aria-label={`${card.rank}${mark}`}
      className={`${className} mock-playing-card ${isRed ? "mock-card-red" : "mock-card-black"}`}
      style={style}
    >
      <span className="mock-card-corner mock-card-corner-top">
        {card.rank}
        {mark}
      </span>
      <span className="mock-card-face">{faceText}</span>
      <span className="mock-card-corner mock-card-corner-bottom">
        {card.rank}
        {mark}
      </span>
    </article>
  );
}

function faceGlyph(face: NonNullable<MockCard["face"]>): string {
  switch (face) {
    case "king":
      return "♚";
    case "queen":
      return "♛";
    case "jack":
      return "♝";
  }
}

export function selfRiverWidth(layout: TableDesignMockLayout): number {
  return layout.center.width * layout.riverGrid.selfSideWidthRatio;
}

export function createRiverPlacements(
  cardCount: number,
  layout: TableDesignMockLayout
): Array<Point & { rotation: number }> {
  const maxCards = layout.riverGrid.maxColumns * layout.riverGrid.maxRows;
  const boundedCardCount = Math.min(cardCount, maxCards);
  const placements: Array<Point & { rotation: number }> = [];
  const d = selfRiverWidth(layout);
  const cardWidth = layout.cardSizes.river.width;
  const cardHeight = layout.cardSizes.river.height;
  const rowPitch = cardHeight + layout.riverGrid.rowGap;

  for (let row = 0; row < layout.riverGrid.maxRows; row += 1) {
    const rowStart = row * layout.riverGrid.maxColumns;
    const rowCount = Math.min(layout.riverGrid.maxColumns, boundedCardCount - rowStart);

    if (rowCount <= 0) {
      break;
    }

    const rowLength = d * (0.5 + 0.125 * (rowCount - 1));
    const rowLeft = (d - rowLength) / 2;
    const usableTravel = Math.max(rowLength - cardWidth, 0);

    for (let column = 0; column < rowCount; column += 1) {
      const x =
        rowLeft +
        (rowCount === 1 ? usableTravel / 2 : (usableTravel * column) / (rowCount - 1));

      placements.push({
        x,
        y: row * rowPitch,
        rotation: 0
      });
    }
  }

  return placements;
}

function selfRiverHeight(cardCount: number, layout: TableDesignMockLayout): number {
  const maxCards = layout.riverGrid.maxColumns * layout.riverGrid.maxRows;
  const boundedCardCount = Math.min(cardCount, maxCards);
  const rows = Math.max(1, Math.ceil(boundedCardCount / layout.riverGrid.maxColumns));

  return layout.cardSizes.river.height * rows + layout.riverGrid.rowGap * (rows - 1);
}

function riverStyle(
  seat: SeatLayout,
  cardCount: number,
  layout: TableDesignMockLayout
): CSSProperties {
  const style = pointWithRotationStyle(seat.river);

  if (seat.id !== "self") {
    return style;
  }

  return {
    ...style,
    "--mock-self-river-height": `${selfRiverHeight(cardCount, layout)}px`,
    "--mock-self-river-width": `${selfRiverWidth(layout)}px`
  } as CSSProperties;
}

function pointStyle(point: Point): CSSProperties {
  return {
    "--mock-x": `${point.x}px`,
    "--mock-y": `${point.y}px`
  } as CSSProperties;
}

function pointWithRotationStyle(point: Point & { rotation: number }): CSSProperties {
  return {
    ...pointStyle(point),
    "--mock-rotation": `${point.rotation}deg`
  } as CSSProperties;
}

function boxStyle(box: Box, origin: "center" | "top-left" = "center"): CSSProperties {
  return {
    "--mock-box-height": `${box.height}px`,
    "--mock-box-width": `${box.width}px`,
    "--mock-x": `${box.x}px`,
    "--mock-y": `${box.y}px`,
    "--mock-origin": origin === "center" ? "translate(-50%, -50%)" : "translate(0, 0)"
  } as CSSProperties;
}
