import type { CSSProperties } from "react";
import "./TableDesignMock.css";

type Suit = "spades" | "hearts" | "diamonds" | "clubs";
type SeatId = "top-left" | "top-right" | "right" | "self" | "left";

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
  id: SeatId;
  label: string;
}

interface TableDesignMockLayout {
  cardSizes: {
    selfHand: { height: number; width: number };
    trick: { height: number; width: number };
  };
  currentTrickZone: {
    gapFromRiver: number;
    paddingBlock: number;
    paddingInline: number;
  };
  center: Box;
  hud: Box;
  page: {
    background: string;
    height: number;
    width: number;
  };
  riverGrid: {
    maxColumns: number;
    maxRows: number;
    rowGap: number;
  };
  seats: readonly SeatLayout[];
}

const roleBoardCenter: Box = {
  x: 1120,
  y: 890,
  width: 338,
  height: 303
};

const roleBoardPentagon = {
  top: { x: 0.5, y: 0 },
  topRight: { x: 1, y: 0.38 },
  bottomRight: { x: 0.81, y: 1 },
  bottomLeft: { x: 0.19, y: 1 },
  topLeft: { x: 0, y: 0.38 }
} as const satisfies Record<string, Point>;

const cardAspectRatio = 178 / 132;
const trickCardWidth = 118;
const riverGap = 18;

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
  center: roleBoardCenter,
  cardSizes: {
    trick: { width: trickCardWidth, height: toLayoutPrecision(trickCardWidth * cardAspectRatio) },
    selfHand: { width: 172, height: 228 }
  },
  currentTrickZone: {
    gapFromRiver: 28,
    paddingBlock: 52,
    paddingInline: 44
  },
  riverGrid: {
    maxColumns: 5,
    maxRows: 4,
    rowGap: 24
  },
  seats: [
    {
      id: "top-left",
      label: "北西",
      avatar: { x: 586, y: 132 },
      hand: { x: 672, y: 292, rotation: -19 }
    },
    {
      id: "top-right",
      label: "北東",
      avatar: { x: 1590, y: 144 },
      hand: { x: 1538, y: 294, rotation: 19 }
    },
    {
      id: "right",
      label: "右席",
      avatar: { x: 2112, y: 1074 },
      hand: { x: 1942, y: 1080, rotation: 55 }
    },
    {
      id: "self",
      label: "自分",
      avatar: { x: 808, y: 1492 },
      hand: { x: 1118, y: 1684, rotation: 0 }
    },
    {
      id: "left",
      label: "左席",
      avatar: { x: 136, y: 1070 },
      hand: { x: 308, y: 1086, rotation: -54 }
    }
  ]
};

const selfCards: readonly MockCard[] = [
  { rank: "5", suit: "spades" },
  { rank: "7", suit: "spades" }
];

const trickCards: Partial<Record<SeatId, MockCard>> = {
  "top-left": { rank: "10", suit: "hearts" },
  "top-right": { rank: "Q", suit: "hearts", face: "queen" },
  right: { rank: "K", suit: "spades", face: "king" },
  left: { rank: "A", suit: "spades" },
  self: { rank: "J", suit: "spades", face: "jack" }
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
      aria-label="Issue 325 table design mock"
      className="table-design-mock-page"
      style={
        {
          "--mock-page-background": layout.page.background,
          "--mock-page-height": `${layout.page.height}px`,
          "--mock-page-width": `${layout.page.width}px`,
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

      <CurrentTrickZone seat={seat} />
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

function CurrentTrickZone({ seat }: { seat: SeatLayout }) {
  const cardCount = riverCards[seat.id]?.length ?? 0;
  const geometry = createCurrentTrickZoneGeometry(tableDesignMockLayout, seat.id, cardCount);

  return (
    <section
      aria-label={`${seat.label}の現在トリック置き場`}
      className={`mock-current-trick-zone mock-current-trick-zone-${seat.id}`}
      style={trickZoneStyle(geometry)}
    >
      <PlayingCard card={trickCards[seat.id]} className="mock-trick-card" />
    </section>
  );
}

function PointRiver({ seat }: { seat: SeatLayout }) {
  const cards = riverCards[seat.id] ?? [];
  const layout = tableDesignMockLayout;
  const riverGeometry = createRiverGeometry(layout, seat.id);
  const riverPlacements = createRiverPlacements(cards.length, layout, seat.id);

  return (
    <section
      aria-label={`${seat.label}のポイント札の河`}
      className={`mock-point-river mock-point-river-${seat.id}`}
      style={riverStyle(riverGeometry)}
    >
      {cards.slice(0, layout.riverGrid.maxColumns * layout.riverGrid.maxRows).map((card, index) => (
        <PlayingCard
          card={card}
          className="mock-river-card"
          key={`${card.rank}-${card.suit}-${index}`}
          style={{
            "--mock-river-card-index": index,
            "--mock-river-card-rotation": `${riverPlacements[index]?.rotation ?? 0}deg`,
            "--mock-river-card-height": `${riverGeometry.cardSize.height}px`,
            "--mock-river-card-width": `${riverGeometry.cardSize.width}px`,
            "--mock-river-card-x": `${riverPlacements[index]?.x ?? 0}px`,
            "--mock-river-card-y": `${riverPlacements[index]?.y ?? 0}px`
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
      style={roleBoardStyle(layout)}
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
  return createRiverGeometry(layout, "self").d;
}

export function roleBoardSelfSideLength(layout: Box): number {
  return createRoleBoardEdgeGeometry(layout, "self").d;
}

interface RoleBoardEdgeGeometry {
  d: number;
  direction: Point;
  end: Point;
  normal: Point;
  rotation: number;
  start: Point;
}

interface RiverGeometry extends RoleBoardEdgeGeometry {
  cardSize: { height: number; width: number };
  height: number;
  rowPitch: number;
  width: number;
  x: number;
  y: number;
}

interface CurrentTrickZoneGeometry extends RoleBoardEdgeGeometry {
  height: number;
  width: number;
  x: number;
  y: number;
}

const roleBoardEdges: Record<SeatId, { end: Point; start: Point }> = {
  "top-left": { start: roleBoardPentagon.top, end: roleBoardPentagon.topLeft },
  "top-right": { start: roleBoardPentagon.topRight, end: roleBoardPentagon.top },
  right: { start: roleBoardPentagon.bottomRight, end: roleBoardPentagon.topRight },
  self: { start: roleBoardPentagon.bottomLeft, end: roleBoardPentagon.bottomRight },
  left: { start: roleBoardPentagon.topLeft, end: roleBoardPentagon.bottomLeft }
};

export function createRoleBoardEdgeGeometry(
  layout: Box,
  seatId: SeatId
): RoleBoardEdgeGeometry {
  const edge = roleBoardEdges[seatId];
  const start = roleBoardAbsolutePoint(layout, edge.start);
  const end = roleBoardAbsolutePoint(layout, edge.end);
  const vector = { x: end.x - start.x, y: end.y - start.y };
  const d = toLayoutPrecision(distance(start, end));
  const direction = normalizeVector(vector);
  const normal = normalizeVector({ x: -direction.y, y: direction.x });

  return {
    d,
    direction,
    end,
    normal,
    rotation: toLayoutPrecision((Math.atan2(direction.y, direction.x) * 180) / Math.PI),
    start
  };
}

export function createRiverGeometry(
  layout: TableDesignMockLayout,
  seatId: SeatId
): RiverGeometry {
  const edge = createRoleBoardEdgeGeometry(layout.center, seatId);
  const cardWidth = toLayoutPrecision(edge.d * 0.5);
  const cardHeight = toLayoutPrecision(cardWidth * cardAspectRatio);
  const rowPitch = toLayoutPrecision(cardHeight + layout.riverGrid.rowGap);
  const height = toLayoutPrecision(cardHeight + rowPitch * (layout.riverGrid.maxRows - 1));
  const offset = riverGap;

  return {
    ...edge,
    cardSize: { width: cardWidth, height: cardHeight },
    height,
    rowPitch,
    width: edge.d,
    x: toLayoutPrecision(edge.start.x + edge.normal.x * offset),
    y: toLayoutPrecision(edge.start.y + edge.normal.y * offset)
  };
}

export function createCurrentTrickZoneGeometry(
  layout: TableDesignMockLayout,
  seatId: SeatId,
  riverCardCount = 0
): CurrentTrickZoneGeometry {
  const edge = createRoleBoardEdgeGeometry(layout.center, seatId);
  const river = createRiverGeometry(layout, seatId);
  const visibleRiverRows = Math.max(1, Math.ceil(Math.min(riverCardCount, 20) / layout.riverGrid.maxColumns));
  const visibleRiverDepth = toLayoutPrecision(
    river.cardSize.height * visibleRiverRows + layout.riverGrid.rowGap * (visibleRiverRows - 1)
  );
  const width = toLayoutPrecision(layout.cardSizes.trick.width + layout.currentTrickZone.paddingInline);
  const height = toLayoutPrecision(layout.cardSizes.trick.height + layout.currentTrickZone.paddingBlock);
  const midpoint = midpointBetween(edge.start, edge.end);
  const centerOffset = toLayoutPrecision(riverGap + visibleRiverDepth + layout.currentTrickZone.gapFromRiver + height / 2);

  return {
    ...edge,
    height,
    width,
    x: toLayoutPrecision(midpoint.x + edge.normal.x * centerOffset),
    y: toLayoutPrecision(midpoint.y + edge.normal.y * centerOffset)
  };
}

export function createRiverPlacements(
  cardCount: number,
  layout: TableDesignMockLayout,
  seatId: SeatId = "self"
): Array<Point & { rotation: number }> {
  const maxCards = layout.riverGrid.maxColumns * layout.riverGrid.maxRows;
  const boundedCardCount = Math.min(cardCount, maxCards);
  const geometry = createRiverGeometry(layout, seatId);
  const placements: Array<Point & { rotation: number }> = [];
  const columnOffset = toLayoutPrecision(geometry.d * 0.125);

  for (let index = 0; index < boundedCardCount; index += 1) {
    const column = index % layout.riverGrid.maxColumns;
    const row = Math.floor(index / layout.riverGrid.maxColumns);

    placements.push({
      x: toLayoutPrecision(column * columnOffset),
      y: toLayoutPrecision(row * geometry.rowPitch),
      rotation: 0
    });
  }

  return placements;
}

function riverStyle(geometry: RiverGeometry): CSSProperties {
  return {
    "--mock-river-height": `${geometry.height}px`,
    "--mock-river-rotation": `${geometry.rotation}deg`,
    "--mock-river-width": `${geometry.width}px`,
    "--mock-x": `${geometry.x}px`,
    "--mock-y": `${geometry.y}px`
  } as CSSProperties;
}

function trickZoneStyle(zone: CurrentTrickZoneGeometry): CSSProperties {
  return {
    ...pointWithRotationStyle(zone),
    "--mock-trick-zone-height": `${zone.height}px`,
    "--mock-trick-zone-width": `${zone.width}px`
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

function roleBoardStyle(box: Box): CSSProperties {
  return {
    ...boxStyle(box),
    "--mock-role-board-polygon": roleBoardClipPath()
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

function roleBoardAbsolutePoint(layout: Box, point: Point): Point {
  return {
    x: layout.x - layout.width / 2 + point.x * layout.width,
    y: layout.y - layout.height / 2 + point.y * layout.height
  };
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpointBetween(a: Point, b: Point): Point {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2
  };
}

function normalizeVector(vector: Point): Point {
  const length = distance({ x: 0, y: 0 }, vector);

  return {
    x: vector.x / length,
    y: vector.y / length
  };
}

function roleBoardClipPath(): string {
  return [
    roleBoardPentagon.top,
    roleBoardPentagon.topRight,
    roleBoardPentagon.bottomRight,
    roleBoardPentagon.bottomLeft,
    roleBoardPentagon.topLeft
  ]
    .map((point) => `${point.x * 100}% ${point.y * 100}%`)
    .join(", ");
}

function toLayoutPrecision(value: number): number {
  return Number(value.toFixed(3));
}
