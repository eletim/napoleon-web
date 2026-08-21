import type { CSSProperties } from "react";
import {
  mockCardBackComponent,
  mockCardBackComponentName,
  mockPlayingCardComponent,
  mockPlayingCardComponentName,
  type MockPlayingCard
} from "./mockPlayingCardAdapter";
import "./TableDesignMock.css";

type SeatId = "top-left" | "top-right" | "right" | "self" | "left";
type OpponentSeatId = Exclude<SeatId, "self">;
type TableDesignMockVariant = "projected" | "world";

interface Point {
  x: number;
  y: number;
}

interface Point3 extends Point {
  z: number;
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
  camera: PerspectiveCameraConfig;
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
  roleBoard: {
    innerPentagonScale: number;
  };
  riverGrid: {
    maxColumns: number;
    maxRows: number;
    rowGap: number;
  };
  roleMarker: {
    height: number;
    sectorMidpointRatio: number;
    width: number;
  };
  opponentHand: {
    baselineOffset: number;
    cardCounts: Readonly<Record<OpponentSeatId, number>>;
    cardGapRatio: number;
    cardThickness: number;
    maxCardCount: number;
  };
  seats: readonly SeatLayout[];
  tableSurface: readonly Point[];
  tabletopWorld: {
    roleBoardRadius: number;
    scale: number;
    tableSurfaceRadius: number;
  };
}

interface PerspectiveCameraConfig {
  focalLength: number;
  position: Point3;
  screenCenter: Point;
  target: Point3;
}

const pentagonCenter: Point = { x: 1120, y: 910 };
const pentagonStartAngle = -90;
const tabletopWorldScale = 1.8;
const tableSurfaceRadius = scaleTabletopDimension(700);
const roleBoardRadius = scaleTabletopDimension(175);

const roleBoardCenter: Box = {
  x: pentagonCenter.x,
  y: pentagonCenter.y,
  width: roleBoardRadius * 2,
  height: roleBoardRadius * 2
};

const roleBoardVertexOrder = ["top", "topRight", "bottomRight", "bottomLeft", "topLeft"] as const;
type RoleBoardVertexId = (typeof roleBoardVertexOrder)[number];

const roleBoardPentagon = Object.fromEntries(
  roleBoardVertexOrder.map((vertexId, index) => [
    vertexId,
    regularPentagon({ x: 0.5, y: 0.5 }, 0.5, pentagonStartAngle)[index]
  ])
) as Record<RoleBoardVertexId, Point>;

const tableSurfacePentagon = regularPentagon(pentagonCenter, tableSurfaceRadius, pentagonStartAngle);

const cardAspectRatio = 7 / 5;
const trickCardWidth = scaleTabletopDimension(118);
const riverGap = scaleTabletopDimension(18);
const opponentHandCardWidthRatio = 0.08;
const opponentHandCardGapRatio = 0.02;

const projectedSelfArtifacts: Pick<SeatLayout, "avatar" | "hand"> = {
  avatar: { x: 808, y: 1570 },
  hand: { x: 1118, y: 1660, rotation: 0 }
};

// Source of Truth: https://github.com/eletim/napoleon-web/issues/308#issuecomment-5348323047
// Keep the screenshot-facing coordinates here so the mock can be tuned without
// hunting through individual elements.
export const tableDesignMockLayout: TableDesignMockLayout = {
  page: {
    width: 2200,
    height: 1830,
    background: "#1d1d1d"
  },
  camera: {
    position: { x: 1120, y: 2450, z: 2750 },
    target: { x: 1120, y: 910, z: 0 },
    focalLength: 2300,
    screenCenter: { x: 1100, y: 671.144 }
  },
  tabletopWorld: {
    scale: tabletopWorldScale,
    tableSurfaceRadius,
    roleBoardRadius
  },
  tableSurface: tableSurfacePentagon,
  roleBoard: {
    innerPentagonScale: 0.42
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
    selfHand: { width: 172, height: toLayoutPrecision(172 * cardAspectRatio) }
  },
  opponentHand: {
    baselineOffset: scaleTabletopDimension(40),
    cardCounts: {
      "top-left": 1,
      "top-right": 5,
      right: 10,
      left: 7
    },
    cardGapRatio: opponentHandCardGapRatio,
    cardThickness: scaleTabletopDimension(6),
    maxCardCount: 10
  },
  currentTrickZone: {
    gapFromRiver: scaleTabletopDimension(28),
    paddingBlock: scaleTabletopDimension(52),
    paddingInline: scaleTabletopDimension(44)
  },
  riverGrid: {
    maxColumns: 5,
    maxRows: 4,
    rowGap: scaleTabletopDimension(24)
  },
  roleMarker: {
    width: scaleTabletopDimension(58),
    height: scaleTabletopDimension(34),
    sectorMidpointRatio: 0.5
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
      hand: { x: 1118, y: 1640, rotation: 0 }
    },
    {
      id: "left",
      label: "左席",
      avatar: { x: 136, y: 1070 },
      hand: { x: 308, y: 1086, rotation: -54 }
    }
  ]
};

const selfCards: readonly MockPlayingCard[] = [
  { rank: "5", suit: "spades" },
  { rank: "7", suit: "spades" }
];

const trickCards: Partial<Record<SeatId, MockPlayingCard>> = {
  "top-left": { rank: "10", suit: "hearts" },
  "top-right": { rank: "Q", suit: "hearts" },
  right: { rank: "K", suit: "spades" },
  left: { rank: "A", suit: "spades" },
  self: { rank: "J", suit: "spades" }
};

const riverCards: Record<string, readonly MockPlayingCard[]> = {
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

const roleMarkers: Record<SeatId, string> = {
  "top-left": "?",
  "top-right": "?",
  right: "?",
  self: "ナポ",
  left: "副"
};

const roleMarkerSeatOrder = ["top-left", "top-right", "right", "self", "left"] as const satisfies readonly SeatId[];
const opponentSeatOrder = ["top-left", "top-right", "right", "left"] as const satisfies readonly OpponentSeatId[];

export function TableDesignMock({ variant = "world" }: { variant?: TableDesignMockVariant }) {
  const layout = tableDesignMockLayout;
  const isProjected = variant === "projected";

  return (
    <main
      aria-label={`Issue 342 table design ${variant} mock`}
      className={`table-design-mock-page table-design-mock-page-${variant}`}
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
        {isProjected ? (
          <ProjectedTabletop layout={layout} />
        ) : (
          <>
            <TableSurfaceWorld points={layout.tableSurface} />
            {layout.seats.map((seat) => (
              <CurrentTrickZone key={`trick-${seat.id}`} seat={seat} />
            ))}
            {layout.seats.map((seat) => (
              <PointRiver key={`river-${seat.id}`} seat={seat} />
            ))}
            <OpponentHandsWorld layout={layout} />
            <RoleBoard layout={layout.center} />
          </>
        )}
        {layout.seats.map((seat) => (
          <PlayerArtifacts key={seat.id} seat={playerArtifactSeat(seat, isProjected)} />
        ))}
      </div>
    </main>
  );
}

function playerArtifactSeat(seat: SeatLayout, isProjected: boolean): SeatLayout {
  if (!isProjected || seat.id !== "self") {
    return seat;
  }

  return {
    ...seat,
    ...projectedSelfArtifacts
  };
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

function PlayerArtifacts({ seat }: { seat: SeatLayout }) {
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
      ) : null}
    </>
  );
}

function TableSurfaceWorld({ points }: { points: readonly Point[] }) {
  return (
    <svg
      aria-hidden="true"
      className="mock-table-surface mock-table-surface-world"
      viewBox={`0 0 ${tableDesignMockLayout.page.width} ${tableDesignMockLayout.page.height}`}
    >
      <polygon className="mock-table-surface-polygon" points={svgPoints(points)} />
    </svg>
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
  const sectorLines = createRoleBoardSectorLines(layout);
  const innerPentagonPoints = roleBoardInnerPolygonPoints(layout);

  return (
    <section
      aria-label="中央役職表示"
      className="mock-role-board"
      style={roleBoardStyle(layout)}
    >
      <div className="mock-role-board-shape">
        <svg
          aria-hidden="true"
          className="mock-role-board-sector-geometry"
          viewBox={`0 0 ${layout.width} ${layout.height}`}
        >
          {sectorLines.map((line, index) => (
            <line
              className="mock-role-board-sector-line"
              key={index}
              x1={line.outer.x}
              x2={line.inner.x}
              y1={line.outer.y}
              y2={line.inner.y}
            />
          ))}
          <polygon className="mock-role-board-inner-pentagon" points={innerPentagonPoints} />
        </svg>
        {roleMarkerSeatOrder.map((seatId) => {
          const marker = createRoleMarkerGeometry(layout, seatId);

          return (
            <span
              className={`role-marker role-marker-${seatId}`}
              key={seatId}
              style={roleMarkerStyle(marker)}
            >
              {roleMarkers[seatId]}
            </span>
          );
        })}
      </div>
    </section>
  );
}

function ProjectedTabletop({ layout }: { layout: TableDesignMockLayout }) {
  const camera = layout.camera;
  const tablePoints = projectTablePolygon(layout.tableSurface, camera);
  const roleOuterPoints = projectTablePolygon(roleBoardOuterPolygon(layout.center), camera);
  const roleInnerPoints = projectTablePolygon(roleBoardInnerPolygon(layout.center), camera);
  const sectorLines = createRoleBoardSectorLines(layout.center).map((line) => ({
    inner: projectTablePoint(roleBoardLocalToAbsolute(layout.center, line.inner), camera),
    outer: projectTablePoint(roleBoardLocalToAbsolute(layout.center, line.outer), camera)
  }));

  return (
    <>
      <svg
        aria-label="投影後の卓上Geometry"
        className="mock-projected-tabletop"
        viewBox={`0 0 ${layout.page.width} ${layout.page.height}`}
      >
        <polygon className="mock-table-surface-polygon" points={svgPoints(tablePoints)} />
        {layout.seats.map((seat) => (
          <ProjectedCurrentTrickZone key={`projected-trick-${seat.id}`} layout={layout} seat={seat} />
        ))}
        <g className="mock-projected-role-board">
          <polygon className="mock-projected-role-board-outer" points={svgPoints(roleOuterPoints)} />
          {sectorLines.map((line, index) => (
            <line
              className="mock-projected-role-board-sector-line"
              key={index}
              x1={line.outer.x}
              x2={line.inner.x}
              y1={line.outer.y}
              y2={line.inner.y}
            />
          ))}
          <polygon className="mock-projected-role-board-inner" points={svgPoints(roleInnerPoints)} />
          {roleMarkerSeatOrder.map((seatId) => (
            <ProjectedRoleMarker key={`projected-role-${seatId}`} layout={layout} seatId={seatId} />
          ))}
        </g>
      </svg>
      <div aria-hidden="true" className="mock-projected-card-layer">
        {layout.seats.map((seat) => (
          <ProjectedCurrentTrickCard key={`projected-trick-card-${seat.id}`} layout={layout} seat={seat} />
        ))}
        {layout.seats.map((seat) => (
          <ProjectedPointRiverCards key={`projected-river-cards-${seat.id}`} layout={layout} seat={seat} />
        ))}
        <ProjectedOpponentHands layout={layout} />
      </div>
    </>
  );
}

function OpponentHandsWorld({ layout }: { layout: TableDesignMockLayout }) {
  const hands = createOpponentHandsGeometry(layout);

  return (
    <svg
      aria-label="相手の垂直手札World Geometry"
      className="mock-opponent-hands-world"
      viewBox={`0 0 ${layout.page.width} ${layout.page.height}`}
    >
      {hands.map((hand) => (
        <g
          aria-label={`${seatLabel(hand.seatId)}の裏向き手札`}
          className={`mock-opponent-hand-world mock-opponent-hand-world-${hand.seatId}`}
          key={hand.seatId}
        >
          {hand.cards.map((card) => (
            <polygon
              className="mock-opponent-hand-world-card"
              key={card.index}
              points={svgPoints(
                verticalCardTopDownThicknessPolygon(card, hand.edge.normal, layout.opponentHand.cardThickness)
              )}
            />
          ))}
        </g>
      ))}
    </svg>
  );
}

function ProjectedOpponentHands({ layout }: { layout: TableDesignMockLayout }) {
  const hands = createOpponentHandsGeometry(layout);

  return (
    <>
      {hands.map((hand) => (
        <section
          aria-label={`${seatLabel(hand.seatId)}の裏向き手札`}
          className={`mock-projected-opponent-hand mock-projected-opponent-hand-${hand.seatId}`}
          key={hand.seatId}
        >
          {hand.cards.map((card) => (
            <ProjectedPlayingCardBack
              corners={projectVerticalCard(card, layout.camera)}
              key={card.index}
              size={hand.cardSize}
            />
          ))}
        </section>
      ))}
    </>
  );
}

function ProjectedCurrentTrickZone({ layout, seat }: { layout: TableDesignMockLayout; seat: SeatLayout }) {
  const cardCount = riverCards[seat.id]?.length ?? 0;
  const geometry = createCurrentTrickZoneGeometry(layout, seat.id, cardCount);
  const zoneCorners = projectTableCard(geometry, layout.camera);

  return (
    <g className={`mock-projected-current-trick-zone mock-projected-current-trick-zone-${seat.id}`}>
      <polygon className="mock-projected-current-trick-zone-fill" points={svgPoints(zoneCorners)} />
    </g>
  );
}

function ProjectedCurrentTrickCard({ layout, seat }: { layout: TableDesignMockLayout; seat: SeatLayout }) {
  const cardCount = riverCards[seat.id]?.length ?? 0;
  const geometry = createCurrentTrickZoneGeometry(layout, seat.id, cardCount);
  const size = layout.cardSizes.trick;
  const corners = projectTableCard(
    {
      ...geometry,
      height: size.height,
      width: size.width
    },
    layout.camera
  );

  return <ProjectedPlayingCard card={trickCards[seat.id]} corners={corners} size={size} variant="trick" />;
}

function ProjectedPointRiverCards({ layout, seat }: { layout: TableDesignMockLayout; seat: SeatLayout }) {
  const cards = riverCards[seat.id] ?? [];
  const river = createRiverGeometry(layout, seat.id);
  const placements = createRiverPlacements(cards.length, layout, seat.id);

  return (
    <g className={`mock-projected-point-river mock-projected-point-river-${seat.id}`}>
      {cards.slice(0, layout.riverGrid.maxColumns * layout.riverGrid.maxRows).map((card, index) => {
        const placement = placements[index] ?? { x: 0, y: 0, rotation: 0 };
        const corners = projectTableCard(
          {
            direction: river.direction,
            height: river.cardSize.height,
            normal: river.normal,
            width: river.cardSize.width,
            x: toLayoutPrecision(river.x + river.direction.x * placement.x + river.normal.x * placement.y),
            y: toLayoutPrecision(river.y + river.direction.y * placement.x + river.normal.y * placement.y)
          },
          layout.camera,
          "top-left"
        );

        return (
          <ProjectedPlayingCard
            card={card}
            corners={corners}
            key={`${card.rank}-${card.suit}-${index}`}
            size={river.cardSize}
            variant={seat.id === "self" ? "self-river" : "river"}
          />
        );
      })}
    </g>
  );
}

function ProjectedRoleMarker({ layout, seatId }: { layout: TableDesignMockLayout; seatId: SeatId }) {
  const marker = createRoleMarkerGeometry(layout.center, seatId);
  const center = roleBoardLocalToAbsolute(layout.center, marker);
  const corners = projectTableCard(
    {
      direction: { x: 1, y: 0 },
      height: marker.height,
      normal: { x: 0, y: 1 },
      width: marker.width,
      x: center.x,
      y: center.y
    },
    layout.camera
  );
  const labelCenter = polygonCenter(corners);

  return (
    <g className={`mock-projected-role-marker mock-projected-role-marker-${seatId}`}>
      <polygon className="mock-projected-role-marker-fill" points={svgPoints(corners)} />
      <text
        className="mock-projected-role-marker-text"
        dominantBaseline="central"
        textAnchor="middle"
        x={labelCenter.x}
        y={labelCenter.y}
      >
        {roleMarkers[seatId]}
      </text>
    </g>
  );
}

function ProjectedPlayingCard({
  card,
  corners,
  size,
  variant
}: {
  card: MockPlayingCard | undefined;
  corners: readonly Point[];
  size: { height: number; width: number };
  variant: "river" | "self-river" | "trick";
}) {
  if (card === undefined) {
    return null;
  }

  const transform = projectiveTransformForRectangle(corners, size.width, size.height);

  return (
    <PlayingCard
      card={card}
      className={`mock-projected-playing-card mock-projected-playing-card-${variant}`}
      style={
        {
          "--mock-projected-card-height": `${size.height}px`,
          "--mock-projected-card-transform": transform,
          "--mock-projected-card-width": `${size.width}px`
        } as CSSProperties
      }
    />
  );
}

function PlayingCard({
  card,
  className,
  style
}: {
  card: MockPlayingCard | undefined;
  className: string;
  style?: CSSProperties;
}) {
  if (card === undefined) {
    return null;
  }

  const CardComponent = mockPlayingCardComponent(card);
  const componentName = mockPlayingCardComponentName(card);

  return (
    <article
      aria-label={componentName}
      className={`${className} mock-playing-card`}
      style={style}
    >
      <CardComponent aria-hidden="true" className="mock-playing-card-svg" focusable="false" />
    </article>
  );
}

function PlayingCardBack({ className, style }: { className: string; style?: CSSProperties }) {
  const CardBackComponent = mockCardBackComponent();

  return (
    <span
      aria-label={mockCardBackComponentName}
      className={`${className} mock-playing-card mock-playing-card-back`}
      style={style}
    >
      <CardBackComponent aria-hidden="true" className="mock-playing-card-svg" focusable="false" />
    </span>
  );
}

function ProjectedPlayingCardBack({
  corners,
  size
}: {
  corners: readonly Point[];
  size: { height: number; width: number };
}) {
  const transform = projectiveTransformForRectangle(corners, size.width, size.height);

  return (
    <PlayingCardBack
      className="mock-projected-playing-card mock-projected-playing-card-opponent-hand"
      style={
        {
          "--mock-projected-card-height": `${size.height}px`,
          "--mock-projected-card-transform": transform,
          "--mock-projected-card-width": `${size.width}px`
        } as CSSProperties
      }
    />
  );
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

interface RoleMarkerGeometry {
  height: number;
  sector: {
    innerEnd: Point;
    innerStart: Point;
    outerEnd: Point;
    outerStart: Point;
  };
  width: number;
  x: number;
  y: number;
}

interface TableCardPlane {
  direction: Point;
  height: number;
  normal: Point;
  width: number;
  x: number;
  y: number;
}

interface VerticalCardGeometry {
  index: number;
  leftBottom: Point3;
  leftTop: Point3;
  rightBottom: Point3;
  rightTop: Point3;
}

interface OpponentHandCardMetrics {
  cardSize: { height: number; width: number };
  edgeLength: number;
  gap: number;
}

interface OpponentHandGeometry {
  baseline: {
    center: Point;
    direction: Point;
    normal: Point;
    offset: number;
  };
  cardSize: { height: number; width: number };
  cards: readonly VerticalCardGeometry[];
  edge: RoleBoardEdgeGeometry;
  gap: number;
  handWidth: number;
  seatId: OpponentSeatId;
}

const roleBoardEdges: Record<SeatId, { end: Point; start: Point }> = {
  "top-left": { start: roleBoardPentagon.top, end: roleBoardPentagon.topLeft },
  "top-right": { start: roleBoardPentagon.topRight, end: roleBoardPentagon.top },
  right: { start: roleBoardPentagon.bottomRight, end: roleBoardPentagon.topRight },
  self: { start: roleBoardPentagon.bottomLeft, end: roleBoardPentagon.bottomRight },
  left: { start: roleBoardPentagon.topLeft, end: roleBoardPentagon.bottomLeft }
};

const tableSurfaceVertexIds = ["top", "topRight", "bottomRight", "bottomLeft", "topLeft"] as const;
const tableSurfaceVertexIndexes = Object.fromEntries(
  tableSurfaceVertexIds.map((vertexId, index) => [vertexId, index])
) as Record<RoleBoardVertexId, number>;

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

export function createTableSurfaceEdgeGeometry(
  layout: TableDesignMockLayout,
  seatId: SeatId
): RoleBoardEdgeGeometry {
  const edge = roleBoardEdges[seatId];
  const start = layout.tableSurface[tableSurfaceVertexIndexes[roleBoardVertexIdForPoint(edge.start)]];
  const end = layout.tableSurface[tableSurfaceVertexIndexes[roleBoardVertexIdForPoint(edge.end)]];

  if (start === undefined || end === undefined) {
    throw new Error(`Missing table surface edge for ${seatId}`);
  }

  return createEdgeGeometry(start, end);
}

export function createOpponentHandsGeometry(layout: TableDesignMockLayout): OpponentHandGeometry[] {
  return opponentSeatOrder.map((seatId) => createOpponentHandGeometry(layout, seatId));
}

export function createOpponentHandGeometry(
  layout: TableDesignMockLayout,
  seatId: OpponentSeatId
): OpponentHandGeometry {
  const edge = createTableSurfaceEdgeGeometry(layout, seatId);
  const cardCount = Math.min(layout.opponentHand.cardCounts[seatId], layout.opponentHand.maxCardCount);
  const metrics = createOpponentHandCardMetrics(edge.d, layout.opponentHand.cardGapRatio);
  const handWidth = opponentHandWidth(cardCount, metrics);
  const midpoint = midpointBetween(edge.start, edge.end);
  const baselineCenter = {
    x: toLayoutPrecision(midpoint.x + edge.normal.x * layout.opponentHand.baselineOffset),
    y: toLayoutPrecision(midpoint.y + edge.normal.y * layout.opponentHand.baselineOffset)
  };
  const cards = Array.from({ length: cardCount }, (_, index) => {
    const cardCenterOffset =
      -handWidth / 2 +
      metrics.cardSize.width / 2 +
      index * (metrics.cardSize.width + metrics.gap);
    const cardCenter = {
      x: baselineCenter.x + edge.direction.x * cardCenterOffset,
      y: baselineCenter.y + edge.direction.y * cardCenterOffset
    };

    return createVerticalCardGeometry({
      baselineCenter: cardCenter,
      direction: edge.direction,
      height: metrics.cardSize.height,
      index,
      width: metrics.cardSize.width
    });
  });

  return {
    baseline: {
      center: baselineCenter,
      direction: edge.direction,
      normal: edge.normal,
      offset: layout.opponentHand.baselineOffset
    },
    cardSize: metrics.cardSize,
    cards,
    edge,
    gap: metrics.gap,
    handWidth,
    seatId
  };
}

export function createOpponentHandCardMetrics(
  edgeLength: number,
  gapRatio = opponentHandCardGapRatio,
  widthRatio = opponentHandCardWidthRatio
): OpponentHandCardMetrics {
  const width = edgeLength * widthRatio;
  const gap = edgeLength * gapRatio;

  return {
    cardSize: {
      width,
      height: width * cardAspectRatio
    },
    edgeLength,
    gap
  };
}

export function opponentHandWidth(cardCount: number, metrics: OpponentHandCardMetrics): number {
  if (cardCount <= 0) {
    return 0;
  }

  return cardCount * metrics.cardSize.width + (cardCount - 1) * metrics.gap;
}

export function createVerticalCardGeometry({
  baselineCenter,
  direction,
  height,
  index,
  width
}: {
  baselineCenter: Point;
  direction: Point;
  height: number;
  index: number;
  width: number;
}): VerticalCardGeometry {
  const halfWidth = width / 2;
  const leftBottom = {
    x: toLayoutPrecision(baselineCenter.x - direction.x * halfWidth),
    y: toLayoutPrecision(baselineCenter.y - direction.y * halfWidth),
    z: 0
  };
  const rightBottom = {
    x: toLayoutPrecision(baselineCenter.x + direction.x * halfWidth),
    y: toLayoutPrecision(baselineCenter.y + direction.y * halfWidth),
    z: 0
  };

  return {
    index,
    leftBottom,
    rightBottom,
    rightTop: { ...rightBottom, z: height },
    leftTop: { ...leftBottom, z: height }
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

export function createRoleMarkerGeometry(
  layout: Box,
  seatId: SeatId,
  marker = tableDesignMockLayout.roleMarker
): RoleMarkerGeometry {
  const sector = createRoleSectorGeometry(layout, seatId);
  const outerMidpoint = midpointBetween(sector.outerStart, sector.outerEnd);
  const innerMidpoint = midpointBetween(sector.innerStart, sector.innerEnd);
  const center = interpolatePoint(outerMidpoint, innerMidpoint, marker.sectorMidpointRatio);

  return {
    height: marker.height,
    sector,
    width: marker.width,
    x: toLayoutPrecision(center.x),
    y: toLayoutPrecision(center.y)
  };
}

export function createRoleSectorGeometry(
  layout: Box,
  seatId: SeatId,
  innerScale = tableDesignMockLayout.roleBoard.innerPentagonScale
): RoleMarkerGeometry["sector"] {
  const edge = roleBoardEdges[seatId];

  return {
    outerStart: roleBoardLocalPoint(layout, edge.start),
    outerEnd: roleBoardLocalPoint(layout, edge.end),
    innerStart: roleBoardScaledLocalPoint(layout, edge.start, innerScale),
    innerEnd: roleBoardScaledLocalPoint(layout, edge.end, innerScale)
  };
}

function createRoleBoardSectorLines(
  layout: Box,
  innerScale = tableDesignMockLayout.roleBoard.innerPentagonScale
): Array<{ inner: Point; outer: Point }> {
  return roleBoardVertexOrder.map((vertexId) => ({
    outer: roleBoardLocalPoint(layout, roleBoardPentagon[vertexId]),
    inner: roleBoardScaledLocalPoint(layout, roleBoardPentagon[vertexId], innerScale)
  }));
}

export function projectTablePoint(point: Point | Point3, camera = tableDesignMockLayout.camera): Point {
  const worldPoint = "z" in point ? point : { ...point, z: 0 };
  const forward = normalizeVector3(subtractPoint3(camera.target, camera.position));
  const right = normalizeVector3(crossPoint3({ x: 0, y: 0, z: 1 }, forward));
  const up = normalizeVector3(crossPoint3(forward, right));
  const delta = subtractPoint3(worldPoint, camera.position);
  const depth = dotPoint3(delta, forward);
  const cameraX = dotPoint3(delta, right);
  const cameraY = dotPoint3(delta, up);

  return {
    x: toLayoutPrecision(camera.screenCenter.x + (cameraX * camera.focalLength) / depth),
    y: toLayoutPrecision(camera.screenCenter.y - (cameraY * camera.focalLength) / depth)
  };
}

export function projectTablePolygon(points: readonly Point[], camera = tableDesignMockLayout.camera): Point[] {
  return points.map((point) => projectTablePoint(point, camera));
}

export function projectTableCard(
  card: TableCardPlane,
  camera = tableDesignMockLayout.camera,
  origin: "center" | "top-left" = "center"
): Point[] {
  const topLeft =
    origin === "center"
      ? {
          x: card.x - card.direction.x * (card.width / 2) - card.normal.x * (card.height / 2),
          y: card.y - card.direction.y * (card.width / 2) - card.normal.y * (card.height / 2)
        }
      : { x: card.x, y: card.y };
  const corners = [
    topLeft,
    {
      x: topLeft.x + card.direction.x * card.width,
      y: topLeft.y + card.direction.y * card.width
    },
    {
      x: topLeft.x + card.direction.x * card.width + card.normal.x * card.height,
      y: topLeft.y + card.direction.y * card.width + card.normal.y * card.height
    },
    {
      x: topLeft.x + card.normal.x * card.height,
      y: topLeft.y + card.normal.y * card.height
    }
  ];

  return projectTablePolygon(corners, camera);
}

export function projectVerticalCard(
  card: VerticalCardGeometry,
  camera = tableDesignMockLayout.camera
): Point[] {
  return [
    projectTablePoint(card.leftTop, camera),
    projectTablePoint(card.rightTop, camera),
    projectTablePoint(card.rightBottom, camera),
    projectTablePoint(card.leftBottom, camera)
  ];
}

function projectiveTransformForRectangle(corners: readonly Point[], width: number, height: number): string {
  const sourceCorners = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height }
  ];
  const [a, b, c, d, e, f, g, h] = solveHomography(sourceCorners, corners);
  const matrix = [a, d, 0, g, b, e, 0, h, 0, 0, 1, 0, c, f, 0, 1].map(cssMatrixNumber);

  return `matrix3d(${matrix.join(",")})`;
}

function solveHomography(sourceCorners: readonly Point[], targetCorners: readonly Point[]): number[] {
  const rows: number[][] = [];

  for (let index = 0; index < 4; index += 1) {
    const source = sourceCorners[index];
    const target = targetCorners[index];

    rows.push([source.x, source.y, 1, 0, 0, 0, -target.x * source.x, -target.x * source.y, target.x]);
    rows.push([0, 0, 0, source.x, source.y, 1, -target.y * source.x, -target.y * source.y, target.y]);
  }

  for (let column = 0; column < 8; column += 1) {
    let pivotRow = column;

    for (let row = column + 1; row < rows.length; row += 1) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivotRow][column])) {
        pivotRow = row;
      }
    }

    [rows[column], rows[pivotRow]] = [rows[pivotRow], rows[column]];

    const pivot = rows[column][column];
    if (Math.abs(pivot) < 1e-9) {
      throw new Error("Projected card corners cannot define a CSS transform");
    }

    for (let cell = column; cell < 9; cell += 1) {
      rows[column][cell] /= pivot;
    }

    for (let row = 0; row < rows.length; row += 1) {
      if (row === column) {
        continue;
      }

      const factor = rows[row][column];
      for (let cell = column; cell < 9; cell += 1) {
        rows[row][cell] -= factor * rows[column][cell];
      }
    }
  }

  return rows.map((row) => row[8]);
}

function cssMatrixNumber(value: number): string {
  const rounded = Math.abs(value) < 1e-10 ? 0 : Number(value.toFixed(10));

  return String(rounded);
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

function verticalCardTopDownThicknessPolygon(
  card: VerticalCardGeometry,
  normal: Point,
  thickness: number
): Point[] {
  const halfThickness = thickness / 2;

  return [
    {
      x: card.leftBottom.x - normal.x * halfThickness,
      y: card.leftBottom.y - normal.y * halfThickness
    },
    {
      x: card.rightBottom.x - normal.x * halfThickness,
      y: card.rightBottom.y - normal.y * halfThickness
    },
    {
      x: card.rightBottom.x + normal.x * halfThickness,
      y: card.rightBottom.y + normal.y * halfThickness
    },
    {
      x: card.leftBottom.x + normal.x * halfThickness,
      y: card.leftBottom.y + normal.y * halfThickness
    }
  ];
}

function trickZoneStyle(zone: CurrentTrickZoneGeometry): CSSProperties {
  return {
    ...pointWithRotationStyle(zone),
    "--mock-trick-zone-height": `${zone.height}px`,
    "--mock-trick-zone-width": `${zone.width}px`
  } as CSSProperties;
}

function roleMarkerStyle(marker: RoleMarkerGeometry): CSSProperties {
  return {
    "--mock-role-marker-height": `${marker.height}px`,
    "--mock-role-marker-width": `${marker.width}px`,
    "--mock-role-marker-x": `${marker.x}px`,
    "--mock-role-marker-y": `${marker.y}px`
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

function createEdgeGeometry(start: Point, end: Point): RoleBoardEdgeGeometry {
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

function roleBoardVertexIdForPoint(point: Point): RoleBoardVertexId {
  const entry = roleBoardVertexOrder.find((vertexId) => roleBoardPentagon[vertexId] === point);

  if (entry === undefined) {
    throw new Error("Unknown role-board vertex");
  }

  return entry;
}

function seatLabel(seatId: SeatId): string {
  return tableDesignMockLayout.seats.find((seat) => seat.id === seatId)?.label ?? seatId;
}

function roleBoardLocalToAbsolute(layout: Box, point: Point): Point {
  return {
    x: toLayoutPrecision(layout.x - layout.width / 2 + point.x),
    y: toLayoutPrecision(layout.y - layout.height / 2 + point.y)
  };
}

function roleBoardLocalPoint(layout: Box, point: Point): Point {
  return {
    x: point.x * layout.width,
    y: point.y * layout.height
  };
}

function roleBoardScaledLocalPoint(layout: Box, point: Point, scale: number): Point {
  const center = { x: layout.width / 2, y: layout.height / 2 };
  const outer = roleBoardLocalPoint(layout, point);

  return {
    x: toLayoutPrecision(center.x + (outer.x - center.x) * scale),
    y: toLayoutPrecision(center.y + (outer.y - center.y) * scale)
  };
}

function roleBoardInnerPolygonPoints(
  layout: Box,
  innerScale = tableDesignMockLayout.roleBoard.innerPentagonScale
): string {
  return roleBoardVertexOrder
    .map((vertexId) => {
      const point = roleBoardScaledLocalPoint(layout, roleBoardPentagon[vertexId], innerScale);

      return `${point.x},${point.y}`;
    })
    .join(" ");
}

function roleBoardOuterPolygon(layout: Box): Point[] {
  return roleBoardVertexOrder.map((vertexId) => roleBoardAbsolutePoint(layout, roleBoardPentagon[vertexId]));
}

function roleBoardInnerPolygon(
  layout: Box,
  innerScale = tableDesignMockLayout.roleBoard.innerPentagonScale
): Point[] {
  return roleBoardVertexOrder.map((vertexId) =>
    roleBoardLocalToAbsolute(layout, roleBoardScaledLocalPoint(layout, roleBoardPentagon[vertexId], innerScale))
  );
}

export function regularPentagon(center: Point, radius: number, startAngle: number): Point[] {
  return Array.from({ length: 5 }, (_, index) => {
    const angle = ((startAngle + index * 72) * Math.PI) / 180;

    return {
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius
    };
  });
}

function svgPoints(points: readonly Point[]): string {
  return points.map((point) => `${toLayoutPrecision(point.x)},${toLayoutPrecision(point.y)}`).join(" ");
}

function polygonCenter(points: readonly Point[]): Point {
  const total = points.reduce(
    (sum, point) => ({
      x: sum.x + point.x,
      y: sum.y + point.y
    }),
    { x: 0, y: 0 }
  );

  return {
    x: toLayoutPrecision(total.x / points.length),
    y: toLayoutPrecision(total.y / points.length)
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

function interpolatePoint(start: Point, end: Point, ratio: number): Point {
  return {
    x: start.x + (end.x - start.x) * ratio,
    y: start.y + (end.y - start.y) * ratio
  };
}

function normalizeVector(vector: Point): Point {
  const length = distance({ x: 0, y: 0 }, vector);

  return {
    x: vector.x / length,
    y: vector.y / length
  };
}

function subtractPoint3(a: Point3, b: Point3): Point3 {
  return {
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z
  };
}

function dotPoint3(a: Point3, b: Point3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function crossPoint3(a: Point3, b: Point3): Point3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  };
}

function normalizeVector3(vector: Point3): Point3 {
  const length = Math.hypot(vector.x, vector.y, vector.z);

  return {
    x: vector.x / length,
    y: vector.y / length,
    z: vector.z / length
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

function scaleTabletopDimension(value: number): number {
  return toLayoutPrecision(value * tabletopWorldScale);
}
