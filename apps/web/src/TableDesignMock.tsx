import { useEffect, useState, type CSSProperties } from "react";
import { cardDesignSuitSymbols } from "./CardDesignCard";
import {
  CardmeisterPlayingCard,
  cardmeisterCardId,
  useCardmeisterScript
} from "./CardmeisterPlayingCard";
import { fourColorSuitColors } from "./cardSuitTheme";
import {
  mockCardBackComponent,
  mockCardBackComponentName,
  type MockPlayingCard
} from "./mockPlayingCardAdapter";
import "./TableDesignMock.css";

type SeatId = "top-left" | "top-right" | "right" | "self" | "left";
type OpponentSeatId = Exclude<SeatId, "self">;
type TableDesignMockVariant = "bidding" | "projected" | "world";

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

interface ViewportSize {
  height: number;
  width: number;
}

interface SeatLayout {
  hand: Point & { rotation: number };
  id: SeatId;
  label: string;
}

interface TableDesignMockLayout {
  camera: PerspectiveCameraConfig;
  cardSizes: {
    selfHand: { height: number; width: number };
  };
  currentTrickZone: {
    cardWidthRatio: number;
    positionRatio: number;
    zoneToCardRatio: number;
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
    cardExposureRatio: number;
    cellHeightRatio: number;
    columnGap: number;
    maxColumns: number;
    maxRows: number;
    rowGap: number;
    widthRatio: number;
  };
  riverFace: {
    borderWidthRatio: number;
    gapRatio: number;
    paddingRatio: number;
    rankFontRatio: number;
    suitFontRatio: number;
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
  playerInfo: {
    avatarSize: number;
    gap: number;
    offsetFromHand: number;
    selfGap: number;
    unitHeight: number;
    unitWidth: number;
    viewportMargin: number;
  };
  bidding: {
    bubble: {
      gap: number;
      height: number;
      viewportMargin: number;
      width: number;
    };
    overlay: {
      gapFromSelfHand: number;
      height: number;
      maxWidth: number;
      minWidth: number;
      viewportMargin: number;
      widthRatio: number;
      yOffsetFromTableCenter: number;
    };
  };
  selfHandUi: {
    bottomInset: number;
    maxCardCount: number;
  };
  projectedFit: {
    tableHeightRatio: number;
    topInsetRatio: number;
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
const mockPageWidth = 2200;
const mockPageHeight = 1830;
const maxSelfHandCardCount = 13;
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
const selfHandCardWidth = (0.8 * mockPageWidth) / maxSelfHandCardCount;
const opponentHandCardWidthRatio = 0.08;
const opponentHandCardGapRatio = 0.02;

// Source of Truth: https://github.com/eletim/napoleon-web/issues/308#issuecomment-5348323047
// Keep the screenshot-facing coordinates here so the mock can be tuned without
// hunting through individual elements.
export const tableDesignMockLayout: TableDesignMockLayout = {
  page: {
    width: mockPageWidth,
    height: mockPageHeight,
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
    selfHand: { width: selfHandCardWidth, height: selfHandCardWidth * cardAspectRatio }
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
  playerInfo: {
    avatarSize: 38,
    gap: 12,
    offsetFromHand: 18,
    selfGap: 8,
    unitHeight: 52,
    unitWidth: 210,
    viewportMargin: 18
  },
  bidding: {
    bubble: {
      gap: 18,
      height: 92,
      viewportMargin: 18,
      width: 168
    },
    overlay: {
      gapFromSelfHand: 36,
      height: 430,
      maxWidth: 820,
      minWidth: 680,
      viewportMargin: 24,
      widthRatio: 0.48,
      yOffsetFromTableCenter: 34
    }
  },
  selfHandUi: {
    bottomInset: 16,
    maxCardCount: maxSelfHandCardCount
  },
  projectedFit: {
    tableHeightRatio: 0.76,
    topInsetRatio: 0.015
  },
  currentTrickZone: {
    cardWidthRatio: 0.9261,
    positionRatio: 0.5,
    zoneToCardRatio: 1.12
  },
  riverGrid: {
    cardExposureRatio: 0.25,
    cellHeightRatio: 1.5,
    columnGap: scaleTabletopDimension(8),
    maxColumns: 10,
    maxRows: 2,
    rowGap: scaleTabletopDimension(8),
    widthRatio: 0.94
  },
  riverFace: {
    borderWidthRatio: 0.03,
    gapRatio: 0.025,
    paddingRatio: 0.025,
    rankFontRatio: 0.9,
    suitFontRatio: 0.82
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
      hand: { x: 672, y: 292, rotation: -19 }
    },
    {
      id: "top-right",
      label: "北東",
      hand: { x: 1538, y: 294, rotation: 19 }
    },
    {
      id: "right",
      label: "右席",
      hand: { x: 1942, y: 1080, rotation: 55 }
    },
    {
      id: "self",
      label: "自分",
      hand: { x: 1118, y: 1640, rotation: 0 }
    },
    {
      id: "left",
      label: "左席",
      hand: { x: 308, y: 1086, rotation: -54 }
    }
  ]
};

const selfCards: readonly MockPlayingCard[] = [
  { rank: "A", suit: "spades" },
  { rank: "2", suit: "clubs" },
  { rank: "3", suit: "hearts" },
  { rank: "4", suit: "diamonds" },
  { rank: "5", suit: "spades" },
  { rank: "6", suit: "clubs" },
  { rank: "7", suit: "hearts" },
  { rank: "8", suit: "diamonds" },
  { rank: "9", suit: "spades" },
  { rank: "10", suit: "clubs" },
  { rank: "J", suit: "hearts" },
  { rank: "Q", suit: "diamonds" },
  { rank: "K", suit: "spades" }
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
    { rank: "10", suit: "diamonds" },
    { rank: "Q", suit: "hearts" },
    { rank: "J", suit: "hearts" },
    { rank: "9", suit: "hearts" },
    { rank: "8", suit: "hearts" },
    { rank: "7", suit: "hearts" },
    { rank: "6", suit: "hearts" },
    { rank: "5", suit: "hearts" },
    { rank: "4", suit: "hearts" },
    { rank: "A", suit: "diamonds" },
    { rank: "K", suit: "diamonds" },
    { rank: "Q", suit: "diamonds" },
    { rank: "J", suit: "diamonds" },
    { rank: "9", suit: "diamonds" },
    { rank: "8", suit: "diamonds" },
    { rank: "7", suit: "diamonds" },
    { rank: "6", suit: "diamonds" },
    { rank: "5", suit: "diamonds" },
    { rank: "4", suit: "diamonds" }
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

type BiddingMockAction =
  | { type: "bid"; suit: MockPlayingCard["suit"]; value: number }
  | { type: "pass" };

const biddingMockActions: Readonly<Record<SeatId, BiddingMockAction>> = {
  "top-left": { type: "pass" },
  "top-right": { type: "bid", suit: "diamonds", value: 14 },
  right: { type: "bid", suit: "spades", value: 15 },
  left: { type: "bid", suit: "hearts", value: 14 },
  self: { type: "bid", suit: "clubs", value: 13 }
};

const biddingMockCurrentHighest = { type: "bid", suit: "spades", value: 15 } as const satisfies BiddingMockAction;
const biddingMockSelectedBid = { type: "bid", suit: "spades", value: 16 } as const satisfies BiddingMockAction;
const biddingSuitOptions = ["spades", "hearts", "diamonds", "clubs"] as const satisfies readonly MockPlayingCard["suit"][];

export function TableDesignMock({ variant = "world" }: { variant?: TableDesignMockVariant }) {
  const layout = tableDesignMockLayout;
  const isProjected = variant === "projected" || variant === "bidding";
  const isBidding = variant === "bidding";
  const viewportSize = useViewportSize(layout.page);

  useCardmeisterScript();

  return (
    <main
      aria-label={`Issue 348 table design ${variant} mock`}
      className={`table-design-mock-page table-design-mock-page-${variant}`}
      style={
        {
          "--mock-page-background": layout.page.background,
          "--mock-page-height": `${layout.page.height}px`,
          "--mock-page-width": `${layout.page.width}px`
        } as CSSProperties
      }
    >
      <div className="table-design-stage">
        <HudBox layout={layout.hud} />
        {isProjected ? (
          <ProjectedTabletop layout={layout} viewportSize={viewportSize} />
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
        {isBidding ? <BiddingMockOverlay layout={layout} viewportSize={viewportSize} /> : null}
        <PlayerInfoOverlay isProjected={isProjected} layout={layout} viewportSize={viewportSize} />
        {isBidding ? <BiddingBubbleOverlay layout={layout} viewportSize={viewportSize} /> : null}
        <SelfHand cards={selfCards} layout={layout} viewportSize={viewportSize} />
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

function PlayerInfoOverlay({
  isProjected,
  layout,
  viewportSize
}: {
  isProjected: boolean;
  layout: TableDesignMockLayout;
  viewportSize: ViewportSize;
}) {
  const playerInfos = createPlayerInfoLayouts(layout, viewportSize, isProjected);

  return (
    <>
      {playerInfos.map((info) => (
        <PlayerInfoUnit info={info} key={info.seatId} />
      ))}
    </>
  );
}

function PlayerInfoUnit({ info }: { info: PlayerInfoGeometry }) {
  return (
    <div
      aria-label={`${info.label} プレイヤー`}
      className={`mock-player-info mock-player-info-${info.seatId}`}
      style={playerInfoStyle(info)}
    >
      <span aria-hidden="true" className="mock-player-info-avatar">
        <span className="mock-player-info-avatar-head" />
        <span className="mock-player-info-avatar-body" />
      </span>
      <span className="mock-player-info-label">{info.label}</span>
    </div>
  );
}

function BiddingMockOverlay({
  layout,
  viewportSize
}: {
  layout: TableDesignMockLayout;
  viewportSize: ViewportSize;
}) {
  const geometry = createBiddingOverlayGeometry(layout, viewportSize);
  const selectedSuit = biddingMockSelectedBid.suit;

  return (
    <section
      aria-label="競り操作Overlay"
      className="mock-bidding-overlay"
      style={biddingOverlayStyle(geometry)}
    >
      <div className="mock-bidding-highest">
        <span className="mock-bidding-label">現在の最高入札</span>
        <strong
          className="mock-bidding-highest-value"
          style={biddingActionColorStyle(biddingMockCurrentHighest)}
        >
          {biddingActionLabel(biddingMockCurrentHighest)}
        </strong>
      </div>
      <div aria-label="スート選択" className="mock-bidding-suit-selector">
        {biddingSuitOptions.map((suit) => (
          <button
            aria-pressed={suit === selectedSuit}
            className="mock-bidding-suit-button"
            key={suit}
            style={
              {
                "--mock-bidding-suit-color": fourColorSuitColors[suit]
              } as CSSProperties
            }
            type="button"
          >
            {cardDesignSuitSymbols[suit]}
          </button>
        ))}
      </div>
      <div aria-label="入札数値選択" className="mock-bidding-number-selector">
        <button className="mock-bidding-step-button" type="button">-</button>
        <strong className="mock-bidding-number-value">{biddingMockSelectedBid.value}</strong>
        <button className="mock-bidding-step-button" type="button">+</button>
      </div>
      <div className="mock-bidding-actions">
        <button className="mock-bidding-declare-button" type="button">宣言</button>
        <button className="mock-bidding-pass-button" type="button">PASS</button>
      </div>
    </section>
  );
}

function BiddingBubbleOverlay({
  layout,
  viewportSize
}: {
  layout: TableDesignMockLayout;
  viewportSize: ViewportSize;
}) {
  const bubbles = createBiddingBubbleLayouts(layout, viewportSize);

  return (
    <div aria-label="各プレイヤーの最新競り宣言" className="mock-bidding-bubble-layer">
      {bubbles.map((bubble) => (
        <output
          aria-label={`${bubble.label} 最新宣言 ${biddingActionLabel(bubble.action)}`}
          className={`mock-bidding-bubble mock-bidding-bubble-${bubble.seatId}`}
          key={bubble.seatId}
          style={biddingBubbleStyle(bubble)}
        >
          {biddingActionLabel(bubble.action)}
        </output>
      ))}
    </div>
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

function SelfHand({
  cards,
  layout,
  viewportSize
}: {
  cards: readonly MockPlayingCard[];
  layout: TableDesignMockLayout;
  viewportSize: ViewportSize;
}) {
  const selfHandLayout = createSelfHandViewportLayout(layout, cards.length, viewportSize);

  return (
    <div
      aria-label="自分の表向き手札"
      className="mock-self-hand"
      style={selfHandViewportStyle(selfHandLayout)}
    >
      {cards.map((card, index) => (
        <PlayingCard card={card} className="mock-self-hand-card" key={`${card.rank}-${card.suit}-${index}`} />
      ))}
    </div>
  );
}

function CurrentTrickZone({ seat }: { seat: SeatLayout }) {
  const cardCount = riverCards[seat.id]?.length ?? 0;
  const geometry = createCurrentTrickZoneGeometry(tableDesignMockLayout, seat.id, cardCount);
  const cardSize = createCurrentTrickCardSize(tableDesignMockLayout, seat.id);

  return (
    <section
      aria-label={`${seat.label}の現在トリック置き場`}
      className={`mock-current-trick-zone mock-current-trick-zone-${seat.id}`}
      style={trickZoneStyle(geometry, cardSize)}
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
        <RiverCardFace
          card={card}
          className="mock-river-card"
          key={`${card.rank}-${card.suit}-${index}`}
          metrics={createRiverFaceMetrics(riverGeometry.visibleCardSize)}
          style={{
            "--mock-river-card-index": index,
            "--mock-river-card-rotation": `${riverPlacements[index]?.rotation ?? 0}deg`,
            "--mock-river-card-height": `${riverGeometry.visibleCardSize.height}px`,
            "--mock-river-card-width": `${riverGeometry.visibleCardSize.width}px`,
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

function ProjectedTabletop({ layout, viewportSize }: { layout: TableDesignMockLayout; viewportSize: ViewportSize }) {
  const camera = layout.camera;
  const tablePoints = projectTablePolygon(layout.tableSurface, camera);
  const roleOuterPoints = projectTablePolygon(roleBoardOuterPolygon(layout.center), camera);
  const roleInnerPoints = projectTablePolygon(roleBoardInnerPolygon(layout.center), camera);
  const fit = createProjectedBoardFit(layout, viewportSize);
  const sectorLines = createRoleBoardSectorLines(layout.center).map((line) => ({
    inner: projectTablePoint(roleBoardLocalToAbsolute(layout.center, line.inner), camera),
    outer: projectTablePoint(roleBoardLocalToAbsolute(layout.center, line.outer), camera)
  }));

  return (
    <div className="mock-projected-board-fit" style={projectedBoardFitStyle(fit)}>
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
    </div>
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
  const cardPlane = createCurrentTrickCardPlane(layout, seat.id);
  const size = { width: cardPlane.width, height: cardPlane.height };
  const corners = projectTableCard(cardPlane, layout.camera);

  return <ProjectedPlayingCard card={trickCards[seat.id]} corners={corners} size={size} variant="trick" />;
}

function ProjectedPointRiverCards({ layout, seat }: { layout: TableDesignMockLayout; seat: SeatLayout }) {
  const cards = riverCards[seat.id] ?? [];
  const river = createRiverGeometry(layout, seat.id);
  const placements = createRiverPlacements(cards.length, layout, seat.id);

  return (
    <>
      {cards.slice(0, layout.riverGrid.maxColumns * layout.riverGrid.maxRows).map((card, index) => {
        const placement = placements[index] ?? { x: 0, y: 0, rotation: 0 };
        const corners = projectTableCard(
          {
            direction: river.direction,
            height: river.visibleCardSize.height,
            normal: river.normal,
            width: river.visibleCardSize.width,
            x: toLayoutPrecision(river.x + river.direction.x * placement.x + river.normal.x * placement.y),
            y: toLayoutPrecision(river.y + river.direction.y * placement.x + river.normal.y * placement.y)
          },
          layout.camera,
          "top-left"
        );

        return (
          <ProjectedRiverCardFace
            card={card}
            corners={corners}
            key={`${card.rank}-${card.suit}-${index}`}
            size={river.visibleCardSize}
          />
        );
      })}
    </>
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

function ProjectedRiverCardFace({
  card,
  corners,
  size
}: {
  card: MockPlayingCard;
  corners: readonly Point[];
  size: { height: number; width: number };
}) {
  const transform = projectiveTransformForRectangle(corners, size.width, size.height);

  return (
    <RiverCardFace
      card={card}
      className="mock-projected-playing-card mock-projected-river-card-face"
      metrics={createRiverFaceMetrics(size)}
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

  return (
    <article
      aria-label={cardmeisterCardId(card)}
      className={`${className} mock-playing-card`}
      style={style}
    >
      <CardmeisterPlayingCard card={card} className="mock-cardmeister-playing-card" />
    </article>
  );
}

function RiverCardFace({
  card,
  className,
  metrics,
  style
}: {
  card: MockPlayingCard;
  className: string;
  metrics: RiverFaceMetrics;
  style?: CSSProperties;
}) {
  const suitColor = fourColorSuitColors[card.suit];
  const symbol = cardDesignSuitSymbols[card.suit];

  return (
    <article
      aria-label={`${card.rank}${symbol}`}
      className={`${className} mock-river-card-face`}
      style={
        {
          ...style,
          "--mock-river-face-border-width": `${metrics.borderWidth}px`,
          "--mock-river-face-color": suitColor,
          "--mock-river-face-gap": `${metrics.gap}px`,
          "--mock-river-face-padding": `${metrics.padding}px`,
          "--mock-river-face-rank-font-size": `${metrics.rankFontSize}px`,
          "--mock-river-face-suit-font-size": `${metrics.suitFontSize}px`
        } as CSSProperties
      }
    >
      <span className="mock-river-card-rank">{card.rank}</span>
      <span className="mock-river-card-suit">{symbol}</span>
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
  return createRiverGeometry(layout, "self").width;
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
  visibleCardSize: { height: number; width: number };
  width: number;
  x: number;
  y: number;
}

interface PlayerInfoGeometry extends Box {
  avatarSize: number;
  gap: number;
  label: string;
  seatId: SeatId;
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

interface SelfHandViewportLayout {
  bottom: number;
  cardSize: { height: number; width: number };
  center: Point;
  gap: number;
  handWidth: number;
  left: number;
  top: number;
}

interface RiverFaceMetrics {
  borderWidth: number;
  gap: number;
  padding: number;
  rankFontSize: number;
  suitFontSize: number;
}

interface BoundingBox extends Box {
  bottom: number;
  left: number;
  right: number;
  top: number;
}

interface ProjectedBoardFit {
  counterScale: number;
  projectedTableBox: BoundingBox;
  scale: number;
  tableHeightRatio: number;
  transformedTableBox: BoundingBox;
  translate: Point;
  viewport: ViewportSize;
}

export const projectedTextMinimumScale = 0.75;

interface BiddingBubbleLayout extends Box {
  action: BiddingMockAction;
  label: string;
  seatId: SeatId;
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

export function createSelfHandViewportMetrics(viewportWidth: number): Pick<SelfHandViewportLayout, "cardSize" | "gap"> {
  const width = (0.8 * viewportWidth) / tableDesignMockLayout.selfHandUi.maxCardCount;

  return {
    cardSize: {
      width,
      height: width * cardAspectRatio
    },
    gap: (0.16 * viewportWidth) / (tableDesignMockLayout.selfHandUi.maxCardCount - 1)
  };
}

export function selfHandWidth(cardCount: number, metrics: Pick<SelfHandViewportLayout, "cardSize" | "gap">): number {
  if (cardCount <= 0) {
    return 0;
  }

  return cardCount * metrics.cardSize.width + (cardCount - 1) * metrics.gap;
}

export function createSelfHandViewportLayout(
  layout: TableDesignMockLayout,
  cardCount: number,
  viewport: ViewportSize = layout.page
): SelfHandViewportLayout {
  const metrics = createSelfHandViewportMetrics(viewport.width);
  const handWidth = selfHandWidth(cardCount, metrics);
  const left = toLayoutPrecision((viewport.width - handWidth) / 2);
  const bottom = toLayoutPrecision(viewport.height - layout.selfHandUi.bottomInset);
  const top = toLayoutPrecision(bottom - metrics.cardSize.height);

  return {
    ...metrics,
    bottom,
    center: {
      x: toLayoutPrecision(left + handWidth / 2),
      y: toLayoutPrecision(top + metrics.cardSize.height / 2)
    },
    handWidth,
    left,
    top
  };
}

export function createProjectedTableBoundingBox(layout: TableDesignMockLayout): BoundingBox {
  return boundingBox(projectTablePolygon(layout.tableSurface, layout.camera));
}

export function createProjectedBoardFit(
  layout: TableDesignMockLayout,
  viewport: ViewportSize
): ProjectedBoardFit {
  const projectedTableBox = createProjectedTableBoundingBox(layout);
  const targetTableHeight = viewport.height * layout.projectedFit.tableHeightRatio;
  const scale = targetTableHeight / projectedTableBox.height;
  const targetTop = viewport.height * layout.projectedFit.topInsetRatio;
  const translate = {
    x: toLayoutPrecision(viewport.width / 2 - projectedTableBox.x * scale),
    y: toLayoutPrecision(targetTop - projectedTableBox.top * scale)
  };
  const transformedTableBox = transformBoundingBox(projectedTableBox, scale, translate);

  return {
    counterScale: Math.max(1, projectedTextMinimumScale / scale),
    projectedTableBox,
    scale,
    tableHeightRatio: layout.projectedFit.tableHeightRatio,
    transformedTableBox,
    translate,
    viewport
  };
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
  return createRiverGeometryWithCellHeightRatio(layout, seatId, layout.riverGrid.cellHeightRatio);
}

export function createCurrentTrickReferenceRiverGeometry(
  layout: TableDesignMockLayout,
  seatId: SeatId
): RiverGeometry {
  return createRiverGeometryWithCellHeightRatio(layout, seatId, 1);
}

function createRiverGeometryWithCellHeightRatio(
  layout: TableDesignMockLayout,
  seatId: SeatId,
  cellHeightRatio: number
): RiverGeometry {
  const edge = createTableSurfaceEdgeGeometry(layout, seatId);
  const gridWidth = toLayoutPrecision(edge.d * layout.riverGrid.widthRatio);
  const totalColumnGap = layout.riverGrid.columnGap * (layout.riverGrid.maxColumns - 1);
  const cardWidth = toLayoutPrecision((gridWidth - totalColumnGap) / layout.riverGrid.maxColumns);
  const cardHeight = toLayoutPrecision(cardWidth * cardAspectRatio);
  const visibleCardHeight = toLayoutPrecision(cardHeight * layout.riverGrid.cardExposureRatio * cellHeightRatio);
  const rowPitch = toLayoutPrecision(visibleCardHeight + layout.riverGrid.rowGap);
  const height = toLayoutPrecision(visibleCardHeight + rowPitch * (layout.riverGrid.maxRows - 1));
  const edgeMargin = toLayoutPrecision((edge.d - gridWidth) / 2);

  return {
    ...edge,
    cardSize: { width: cardWidth, height: cardHeight },
    height,
    rowPitch,
    visibleCardSize: { width: cardWidth, height: visibleCardHeight },
    width: gridWidth,
    x: toLayoutPrecision(edge.start.x + edge.direction.x * edgeMargin),
    y: toLayoutPrecision(edge.start.y + edge.direction.y * edgeMargin)
  };
}

export function createRiverFaceMetrics(
  cellSize: { height: number; width: number },
  face = tableDesignMockLayout.riverFace
): RiverFaceMetrics {
  const basis = Math.min(cellSize.width, cellSize.height);

  return {
    borderWidth: toLayoutPrecision(Math.max(1, basis * face.borderWidthRatio)),
    gap: toLayoutPrecision(cellSize.width * face.gapRatio),
    padding: toLayoutPrecision(basis * face.paddingRatio),
    rankFontSize: toLayoutPrecision(cellSize.height * face.rankFontRatio),
    suitFontSize: toLayoutPrecision(cellSize.height * face.suitFontRatio)
  };
}

export function createCurrentTrickCardSize(
  layout: TableDesignMockLayout,
  seatId: SeatId
): { height: number; width: number } {
  const roleEdge = createRoleBoardEdgeGeometry(layout.center, seatId);
  const width = toLayoutPrecision(roleEdge.d * layout.currentTrickZone.cardWidthRatio);

  return {
    width,
    height: toLayoutPrecision(width * cardAspectRatio)
  };
}

export function createCurrentTrickCardPlane(
  layout: TableDesignMockLayout,
  seatId: SeatId
): TableCardPlane {
  const zone = createCurrentTrickZoneGeometry(layout, seatId);
  const size = createCurrentTrickCardSize(layout, seatId);

  return {
    ...zone,
    ...size,
    x: zone.x,
    y: zone.y
  };
}

export function createCurrentTrickZoneGeometry(
  layout: TableDesignMockLayout,
  seatId: SeatId,
  _riverCardCount = 0
): CurrentTrickZoneGeometry {
  const edge = createTableSurfaceEdgeGeometry(layout, seatId);
  const roleEdge = createRoleBoardEdgeGeometry(layout.center, seatId);
  const river = createCurrentTrickReferenceRiverGeometry(layout, seatId);
  const cardSize = createCurrentTrickCardSize(layout, seatId);
  const roleEdgeCenter = midpointBetween(roleEdge.start, roleEdge.end);
  const riverInnerEdgeCenter = {
    x: toLayoutPrecision(river.x + river.direction.x * (river.width / 2) - river.normal.x * river.height),
    y: toLayoutPrecision(river.y + river.direction.y * (river.width / 2) - river.normal.y * river.height)
  };
  const availableDepth = distance(roleEdgeCenter, riverInnerEdgeCenter);
  const width = toLayoutPrecision(cardSize.width * layout.currentTrickZone.zoneToCardRatio);
  const height = toLayoutPrecision(cardSize.height * layout.currentTrickZone.zoneToCardRatio);
  const remainingDepth = availableDepth - height;
  const centerOffsetFromRole = toLayoutPrecision(
    height / 2 + Math.max(remainingDepth, 0) * layout.currentTrickZone.positionRatio
  );

  return {
    ...edge,
    height,
    width,
    x: toLayoutPrecision(roleEdgeCenter.x + edge.normal.x * centerOffsetFromRole),
    y: toLayoutPrecision(roleEdgeCenter.y + edge.normal.y * centerOffsetFromRole)
  };
}

export function createPlayerInfoLayouts(
  layout: TableDesignMockLayout,
  viewport: ViewportSize = layout.page,
  isProjected = false
): PlayerInfoGeometry[] {
  const effectiveLayout = createViewportPlayerInfoLayout(layout, viewport);
  const selfHandLayout = createSelfHandViewportLayout(effectiveLayout, selfCards.length, viewport);
  const opponents = opponentSeatOrder.map((seatId) =>
    isProjected
      ? createProjectedOpponentPlayerInfoLayout(effectiveLayout, seatId, viewport)
      : createWorldOpponentPlayerInfoLayout(effectiveLayout, seatId)
  );

  return [
    ...opponents,
    createSelfPlayerInfoLayout(effectiveLayout, selfHandLayout, viewport)
  ];
}

function createViewportPlayerInfoLayout(
  layout: TableDesignMockLayout,
  viewport: ViewportSize
): TableDesignMockLayout {
  if (viewport.height > 520 || viewport.width <= viewport.height) {
    return layout;
  }

  return {
    ...layout,
    playerInfo: {
      ...layout.playerInfo,
      avatarSize: 24,
      gap: 7,
      offsetFromHand: 12,
      selfGap: 16,
      unitHeight: 36,
      unitWidth: 138
    }
  };
}

export function createBiddingOverlayGeometry(
  layout: TableDesignMockLayout,
  viewport: ViewportSize = layout.page
): Box {
  const fit = createProjectedBoardFit(layout, viewport);
  const roleBoardBox = createProjectedRoleBoardBoundingBox(layout, viewport);
  const selfHand = createSelfHandViewportLayout(layout, selfCards.length, viewport);
  const config = layout.bidding.overlay;
  const isCompactLandscape = viewport.height <= 520 && viewport.width > viewport.height;
  const minimumHeight = isCompactLandscape ? 150 : 240;
  const desiredHeight = isCompactLandscape
    ? Math.min(config.height, viewport.height * 0.46)
    : config.height;
  const requestedWidth = Math.min(
    viewport.width - config.viewportMargin * 2,
    clamp(viewport.width * config.widthRatio, config.minWidth, config.maxWidth)
  );
  const availableLeftWidth = roleBoardBox.left - config.viewportMargin * 2;
  const availableRightWidth = viewport.width - roleBoardBox.right - config.viewportMargin * 2;
  const placeOnRight = availableRightWidth >= availableLeftWidth;
  const width = toLayoutPrecision(Math.min(
    requestedWidth,
    Math.max(placeOnRight ? availableRightWidth : availableLeftWidth, 0)
  ));
  const height = toLayoutPrecision(Math.min(
    desiredHeight,
    Math.max(minimumHeight, selfHand.top - config.gapFromSelfHand - config.viewportMargin * 2)
  ));
  const x = toLayoutPrecision(placeOnRight
    ? roleBoardBox.right + config.viewportMargin + width / 2
    : roleBoardBox.left - config.viewportMargin - width / 2);
  const y = toLayoutPrecision(clamp(
    fit.transformedTableBox.y + config.yOffsetFromTableCenter,
    config.viewportMargin + height / 2,
    selfHand.top - config.gapFromSelfHand - height / 2
  ));

  return {
    height,
    width,
    x,
    y
  };
}

export function createProjectedRoleBoardBoundingBox(
  layout: TableDesignMockLayout,
  viewport: ViewportSize = layout.page
): BoundingBox {
  const fit = createProjectedBoardFit(layout, viewport);
  const projectedRoleBoard = boundingBox(
    projectTablePolygon(roleBoardOuterPolygon(layout.center), layout.camera)
  );

  return transformBoundingBox(projectedRoleBoard, fit.scale, fit.translate);
}

export function createBiddingBubbleLayouts(
  layout: TableDesignMockLayout,
  viewport: ViewportSize = layout.page
): BiddingBubbleLayout[] {
  const fit = createProjectedBoardFit(layout, viewport);
  const playerInfos = createPlayerInfoLayouts(layout, viewport, true);
  const selfHand = createSelfHandViewportLayout(layout, selfCards.length, viewport);
  const selfHandBox = boundingBoxFromTopLeft({
    height: selfHand.cardSize.height,
    width: selfHand.handWidth,
    x: selfHand.left,
    y: selfHand.top
  });
  const opponentHandBoxes = opponentSeatOrder.map((seatId) =>
    createProjectedOpponentHandBoundingBox(layout, seatId, fit)
  );
  const overlayBox = boundingBoxFromCenter(createBiddingOverlayGeometry(layout, viewport));
  const roleBoardBox = createProjectedRoleBoardBoundingBox(layout, viewport);
  const hudBox = boundingBoxFromTopLeft(layout.hud);
  const staticAvoidBoxes = [
    ...playerInfos.map((info) => boundingBoxFromCenter(info)),
    hudBox,
    selfHandBox,
    ...opponentHandBoxes,
    overlayBox,
    roleBoardBox
  ];
  const placedBubbleBoxes: BoundingBox[] = [];

  return playerInfos.map((info) => {
    const action = biddingMockActions[info.seatId];
    const bubble = chooseBiddingBubbleBox(
      info,
      fit.transformedTableBox,
      [...staticAvoidBoxes, ...placedBubbleBoxes],
      viewport,
      layout.bidding.bubble
    );

    placedBubbleBoxes.push(boundingBoxFromCenter(bubble));

    return {
      ...bubble,
      action,
      label: info.label,
      seatId: info.seatId
    };
  });
}

function chooseBiddingBubbleBox(
  info: PlayerInfoGeometry,
  tableBox: BoundingBox,
  avoidBoxes: readonly BoundingBox[],
  viewport: ViewportSize,
  config: TableDesignMockLayout["bidding"]["bubble"]
): Box {
  const compactScale = viewport.height <= 520 && viewport.width > viewport.height ? 0.58 : 1;
  const size = {
    width: toLayoutPrecision(config.width * compactScale),
    height: toLayoutPrecision(config.height * compactScale)
  };
  const outward = normalizeVector({ x: info.x - tableBox.x, y: info.y - tableBox.y });
  const tangent = { x: -outward.y, y: outward.x };
  const compactSelfDirections = [
    { x: 0, y: -1 },
    { x: -1, y: 0 },
    { x: 1, y: 0 }
  ];
  const selfDirections = compactScale < 1
    ? compactSelfDirections
    : [
        { x: 0, y: -1 },
        { x: -1, y: 0 },
        { x: 1, y: 0 },
        outward,
        { x: -outward.x, y: -outward.y },
        tangent,
        { x: -tangent.x, y: -tangent.y }
      ];
  const opponentDirections = [
    outward,
    tangent,
    { x: -tangent.x, y: -tangent.y },
    { x: -outward.x, y: -outward.y },
    { x: 0, y: -1 },
    { x: 1, y: 0 },
    { x: 0, y: 1 },
    { x: -1, y: 0 }
  ];
  const directions = info.seatId === "self" ? selfDirections : opponentDirections;
  const candidates = uniquePoints(
    directions.map((direction) => {
      const offset =
        rectHalfExtentAlong(info, direction) +
        rectHalfExtentAlong(size, direction) +
        config.gap;

      return clampBiddingBubbleCenter({
        x: toLayoutPrecision(info.x + direction.x * offset),
        y: toLayoutPrecision(info.y + direction.y * offset)
      }, size, viewport, config.viewportMargin);
    })
  ).map((center) => ({ ...center, ...size }));
  const nonOverlapping = candidates.filter((candidate) =>
    avoidBoxes.every((avoidBox) => !boxesOverlap(boundingBoxFromCenter(candidate), avoidBox))
  );

  if (nonOverlapping.length > 0) {
    return nonOverlapping[0] ?? candidates[0];
  }

  return [...candidates].sort((a, b) => {
    const overlapA = totalOverlapArea(boundingBoxFromCenter(a), avoidBoxes);
    const overlapB = totalOverlapArea(boundingBoxFromCenter(b), avoidBoxes);

    if (overlapA !== overlapB) {
      return overlapA - overlapB;
    }

    return distance(a, info) - distance(b, info);
  })[0] ?? {
    ...size,
    x: info.x,
    y: info.y
  };
}

function clampBiddingBubbleCenter(
  center: Point,
  size: Pick<Box, "height" | "width">,
  viewport: ViewportSize,
  margin: number
): Point {
  return {
    x: toLayoutPrecision(clamp(center.x, margin + size.width / 2, viewport.width - margin - size.width / 2)),
    y: toLayoutPrecision(clamp(center.y, margin + size.height / 2, viewport.height - margin - size.height / 2))
  };
}

function createProjectedOpponentHandBoundingBox(
  layout: TableDesignMockLayout,
  seatId: OpponentSeatId,
  fit: ProjectedBoardFit
): BoundingBox {
  const hand = createOpponentHandGeometry(layout, seatId);

  return transformBoundingBox(
    boundingBox(hand.cards.flatMap((card) => projectVerticalCard(card, layout.camera))),
    fit.scale,
    fit.translate
  );
}

function totalOverlapArea(box: BoundingBox, avoidBoxes: readonly BoundingBox[]): number {
  return avoidBoxes.reduce((total, avoidBox) => total + overlapArea(box, avoidBox), 0);
}

function createWorldOpponentPlayerInfoLayout(
  layout: TableDesignMockLayout,
  seatId: OpponentSeatId
): PlayerInfoGeometry {
  const hand = createOpponentHandGeometry(layout, seatId);
  const handBox = boundingBox(
    hand.cards.flatMap((card) =>
      verticalCardTopDownThicknessPolygon(card, hand.edge.normal, layout.opponentHand.cardThickness)
    )
  );
  const center = playerInfoCenterOutsideBox(handBox, hand.baseline.normal, layout.playerInfo);
  const clampedCenter = clampPlayerInfoCenter(center, layout.page, layout.playerInfo);
  const hudBox = boundingBoxFromTopLeft(layout.hud);
  const riverBox = createWorldRiverCardsBoundingBox(layout, seatId);

  return createPlayerInfoGeometry(
    layout,
    seatId,
    avoidPlayerInfoOverlaps(
      clampedCenter,
      [handBox, hudBox, ...optionalBox(riverBox)],
      handBox,
      hand.baseline.normal,
      layout.page,
      layout.playerInfo
    )
  );
}

function createProjectedOpponentPlayerInfoLayout(
  layout: TableDesignMockLayout,
  seatId: OpponentSeatId,
  viewport: ViewportSize
): PlayerInfoGeometry {
  const context = createProjectedOpponentPlayerInfoContext(layout, seatId, viewport);
  const preferredCenter = createProjectedOpponentPlayerInfoBalancedCenter(layout, seatId, viewport, context);
  const shouldPreferCenter =
    seatId === "top-right" && distance(preferredCenter, context.clampedCenter) > 0.5;
  const hudBox = boundingBoxFromTopLeft(layout.hud);
  const riverBox = createProjectedRiverCardsBoundingBox(layout, seatId, context.fit);
  const selfHand = createSelfHandViewportLayout(layout, selfCards.length, viewport);
  const selfHandBox = boundingBoxFromTopLeft({
    height: selfHand.cardSize.height,
    width: selfHand.handWidth,
    x: selfHand.left,
    y: selfHand.top
  });

  return createPlayerInfoGeometry(
    layout,
    seatId,
    avoidPlayerInfoOverlaps(
      preferredCenter,
      [context.handBox, hudBox, selfHandBox, ...optionalBox(riverBox)],
      context.handBox,
      context.outward,
      viewport,
      layout.playerInfo,
      shouldPreferCenter ? preferredCenter : undefined
    )
  );
}

function createProjectedOpponentPlayerInfoContext(
  layout: TableDesignMockLayout,
  seatId: OpponentSeatId,
  viewport: ViewportSize
): {
  clampedCenter: Point;
  fit: ProjectedBoardFit;
  handBox: BoundingBox;
  outward: Point;
} {
  const hand = createOpponentHandGeometry(layout, seatId);
  const fit = createProjectedBoardFit(layout, viewport);
  const handBox = transformBoundingBox(
    boundingBox(hand.cards.flatMap((card) => projectVerticalCard(card, layout.camera))),
    fit.scale,
    fit.translate
  );
  const outward = normalizeVector({
    x: handBox.x - fit.transformedTableBox.x,
    y: handBox.y - fit.transformedTableBox.y
  });
  const center = playerInfoCenterOutsideBox(handBox, outward, layout.playerInfo);

  return {
    clampedCenter: clampPlayerInfoCenter(center, viewport, layout.playerInfo),
    fit,
    handBox,
    outward
  };
}

function createProjectedOpponentPlayerInfoBalancedCenter(
  layout: TableDesignMockLayout,
  seatId: OpponentSeatId,
  viewport: ViewportSize,
  context: ReturnType<typeof createProjectedOpponentPlayerInfoContext>
): Point {
  if (seatId !== "top-right") {
    return context.clampedCenter;
  }

  const topLeft = createProjectedOpponentPlayerInfoLayout(layout, "top-left", viewport);
  const tableCenterX = context.fit.transformedTableBox.x;

  if (topLeft.x >= tableCenterX) {
    return context.clampedCenter;
  }

  const mirroredTopLeftX = tableCenterX + Math.abs(tableCenterX - topLeft.x);

  return clampPlayerInfoCenter(
    {
      x: mirroredTopLeftX,
      y: context.clampedCenter.y
    },
    viewport,
    layout.playerInfo
  );
}

function createSelfPlayerInfoLayout(
  layout: TableDesignMockLayout,
  selfHandLayout: SelfHandViewportLayout,
  viewport: ViewportSize
): PlayerInfoGeometry {
  const info = layout.playerInfo;
  const x = clamp(
    selfHandLayout.left,
    info.viewportMargin,
    viewport.width - info.viewportMargin - info.unitWidth
  );
  const y = clamp(
    selfHandLayout.top - info.unitHeight - info.selfGap,
    info.viewportMargin,
    selfHandLayout.top - info.unitHeight - info.selfGap
  );

  return createPlayerInfoGeometry(layout, "self", {
    x: toLayoutPrecision(x + info.unitWidth / 2),
    y: toLayoutPrecision(y + info.unitHeight / 2)
  });
}

function playerInfoCenterOutsideBox(
  box: Pick<Box, "height" | "width" | "x" | "y">,
  outward: Point,
  info: TableDesignMockLayout["playerInfo"]
): Point {
  const offset =
    rectHalfExtentAlong(box, outward) +
    rectHalfExtentAlong({ width: info.unitWidth, height: info.unitHeight }, outward) +
    info.offsetFromHand;

  return {
    x: toLayoutPrecision(box.x + outward.x * offset),
    y: toLayoutPrecision(box.y + outward.y * offset)
  };
}

function createPlayerInfoGeometry(
  layout: TableDesignMockLayout,
  seatId: SeatId,
  center: Point
): PlayerInfoGeometry {
  const seat = layout.seats.find((entry) => entry.id === seatId);

  if (seat === undefined) {
    throw new Error(`Missing seat layout for ${seatId}`);
  }

  return {
    avatarSize: layout.playerInfo.avatarSize,
    gap: layout.playerInfo.gap,
    height: layout.playerInfo.unitHeight,
    label: seat.label,
    seatId,
    width: layout.playerInfo.unitWidth,
    x: center.x,
    y: center.y
  };
}

function createWorldRiverCardsBoundingBox(layout: TableDesignMockLayout, seatId: SeatId): BoundingBox | undefined {
  const cardBoxes = createRiverCardPlanes(layout, seatId).map((card) => boundingBoxFromTableCard(card));

  return cardBoxes.length === 0 ? undefined : boundingBoxAroundBoxes(cardBoxes);
}

function createProjectedRiverCardsBoundingBox(
  layout: TableDesignMockLayout,
  seatId: SeatId,
  fit: ProjectedBoardFit
): BoundingBox | undefined {
  const cardBoxes = createRiverCardPlanes(layout, seatId).map((card) =>
    transformBoundingBox(boundingBox(projectTableCard(card, layout.camera, "top-left")), fit.scale, fit.translate)
  );

  return cardBoxes.length === 0 ? undefined : boundingBoxAroundBoxes(cardBoxes);
}

function createRiverCardPlanes(layout: TableDesignMockLayout, seatId: SeatId): TableCardPlane[] {
  const cards = riverCards[seatId] ?? [];
  const river = createRiverGeometry(layout, seatId);
  const placements = createRiverPlacements(cards.length, layout, seatId);

  return placements.map((placement) => ({
    direction: river.direction,
    height: river.visibleCardSize.height,
    normal: river.normal,
    width: river.visibleCardSize.width,
    x: toLayoutPrecision(river.x + river.direction.x * placement.x + river.normal.x * placement.y),
    y: toLayoutPrecision(river.y + river.direction.y * placement.x + river.normal.y * placement.y)
  }));
}

function optionalBox(box: BoundingBox | undefined): BoundingBox[] {
  return box === undefined ? [] : [box];
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

export function createRoleBoardSectorLines(
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

export function projectiveTransformForRectangle(corners: readonly Point[], width: number, height: number): string {
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
  const columnOffset = toLayoutPrecision(geometry.visibleCardSize.width + layout.riverGrid.columnGap);

  for (let index = 0; index < boundedCardCount; index += 1) {
    const column = index % layout.riverGrid.maxColumns;
    const row = Math.floor(index / layout.riverGrid.maxColumns);

    placements.push({
      x: toLayoutPrecision(column * columnOffset),
      y: toLayoutPrecision(-geometry.visibleCardSize.height - row * geometry.rowPitch),
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

function playerInfoStyle(info: PlayerInfoGeometry): CSSProperties {
  return {
    "--mock-player-avatar-size": `${info.avatarSize}px`,
    "--mock-player-gap": `${info.gap}px`,
    "--mock-player-height": `${info.height}px`,
    "--mock-player-width": `${info.width}px`,
    "--mock-x": `${info.x}px`,
    "--mock-y": `${info.y}px`
  } as CSSProperties;
}

function biddingOverlayStyle(geometry: Box): CSSProperties {
  return {
    "--mock-bidding-overlay-height": `${geometry.height}px`,
    "--mock-bidding-overlay-width": `${geometry.width}px`,
    "--mock-x": `${geometry.x}px`,
    "--mock-y": `${geometry.y}px`
  } as CSSProperties;
}

function biddingBubbleStyle(bubble: BiddingBubbleLayout): CSSProperties {
  return {
    ...biddingActionColorStyle(bubble.action),
    "--mock-bidding-bubble-height": `${bubble.height}px`,
    "--mock-bidding-bubble-width": `${bubble.width}px`,
    "--mock-x": `${bubble.x}px`,
    "--mock-y": `${bubble.y}px`
  } as CSSProperties;
}

function biddingActionColorStyle(action: BiddingMockAction): CSSProperties {
  return {
    "--mock-bidding-action-color": action.type === "bid" ? fourColorSuitColors[action.suit] : "#64748b"
  } as CSSProperties;
}

function biddingActionLabel(action: BiddingMockAction): string {
  if (action.type === "pass") {
    return "PASS";
  }

  return `${cardDesignSuitSymbols[action.suit]}${action.value}`;
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

function trickZoneStyle(
  zone: CurrentTrickZoneGeometry,
  cardSize: { height: number; width: number }
): CSSProperties {
  return {
    ...pointWithRotationStyle(zone),
    "--mock-trick-card-height": `${cardSize.height}px`,
    "--mock-trick-card-width": `${cardSize.width}px`,
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

function selfHandViewportStyle(layout: SelfHandViewportLayout): CSSProperties {
  return {
    "--mock-self-card-gap": `${layout.gap}px`,
    "--mock-self-card-height": `${layout.cardSize.height}px`,
    "--mock-self-card-width": `${layout.cardSize.width}px`,
    "--mock-self-hand-left": `${layout.left}px`,
    "--mock-self-hand-top": `${layout.top}px`,
    "--mock-self-hand-width": `${layout.handWidth}px`
  } as CSSProperties;
}

function projectedBoardFitStyle(fit: ProjectedBoardFit): CSSProperties {
  return {
    "--mock-projected-board-counter-scale": fit.counterScale,
    "--mock-projected-board-transform": `translate(${fit.translate.x}px, ${fit.translate.y}px) scale(${fit.scale})`
  } as CSSProperties;
}

function pointStyle(point: Point): CSSProperties {
  return {
    "--mock-x": `${point.x}px`,
    "--mock-y": `${point.y}px`
  } as CSSProperties;
}

function useViewportSize(fallback: ViewportSize): ViewportSize {
  const [viewportSize, setViewportSize] = useState<ViewportSize>(fallback);

  useEffect(() => {
    const readViewportSize = () => ({
      width: window.innerWidth,
      height: window.innerHeight
    });

    const updateViewportSize = () => {
      setViewportSize(readViewportSize());
    };

    updateViewportSize();
    window.addEventListener("resize", updateViewportSize);

    return () => {
      window.removeEventListener("resize", updateViewportSize);
    };
  }, []);

  return viewportSize;
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

export function roleBoardLocalToAbsolute(layout: Box, point: Point): Point {
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

export function roleBoardOuterPolygon(layout: Box): Point[] {
  return roleBoardVertexOrder.map((vertexId) => roleBoardAbsolutePoint(layout, roleBoardPentagon[vertexId]));
}

export function roleBoardInnerPolygon(
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

function boundingBox(points: readonly Point[]): BoundingBox {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);
  const width = right - left;
  const height = bottom - top;

  return {
    bottom: toLayoutPrecision(bottom),
    height: toLayoutPrecision(height),
    left: toLayoutPrecision(left),
    right: toLayoutPrecision(right),
    top: toLayoutPrecision(top),
    width: toLayoutPrecision(width),
    x: toLayoutPrecision(left + width / 2),
    y: toLayoutPrecision(top + height / 2)
  };
}

function boundingBoxFromTableCard(card: TableCardPlane): BoundingBox {
  const topLeft = {
    x: card.x,
    y: card.y
  };

  return boundingBox([
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
  ]);
}

function transformBoundingBox(box: BoundingBox, scale: number, translate: Point): BoundingBox {
  const left = box.left * scale + translate.x;
  const right = box.right * scale + translate.x;
  const top = box.top * scale + translate.y;
  const bottom = box.bottom * scale + translate.y;
  const width = right - left;
  const height = bottom - top;

  return {
    bottom: toLayoutPrecision(bottom),
    height: toLayoutPrecision(height),
    left: toLayoutPrecision(left),
    right: toLayoutPrecision(right),
    top: toLayoutPrecision(top),
    width: toLayoutPrecision(width),
    x: toLayoutPrecision(left + width / 2),
    y: toLayoutPrecision(top + height / 2)
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

function rectHalfExtentAlong(size: Pick<Box, "height" | "width">, direction: Point): number {
  return Math.abs(direction.x) * size.width / 2 + Math.abs(direction.y) * size.height / 2;
}

function clampPlayerInfoCenter(
  center: Point,
  viewport: ViewportSize,
  info: TableDesignMockLayout["playerInfo"]
): Point {
  return {
    x: toLayoutPrecision(clamp(
      center.x,
      info.viewportMargin + info.unitWidth / 2,
      viewport.width - info.viewportMargin - info.unitWidth / 2
    )),
    y: toLayoutPrecision(clamp(
      center.y,
      info.viewportMargin + info.unitHeight / 2,
      viewport.height - info.viewportMargin - info.unitHeight / 2
    ))
  };
}

function avoidPlayerInfoOverlaps(
  center: Point,
  avoidBoxes: readonly BoundingBox[],
  anchorBox: BoundingBox,
  outward: Point,
  viewport: ViewportSize,
  info: TableDesignMockLayout["playerInfo"],
  preferredCenter?: Point
): Point {
  const boxForCenter = (candidate: Point): BoundingBox => boundingBoxFromCenter({
    ...candidate,
    width: info.unitWidth,
    height: info.unitHeight
  });
  const overlapsAnyBox = (candidate: Point) =>
    avoidBoxes.some((avoidBox) => boxesOverlap(boxForCenter(candidate), avoidBox));

  if (!overlapsAnyBox(center)) {
    return center;
  }

  const gap = info.offsetFromHand;
  const candidateBoxes = [...avoidBoxes, boundingBoxAroundBoxes(avoidBoxes)];
  const preferredOffsets =
    preferredCenter === undefined
      ? []
      : [gap, info.unitHeight / 2 + gap, info.unitHeight + gap, info.unitWidth / 2 + gap];
  const preferredCandidates = preferredOffsets.flatMap((offset) =>
    preferredCenter === undefined
      ? []
      : [
          { x: preferredCenter.x, y: preferredCenter.y - offset },
          { x: preferredCenter.x, y: preferredCenter.y + offset },
          { x: preferredCenter.x - offset, y: preferredCenter.y },
          { x: preferredCenter.x + offset, y: preferredCenter.y },
          {
            x: preferredCenter.x + outward.x * offset,
            y: preferredCenter.y + outward.y * offset
          }
        ]
  );
  const candidates = candidateBoxes
    .flatMap((avoidBox) => [
      { x: center.x, y: avoidBox.top - info.unitHeight / 2 - gap },
      { x: center.x, y: avoidBox.bottom + info.unitHeight / 2 + gap },
      { x: avoidBox.left - info.unitWidth / 2 - gap, y: center.y },
      { x: avoidBox.right + info.unitWidth / 2 + gap, y: center.y }
    ])
    .concat(preferredCandidates)
    .map((candidate) => clampPlayerInfoCenter(candidate, viewport, info));
  const nonOverlapping = uniquePoints(candidates).filter((candidate) => !overlapsAnyBox(candidate));

  if (nonOverlapping.length === 0) {
    return center;
  }

  const outwardValue = (candidate: Point) =>
    (candidate.x - anchorBox.x) * outward.x + (candidate.y - anchorBox.y) * outward.y;
  const outwardCandidates = nonOverlapping.filter((candidate) => outwardValue(candidate) > 0);
  const sortableCandidates = outwardCandidates.length > 0 ? outwardCandidates : nonOverlapping;

  return [...sortableCandidates].sort((a, b) => {
    if (preferredCenter !== undefined) {
      const aPreferredDistance = distance(a, preferredCenter);
      const bPreferredDistance = distance(b, preferredCenter);

      if (aPreferredDistance !== bPreferredDistance) {
        return aPreferredDistance - bPreferredDistance;
      }
    }

    const aOutward = (a.x - anchorBox.x) * outward.x + (a.y - anchorBox.y) * outward.y;
    const bOutward = (b.x - anchorBox.x) * outward.x + (b.y - anchorBox.y) * outward.y;

    if (aOutward !== bOutward) {
      return bOutward - aOutward;
    }

    return distance(a, center) - distance(b, center);
  })[0] ?? center;
}

function boundingBoxAroundBoxes(boxes: readonly BoundingBox[]): BoundingBox {
  const left = Math.min(...boxes.map((box) => box.left));
  const right = Math.max(...boxes.map((box) => box.right));
  const top = Math.min(...boxes.map((box) => box.top));
  const bottom = Math.max(...boxes.map((box) => box.bottom));

  return boundingBoxFromEdges(left, right, top, bottom);
}

function boundingBoxFromEdges(left: number, right: number, top: number, bottom: number): BoundingBox {
  const width = right - left;
  const height = bottom - top;

  return {
    bottom: toLayoutPrecision(bottom),
    height: toLayoutPrecision(height),
    left: toLayoutPrecision(left),
    right: toLayoutPrecision(right),
    top: toLayoutPrecision(top),
    width: toLayoutPrecision(width),
    x: toLayoutPrecision(left + width / 2),
    y: toLayoutPrecision(top + height / 2)
  };
}

function uniquePoints(points: readonly Point[]): Point[] {
  const seen = new Set<string>();
  const unique: Point[] = [];

  for (const point of points) {
    const key = `${point.x}:${point.y}`;

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    unique.push(point);
  }

  return unique;
}

function boundingBoxFromCenter(box: Box): BoundingBox {
  return {
    bottom: toLayoutPrecision(box.y + box.height / 2),
    height: box.height,
    left: toLayoutPrecision(box.x - box.width / 2),
    right: toLayoutPrecision(box.x + box.width / 2),
    top: toLayoutPrecision(box.y - box.height / 2),
    width: box.width,
    x: box.x,
    y: box.y
  };
}

function boundingBoxFromTopLeft(box: Box): BoundingBox {
  return {
    bottom: toLayoutPrecision(box.y + box.height),
    height: box.height,
    left: box.x,
    right: toLayoutPrecision(box.x + box.width),
    top: box.y,
    width: box.width,
    x: toLayoutPrecision(box.x + box.width / 2),
    y: toLayoutPrecision(box.y + box.height / 2)
  };
}

function boxesOverlap(a: BoundingBox, b: BoundingBox): boolean {
  return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
}

function overlapArea(a: BoundingBox, b: BoundingBox): number {
  const width = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
  const height = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));

  return width * height;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function normalizeVector(vector: Point): Point {
  const length = distance({ x: 0, y: 0 }, vector);

  if (length === 0) {
    return { x: 0, y: -1 };
  }

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
