import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
  type RefObject
} from "react";
import type {
  PublicBidAction,
  PublicBiddingState,
  PublicCard,
  PublicGameState,
  PublicMatchState,
  PublicPlayedCard,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { determineCurrentWinningPlayer } from "@napoleon/game-core";
import { CardmeisterPlayingCard, useCardmeisterScript } from "./CardmeisterPlayingCard";
import { cardDesignSuitSymbols } from "./CardDesignCard";
import {
  createBiddingBubbleLayouts,
  createBiddingOverlayGeometry,
  createCompactBiddingContentMetrics,
  createCurrentTrickCardPlaneForViewport,
  createCurrentTrickZoneGeometry,
  createOpponentHandGeometry,
  createPlayerInfoLayouts,
  createProjectedBoardFit,
  createProjectedRoleTextCenters,
  createRiverFaceMetrics,
  createRiverGeometry,
  createRiverPlacements,
  createRoleBoardSectorLines,
  createRoleMarkerGeometry,
  createSelfHandViewportLayout,
  polygonCenter,
  projectTableCard,
  projectTablePoint,
  projectTablePolygon,
  projectVerticalCard,
  projectiveTransformForRectangle,
  roleBoardInnerPolygon,
  roleBoardLocalToAbsolute,
  roleBoardOuterPolygon,
  selfHandCardIndexStyle,
  selfHandViewportStyle,
  tableDesignMockLayout
} from "./TableDesignMock";
import { fourColorSuitColors } from "./cardSuitTheme";
import {
  findBidAction,
  getBidStepperState,
  getBidTargetsForSuit,
  normalizeBidSelection,
  type BidSelection
} from "./biddingOptions";
import { mockCardBackComponent, mockCardBackComponentName, type MockPlayingCard } from "./mockPlayingCardAdapter";
import { getDisplayedHandCards, type HandOrderMode } from "./handSorting";
import type { Seat, TablePlayer } from "./tableTypes";

interface TableSurfaceProps {
  actionPanel: ReactNode;
  canExchange: boolean;
  canPass?: boolean;
  collectingWinnerId?: string;
  currentTrick: readonly PublicPlayedCard[];
  highlightWinningCard: boolean;
  isBusy: boolean;
  isResultEmphasisActive?: boolean;
  legalBidActions?: readonly PublicBidAction[];
  legalCardIds: ReadonlySet<string>;
  match?: PublicMatchState;
  onBid?: (action: PublicBidAction) => void;
  onPass?: () => void;
  onPlay: (card: PublicCard) => void;
  onToggleWinningCardHighlight: () => void;
  players: readonly TablePlayer[];
  selectedDiscardCardIds: readonly string[];
  selfPlayerId?: string;
  state: PublicGameState | undefined;
  trickNumber: number | undefined;
  trumpSuit: PublicSuit | null | undefined;
}

type TableSeatId = Seat;

interface ViewportSize {
  height: number;
  width: number;
}

interface TablePlayerAdapter {
  biddingDeclaration: TablePlayer["biddingDeclaration"];
  capturedPointCards: readonly PublicStandardCard[];
  handCount: number;
  id: string;
  isSelf: boolean;
  label: string;
  seat: TableSeatId;
  selfHand: readonly PublicCard[];
}

const seatOrder = ["top-left", "top-right", "right", "left", "self"] as const satisfies readonly TableSeatId[];
const opponentSeatOrder = ["top-left", "top-right", "right", "left"] as const satisfies readonly Exclude<TableSeatId, "self">[];
const biddingSuitOptions = ["spades", "hearts", "diamonds", "clubs"] as const satisfies readonly PublicSuit[];

export function TableSurface({
  actionPanel,
  canExchange,
  canPass = false,
  collectingWinnerId,
  currentTrick,
  highlightWinningCard,
  isBusy,
  isResultEmphasisActive = false,
  legalBidActions = [],
  legalCardIds,
  match,
  onBid,
  onPass,
  onPlay,
  onToggleWinningCardHighlight,
  players,
  selectedDiscardCardIds,
  selfPlayerId,
  state,
  trickNumber,
  trumpSuit
}: TableSurfaceProps) {
  const [handOrderMode, setHandOrderMode] = useState<HandOrderMode>("riipai");
  const surfaceRef = useRef<HTMLDivElement>(null);
  const viewportSize = useViewportSize(tableDesignMockLayout.page, surfaceRef);
  const adapters = useMemo(() => createProductionTablePlayers(players, state, handOrderMode), [handOrderMode, players, state]);
  const playedCardsByPlayerId = useMemo(
    () => new Map(currentTrick.map((played) => [played.playerId, played] as const)),
    [currentTrick]
  );
  const winningPlayerId =
    highlightWinningCard && trumpSuit !== null && trumpSuit !== undefined
      ? getCurrentWinningPlayerId(currentTrick, trumpSuit, trickNumber)
      : undefined;
  const collectingSeat = adapters.find((player) => player.id === collectingWinnerId)?.seat;
  const className = [
    "production-table-surface",
    state?.phase === "bidding" ? "production-table-surface-bidding" : "production-table-surface-playing",
    isResultEmphasisActive ? "production-table-surface-result" : "",
    collectingSeat === undefined ? "" : `production-table-surface-collecting production-table-surface-collecting-to-${collectingSeat}`
  ]
    .filter(Boolean)
    .join(" ");

  useCardmeisterScript();

  return (
    <div
      className={className}
      aria-label="ゲームテーブル"
      ref={surfaceRef}
      style={tableSurfaceStyle(tableDesignMockLayout)}
    >
      <ProjectedProductionBoard
        adapters={adapters}
        currentTrickByPlayerId={playedCardsByPlayerId}
        isCollecting={collectingSeat !== undefined}
        match={match}
        viewportSize={viewportSize}
        winningPlayerId={winningPlayerId}
        state={state}
      />
      <PlayerInfoLayer
        adapters={adapters}
        currentPlayerId={state?.currentPlayerId}
        viewportSize={viewportSize}
      />
      {state?.phase !== "bidding" ? <ProductionContractHud state={state} /> : null}
      {state?.phase === "bidding" ? (
        <>
          <ProductionBiddingOverlay
            bidding={state.bidding}
            canPass={canPass}
            currentPlayerId={state.currentPlayerId}
            isBusy={isBusy}
            legalBidActions={legalBidActions}
            onBid={onBid}
            onPass={onPass}
            selfPlayerId={selfPlayerId}
            viewportSize={viewportSize}
          />
          <BiddingBubbleLayer adapters={adapters} viewportSize={viewportSize} />
        </>
      ) : null}
      <SelfHandLayer
        canExchange={canExchange}
        handOrderMode={handOrderMode}
        isBusy={isBusy}
        legalCardIds={legalCardIds}
        onPlay={onPlay}
        selectedDiscardCardIds={selectedDiscardCardIds}
        self={adapters.find((player) => player.isSelf)}
        state={state}
        viewportSize={viewportSize}
      />
      <button
        aria-label={handOrderMode === "riipai" ? "理牌オン" : "理牌オフ"}
        aria-pressed={handOrderMode === "riipai"}
        className={[
          "production-riipai-sidebar",
          handOrderMode === "riipai" ? "production-riipai-sidebar-active" : ""
        ].filter(Boolean).join(" ")}
        onClick={() => setHandOrderMode((current) => (current === "riipai" ? "original" : "riipai"))}
        type="button"
      >
        <span>理</span>
        <strong>{handOrderMode === "riipai" ? "ON" : "OFF"}</strong>
      </button>
      <div className="production-table-tools" aria-label="補助操作">
        <ProductionRoundScoreHistory adapters={adapters} match={match} />
        <button
          aria-label={highlightWinningCard ? "暫定勝ち札強調オン" : "暫定勝ち札強調オフ"}
          aria-pressed={highlightWinningCard}
          className={highlightWinningCard ? "production-tool-button production-tool-button-active" : "production-tool-button"}
          onClick={onToggleWinningCardHighlight}
          type="button"
        >
          勝札
        </button>
      </div>
      {actionPanel === null || state?.phase === "bidding" ? null : (
        <aside className="production-action-overlay" aria-label="操作">
          {actionPanel}
        </aside>
      )}
    </div>
  );
}

function ProductionContractHud({ state }: { state: PublicGameState | undefined }) {
  if (state?.contract === null || state?.contract === undefined) {
    return null;
  }

  const calledCard = state.adjutant?.calledCardId;

  return (
    <aside aria-label="契約と呼札" className="production-contract-hud">
      <span className="production-contract-chip">
        <span>宣言</span>
        <strong style={biddingActionColorStyle(state.contract.trumpSuit)}>
          {cardDesignSuitSymbols[state.contract.trumpSuit]}
          {state.contract.targetPointCards}
        </strong>
      </span>
      <span className="production-contract-chip">
        <span>副官</span>
        <strong style={calledCardColorStyle(calledCard)}>{formatCalledCardId(calledCard)}</strong>
      </span>
    </aside>
  );
}

function ProjectedProductionBoard({
  adapters,
  currentTrickByPlayerId,
  isCollecting,
  match,
  state,
  viewportSize,
  winningPlayerId
}: {
  adapters: readonly TablePlayerAdapter[];
  currentTrickByPlayerId: ReadonlyMap<string, PublicPlayedCard>;
  isCollecting: boolean;
  match: PublicMatchState | undefined;
  state: PublicGameState | undefined;
  viewportSize: ViewportSize;
  winningPlayerId: string | undefined;
}) {
  const layout = tableDesignMockLayout;
  const selfHandCardCount = adapters.find((adapter) => adapter.isSelf)?.selfHand.length
    ?? 10;
  const fit = createProjectedBoardFit(layout, viewportSize, selfHandCardCount);
  const tablePoints = projectTablePolygon(layout.tableSurface, layout.camera);
  const roleOuterPoints = projectTablePolygon(roleBoardOuterPolygon(layout.center), layout.camera);
  const roleInnerPoints = projectTablePolygon(roleBoardInnerPolygon(layout.center), layout.camera);
  const sectorLines = createRoleBoardSectorLines(layout.center).map((line) => ({
    inner: projectTablePoint(roleBoardLocalToAbsolute(layout.center, line.inner), layout.camera),
    outer: projectTablePoint(roleBoardLocalToAbsolute(layout.center, line.outer), layout.camera)
  }));
  const roleTextCenters = useMemo(() => createProjectedRoleTextCenters(layout, viewportSize, {
    isBidding: state?.phase === "bidding",
    opponentHandCounts: Object.fromEntries(adapters
      .filter((adapter) => adapter.seat !== "self")
      .map((adapter) => [adapter.seat, adapter.handCount])),
    riverCardCounts: Object.fromEntries(adapters
      .map((adapter) => [adapter.seat, adapter.capturedPointCards.length])),
    roleTextLabels: Object.fromEntries(seatOrder.map((seat) => [
      seat,
      createProductionRoleText(adapters, match, state, seat).compact
    ])),
    selfHandCardCount: adapters.find((adapter) => adapter.isSelf)?.selfHand.length ?? 0
  }), [adapters, match, state, viewportSize]);

  return (
    <div className="mock-projected-board-fit" style={projectedBoardFitStyle(fit)}>
      <svg
        aria-label="投影後の実ゲーム卓"
        className="mock-projected-tabletop production-projected-tabletop"
        viewBox={`0 0 ${layout.page.width} ${layout.page.height}`}
      >
        <polygon className="mock-table-surface-polygon" points={svgPoints(tablePoints)} />
        {seatOrder.map((seat) => {
          const zone = createCurrentTrickZoneGeometry(layout, seat);
          return (
            <polygon
              className="mock-projected-current-trick-zone-fill"
              key={`production-trick-zone-${seat}`}
              points={svgPoints(projectTableCard(zone, layout.camera))}
            />
          );
        })}
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
          {seatOrder.map((seat) => (
            <ProductionRoleMarker
              adapters={adapters}
              compact={viewportSize.height <= 500 && viewportSize.width > viewportSize.height}
              key={`production-role-${seat}`}
              labelCenter={roleTextCenters[seat]}
              match={match}
              seat={seat}
              state={state}
            />
          ))}
          <ProductionMatchRound match={match} />
        </g>
      </svg>
      <div className="mock-projected-card-layer">
        {seatOrder.map((seat) => {
          const player = adapters.find((entry) => entry.seat === seat);
          const played = player === undefined ? undefined : currentTrickByPlayerId.get(player.id);

          return (
            <ProjectedCurrentTrickCard
              isCollecting={isCollecting}
              isWinning={played !== undefined && played.playerId === winningPlayerId}
              key={`production-trick-card-${seat}`}
              played={played}
              seat={seat}
              viewportSize={viewportSize}
            />
          );
        })}
        {seatOrder.map((seat) => {
          const player = adapters.find((entry) => entry.seat === seat);

          return (
            <ProjectedPointRiverCards
              cards={player?.capturedPointCards ?? []}
              key={`production-river-${seat}`}
              seat={seat}
            />
          );
        })}
        {opponentSeatOrder.map((seat) => {
          const player = adapters.find((entry) => entry.seat === seat);

          return (
            <ProjectedOpponentHand
              count={player?.handCount ?? 0}
              key={`production-opponent-hand-${seat}`}
              label={player?.label ?? seat}
              seat={seat}
            />
          );
        })}
      </div>
    </div>
  );
}

function projectRoleMarkerBox(
  layout: typeof tableDesignMockLayout,
  marker: ReturnType<typeof createRoleMarkerGeometry>
): { center: { x: number; y: number }; corners: readonly { x: number; y: number }[]; rotate: string } {
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
  const boxCenter = polygonCenter(corners);

  return {
    center: boxCenter,
    corners,
    rotate: `rotate(${marker.rotation} ${boxCenter.x} ${boxCenter.y})`
  };
}

function ProductionRoleMarker({
  adapters,
  compact,
  labelCenter,
  match,
  seat,
  state
}: {
  adapters: readonly TablePlayerAdapter[];
  compact: boolean;
  labelCenter: { x: number; y: number };
  match: PublicMatchState | undefined;
  seat: TableSeatId;
  state: PublicGameState | undefined;
}) {
  const layout = tableDesignMockLayout;
  const roleText = createProductionRoleText(adapters, match, state, seat);
  const ariaLabel = `${roleText.player?.label ?? seat}: 役職 ${roleText.role}, 累積試合スコア ${roleText.score}`;
  const groupClassName = `mock-projected-role-marker mock-projected-role-marker-${seat}`;

  if (compact) {
    // Compact (short-landscape) viewports keep the original single combined
    // box: there isn't room to split role and score into two layers, and
    // the collision solver may have nudged labelCenter away from its
    // nominal spot to dodge nearby cards, so the fill rotates around its
    // own (fixed) center while the text rotates around that adjusted point.
    const marker = createRoleMarkerGeometry(layout.center, seat);
    const box = projectRoleMarkerBox(layout, marker);
    const textRotate = `rotate(${marker.rotation} ${labelCenter.x} ${labelCenter.y})`;

    return (
      <g aria-label={ariaLabel} className={groupClassName}>
        <polygon
          className={`mock-projected-role-marker-fill mock-projected-role-marker-fill-${roleText.kind}`}
          points={svgPoints(box.corners)}
          transform={box.rotate}
        />
        {/* The text's own CSS transform (counter-scale) would override an
            SVG transform attribute on the same element, so the rotation
            goes on a wrapping <g> instead. */}
        <g transform={textRotate}>
          <text
            className="mock-projected-role-marker-text mock-projected-role-marker-compact"
            dominantBaseline="central"
            textAnchor="middle"
            x={labelCenter.x}
            y={labelCenter.y}
          >
            {roleText.compact}
          </text>
        </g>
      </g>
    );
  }

  // Score sits in its own box toward the sector's outer (seat) edge, and
  // the role glyph in its own box toward the inner (pentagon-center) edge,
  // so the two read as separate layers instead of being packed together.
  const scoreBox = projectRoleMarkerBox(layout, createRoleMarkerGeometry(layout.center, seat, layout.roleScoreMarker));
  const glyphBox = projectRoleMarkerBox(layout, createRoleMarkerGeometry(layout.center, seat, layout.roleGlyphMarker));

  return (
    <g aria-label={ariaLabel} className={groupClassName}>
      <polygon
        className="mock-projected-role-score-fill"
        points={svgPoints(scoreBox.corners)}
        transform={scoreBox.rotate}
      />
      <g transform={scoreBox.rotate}>
        <text
          className="mock-projected-role-marker-text mock-projected-role-marker-score"
          dominantBaseline="central"
          textAnchor="middle"
          x={scoreBox.center.x}
          y={scoreBox.center.y}
        >
          {roleText.score}
        </text>
      </g>
      <polygon
        className={`mock-projected-role-marker-fill mock-projected-role-marker-fill-${roleText.kind}`}
        points={svgPoints(glyphBox.corners)}
        transform={glyphBox.rotate}
      />
      <g transform={glyphBox.rotate}>
        <text
          className="mock-projected-role-marker-text mock-projected-role-marker-role"
          dominantBaseline="central"
          textAnchor="middle"
          x={glyphBox.center.x}
          y={glyphBox.center.y}
        >
          {roleText.glyph}
        </text>
      </g>
    </g>
  );
}

type RoleMarkerKind = "adjutant" | "citizen" | "napoleon" | "napoleon-adjutant" | "unknown";

// Short glyphs instead of spelled-out role names: the badge should read as
// "this seat holds a role," not "this seat's role name is printed here."
// 市民/連合軍 (the non-Napoleon side) share one team marker regardless of player.
function roleMarkerGlyph(role: string): string {
  switch (role) {
    case "ナポレオン":
      return "♛";
    case "副官":
      return "★";
    case "市民":
      return "⚑";
    case "ナ/副":
      return "♛★";
    default:
      return "?";
  }
}

function roleMarkerKind(role: string): RoleMarkerKind {
  switch (role) {
    case "ナポレオン":
      return "napoleon";
    case "副官":
      return "adjutant";
    case "市民":
      return "citizen";
    case "ナ/副":
      return "napoleon-adjutant";
    default:
      return "unknown";
  }
}

function createProductionRoleText(
  adapters: readonly TablePlayerAdapter[],
  match: PublicMatchState | undefined,
  state: PublicGameState | undefined,
  seat: TableSeatId
): {
  compact: string;
  glyph: string;
  kind: RoleMarkerKind;
  player: TablePlayerAdapter | undefined;
  role: string;
  score: string;
} {
  const player = adapters.find((entry) => entry.seat === seat);
  const role = player === undefined ? "?" : playerRoleLabel(player.id, state);
  const rawMatchScore = player === undefined
    ? undefined
    : match?.players.find((entry) => entry.playerId === player.id)?.rawMatchScore;
  const score = rawMatchScore === undefined ? "—" : formatMatchScore(rawMatchScore);
  const glyph = roleMarkerGlyph(role);

  return {
    compact: `${glyph} ${score}`,
    glyph,
    kind: roleMarkerKind(role),
    player,
    role,
    score
  };
}

function ProductionRoundScoreHistory({
  adapters,
  match
}: {
  adapters: readonly TablePlayerAdapter[];
  match: PublicMatchState | undefined;
}) {
  if (match === undefined || match.completed) {
    return null;
  }

  const labels = new Map(adapters.map((player) => [player.id, player.label]));
  const completedRounds = Array.from(
    { length: match.completedRoundCount },
    (_, index) => index + 1
  );

  return (
    <details className="production-round-score-history">
      <summary aria-label="局ごとの得点履歴を表示">局別</summary>
      <div className="production-round-score-history-panel">
        <table>
          <caption>局ごとの得点</caption>
          <thead>
            <tr>
              <th scope="col">プレイヤー</th>
              {completedRounds.map((round) => <th key={round} scope="col">第{round}局</th>)}
            </tr>
          </thead>
          <tbody>
            {match.players.map((player) => (
              <tr key={player.playerId}>
                <th scope="row">{labels.get(player.playerId) ?? player.playerId}</th>
                {completedRounds.map((round) => (
                  <td key={round}>
                    {player.roundScores[round - 1] === undefined
                      ? "—"
                      : formatMatchScore(player.roundScores[round - 1])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {completedRounds.length === 0 ? <p>終了した局はまだありません。</p> : null}
      </div>
    </details>
  );
}

function ProductionMatchRound({ match }: { match: PublicMatchState | undefined }) {
  if (match === undefined) {
    return null;
  }

  const layout = tableDesignMockLayout;
  const center = projectTablePoint({
    x: layout.center.x,
    y: layout.center.y
  }, layout.camera);

  return (
    <text
      aria-label={`現在 第${match.currentRound}局 / 全${match.roundCount}局`}
      className="production-match-round"
      dominantBaseline="central"
      textAnchor="middle"
      x={center.x}
      y={center.y}
    >
      第{match.currentRound}局
    </text>
  );
}

function ProjectedCurrentTrickCard({
  isCollecting,
  isWinning,
  played,
  seat,
  viewportSize
}: {
  isCollecting: boolean;
  isWinning: boolean;
  played: PublicPlayedCard | undefined;
  seat: TableSeatId;
  viewportSize: { height: number; width: number };
}) {
  if (played === undefined) {
    return null;
  }

  const layout = tableDesignMockLayout;
  const cardPlane = createCurrentTrickCardPlaneForViewport(layout, seat, viewportSize);
  const size = { width: cardPlane.width, height: cardPlane.height };
  const corners = projectTableCard(cardPlane, layout.camera);

  return (
    <ProjectedPlayingCard
      card={played.card}
      className="mock-projected-playing-card-trick"
      contentClassName={[
        "production-trick-card",
        `production-trick-card-from-${seat}`,
        isWinning ? "production-trick-card-winning" : "",
        isCollecting ? "production-trick-card-collecting" : ""
      ]
        .filter(Boolean)
        .join(" ")}
      corners={corners}
      size={size}
    />
  );
}

function ProjectedPointRiverCards({
  cards,
  seat
}: {
  cards: readonly PublicStandardCard[];
  seat: TableSeatId;
}) {
  const layout = tableDesignMockLayout;
  const river = createRiverGeometry(layout, seat);
  const placements = createRiverPlacements(cards.length, layout, seat);

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
            key={`${card.id}-${index}`}
            size={river.visibleCardSize}
          />
        );
      })}
    </>
  );
}

function ProjectedOpponentHand({
  count,
  label,
  seat
}: {
  count: number;
  label: string;
  seat: Exclude<TableSeatId, "self">;
}) {
  const layout = tableDesignMockLayout;
  const geometry = createOpponentHandGeometry(
    {
      ...layout,
      opponentHand: {
        ...layout.opponentHand,
        cardCounts: {
          ...layout.opponentHand.cardCounts,
          [seat]: count
        }
      }
    },
    seat
  );

  return (
    <section
      aria-label={`${label}の裏向き手札 ${count}枚`}
      className={`mock-projected-opponent-hand mock-projected-opponent-hand-${seat}`}
    >
      {geometry.cards.map((card) => (
        <ProjectedPlayingCardBack corners={projectVerticalCard(card, layout.camera)} key={card.index} size={geometry.cardSize} />
      ))}
    </section>
  );
}

function PlayerInfoLayer({
  adapters,
  currentPlayerId,
  viewportSize
}: {
  adapters: readonly TablePlayerAdapter[];
  currentPlayerId: string | undefined;
  viewportSize: ViewportSize;
}) {
  const selfHandCardCount = adapters.find((adapter) => adapter.isSelf)?.selfHand.length ?? 0;
  const infos = createPlayerInfoLayouts(
    tableDesignMockLayout,
    viewportSize,
    true,
    selfHandCardCount
  );

  return (
    <>
      {infos.map((info) => {
        const player = adapters.find((entry) => entry.seat === info.seatId);
        const isCurrent = player?.id === currentPlayerId;

        return (
          <div
            aria-label={`${player?.label ?? info.label} プレイヤー${isCurrent ? " 現在の手番" : ""}`}
            className={[
              "mock-player-info",
              `mock-player-info-${info.seatId}`,
              "production-player-info",
              isCurrent ? "production-player-info-current" : ""
            ]
              .filter(Boolean)
              .join(" ")}
            key={info.seatId}
            style={playerInfoStyle(info)}
          >
            <span aria-hidden="true" className="mock-player-info-avatar">
              <span className="mock-player-info-avatar-head" />
              <span className="mock-player-info-avatar-body" />
            </span>
            <span className="mock-player-info-label">{player?.label ?? info.label}</span>
          </div>
        );
      })}
    </>
  );
}

function BiddingBubbleLayer({
  adapters,
  viewportSize
}: {
  adapters: readonly TablePlayerAdapter[];
  viewportSize: ViewportSize;
}) {
  const bubbles = createBiddingBubbleLayouts(tableDesignMockLayout, viewportSize);

  return (
    <div aria-label="各プレイヤーの最新競り宣言" className="mock-bidding-bubble-layer">
      {bubbles.map((bubble) => {
        const player = adapters.find((entry) => entry.seat === bubble.seatId);
        const declaration = player?.isSelf === true || player !== undefined ? player?.biddingDeclaration : undefined;

        return (
          <output
            aria-label={`${player?.label ?? bubble.label} 最新宣言 ${formatBiddingDeclaration(declaration)}`}
            className={`mock-bidding-bubble mock-bidding-bubble-${bubble.seatId}`}
            key={bubble.seatId}
            style={biddingBubbleStyle(bubble, declaration)}
          >
            {formatBiddingDeclaration(declaration)}
          </output>
        );
      })}
    </div>
  );
}

function ProductionBiddingOverlay({
  bidding,
  canPass,
  currentPlayerId,
  isBusy,
  legalBidActions,
  onBid,
  onPass,
  selfPlayerId,
  viewportSize
}: {
  bidding: PublicBiddingState | null;
  canPass: boolean;
  currentPlayerId: string;
  isBusy: boolean;
  legalBidActions: readonly PublicBidAction[];
  onBid: ((action: PublicBidAction) => void) | undefined;
  onPass: (() => void) | undefined;
  selfPlayerId: string | undefined;
  viewportSize: ViewportSize;
}) {
  const [selection, setSelection] = useState<BidSelection | null>(() => normalizeBidSelection(legalBidActions, null));
  const geometry = createBiddingOverlayGeometry(tableDesignMockLayout, viewportSize);
  const contentMetrics = createCompactBiddingContentMetrics(geometry);
  const isSelfTurn = selfPlayerId !== undefined && currentPlayerId === selfPlayerId;
  const canOperate = isSelfTurn && !isBusy;
  const selectedAction = useMemo(() => findBidAction(legalBidActions, selection), [legalBidActions, selection]);
  const stepper = useMemo(() => getBidStepperState(legalBidActions, selection), [legalBidActions, selection]);

  useEffect(() => {
    setSelection((current) => normalizeBidSelection(legalBidActions, current));
  }, [legalBidActions]);

  return (
    <section
      aria-label="競り操作Overlay"
      className={`mock-bidding-overlay production-bidding-overlay${contentMetrics.isNarrow ? " production-bidding-overlay-narrow" : ""}`}
      style={biddingOverlayStyle(geometry)}
    >
      <div className="mock-bidding-highest">
        <span className="mock-bidding-label">現在の最高入札</span>
        <strong
          className="mock-bidding-highest-value"
          style={biddingActionColorStyle(bidding?.highestBid?.suit)}
        >
          {formatHighestBid(bidding?.highestBid ?? null)}
        </strong>
      </div>
      <div aria-label="スート選択" className="mock-bidding-suit-selector">
        {biddingSuitOptions.map((suit) => {
          const isAvailable = getBidTargetsForSuit(legalBidActions, suit).length > 0;

          return (
            <button
              aria-label={`${cardDesignSuitSymbols[suit]}を選択`}
              aria-pressed={selection?.suit === suit}
              className="mock-bidding-suit-button"
              disabled={!canOperate || !isAvailable}
              key={suit}
              onClick={() => handleSuitSelect(suit, legalBidActions, selection, setSelection)}
              style={{ "--mock-bidding-suit-color": fourColorSuitColors[suit] } as CSSProperties}
              type="button"
            >
              {cardDesignSuitSymbols[suit]}
            </button>
          );
        })}
      </div>
      <div aria-label="入札数値選択" className="mock-bidding-number-selector">
        <button
          aria-label="前の宣言枚数"
          className="mock-bidding-step-button"
          disabled={!canOperate || stepper.previousTarget === null}
          onClick={() => moveTarget(selection, stepper.previousTarget, setSelection)}
          type="button"
        >
          -
        </button>
        <strong aria-label={selection === null ? "宣言枚数未選択" : `宣言枚数 ${selection.targetPointCards}`} className="mock-bidding-number-value">
          {selection?.targetPointCards ?? "-"}
        </strong>
        <button
          aria-label="次の宣言枚数"
          className="mock-bidding-step-button"
          disabled={!canOperate || stepper.nextTarget === null}
          onClick={() => moveTarget(selection, stepper.nextTarget, setSelection)}
          type="button"
        >
          +
        </button>
      </div>
      <div className="mock-bidding-actions">
        <button
          className="mock-bidding-declare-button"
          disabled={!canOperate || selectedAction === undefined || onBid === undefined}
          onClick={() => {
            if (selectedAction !== undefined) {
              onBid?.(selectedAction);
            }
          }}
          type="button"
        >
          宣言
        </button>
        <button
          className="mock-bidding-pass-button"
          disabled={!canOperate || !canPass || onPass === undefined}
          onClick={() => onPass?.()}
          type="button"
        >
          PASS
        </button>
      </div>
    </section>
  );
}

function SelfHandLayer({
  canExchange,
  isBusy,
  legalCardIds,
  onPlay,
  selectedDiscardCardIds,
  self,
  state,
  viewportSize
}: {
  canExchange: boolean;
  handOrderMode: HandOrderMode;
  isBusy: boolean;
  legalCardIds: ReadonlySet<string>;
  onPlay: (card: PublicCard) => void;
  selectedDiscardCardIds: readonly string[];
  self: TablePlayerAdapter | undefined;
  state: PublicGameState | undefined;
  viewportSize: ViewportSize;
}) {
  const cards = self?.selfHand ?? [];
  const hand = createSelfHandViewportLayout(tableDesignMockLayout, cards.length, viewportSize);

  return (
    <div
      aria-label="自分の手札"
      className="mock-self-hand production-self-hand"
      style={selfHandViewportStyle(hand)}
    >
      {cards.map((card, index) => {
        const interactionState = getCardInteractionState(card, state, legalCardIds, canExchange);
        const selected = selectedDiscardCardIds.includes(card.id);

        return (
          <button
            aria-label={formatCardForAria(card)}
            className={[
              "mock-self-hand-card",
              "mock-playing-card",
              "production-self-hand-card",
              `production-card-${interactionState}`,
              selected ? "production-card-selected" : ""
            ]
              .filter(Boolean)
              .join(" ")}
            disabled={
              isBusy ||
              (state?.phase === "playing"
                ? !legalCardIds.has(card.id)
                : state?.phase === "exchanging"
                  ? !canExchange
                  : true)
            }
            key={card.id}
            onClick={() => onPlay(card)}
            style={selfHandCardIndexStyle(index)}
            type="button"
          >
            <CardFace card={card} />
            <span className="visually-hidden">{interactionState}</span>
          </button>
        );
      })}
    </div>
  );
}

function CardFace({ card }: { card: PublicCard }) {
  if (card.type === "joker") {
    return <JokerCardFace />;
  }

  return <CardmeisterPlayingCard card={standardToMock(card)} className="mock-cardmeister-playing-card" />;
}

function ProjectedPlayingCard({
  card,
  className,
  contentClassName,
  corners,
  size
}: {
  card: PublicCard;
  className: string;
  contentClassName?: string;
  corners: readonly { x: number; y: number }[];
  size: { height: number; width: number };
}) {
  const transform = projectiveTransformForRectangle(corners, size.width, size.height);

  return (
    <article
      aria-label={formatCardForAria(card)}
      className={`mock-projected-playing-card mock-playing-card ${className}`}
      style={{
        "--mock-projected-card-height": `${size.height}px`,
        "--mock-projected-card-transform": transform,
        "--mock-projected-card-width": `${size.width}px`
      } as CSSProperties}
    >
      {contentClassName === undefined ? (
        <CardFace card={card} />
      ) : (
        <div className={contentClassName}>
          <CardFace card={card} />
        </div>
      )}
    </article>
  );
}

function ProjectedRiverCardFace({
  card,
  corners,
  size
}: {
  card: PublicStandardCard;
  corners: readonly { x: number; y: number }[];
  size: { height: number; width: number };
}) {
  const metrics = createRiverFaceMetrics(size);
  const transform = projectiveTransformForRectangle(corners, size.width, size.height);
  const symbol = cardDesignSuitSymbols[card.suit];
  const color = fourColorSuitColors[card.suit];

  return (
    <article
      aria-label={`${card.rank}${symbol}`}
      className="mock-projected-playing-card mock-projected-river-card-face mock-river-card-face"
      style={{
        "--mock-projected-card-height": `${size.height}px`,
        "--mock-projected-card-transform": transform,
        "--mock-projected-card-width": `${size.width}px`,
        "--mock-river-face-border-width": `${metrics.borderWidth}px`,
        "--mock-river-face-color": color,
        "--mock-river-face-gap": `${metrics.gap}px`,
        "--mock-river-face-padding": `${metrics.padding}px`,
        "--mock-river-face-rank-font-size": `${metrics.rankFontSize}px`,
        "--mock-river-face-suit-font-size": `${metrics.suitFontSize}px`
      } as CSSProperties}
    >
      <span className="mock-river-card-rank">{card.rank}</span>
      <span className="mock-river-card-suit">{symbol}</span>
    </article>
  );
}

function ProjectedPlayingCardBack({
  corners,
  size
}: {
  corners: readonly { x: number; y: number }[];
  size: { height: number; width: number };
}) {
  const CardBack = mockCardBackComponent();
  const transform = projectiveTransformForRectangle(corners, size.width, size.height);

  return (
    <span
      aria-label={mockCardBackComponentName}
      className="mock-projected-playing-card mock-projected-playing-card-opponent-hand mock-playing-card mock-playing-card-back"
      style={{
        "--mock-projected-card-height": `${size.height}px`,
        "--mock-projected-card-transform": transform,
        "--mock-projected-card-width": `${size.width}px`
      } as CSSProperties}
    >
      <CardBack aria-hidden="true" className="mock-playing-card-svg" focusable="false" />
    </span>
  );
}

function JokerCardFace() {
  return (
    <span className="production-joker-card-face">
      <span>JOKER</span>
      <strong>Joker</strong>
      <span>JOKER</span>
    </span>
  );
}

function createProductionTablePlayers(
  players: readonly TablePlayer[],
  state: PublicGameState | undefined,
  handOrderMode: HandOrderMode
): readonly TablePlayerAdapter[] {
  return seatOrder.map((seat) => {
    const player = players.find((entry) => entry.seat === seat);
    const selfHand = player?.isSelf === true ? getDisplayedHandCards(state?.self.hand ?? [], handOrderMode) : [];

    return {
      biddingDeclaration: player?.biddingDeclaration,
      capturedPointCards: player?.isSelf === true
        ? (state?.self.capturedPointCards ?? player.capturedPointCards)
        : (player?.capturedPointCards ?? []),
      handCount: player?.handCount ?? 0,
      id: player?.id ?? seat,
      isSelf: player?.isSelf ?? seat === "self",
      label: player?.label ?? seat,
      seat,
      selfHand
    };
  });
}

function handleSuitSelect(
  suit: PublicSuit,
  legalBidActions: readonly PublicBidAction[],
  selection: BidSelection | null,
  setSelection: (selection: BidSelection | null) => void
): void {
  const targets = getBidTargetsForSuit(legalBidActions, suit);

  if (targets.length === 0) {
    return;
  }

  setSelection({
    suit,
    targetPointCards:
      selection !== null && targets.includes(selection.targetPointCards)
        ? selection.targetPointCards
        : targets[0]
  });
}

function moveTarget(
  selection: BidSelection | null,
  targetPointCards: number | null,
  setSelection: (selection: BidSelection | null) => void
): void {
  if (selection === null || targetPointCards === null) {
    return;
  }

  setSelection({
    suit: selection.suit,
    targetPointCards
  });
}

function standardToMock(card: PublicStandardCard): MockPlayingCard {
  return {
    rank: card.rank,
    suit: card.suit
  };
}

function formatBiddingDeclaration(declaration: TablePlayer["biddingDeclaration"] | undefined): string {
  if (declaration === undefined || declaration.type === "none") {
    return "?";
  }

  if (declaration.type === "pass") {
    return "PASS";
  }

  return `${cardDesignSuitSymbols[declaration.suit]}${declaration.targetPointCards}`;
}

function formatHighestBid(bid: PublicBiddingState["highestBid"]): string {
  if (bid === null) {
    return "-";
  }

  return `${cardDesignSuitSymbols[bid.suit]}${bid.targetPointCards}`;
}

function biddingBubbleStyle(
  bubble: { height: number; width: number; x: number; y: number },
  declaration: TablePlayer["biddingDeclaration"] | undefined
): CSSProperties {
  const suit = declaration?.type === "bid" ? declaration.suit : undefined;

  return {
    ...biddingActionColorStyle(suit),
    "--mock-bidding-bubble-height": `${bubble.height}px`,
    "--mock-bidding-bubble-width": `${bubble.width}px`,
    "--mock-x": `${bubble.x}px`,
    "--mock-y": `${bubble.y}px`
  } as CSSProperties;
}

function biddingActionColorStyle(suit: PublicSuit | undefined): CSSProperties {
  return {
    "--mock-bidding-action-color": suit === undefined ? "#64748b" : fourColorSuitColors[suit]
  } as CSSProperties;
}

function playerInfoStyle(info: { avatarSize: number; gap: number; height: number; width: number; x: number; y: number }): CSSProperties {
  return {
    "--mock-player-avatar-size": `${info.avatarSize}px`,
    "--mock-player-gap": `${info.gap}px`,
    "--mock-player-height": `${info.height}px`,
    "--mock-player-width": `${info.width}px`,
    "--mock-x": `${info.x}px`,
    "--mock-y": `${info.y}px`
  } as CSSProperties;
}

function biddingOverlayStyle(geometry: { height: number; width: number; x: number; y: number }): CSSProperties {
  const compactContent = createCompactBiddingContentMetrics(geometry);

  return {
    "--mock-bidding-column-gap": `${compactContent.columnGap}px`,
    "--mock-bidding-overlay-height": `${geometry.height}px`,
    "--mock-bidding-overlay-width": `${geometry.width}px`,
    "--mock-bidding-padding-inline": `${compactContent.paddingInline}px`,
    "--mock-bidding-primary-column-width": `${compactContent.primaryColumnWidth}px`,
    "--mock-bidding-secondary-column-width": `${compactContent.secondaryColumnWidth}px`,
    "--mock-x": `${geometry.x}px`,
    "--mock-y": `${geometry.y}px`
  } as CSSProperties;
}

function projectedBoardFitStyle(fit: {
  counterScale: number;
  scale: number;
  translate: { x: number; y: number };
}): CSSProperties {
  return {
    "--mock-projected-board-counter-scale": fit.counterScale,
    "--mock-projected-board-transform": `translate(${fit.translate.x}px, ${fit.translate.y}px) scale(${fit.scale})`
  } as CSSProperties;
}

function tableSurfaceStyle(layout: typeof tableDesignMockLayout): CSSProperties {
  return {
    "--mock-page-background": layout.page.background,
    "--mock-page-height": `${layout.page.height}px`,
    "--mock-page-width": `${layout.page.width}px`
  } as CSSProperties;
}

function playerRoleLabel(playerId: string, state: PublicGameState | undefined): string {
  if (state === undefined) {
    return "?";
  }

  if (state.phase === "finished" && state.result?.resultType === "standard") {
    const isNapoleon = state.result.napoleonPlayerId === playerId;
    const isAdjutant = state.result.adjutantPlayerId === playerId;

    if (isNapoleon && isAdjutant) {
      return "ナ/副";
    }

    if (isNapoleon) {
      return "ナポレオン";
    }

    if (isAdjutant) {
      return "副官";
    }

    return "市民";
  }

  if (state.contract === null) {
    return "?";
  }

  const isNapoleon = state.contract.napoleonPlayerId === playerId;
  const isAdjutant = state.adjutant?.revealedPlayerId === playerId;

  if (isNapoleon && isAdjutant) {
    return "ナ/副";
  }

  if (isNapoleon) {
    return "ナポレオン";
  }

  if (isAdjutant) {
    return "副官";
  }

  if (
    (state.phase === "playing" || state.phase === "finished") &&
    state.adjutant?.revealedPlayerId !== null &&
    state.adjutant?.revealedPlayerId !== undefined
  ) {
    return "市民";
  }

  return "?";
}

function formatMatchScore(score: number): string {
  return `${score > 0 ? "+" : ""}${Number.isInteger(score) ? score : score.toFixed(2)}`;
}

function getCardInteractionState(
  card: PublicCard,
  state: PublicGameState | undefined,
  legalCardIds: ReadonlySet<string>,
  canExchange: boolean
): "legal" | "blocked" | "selectable" {
  if (state?.phase === "exchanging" && canExchange) {
    return "selectable";
  }

  if (state?.phase === "playing" && legalCardIds.has(card.id)) {
    return "legal";
  }

  return "blocked";
}

function getCurrentWinningPlayerId(
  currentTrick: readonly PublicPlayedCard[],
  trumpSuit: PublicSuit,
  trickNumber: number | undefined
): string | undefined {
  if (currentTrick.length === 0) {
    return undefined;
  }

  return determineCurrentWinningPlayer(currentTrick, { trumpSuit }, { trickNumber });
}

function formatCardForAria(card: PublicCard): string {
  if (card.type === "joker") {
    return "JOKER";
  }

  return `${card.rank}${cardDesignSuitSymbols[card.suit]}`;
}

function formatCalledCardId(cardId: string | undefined): string {
  if (cardId === undefined) {
    return "?";
  }

  if (cardId === "joker") {
    return "JOKER";
  }

  const [suit, rank] = cardId.split("-");

  return isPublicSuit(suit) && rank !== undefined
    ? `${rank}${cardDesignSuitSymbols[suit]}`
    : cardId;
}

function calledCardColorStyle(cardId: string | undefined): CSSProperties {
  const [suit] = cardId?.split("-") ?? [];

  return biddingActionColorStyle(isPublicSuit(suit) ? suit : undefined);
}

function isPublicSuit(value: string | undefined): value is PublicSuit {
  return value === "spades" || value === "hearts" || value === "diamonds" || value === "clubs";
}

function svgPoints(points: readonly { x: number; y: number }[]): string {
  return points.map((point) => `${toLayoutPrecision(point.x)},${toLayoutPrecision(point.y)}`).join(" ");
}

function useViewportSize(
  fallback: ViewportSize,
  elementRef: RefObject<HTMLElement | null>
): ViewportSize {
  const [viewportSize, setViewportSize] = useState<ViewportSize>(fallback);

  useEffect(() => {
    const element = elementRef.current;
    if (element === null) {
      return;
    }

    const readViewportSize = (): ViewportSize | undefined => {
      const bounds = element.getBoundingClientRect();
      const width = element.clientWidth || bounds.width;
      const height = element.clientHeight || bounds.height;

      return width > 0 && height > 0 ? { width, height } : undefined;
    };

    const updateViewportSize = () => {
      const nextViewportSize = readViewportSize();
      if (nextViewportSize !== undefined) {
        setViewportSize((current) =>
          current.width === nextViewportSize.width && current.height === nextViewportSize.height
            ? current
            : nextViewportSize
        );
      }
    };

    const resizeObserver = typeof ResizeObserver === "undefined"
      ? undefined
      : new ResizeObserver(updateViewportSize);

    updateViewportSize();
    resizeObserver?.observe(element);
    window.addEventListener("resize", updateViewportSize);

    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener("resize", updateViewportSize);
    };
  }, [elementRef]);

  return viewportSize;
}

function toLayoutPrecision(value: number): number {
  return Number(value.toFixed(3));
}

export const productionTableTestExports = {
  createProductionTablePlayers,
  formatBiddingDeclaration,
  formatHighestBid,
  playerRoleLabel,
  standardToMock
};
