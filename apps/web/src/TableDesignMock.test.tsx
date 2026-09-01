import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  TableDesignMock,
  createBiddingBubbleLayouts,
  createBiddingOverlayGeometry,
  createCompactBiddingContentMetrics,
  createCurrentTrickCardPlane,
  createCurrentTrickCardSize,
  createCurrentTrickReferenceRiverGeometry,
  createCurrentTrickZoneGeometry,
  createOpponentHandCardMetrics,
  createOpponentHandGeometry,
  createOpponentHandsGeometry,
  createPlayerInfoLayouts,
  createProjectedBoardFit,
  createProjectedRoleTextCandidates,
  createProjectedRoleTextCenters,
  createProjectedRoleTextObstacles,
  createProjectedRoleTextSectorGeometry,
  createProjectedRoleBoardBoundingBox,
  createProjectedTableBoundingBox,
  createRiverFaceMetrics,
  createRiverGeometry,
  createRiverPlacements,
  createRoleBoardEdgeGeometry,
  createRoleMarkerGeometry,
  createRoleSectorGeometry,
  createSelfHandCardPlacements,
  createSelfHandViewportLayout,
  createSelfHandViewportMetrics,
  createTableSurfaceEdgeGeometry,
  opponentHandWidth,
  projectTableCard,
  projectVerticalCard,
  projectTablePoint,
  projectTablePolygon,
  projectedRoleTextBoardMaxDistance,
  projectedRoleTextSectorMinimumAlignment,
  projectedTextMinimumScale,
  regularPentagon,
  roleBoardSelfSideLength,
  selfHandWidth,
  selfRiverWidth,
  tableDesignMockLayout
} from "./TableDesignMock";
import { cardmeisterFourColorCsv, fourColorSuitColors } from "./cardSuitTheme";

describe("TableDesignMock", () => {
  it("renders the issue 348 world mock with unprojected tabletop geometry", () => {
    const html = renderToStaticMarkup(<TableDesignMock />);
    const selfHandStyle = html.match(/aria-label="自分の表向き手札" class="mock-self-hand" style="([^"]+)"/)?.[1] ?? "";

    expect(tableDesignMockLayout.seats).toHaveLength(5);
    expect(html).toContain("Issue 348 table design world mock");
    expect(html).toContain("契約HUD");
    expect(html).toContain("中央役職表示");
    expect(html).toContain("自分の表向き手札");
    expect(html).toContain("北西の裏向き手札");
    expect(html).toContain("北西の現在トリック置き場");
    expect(html).toContain("自分のポイント札の河");
    expect(html).not.toContain("自席操作UI");
    expect(html).toContain("--mock-page-width:2200px");
    expect(selfHandStyle).toContain("--mock-self-card-width:");
    expect(selfHandStyle).toContain("--mock-self-card-height:");
    expect(selfHandStyle).toContain("--mock-self-card-gap:");
    expect(selfHandStyle).toContain("--mock-self-hand-left:");
    expect(selfHandStyle).toContain("--mock-self-hand-top:");
    expect(selfHandStyle).toContain("--mock-self-hand-width:");
    expect(html).toContain("--mock-trick-card-width:342.939px");
    expect(html).toContain("--mock-trick-card-height:480.115px");
    expect(html).toContain("--mock-trick-zone-width:384.092px");
    expect(html).toContain("--mock-trick-zone-height:537.729px");
    expect(html).not.toContain("--mock-river-card-width:56px");
    expect(html).not.toContain("mock-avatar");
    expect((html.match(/mock-player-info mock-player-info-/g) ?? [])).toHaveLength(5);
    expect(html).toContain("mock-table-surface-world");
    expect(html).toContain("1120,-350 2318.331,520.639 1860.609,1929.361 379.391,1929.361 -78.331,520.639");
    expect(tableDesignMockLayout.tabletopWorld).toMatchObject({
      scale: 1.8,
      roleBoardRadius: 315,
      tableSurfaceRadius: 1260
    });
    expect(tableDesignMockLayout.opponentHand).toMatchObject({
      baselineOffset: 72,
      cardCounts: {
        "top-left": 1,
        "top-right": 5,
        right: 10,
        left: 7
      },
      cardGapRatio: 0.02,
      cardThickness: 10.8,
      maxCardCount: 10
    });
    expect(tableDesignMockLayout.center).toMatchObject({ height: 630, width: 630, x: 1120, y: 910 });
    expect(tableDesignMockLayout.tableSurface).toHaveLength(5);
    expect(tableDesignMockLayout.tableSurface).toEqual(regularPentagon({ x: 1120, y: 910 }, 1260, -90));
    expect(tableDesignMockLayout.seats.find((seat) => seat.id === "self")).toMatchObject({ hand: { y: 1640 } });
    expect(tableDesignMockLayout.seats.some((seat) => "trickZone" in seat)).toBe(false);
    expect(tableDesignMockLayout.riverGrid).toMatchObject({
      cardExposureRatio: 0.25,
      cellHeightRatio: 1.5,
      maxColumns: 10,
      maxRows: 2
    });
    expect(tableDesignMockLayout.riverFace).toMatchObject({
      borderWidthRatio: 0.03,
      gapRatio: 0.025,
      paddingRatio: 0.025,
      rankFontRatio: 0.9,
      suitFontRatio: 0.82
    });
    expect("edgeInset" in tableDesignMockLayout.riverGrid).toBe(false);
    expect((html.match(/mock-current-trick-zone/g) ?? [])).toHaveLength(10);
    expect((html.match(/mock-trick-card mock-playing-card/g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-self-hand-card/g) ?? [])).toHaveLength(10);
    expect((html.match(/class="mock-cardmeister-playing-card"/g) ?? [])).toHaveLength(15);
    expect((html.match(/<playing-card/g) ?? [])).toHaveLength(15);
    expect((html.match(new RegExp(`suitcolor="${cardmeisterFourColorCsv}"`, "g")) ?? [])).toHaveLength(15);
    expect((html.match(/mock-river-card mock-river-card-face/g) ?? [])).toHaveLength(32);
    expect(html).toContain('aria-label="10♦"');
    expect(html).toContain('aria-label="A♦"');
    expect(html).toContain('aria-label="K♦"');
    expect(html).toContain(`--mock-river-face-color:${fourColorSuitColors.diamonds}`);
    expect(html).not.toContain("--mock-river-card-full-height");
    expect((html.match(/mock-playing-card-svg/g) ?? [])).toHaveLength(0);
    expect((html.match(/mock-opponent-hand-world-card/g) ?? [])).toHaveLength(23);
    expect((html.match(/aria-label="B1"/g) ?? [])).toHaveLength(0);
    expect(html).not.toContain("mock-card-back-fan");
    expect((html.match(/class="role-marker /g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-role-board-sector-line/g) ?? [])).toHaveLength(5);
    expect(html).toContain("mock-role-board-inner-pentagon");
    expect(html).not.toContain("role-cell");
    expect(html).not.toContain("mock-card-corner");
    expect(html).not.toContain("mock-card-face");
    expect(html).toContain("aria-label=\"Js\"");
    expect(html).toContain(">ナポ</span>");
    expect(html).toContain(">副</span>");
  });

  it("renders the issue 348 projected mock from projected tabletop polygons", () => {
    const html = renderToStaticMarkup(<TableDesignMock variant="projected" />);

    expect(html).toContain("Issue 348 table design projected mock");
    expect(html).toContain("投影後の卓上Geometry");
    expect(html).toContain("mock-projected-board-fit");
    expect(html).toContain("--mock-projected-board-transform:");
    expect(html).toContain("mock-projected-tabletop");
    expect(html).toContain("mock-table-surface-polygon");
    expect((html.match(/mock-projected-current-trick-zone mock-projected-current-trick-zone-/g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-projected-role-marker mock-projected-role-marker-/g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-projected-playing-card /g) ?? [])).toHaveLength(60);
    expect((html.match(/mock-projected-river-card-face/g) ?? [])).toHaveLength(32);
    expect((html.match(/class="mock-cardmeister-playing-card"/g) ?? [])).toHaveLength(15);
    expect((html.match(/mock-projected-playing-card-opponent-hand/g) ?? [])).toHaveLength(23);
    expect((html.match(/matrix3d\(/g) ?? [])).toHaveLength(60);
    expect((html.match(/aria-label="B1"/g) ?? [])).toHaveLength(23);
    expect(html).toContain(`rankcolor="${cardmeisterFourColorCsv}"`);
    expect(html).toContain(`--mock-river-face-color:${fourColorSuitColors.clubs}`);
    expect(html).toContain("mock-projected-card-layer");
    expect(html).not.toContain("mock-projected-card-corner");
    expect(html).not.toContain("mock-projected-card-face");
    expect(html).not.toContain("mock-current-trick-zone mock-current-trick-zone");
    expect(html).not.toContain("mock-point-river mock-point-river");
    expect(html).toContain("北西の裏向き手札");
    expect(html).toContain("自分の表向き手札");
    expect((html.match(/mock-self-hand-card/g) ?? [])).toHaveLength(10);
    expect((html.match(/mock-player-info mock-player-info-/g) ?? [])).toHaveLength(5);
    expect(html).toContain('aria-label="自分 プレイヤー" class="mock-player-info mock-player-info-self"');
  });

  it("renders the bidding mock as a projected table variant with bidding-only UI", () => {
    const html = renderToStaticMarkup(<TableDesignMock variant="bidding" />);

    expect(html).toContain("Issue 348 table design bidding mock");
    expect(html).toContain("投影後の卓上Geometry");
    expect(html).toContain("競り操作Overlay");
    expect(html).toContain("現在の最高入札");
    expect(html).toContain("スート選択");
    expect(html).toContain("入札数値選択");
    expect(html).toContain("宣言");
    expect(html).toContain("PASS");
    expect(html).toContain("各プレイヤーの最新競り宣言");
    expect((html.match(/mock-bidding-bubble mock-bidding-bubble-/g) ?? [])).toHaveLength(5);
    expect(html).toContain("北西 最新宣言 PASS");
    expect(html).toContain("北東 最新宣言 ♦14");
    expect(html).toContain("右席 最新宣言 ♠15");
    expect(html).toContain("左席 最新宣言 ♥14");
    expect(html).toContain("自分 最新宣言 ♣13");
    expect(html).toContain(`--mock-bidding-action-color:${fourColorSuitColors.spades}`);
    expect(html).toContain(`--mock-bidding-suit-color:${fourColorSuitColors.diamonds}`);
    expect((html.match(/mock-projected-current-trick-zone mock-projected-current-trick-zone-/g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-projected-river-card-face/g) ?? [])).toHaveLength(32);
    expect((html.match(/mock-self-hand-card/g) ?? [])).toHaveLength(10);
    expect((html.match(/mock-player-info mock-player-info-/g) ?? [])).toHaveLength(5);
  });

  it("does not render bidding UI on the existing world or projected mock routes", () => {
    const world = renderToStaticMarkup(<TableDesignMock variant="world" />);
    const projected = renderToStaticMarkup(<TableDesignMock variant="projected" />);

    expect(world).not.toContain("mock-bidding-overlay");
    expect(world).not.toContain("mock-bidding-bubble");
    expect(projected).not.toContain("mock-bidding-overlay");
    expect(projected).not.toContain("mock-bidding-bubble");
  });

  it("places the bidding overlay over the table without covering the self hand at 1920x1080", () => {
    const viewport = { width: 1920, height: 1080 };
    const overlay = createBiddingOverlayGeometry(tableDesignMockLayout, viewport);
    const overlayBox = boxFromCenter(overlay);
    const selfHand = createSelfHandViewportLayout(tableDesignMockLayout, 10, viewport);
    const visibleSelfHandTop = createSelfHandCardPlacements(
      tableDesignMockLayout,
      10,
      viewport
    )[0]?.y ?? selfHand.bottom;
    const selfHandBox = boxFromTopLeft({
      height: selfHand.bottom - visibleSelfHandTop,
      width: selfHand.handWidth,
      x: selfHand.left,
      y: visibleSelfHandTop
    });

    const roleBoardBox = createProjectedRoleBoardBoundingBox(tableDesignMockLayout, viewport);

    expect(overlay.width).toBeLessThanOrEqual(820);
    expect(overlay.height).toBe(430);
    expect(boxesOverlap(overlayBox, roleBoardBox)).toBe(false);
    expect(overlayBox.top).toBeGreaterThanOrEqual(tableDesignMockLayout.bidding.overlay.viewportMargin);
    expect(overlayBox.bottom).toBeLessThanOrEqual(
      visibleSelfHandTop - tableDesignMockLayout.bidding.overlay.gapFromSelfHand
    );
    expect(boxesOverlap(overlayBox, selfHandBox)).toBe(false);
  });

  it("keeps the projected match-information pentagon visible during bidding at desktop and mobile landscape viewports", () => {
    for (const viewport of [
      { width: 1440, height: 900 },
      { width: 844, height: 390 }
    ]) {
      const overlayBox = boxFromCenter(
        createBiddingOverlayGeometry(tableDesignMockLayout, viewport)
      );
      const roleBoardBox = createProjectedRoleBoardBoundingBox(
        tableDesignMockLayout,
        viewport
      );
      const bubbles = createBiddingBubbleLayouts(tableDesignMockLayout, viewport);

      expect(boxesOverlap(overlayBox, roleBoardBox)).toBe(false);
      for (const bubble of bubbles) {
        expect(boxesOverlap(boxFromCenter(bubble), roleBoardBox)).toBe(false);
      }
      expect(overlayBox.left).toBeGreaterThanOrEqual(0);
      expect(overlayBox.right).toBeLessThanOrEqual(viewport.width);
      expect(overlayBox.top).toBeGreaterThanOrEqual(0);
      expect(overlayBox.bottom).toBeLessThanOrEqual(viewport.height);
    }
  });

  it("keeps compact bidding controls inside narrower landscape overlays", () => {
    for (const viewport of [
      { width: 667, height: 375 },
      { width: 568, height: 320 }
    ]) {
      const overlay = createBiddingOverlayGeometry(tableDesignMockLayout, viewport);
      const overlayBox = boxFromCenter(overlay);
      const metrics = createCompactBiddingContentMetrics(overlay);
      const contentLeft = overlayBox.left + metrics.borderWidth + metrics.paddingInline;
      const contentBox = boxFromTopLeft({
        height: 1,
        width: metrics.contentWidth,
        x: contentLeft,
        y: overlay.y
      });
      const suitButtonGap = 4;
      const suitButtonWidth = (contentBox.width - suitButtonGap * 3) / 4;
      const suitButtons = Array.from({ length: 4 }, (_, index) => boxFromTopLeft({
        height: 36,
        width: suitButtonWidth,
        x: contentBox.left + index * (suitButtonWidth + suitButtonGap),
        y: overlay.y
      }));
      const numberGap = 6;
      const numberTrackUnit = (contentBox.width - numberGap * 2) / 2.6;
      const numberWidths = [numberTrackUnit * 0.8, numberTrackUnit, numberTrackUnit * 0.8];
      let numberControlLeft = contentBox.left;
      const numberControls = numberWidths.map((width) => {
        const control = boxFromTopLeft({
          height: 36,
          width,
          x: numberControlLeft,
          y: overlay.y
        });
        numberControlLeft = control.right + numberGap;
        return control;
      });
      const actionGap = 8;
      const actionWidth = (contentBox.width - actionGap) / 2;
      const actionButtons = [0, 1].map((index) => boxFromTopLeft({
        height: 36,
        width: actionWidth,
        x: contentBox.left + index * (actionWidth + actionGap),
        y: overlay.y
      }));

      expect(metrics.isNarrow).toBe(true);
      expect(contentBox.left).toBeGreaterThanOrEqual(overlayBox.left);
      expect(contentBox.right).toBeLessThanOrEqual(overlayBox.right);
      const highestBid = boxFromCenter({
        height: 32,
        width: 76,
        x: contentBox.right - 38,
        y: overlay.y
      });

      expect(highestBid.left).toBeGreaterThanOrEqual(overlayBox.left);
      expect(highestBid.right).toBeLessThanOrEqual(overlayBox.right);
      for (const control of [...suitButtons, ...numberControls, ...actionButtons]) {
        expect(control.left).toBeGreaterThanOrEqual(overlayBox.left);
        expect(control.right).toBeLessThanOrEqual(overlayBox.right);
        expect(control.width).toBeGreaterThanOrEqual(36);
        expect(control.height).toBeGreaterThanOrEqual(36);
      }
    }
  });

  it("places all bidding bubbles near player info without covering hands or viewport edges", () => {
    const viewport = { width: 1920, height: 1080 };
    const bubbles = createBiddingBubbleLayouts(tableDesignMockLayout, viewport);
    const playerInfos = createPlayerInfoLayouts(tableDesignMockLayout, viewport, true);
    const projectedFit = createProjectedBoardFit(tableDesignMockLayout, viewport);
    const selfHand = createSelfHandViewportLayout(tableDesignMockLayout, 10, viewport);
    const selfHandBox = boxFromTopLeft({
      height: selfHand.handHeight,
      width: selfHand.handWidth,
      x: selfHand.left,
      y: selfHand.top
    });
    const opponentHandBoxes = (["top-left", "top-right", "right", "left"] as const).map((seatId) => {
      const hand = createOpponentHandGeometry(tableDesignMockLayout, seatId);

      return transformTestBox(
        boundingTestBox(hand.cards.flatMap((card) => projectVerticalCard(card))),
        projectedFit.scale,
        projectedFit.translate
      );
    });

    expect(bubbles.map((bubble) => bubble.seatId)).toEqual(["top-left", "top-right", "right", "left", "self"]);
    expect(new Set(bubbles.map((bubble) => `${bubble.x}:${bubble.y}`)).size).toBe(5);

    for (const bubble of bubbles) {
      const bubbleBox = boxFromCenter(bubble);

      expect(bubble.width).toBe(tableDesignMockLayout.bidding.bubble.width);
      expect(bubble.height).toBe(tableDesignMockLayout.bidding.bubble.height);
      expect(bubbleBox.left).toBeGreaterThanOrEqual(tableDesignMockLayout.bidding.bubble.viewportMargin);
      expect(bubbleBox.right).toBeLessThanOrEqual(viewport.width - tableDesignMockLayout.bidding.bubble.viewportMargin);
      expect(bubbleBox.top).toBeGreaterThanOrEqual(tableDesignMockLayout.bidding.bubble.viewportMargin);
      expect(bubbleBox.bottom).toBeLessThanOrEqual(viewport.height - tableDesignMockLayout.bidding.bubble.viewportMargin);
      expect(boxesOverlap(bubbleBox, selfHandBox)).toBe(false);

      for (const info of playerInfos) {
        expect(boxesOverlap(bubbleBox, boxFromCenter(info))).toBe(false);
      }

      for (const handBox of opponentHandBoxes) {
        expect(boxesOverlap(bubbleBox, handBox)).toBe(false);
      }
    }
  });

  it("keeps a fixed self-hand footprint as cards are removed", () => {
    const viewportWidth = tableDesignMockLayout.page.width;
    const metrics = createSelfHandViewportMetrics(viewportWidth);
    const exchangeHand = createSelfHandViewportLayout(tableDesignMockLayout, 13);
    const fullHand = createSelfHandViewportLayout(tableDesignMockLayout, 10);
    const reducedHand = createSelfHandViewportLayout(tableDesignMockLayout, 3);

    expect(metrics.cardSize.width).toBeCloseTo((0.8 * viewportWidth) / 13);
    expect(metrics.cardSize.height).toBeCloseTo(metrics.cardSize.width * 7 / 5);
    expect(metrics.gap).toBeCloseTo((0.16 * viewportWidth) / 12);
    expect(fullHand.columnCount).toBe(5);
    expect(fullHand.rowCount).toBe(2);
    expect(fullHand.handWidth).toBeCloseTo(selfHandWidth(5, metrics));
    expect(fullHand.contentWidth).toBe(fullHand.handWidth);
    expect(fullHand.top).toBeGreaterThan(0);
    expect(fullHand.bottom).toBeLessThanOrEqual(tableDesignMockLayout.page.height);

    expect(reducedHand.cardSize.width).toBe(fullHand.cardSize.width);
    expect(reducedHand.cardSize.height).toBe(fullHand.cardSize.height);
    expect(reducedHand.gap).toBe(fullHand.gap);
    expect(reducedHand.contentWidth).toBeLessThan(fullHand.contentWidth);
    expect(reducedHand.handWidth).toBe(fullHand.handWidth);
    expect(reducedHand.handHeight).toBe(fullHand.handHeight);
    expect(reducedHand.left).toBe(fullHand.left);
    expect(reducedHand.top).toBe(fullHand.top);
    expect(reducedHand.center.x).toBeCloseTo(viewportWidth / 2);
    expect(exchangeHand).toMatchObject({
      bottom: fullHand.bottom,
      handHeight: fullHand.handHeight,
      handWidth: fullHand.handWidth,
      left: fullHand.left,
      top: fullHand.top
    });
  });

  it("places ten self-hand cards in five fixed columns and two rows", () => {
    for (const viewport of [
      { width: 1920, height: 1080 },
      { width: 844, height: 390 },
      { width: 568, height: 320 }
    ]) {
      const hand = createSelfHandViewportLayout(tableDesignMockLayout, 10, viewport);
      const cards = createSelfHandCardPlacements(tableDesignMockLayout, 10, viewport);

      expect(new Set(cards.map((card) => card.x)).size).toBe(5);
      expect(new Set(cards.map((card) => card.y)).size).toBe(2);
      const firstVisibleRowTop = hand.top;
      for (const card of cards.slice(0, 5)) {
        expect(card.y).toBeCloseTo(firstVisibleRowTop);
      }
      for (const card of cards.slice(5)) {
        expect(card.y).toBeCloseTo(firstVisibleRowTop + hand.rowStep);
      }
      expect(cards[0]?.x).toBe(hand.left);
      expect(cards[5]?.x).toBe(hand.left);
      expect(Math.max(...cards.map((card) => card.y + card.height))).toBeCloseTo(hand.bottom);

      const exchangeHand = createSelfHandViewportLayout(tableDesignMockLayout, 13, viewport);
      const exchangeCards = createSelfHandCardPlacements(tableDesignMockLayout, 13, viewport);
      expect(new Set(exchangeCards.map((card) => card.x)).size).toBe(7);
      expect(new Set(exchangeCards.map((card) => card.y)).size).toBe(2);
      expect(Math.min(...exchangeCards.map((card) => card.y))).toBeCloseTo(exchangeHand.top);
      expect(Math.max(...exchangeCards.map((card) => card.y + card.height))).toBeCloseTo(
        exchangeHand.bottom
      );
      for (const [index, card] of exchangeCards.entries()) {
        for (const otherCard of exchangeCards.slice(index + 1)) {
          expect(boxesOverlap(boxFromTopLeft(card), boxFromTopLeft(otherCard))).toBe(false);
        }
      }
    }
  });

  it("keeps the self hand viewport-width based when projected board fit changes", () => {
    const fullHdHand = createSelfHandViewportLayout(tableDesignMockLayout, 10, { width: 1920, height: 1080 });
    const qhdHand = createSelfHandViewportLayout(tableDesignMockLayout, 10, { width: 2560, height: 1440 });

    expect(fullHdHand.cardSize.width).toBeCloseTo((0.8 * 1920) / 13);
    expect(fullHdHand.cardSize.height).toBeCloseTo(fullHdHand.cardSize.width * 7 / 5);
    expect(fullHdHand.gap).toBeCloseTo((0.16 * 1920) / 12);
    expect(fullHdHand.handWidth).toBeCloseTo(selfHandWidth(5, fullHdHand));
    expect(fullHdHand.center.x).toBeCloseTo(960);
    expect(fullHdHand.bottom).toBe(1080 - tableDesignMockLayout.selfHandUi.bottomInset);

    expect(qhdHand.cardSize.width).toBeCloseTo((0.8 * 2560) / 13);
    expect(qhdHand.handWidth).toBeCloseTo(selfHandWidth(5, qhdHand));
    expect(qhdHand.center.x).toBeCloseTo(1280);
    expect(qhdHand.cardSize.width).toBeGreaterThan(fullHdHand.cardSize.width);
  });

  it("keeps the full hand inside desktop, mobile, and safe-area-reduced table surfaces", () => {
    for (const viewport of [
      { width: 1920, height: 1080 },
      { width: 1440, height: 900 },
      { width: 960, height: 500 },
      { width: 844, height: 390 },
      { width: 568, height: 320 },
      { width: 812, height: 341 }
    ]) {
      const fullHand = createSelfHandViewportLayout(tableDesignMockLayout, 10, viewport);
      const reducedHand = createSelfHandViewportLayout(tableDesignMockLayout, 3, viewport);
      const normalCards = createSelfHandCardPlacements(tableDesignMockLayout, 10, viewport);
      const normalHandTop = normalCards[0]?.y ?? fullHand.bottom;
      const normalHandBox = boxFromTopLeft({
        height: fullHand.bottom - normalHandTop,
        width: fullHand.handWidth,
        x: fullHand.left,
        y: normalHandTop
      });
      const exchangeCards = createSelfHandCardPlacements(tableDesignMockLayout, 13, viewport);
      const exchangeHandBox = boundingTestBox(exchangeCards.flatMap((card) => [
        { x: card.x, y: card.y },
        { x: card.x + card.width, y: card.y },
        { x: card.x + card.width, y: card.y + card.height },
        { x: card.x, y: card.y + card.height }
      ]));
      const normalProjectedTable = createProjectedBoardFit(tableDesignMockLayout, viewport, 10);
      const exchangeProjectedTable = createProjectedBoardFit(tableDesignMockLayout, viewport, 13);
      const obstacleContext = {
        opponentHandCounts: { left: 10, right: 10, "top-left": 10, "top-right": 10 },
        riverCardCounts: { left: 10, right: 10, self: 10, "top-left": 10, "top-right": 10 }
      };
      const normalObstacles = createProjectedRoleTextObstacles(tableDesignMockLayout, viewport, {
        ...obstacleContext,
        selfHandCardCount: 10
      });
      const exchangeObstacles = createProjectedRoleTextObstacles(tableDesignMockLayout, viewport, {
        ...obstacleContext,
        selfHandCardCount: 13
      });
      const selfInfo = createPlayerInfoLayouts(tableDesignMockLayout, viewport, true, 13)
        .find((info) => info.seatId === "self");
      expect(fullHand.top).toBeGreaterThanOrEqual(0);
      expect(fullHand.bottom).toBeLessThanOrEqual(
        viewport.height - tableDesignMockLayout.selfHandUi.bottomInset
      );
      expect(fullHand.left).toBeGreaterThanOrEqual(0);
      expect(fullHand.left + fullHand.handWidth).toBeLessThanOrEqual(viewport.width);
      expect(
        normalHandBox.top - normalProjectedTable.transformedTableBox.bottom
      ).toBeCloseTo(tableDesignMockLayout.selfHandUi.gapFromTable);
      expect(exchangeProjectedTable.transformedTableBox.bottom).toBeLessThanOrEqual(
        fullHand.top - tableDesignMockLayout.selfHandUi.gapFromTable
      );
      expect(reducedHand).toMatchObject({
        bottom: fullHand.bottom,
        handWidth: fullHand.handWidth,
        left: fullHand.left,
        top: fullHand.top
      });
      expect(selfInfo).toBeDefined();
      if (selfInfo !== undefined) {
        const selfInfoBox = boxFromCenter(selfInfo);
        expect(boxesOverlap(selfInfoBox, exchangeHandBox)).toBe(false);
        expect(selfInfoBox.top).toBeGreaterThanOrEqual(0);
        expect(selfInfoBox.bottom).toBeLessThanOrEqual(viewport.height);
      }
      for (const [selfHandBox, obstacles] of [
        [normalHandBox, normalObstacles],
        [exchangeHandBox, exchangeObstacles]
      ] as const) {
        for (const gameplayBox of [
          ...obstacles.opponentHands,
          ...obstacles.rivers,
          ...obstacles.trickCards
        ]) {
          expect(boxesOverlap(selfHandBox, gameplayBox)).toBe(false);
        }
      }
    }
  });

  it("fits projected table scale from viewport height and keeps width growth from enlarging it", () => {
    const rawBox = createProjectedTableBoundingBox(tableDesignMockLayout);
    const fullHd = createProjectedBoardFit(tableDesignMockLayout, { width: 1920, height: 1080 });
    const widerSameHeight = createProjectedBoardFit(tableDesignMockLayout, { width: 2560, height: 1080 });
    const qhd = createProjectedBoardFit(tableDesignMockLayout, { width: 2560, height: 1440 });

    expect(rawBox.height).toBeGreaterThan(0);
    expect(fullHd.transformedTableBox.height).toBeCloseTo(1080 * fullHd.tableHeightRatio);
    expect(fullHd.transformedTableBox.top).toBeCloseTo(1080 * 0.015);
    expect(fullHd.transformedTableBox.x).toBeCloseTo(960);
    expect(fullHd.transformedTableBox.width).toBeLessThan(1920);

    expect(widerSameHeight.scale).toBeLessThan(fullHd.scale);
    expect(widerSameHeight.transformedTableBox.height).toBeLessThan(fullHd.transformedTableBox.height);
    expect(widerSameHeight.transformedTableBox.width).toBeLessThan(fullHd.transformedTableBox.width);
    expect(widerSameHeight.transformedTableBox.x).toBeCloseTo(1280);

    expect(qhd.scale).toBeGreaterThan(fullHd.scale);
    expect(qhd.transformedTableBox.height).toBeCloseTo(1440 * qhd.tableHeightRatio);
    expect(qhd.transformedTableBox.top).toBeCloseTo(1440 * 0.015);
    expect(qhd.transformedTableBox.x).toBeCloseTo(1280);
  });

  it("keeps the projected board fixed when the last self-hand card is played", () => {
    const viewport = { width: 844, height: 390 };

    expect(createProjectedBoardFit(tableDesignMockLayout, viewport, 0)).toEqual(
      createProjectedBoardFit(tableDesignMockLayout, viewport, 1)
    );
  });

  it("keeps the projected board fixed through the 13-to-10 card exchange transition", () => {
    const viewport = { width: 812, height: 341 };

    expect(createProjectedBoardFit(tableDesignMockLayout, viewport, 10)).toEqual(
      createProjectedBoardFit(tableDesignMockLayout, viewport, 13)
    );
  });

  it("counter-scales central match text to readable rendered sizes at 844x390", () => {
    const mobileFit = createProjectedBoardFit(tableDesignMockLayout, { width: 844, height: 390 });
    const renderedFontSize = (fontSize: number) => fontSize * mobileFit.scale * mobileFit.counterScale;

    expect(mobileFit.counterScale).toBeCloseTo(projectedTextMinimumScale / mobileFit.scale);
    expect(renderedFontSize(16)).toBeCloseTo(12);
    expect(renderedFontSize(18)).toBeCloseTo(13.5);
    expect(renderedFontSize(20)).toBeCloseTo(15);

    const html = renderToStaticMarkup(<TableDesignMock variant="projected" />);
    expect(html).toContain("--mock-projected-board-counter-scale:");
  });

  it("keeps every compact label visible around occupied table UI during bidding and play", () => {
    const labels = ["ナ/+21", "副/-3", "市/+13", "市/+7", "市/0"];
    const seats = ["top-left", "top-right", "right", "left", "self"] as const;

    for (const viewport of [
      { width: 568, height: 320 },
      { width: 844, height: 390 }
    ]) {
      for (const isBidding of [false, true]) {
        const fit = createProjectedBoardFit(tableDesignMockLayout, viewport);
        const context = {
          isBidding,
          roleTextLabels: Object.fromEntries(seats.map((seatId, index) => [seatId, labels[index]]))
        };
        const centers = createProjectedRoleTextCenters(tableDesignMockLayout, viewport, context);
        const obstacles = createProjectedRoleTextObstacles(tableDesignMockLayout, viewport, context);
        const markerBoxes = seats.map((seatId, index) => {
          const center = centers[seatId];
          const renderedCenter = {
            x: center.x * fit.scale + fit.translate.x,
            y: center.y * fit.scale + fit.translate.y
          };

          return boxFromCenter({
            ...renderedCenter,
            height: 15,
            width: labels[index].length * 9
          });
        });
        const renderedCenters = seats.map((seatId) => {
          const center = centers[seatId];

          return {
            x: center.x * fit.scale + fit.translate.x,
            y: center.y * fit.scale + fit.translate.y
          };
        });

        expect(obstacles.opponentHands).toHaveLength(4);
        expect(obstacles.playerInfos).toHaveLength(5);
        expect(obstacles.trickCards).toHaveLength(isBidding ? 0 : 5);
        expect(obstacles.biddingOverlay).toHaveLength(isBidding ? 1 : 0);
        expect(obstacles.biddingBubbles).toHaveLength(isBidding ? 5 : 0);

        for (const [index, seatId] of seats.entries()) {
          const ownSector = createProjectedRoleTextSectorGeometry(tableDesignMockLayout, viewport, seatId);

          expect(projectedSectorAlignment(renderedCenters[index], ownSector))
            .toBeGreaterThanOrEqual(projectedRoleTextSectorMinimumAlignment);
        }

        for (const [index, markerBox] of markerBoxes.entries()) {
          expect(markerBox.left).toBeGreaterThanOrEqual(0);
          expect(markerBox.right).toBeLessThanOrEqual(viewport.width);
          expect(markerBox.top).toBeGreaterThanOrEqual(0);
          expect(markerBox.bottom).toBeLessThanOrEqual(viewport.height);

          for (const obstacle of Object.values(obstacles).flat()) {
            expect(boxesOverlap(markerBox, obstacle)).toBe(false);
          }
          for (const otherMarkerBox of markerBoxes.slice(index + 1)) {
            expect(boxesOverlap(markerBox, otherMarkerBox)).toBe(false);
          }
        }
      }
    }
  });

  it("keeps initial production bidding labels renderable on common mobile landscape viewports", () => {
    const labels = ["ナ副/+220", "副/-3", "市/+13", "市/+7", "市/0"];
    const seats = ["top-left", "top-right", "right", "left", "self"] as const;
    const context = {
      isBidding: true,
      opponentHandCounts: { "top-left": 10, "top-right": 10, right: 10, left: 10 },
      riverCardCounts: { "top-left": 0, "top-right": 0, right: 0, self: 0, left: 0 },
      roleTextLabels: Object.fromEntries(seats.map((seatId, index) => [seatId, labels[index]]))
    };

    for (const viewport of [
      { width: 568, height: 320 },
      { width: 640, height: 360 },
      { width: 667, height: 375 },
      { width: 720, height: 360 },
      { width: 960, height: 500 }
    ]) {
      const fit = createProjectedBoardFit(tableDesignMockLayout, viewport);
      const roleBoardBox = createProjectedRoleBoardBoundingBox(tableDesignMockLayout, viewport);
      const centers = createProjectedRoleTextCenters(tableDesignMockLayout, viewport, context);
      const obstacles = createProjectedRoleTextObstacles(tableDesignMockLayout, viewport, context);
      const markerBoxes = seats.map((seatId, index) => {
        const center = centers[seatId];

        return boxFromCenter({
          x: center.x * fit.scale + fit.translate.x,
          y: center.y * fit.scale + fit.translate.y,
          height: 15,
          width: labels[index].length * 9
        });
      });
      const renderedCenters = seats.map((seatId) => {
        const center = centers[seatId];

        return {
          x: center.x * fit.scale + fit.translate.x,
          y: center.y * fit.scale + fit.translate.y
        };
      });

      expect(Object.keys(centers).sort()).toEqual(["left", "right", "self", "top-left", "top-right"]);
      for (const center of Object.values(centers)) {
        expect(Number.isFinite(center.x)).toBe(true);
        expect(Number.isFinite(center.y)).toBe(true);
      }
      for (const [index, seatId] of seats.entries()) {
        const ownSector = createProjectedRoleTextSectorGeometry(tableDesignMockLayout, viewport, seatId);

        expect(projectedSectorAlignment(renderedCenters[index], ownSector))
          .toBeGreaterThanOrEqual(projectedRoleTextSectorMinimumAlignment);
        expect(distanceFromTestBox(renderedCenters[index], roleBoardBox))
          .toBeLessThanOrEqual(projectedRoleTextBoardMaxDistance);
      }
      for (const [index, markerBox] of markerBoxes.entries()) {
        expect(markerBox.left).toBeGreaterThanOrEqual(0);
        expect(markerBox.right).toBeLessThanOrEqual(viewport.width);
        expect(markerBox.top).toBeGreaterThanOrEqual(0);
        expect(markerBox.bottom).toBeLessThanOrEqual(viewport.height);
        for (const [obstacleType, obstacleBoxes] of Object.entries(obstacles)) {
          for (const obstacle of obstacleBoxes) {
            expect(
              boxesOverlap(markerBox, obstacle),
              `${viewport.width}x${viewport.height} ${seats[index]} overlaps ${obstacleType}`
            ).toBe(false);
          }
        }
        for (const otherMarkerBox of markerBoxes.slice(index + 1)) {
          expect(boxesOverlap(markerBox, otherMarkerBox)).toBe(false);
        }
      }
    }
  });

  it("keeps the longest top-seat label clear of Current Trick cards during play", () => {
    const context = {
      isBidding: false,
      opponentHandCounts: { "top-left": 9, "top-right": 9, right: 9, left: 9 },
      riverCardCounts: { "top-left": 0, "top-right": 0, right: 0, self: 0, left: 0 },
      roleTextLabels: {
        "top-left": "ナ副/+220",
        "top-right": "副/-3",
        right: "市/+13",
        left: "市/+7",
        self: "市/0"
      }
    };

    for (const viewport of [
      { width: 568, height: 320 },
      { width: 640, height: 360 },
      { width: 720, height: 360 },
      { width: 844, height: 390 }
    ]) {
      const fit = createProjectedBoardFit(tableDesignMockLayout, viewport);
      const center = createProjectedRoleTextCenters(tableDesignMockLayout, viewport, context)["top-left"];
      const markerBox = boxFromCenter({
        x: center.x * fit.scale + fit.translate.x,
        y: center.y * fit.scale + fit.translate.y,
        height: 15,
        width: "ナ副/+220".length * 9
      });
      const obstacles = createProjectedRoleTextObstacles(tableDesignMockLayout, viewport, context);
      const trickCards = obstacles.trickCards;

      expect(trickCards).toHaveLength(5);
      for (const trickCard of trickCards) {
        expect(
          boxesOverlap(markerBox, trickCard),
          `${viewport.width}x${viewport.height} top-left label overlaps Current Trick`
        ).toBe(false);
      }
    }
  });

  it("bounds compact role-label candidate work to the projected role-board region", () => {
    const context = {
      isBidding: true,
      opponentHandCounts: { "top-left": 10, "top-right": 10, right: 10, left: 10 },
      riverCardCounts: { "top-left": 0, "top-right": 0, right: 0, self: 0, left: 0 }
    };

    for (const viewport of [
      { width: 568, height: 320 },
      { width: 844, height: 390 },
      { width: 960, height: 500 }
    ]) {
      const candidates = createProjectedRoleTextCandidates(tableDesignMockLayout, viewport);
      const startedAt = performance.now();

      createProjectedRoleTextCenters(tableDesignMockLayout, viewport, context);

      expect(candidates.length).toBeLessThan(20_000);
      expect(performance.now() - startedAt).toBeLessThan(100);
    }
  });

  it("uses the same unprojected self-hand layout for world and projected mocks", () => {
    const world = renderToStaticMarkup(<TableDesignMock variant="world" />);
    const projected = renderToStaticMarkup(<TableDesignMock variant="projected" />);
    const selfHandStyle = /aria-label="自分の表向き手札" class="mock-self-hand" style="([^"]+)"/;
    const worldStyle = world.match(selfHandStyle)?.[1];
    const projectedStyle = projected.match(selfHandStyle)?.[1];

    expect(worldStyle).toBeDefined();
    expect(projectedStyle).toBeDefined();
    expect(projectedStyle).toBe(worldStyle);
    expect(worldStyle).not.toContain("matrix3d");
    expect(worldStyle).not.toContain("--mock-x");
    expect(worldStyle).not.toContain("--mock-y");
  });

  it("keeps the self-hand layout independent from projected camera settings", () => {
    const baseline = createSelfHandViewportLayout(tableDesignMockLayout, 13);
    const cameraChangedLayout = {
      ...tableDesignMockLayout,
      camera: {
        ...tableDesignMockLayout.camera,
        focalLength: tableDesignMockLayout.camera.focalLength * 1.5,
        position: { x: 620, y: 3120, z: 1800 },
        screenCenter: { x: 970, y: 520 },
        target: { x: 1200, y: 860, z: 0 }
      }
    };

    expect(createSelfHandViewportLayout(cameraChangedLayout, 13)).toEqual(baseline);
  });

  it("places player name and avatar units outside the hands", () => {
    const viewport = { width: 1920, height: 1080 };
    const projectedInfos = createPlayerInfoLayouts(tableDesignMockLayout, viewport, true);
    const worldInfos = createPlayerInfoLayouts(tableDesignMockLayout);
    const projectedFit = createProjectedBoardFit(tableDesignMockLayout, viewport);
    const selfInfo = projectedInfos.find((info) => info.seatId === "self");
    const selfHand = createSelfHandViewportLayout(tableDesignMockLayout, 10, viewport);
    const compactProjectedInfos = createPlayerInfoLayouts(tableDesignMockLayout, { width: 1366, height: 768 }, true);
    const fixtureRiverCardCounts = { "top-left": 2, "top-right": 20, right: 3, left: 2 } as const;

    expect(projectedInfos.map((info) => info.seatId)).toEqual(["top-left", "top-right", "right", "left", "self"]);
    expect(worldInfos.map((info) => info.seatId)).toEqual(["top-left", "top-right", "right", "left", "self"]);
    expect(selfInfo).toBeDefined();

    const hudBox = boxFromTopLeft(tableDesignMockLayout.hud);

    for (const info of worldInfos) {
      expect(boxesOverlap(boxFromCenter(info), hudBox)).toBe(false);
    }

    for (const info of compactProjectedInfos) {
      expect(boxesOverlap(boxFromCenter(info), hudBox)).toBe(false);
    }

    for (const seatId of ["top-left", "top-right", "right", "left"] as const) {
      const info = worldInfos.find((entry) => entry.seatId === seatId);
      const hand = createOpponentHandGeometry(tableDesignMockLayout, seatId);
      const halfThickness = tableDesignMockLayout.opponentHand.cardThickness / 2;
      const handBox = boundingTestBox(
        hand.cards.flatMap((card) => [
          {
            x: card.leftBottom.x - hand.edge.normal.x * halfThickness,
            y: card.leftBottom.y - hand.edge.normal.y * halfThickness
          },
          {
            x: card.rightBottom.x - hand.edge.normal.x * halfThickness,
            y: card.rightBottom.y - hand.edge.normal.y * halfThickness
          },
          {
            x: card.rightBottom.x + hand.edge.normal.x * halfThickness,
            y: card.rightBottom.y + hand.edge.normal.y * halfThickness
          },
          {
            x: card.leftBottom.x + hand.edge.normal.x * halfThickness,
            y: card.leftBottom.y + hand.edge.normal.y * halfThickness
          }
        ])
      );

      expect(info).toBeDefined();

      if (info === undefined) {
        throw new Error(`Expected world player info for ${seatId}`);
      }

      expect(boxesOverlap(boxFromCenter(info), handBox)).toBe(false);
      expect(boxesOverlap(boxFromCenter(info), riverCardsTestBox(seatId, fixtureRiverCardCounts[seatId]))).toBe(false);
    }

    if (selfInfo === undefined) {
      throw new Error("Expected self player info");
    }

    const selfBox = boxFromCenter(selfInfo);

    expect(selfBox.left).toBeGreaterThanOrEqual(tableDesignMockLayout.playerInfo.viewportMargin);
    expect(selfBox.right).toBeLessThanOrEqual(
      selfHand.left - tableDesignMockLayout.playerInfo.selfGap
    );
    expect(boxesOverlap(selfBox, boxFromTopLeft({
      height: selfHand.handHeight,
      width: selfHand.handWidth,
      x: selfHand.left,
      y: selfHand.top
    }))).toBe(false);
    expect(selfBox.top).toBeGreaterThanOrEqual(0);
    expect(selfBox.bottom).toBeLessThanOrEqual(viewport.height);

    const topLeftInfo = projectedInfos.find((entry) => entry.seatId === "top-left");
    const topRightInfo = projectedInfos.find((entry) => entry.seatId === "top-right");

    if (topLeftInfo === undefined || topRightInfo === undefined) {
      throw new Error("Expected top player info units");
    }

    const topLeftDistanceFromCenter = projectedFit.transformedTableBox.x - topLeftInfo.x;
    const topRightDistanceFromCenter = topRightInfo.x - projectedFit.transformedTableBox.x;

    expect(topLeftDistanceFromCenter).toBeGreaterThan(0);
    expect(topRightDistanceFromCenter).toBeGreaterThan(0);
    expect(Math.abs(topLeftDistanceFromCenter - topRightDistanceFromCenter)).toBeLessThanOrEqual(128);

    const tabletProjectedInfos = createPlayerInfoLayouts(tableDesignMockLayout, { width: 1024, height: 768 }, true);
    const tabletTopLeftInfo = tabletProjectedInfos.find((entry) => entry.seatId === "top-left");
    const tabletTopRightInfo = tabletProjectedInfos.find((entry) => entry.seatId === "top-right");

    if (tabletTopLeftInfo === undefined || tabletTopRightInfo === undefined) {
      throw new Error("Expected tablet top player info units");
    }

    expect(Math.abs(tabletTopRightInfo.x - tabletTopLeftInfo.x)).toBeGreaterThan(
      tableDesignMockLayout.playerInfo.unitWidth * 0.8
    );

    for (const compactViewport of [{ width: 720, height: 360 }, { width: 690, height: 320 }]) {
      const compactProjectedInfos = createPlayerInfoLayouts(tableDesignMockLayout, compactViewport, true);
      const compactTopLeftInfo = compactProjectedInfos.find((entry) => entry.seatId === "top-left");
      const compactTopRightInfo = compactProjectedInfos.find((entry) => entry.seatId === "top-right");

      if (compactTopLeftInfo === undefined || compactTopRightInfo === undefined) {
        throw new Error(`Expected compact top player info units for ${compactViewport.width}x${compactViewport.height}`);
      }

      expect(Math.abs(compactTopRightInfo.x - compactTopLeftInfo.x)).toBeGreaterThanOrEqual(138);
    }

    for (const seatId of ["top-left", "top-right", "right", "left"] as const) {
      const info = projectedInfos.find((entry) => entry.seatId === seatId);
      const hand = createOpponentHandGeometry(tableDesignMockLayout, seatId);
      const handBox = transformTestBox(
        boundingTestBox(hand.cards.flatMap((card) => projectVerticalCard(card))),
        projectedFit.scale,
        projectedFit.translate
      );

      expect(info).toBeDefined();

      if (info === undefined) {
        throw new Error(`Expected player info for ${seatId}`);
      }

      const infoBox = boxFromCenter(info);
      const outward = normalizeTestVector({
        x: handBox.x - projectedFit.transformedTableBox.x,
        y: handBox.y - projectedFit.transformedTableBox.y
      });

      expect(boxesOverlap(infoBox, handBox)).toBe(false);
      expect((info.x - handBox.x) * outward.x + (info.y - handBox.y) * outward.y).toBeGreaterThan(0);
      expect(infoBox.left).toBeGreaterThanOrEqual(tableDesignMockLayout.playerInfo.viewportMargin);
      expect(infoBox.right).toBeLessThanOrEqual(viewport.width - tableDesignMockLayout.playerInfo.viewportMargin);
      expect(infoBox.top).toBeGreaterThanOrEqual(tableDesignMockLayout.playerInfo.viewportMargin);
      expect(infoBox.bottom).toBeLessThanOrEqual(viewport.height - tableDesignMockLayout.playerInfo.viewportMargin);
    }

    const compactProjectedFit = createProjectedBoardFit(tableDesignMockLayout, { width: 1366, height: 768 });

    for (const seatId of ["top-left", "top-right", "right", "left"] as const) {
      const info = compactProjectedInfos.find((entry) => entry.seatId === seatId);
      const hand = createOpponentHandGeometry(tableDesignMockLayout, seatId);
      const handBox = transformTestBox(
        boundingTestBox(hand.cards.flatMap((card) => projectVerticalCard(card))),
        compactProjectedFit.scale,
        compactProjectedFit.translate
      );
      const riverBox = transformTestBox(
        projectedRiverCardsTestBox(seatId, fixtureRiverCardCounts[seatId]),
        compactProjectedFit.scale,
        compactProjectedFit.translate
      );

      expect(info).toBeDefined();

      if (info === undefined) {
        throw new Error(`Expected compact projected player info for ${seatId}`);
      }

      expect(boxesOverlap(boxFromCenter(info), handBox)).toBe(false);
      expect(boxesOverlap(boxFromCenter(info), riverBox)).toBe(false);
    }
  });

  it("keeps projected player info panels separate in normal, exchange, and zero-card result layouts", () => {
    for (const { cardCount, viewport } of [
      { cardCount: 10, viewport: { width: 844, height: 390 } },
      { cardCount: 13, viewport: { width: 812, height: 341 } },
      { cardCount: 0, viewport: { width: 1280, height: 720 } },
      { cardCount: 0, viewport: { width: 1024, height: 768 } }
    ]) {
      const boxes = createPlayerInfoLayouts(
        tableDesignMockLayout,
        viewport,
        true,
        cardCount
      ).map((info) => boxFromCenter(info));

      for (const [index, box] of boxes.entries()) {
        expect(box.left).toBeGreaterThanOrEqual(tableDesignMockLayout.playerInfo.viewportMargin);
        expect(box.right).toBeLessThanOrEqual(
          viewport.width - tableDesignMockLayout.playerInfo.viewportMargin
        );
        expect(box.top).toBeGreaterThanOrEqual(tableDesignMockLayout.playerInfo.viewportMargin);
        expect(box.bottom).toBeLessThanOrEqual(
          viewport.height - tableDesignMockLayout.playerInfo.viewportMargin
        );

        for (const otherBox of boxes.slice(index + 1)) {
          expect(boxesOverlap(box, otherBox)).toBe(false);
        }
      }
    }
  });

  it("projects table points through the shared camera so closer geometry is larger", () => {
    const projectedTable = projectTablePolygon(tableDesignMockLayout.tableSurface);
    const projectedTableTop = Math.min(...projectedTable.map((point) => point.y));
    const farSegment = [
      projectTablePoint({ x: 1000, y: 620 }),
      projectTablePoint({ x: 1200, y: 620 })
    ];
    const nearSegment = [
      projectTablePoint({ x: 1000, y: 1420 }),
      projectTablePoint({ x: 1200, y: 1420 })
    ];

    expect(projectedTable).toHaveLength(5);
    expect(tableDesignMockLayout.camera).toMatchObject({
      focalLength: 2300,
      position: { x: 1120, y: 2450, z: 2750 },
      screenCenter: { x: 1100, y: 671.144 },
      target: { x: 1120, y: 910, z: 0 }
    });
    const farLength = distanceBetween(farSegment[0], farSegment[1]);
    const nearLength = distanceBetween(nearSegment[0], nearSegment[1]);

    expect(projectedTableTop).toBeCloseTo(0);
    expect(nearLength).toBeGreaterThan(farLength);
    expect(nearLength / farLength).toBeLessThan(1.2);
  });

  it("defines all river geometry from the corresponding table-surface edge", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;
    const expected = {
      "top-left": { d: 1481.219, rotation: 144 },
      "top-right": { d: 1481.219, rotation: -144 },
      right: { d: 1481.219, rotation: -72 },
      self: { d: 1481.219, rotation: 0 },
      left: { d: 1481.219, rotation: 72 }
    } as const;
    const sideLengths = seatIds.map((seatId) => createTableSurfaceEdgeGeometry(tableDesignMockLayout, seatId).d);

    for (const seatId of seatIds) {
      const edge = createTableSurfaceEdgeGeometry(tableDesignMockLayout, seatId);
      const river = createRiverGeometry(tableDesignMockLayout, seatId);
      const expectedGridWidth = edge.d * tableDesignMockLayout.riverGrid.widthRatio;
      const expectedCardWidth =
        (river.width - tableDesignMockLayout.riverGrid.columnGap * (tableDesignMockLayout.riverGrid.maxColumns - 1)) /
        tableDesignMockLayout.riverGrid.maxColumns;
      const oldCellHeight = river.cardSize.height * tableDesignMockLayout.riverGrid.cardExposureRatio;
      const newCellHeight = oldCellHeight * tableDesignMockLayout.riverGrid.cellHeightRatio;

      expect(river.d).toBeCloseTo(edge.d);
      expect(river.d).toBeCloseTo(expected[seatId].d);
      expect(river.rotation).toBeCloseTo(expected[seatId].rotation);
      expect(river.width).toBeCloseTo(expectedGridWidth);
      expect(river.cardSize.width).toBeCloseTo(expectedCardWidth);
      expect(river.cardSize.height).toBeCloseTo(river.cardSize.width * 7 / 5);
      expect(river.visibleCardSize.height).toBeCloseTo(newCellHeight);
      expect(river.visibleCardSize.height / oldCellHeight).toBeCloseTo(1.5);
      expect(river.rowPitch).toBeCloseTo(river.visibleCardSize.height + tableDesignMockLayout.riverGrid.rowGap);
      const face = createRiverFaceMetrics(river.visibleCardSize);

      expect(face.rankFontSize).toBeCloseTo(river.visibleCardSize.height * tableDesignMockLayout.riverFace.rankFontRatio);
      expect(face.suitFontSize).toBeCloseTo(river.visibleCardSize.height * tableDesignMockLayout.riverFace.suitFontRatio);
      expect(face.gap).toBeCloseTo(river.visibleCardSize.width * tableDesignMockLayout.riverFace.gapRatio);
      expect(face.padding).toBeCloseTo(Math.min(river.visibleCardSize.width, river.visibleCardSize.height) * tableDesignMockLayout.riverFace.paddingRatio);
      expect(river.normal.x * edge.direction.x + river.normal.y * edge.direction.y).toBeCloseTo(0);

      const riverOuterStart = {
        x: river.x,
        y: river.y
      };
      const riverOuterEnd = {
        x: riverOuterStart.x + river.direction.x * river.width,
        y: riverOuterStart.y + river.direction.y * river.width
      };
      const riverInnerStart = {
        x: riverOuterStart.x - river.normal.x * river.height,
        y: riverOuterStart.y - river.normal.y * river.height
      };

      expect(distanceAlongNormal(edge, riverOuterStart)).toBeCloseTo(0);
      expect(distanceAlongNormal(edge, riverOuterEnd)).toBeCloseTo(0);
      expect(distanceAlongNormal(edge, riverInnerStart)).toBeCloseTo(-river.height);
    }

    expect(new Set(sideLengths)).toEqual(new Set([1481.219]));
    expect(selfRiverWidth(tableDesignMockLayout)).toBeCloseTo(
      createTableSurfaceEdgeGeometry(tableDesignMockLayout, "self").d * tableDesignMockLayout.riverGrid.widthRatio
    );
    expect(selfRiverWidth(tableDesignMockLayout)).toBeGreaterThan(roleBoardSelfSideLength(tableDesignMockLayout.center));
  });

  it("keeps the river outer boundary anchored while cell height ratio changes", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;

    for (const seatId of seatIds) {
      const edge = createTableSurfaceEdgeGeometry(tableDesignMockLayout, seatId);
      const baseRiver = createRiverGeometry({
        ...tableDesignMockLayout,
        riverGrid: { ...tableDesignMockLayout.riverGrid, cellHeightRatio: 1 }
      }, seatId);

      for (const cellHeightRatio of [1, 1.5, 2]) {
        const layout = {
          ...tableDesignMockLayout,
          riverGrid: { ...tableDesignMockLayout.riverGrid, cellHeightRatio }
        };
        const river = createRiverGeometry(layout, seatId);
        const outerStart = { x: river.x, y: river.y };
        const outerEnd = {
          x: river.x + river.direction.x * river.width,
          y: river.y + river.direction.y * river.width
        };
        const innerStart = {
          x: river.x - river.normal.x * river.height,
          y: river.y - river.normal.y * river.height
        };

        expect(river.x).toBeCloseTo(baseRiver.x);
        expect(river.y).toBeCloseTo(baseRiver.y);
        expect(river.width).toBeCloseTo(baseRiver.width);
        expect(distanceAlongNormal(edge, outerStart)).toBeCloseTo(0);
        expect(distanceAlongNormal(edge, outerEnd)).toBeCloseTo(0);
        expect(distanceAlongNormal(edge, innerStart)).toBeCloseTo(-river.height);
        expect(river.height).toBeGreaterThanOrEqual(baseRiver.height);
      }
    }
  });

  it("applies the same 10x2 river placement rule to every seat", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;

    for (const seatId of seatIds) {
      const river = createRiverGeometry(tableDesignMockLayout, seatId);
      const oneCard = createRiverPlacements(1, tableDesignMockLayout, seatId);
      const tenCards = createRiverPlacements(10, tableDesignMockLayout, seatId);
      const elevenCards = createRiverPlacements(11, tableDesignMockLayout, seatId);
      const maxCards = createRiverPlacements(21, tableDesignMockLayout, seatId);
      const columnOffset = river.visibleCardSize.width + tableDesignMockLayout.riverGrid.columnGap;

      expect(oneCard).toHaveLength(1);
      expect(oneCard[0]).toMatchObject({ x: 0, rotation: 0 });
      expect(oneCard[0]?.y).toBeCloseTo(-river.visibleCardSize.height);
      expect(tenCards).toHaveLength(10);
      expect(tenCards[1]?.x).toBeCloseTo(columnOffset);
      expect(tenCards[9]?.x).toBeCloseTo(columnOffset * 9);
      expect(tenCards[0]?.y).toBeCloseTo(-river.visibleCardSize.height);
      expect(tenCards[9]?.y).toBeCloseTo(-river.visibleCardSize.height);
      expect((tenCards[9]?.x ?? 0) + river.visibleCardSize.width - (tenCards[0]?.x ?? 0)).toBeCloseTo(
        river.width
      );
      expect(elevenCards[10]?.x).toBeCloseTo(0);
      expect(elevenCards[10]?.y).toBeCloseTo(-river.visibleCardSize.height - river.rowPitch);
      expect(maxCards).toHaveLength(20);
    }
  });

  it("calculates opponent card metrics from each table-surface edge length", () => {
    const seatIds = ["top-left", "top-right", "right", "left"] as const;
    const edgeLengths = new Set<number>();

    for (const seatId of seatIds) {
      const edge = createTableSurfaceEdgeGeometry(tableDesignMockLayout, seatId);
      const metrics = createOpponentHandCardMetrics(edge.d, tableDesignMockLayout.opponentHand.cardGapRatio);

      edgeLengths.add(edge.d);
      expect(metrics.edgeLength).toBe(edge.d);
      expect(metrics.cardSize.width).toBeCloseTo(edge.d * 0.08);
      expect(metrics.gap).toBeCloseTo(edge.d * 0.02);
      expect(metrics.cardSize.height).toBeCloseTo(edge.d * 0.112);
      expect(metrics.cardSize.height).toBeCloseTo(metrics.cardSize.width * 7 / 5);
      expect(opponentHandWidth(1, metrics)).toBeCloseTo(edge.d * 0.08);
      expect(opponentHandWidth(5, metrics)).toBeCloseTo(edge.d * 0.48);
      expect(opponentHandWidth(10, metrics)).toBeCloseTo(edge.d * 0.98);
    }

    expect(edgeLengths).toEqual(new Set([createTableSurfaceEdgeGeometry(tableDesignMockLayout, "top-left").d]));
  });

  it("generates opponent hands as vertical card planes from table-surface edges", () => {
    const seatIds = ["top-left", "top-right", "right", "left"] as const;
    const expectedCounts = {
      "top-left": 1,
      "top-right": 5,
      right: 10,
      left: 7
    } as const;
    const hands = createOpponentHandsGeometry(tableDesignMockLayout);

    expect(hands.map((hand) => hand.seatId)).toEqual(seatIds);

    for (const seatId of seatIds) {
      const edge = createTableSurfaceEdgeGeometry(tableDesignMockLayout, seatId);
      const hand = createOpponentHandGeometry(tableDesignMockLayout, seatId);
      const metrics = createOpponentHandCardMetrics(edge.d, tableDesignMockLayout.opponentHand.cardGapRatio);
      const edgeMidpoint = midpointBetween(edge.start, edge.end);
      const expectedBaseline = {
        x: edgeMidpoint.x + edge.normal.x * tableDesignMockLayout.opponentHand.baselineOffset,
        y: edgeMidpoint.y + edge.normal.y * tableDesignMockLayout.opponentHand.baselineOffset
      };
      const firstCard = hand.cards[0];
      const lastCard = hand.cards[hand.cards.length - 1];

      expect(hand.edge).toEqual(edge);
      expect(hand.baseline.center.x).toBeCloseTo(expectedBaseline.x);
      expect(hand.baseline.center.y).toBeCloseTo(expectedBaseline.y);
      expect(hand.baseline.offset).toBe(tableDesignMockLayout.opponentHand.baselineOffset);
      expect(hand.cardSize.width).toBeCloseTo(metrics.cardSize.width);
      expect(hand.cardSize.height).toBeCloseTo(metrics.cardSize.height);
      expect(hand.gap).toBeCloseTo(metrics.gap);
      expect(hand.handWidth).toBeCloseTo(opponentHandWidth(expectedCounts[seatId], metrics));
      expect(hand.cards).toHaveLength(expectedCounts[seatId]);
      expect(firstCard).toBeDefined();
      expect(lastCard).toBeDefined();

      if (firstCard === undefined || lastCard === undefined) {
        throw new Error(`Expected vertical cards for ${seatId}`);
      }

      expect(distanceAlongNormal(edge, hand.baseline.center)).toBeGreaterThan(0);
      expect(hand.handWidth).toBeLessThan(edge.d);
      expect(edge.d - hand.handWidth).toBeCloseTo(
        edge.d * (1 - (expectedCounts[seatId] * 0.08 + (expectedCounts[seatId] - 1) * 0.02))
      );
      expect(distanceBetween(firstCard.leftBottom, lastCard.rightBottom)).toBeCloseTo(
        hand.handWidth
      );

      for (const card of hand.cards) {
        expect(card.leftBottom.z).toBe(0);
        expect(card.rightBottom.z).toBe(0);
        expect(card.leftTop.z).toBe(hand.cardSize.height);
        expect(card.rightTop.z).toBe(hand.cardSize.height);
        expect(card.leftTop.x).toBe(card.leftBottom.x);
        expect(card.leftTop.y).toBe(card.leftBottom.y);
        expect(card.rightTop.x).toBe(card.rightBottom.x);
        expect(card.rightTop.y).toBe(card.rightBottom.y);
        expect(distanceBetween(card.leftBottom, card.rightBottom)).toBeCloseTo(
          hand.cardSize.width
        );
      }
    }
  });

  it("projects vertical opponent cards through the shared camera", () => {
    const hand = createOpponentHandGeometry(tableDesignMockLayout, "right");
    const centerCard = hand.cards[4];

    if (centerCard === undefined) {
      throw new Error("Expected a center opponent card");
    }

    const projected = projectVerticalCard(centerCard);
    const [leftTop, rightTop, rightBottom, leftBottom] = projected;

    expect(projected).toHaveLength(4);
    expect(leftTop).toBeDefined();
    expect(rightTop).toBeDefined();
    expect(rightBottom).toBeDefined();
    expect(leftBottom).toBeDefined();

    if (
      leftTop === undefined ||
      rightTop === undefined ||
      rightBottom === undefined ||
      leftBottom === undefined
    ) {
      throw new Error("Expected four projected vertical card corners");
    }

    expect(distanceBetween(leftTop, rightTop)).toBeGreaterThan(0);
    expect(distanceBetween(leftBottom, rightBottom)).toBeGreaterThan(0);
    expect(Math.max(leftTop.y, rightTop.y)).toBeLessThan(Math.max(leftBottom.y, rightBottom.y));
  });

  it("generates every current trick zone as an independent card-sized area between role board and river", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;
    const riverCardCounts = {
      "top-left": 2,
      "top-right": 2,
      right: 3,
      self: 5,
      left: 2
    } as const;
    const expected = {
      "top-left": { rotation: 144 },
      "top-right": { rotation: -144 },
      right: { rotation: -72 },
      self: { rotation: 0 },
      left: { rotation: 72 }
    } as const;
    const issue386CardSize = { width: 244.957, height: 342.94 };
    const zones = seatIds.map((seatId) => createCurrentTrickZoneGeometry(tableDesignMockLayout, seatId));

    for (const seatId of seatIds) {
      const edge = createTableSurfaceEdgeGeometry(tableDesignMockLayout, seatId);
      const roleEdge = createRoleBoardEdgeGeometry(tableDesignMockLayout.center, seatId);
      const river = createRiverGeometry(tableDesignMockLayout, seatId);
      const currentTrickReferenceRiver = createCurrentTrickReferenceRiverGeometry(tableDesignMockLayout, seatId);
      const zone = createCurrentTrickZoneGeometry(tableDesignMockLayout, seatId, riverCardCounts[seatId]);
      const cardSize = createCurrentTrickCardSize(tableDesignMockLayout, seatId);
      const cardPlane = createCurrentTrickCardPlane(tableDesignMockLayout, seatId);
      const roleEdgeCenter = midpointBetween(roleEdge.start, roleEdge.end);
      const riverInnerEdgeCenter = {
        x: currentTrickReferenceRiver.x + currentTrickReferenceRiver.direction.x * (currentTrickReferenceRiver.width / 2) - currentTrickReferenceRiver.normal.x * currentTrickReferenceRiver.height,
        y: currentTrickReferenceRiver.y + currentTrickReferenceRiver.direction.y * (currentTrickReferenceRiver.width / 2) - currentTrickReferenceRiver.normal.y * currentTrickReferenceRiver.height
      };
      const actualRiverInnerEdgeCenter = {
        x: river.x + river.direction.x * (river.width / 2) - river.normal.x * river.height,
        y: river.y + river.direction.y * (river.width / 2) - river.normal.y * river.height
      };
      const availableDepth = distanceBetween(roleEdgeCenter, riverInnerEdgeCenter);
      const zoneInnerEdgeCenter = {
        x: zone.x - zone.normal.x * (zone.height / 2),
        y: zone.y - zone.normal.y * (zone.height / 2)
      };
      const zoneOuterEdgeCenter = {
        x: zone.x + zone.normal.x * (zone.height / 2),
        y: zone.y + zone.normal.y * (zone.height / 2)
      };
      const cardInnerEdgeCenter = {
        x: cardPlane.x - cardPlane.normal.x * (cardPlane.height / 2),
        y: cardPlane.y - cardPlane.normal.y * (cardPlane.height / 2)
      };
      const cardOuterEdgeCenter = {
        x: cardPlane.x + cardPlane.normal.x * (cardPlane.height / 2),
        y: cardPlane.y + cardPlane.normal.y * (cardPlane.height / 2)
      };
      const roleGap =
        distanceAlongNormal(edge, zoneInnerEdgeCenter) - distanceAlongNormal(edge, roleEdgeCenter);
      const riverGap =
        distanceAlongNormal(edge, riverInnerEdgeCenter) - distanceAlongNormal(edge, zoneOuterEdgeCenter);
      const actualRiverGap =
        distanceAlongNormal(edge, actualRiverInnerEdgeCenter) - distanceAlongNormal(edge, zoneOuterEdgeCenter);

      expect(zone.rotation).toBeCloseTo(edge.rotation);
      expect(zone.rotation).toBeCloseTo(expected[seatId].rotation);
      expect(cardSize.width).toBeCloseTo(roleEdge.d * tableDesignMockLayout.currentTrickZone.cardWidthRatio);
      expect(cardSize.width / issue386CardSize.width).toBeGreaterThanOrEqual(1.35);
      expect(cardSize.width / issue386CardSize.width).toBeLessThanOrEqual(1.5);
      expect(cardSize.height).toBeCloseTo(cardSize.width * 7 / 5);
      expect(cardSize.height / issue386CardSize.height).toBeGreaterThanOrEqual(1.35);
      expect(cardSize.height / issue386CardSize.height).toBeLessThanOrEqual(1.5);
      expect(cardSize.width).toBeGreaterThan(river.cardSize.width);
      expect(zone.width).toBeCloseTo(cardSize.width * tableDesignMockLayout.currentTrickZone.zoneToCardRatio);
      expect(zone.height).toBeCloseTo(cardSize.height * tableDesignMockLayout.currentTrickZone.zoneToCardRatio);
      expect(zone.width / cardSize.width).toBeCloseTo(tableDesignMockLayout.currentTrickZone.zoneToCardRatio);
      expect(zone.height / cardSize.height).toBeCloseTo(tableDesignMockLayout.currentTrickZone.zoneToCardRatio);
      expect(zone.height).toBeLessThan(availableDepth);
      expect(zone.height / availableDepth).toBeLessThan(0.85);
      expect(zone.width).toBeGreaterThan(roleEdge.d);
      expect(roleGap).toBeGreaterThan(0);
      expect(riverGap).toBeGreaterThan(0);
      expect(actualRiverGap).toBeGreaterThan(0);
      expect(roleGap).toBeCloseTo(riverGap);
      expect(roleGap).toBeCloseTo((availableDepth - zone.height) / 2);
      expect(cardPlane.width).toBe(cardSize.width);
      expect(cardPlane.height).toBe(cardSize.height);
      expect(cardPlane.x).toBe(zone.x);
      expect(cardPlane.y).toBe(zone.y);
      expect(distanceAlongNormal(edge, cardInnerEdgeCenter)).toBeGreaterThan(distanceAlongNormal(edge, zoneInnerEdgeCenter));
      expect(distanceAlongNormal(edge, cardOuterEdgeCenter)).toBeLessThan(distanceAlongNormal(edge, zoneOuterEdgeCenter));
      expect(distanceAlongNormal(edge, cardInnerEdgeCenter)).toBeGreaterThan(distanceAlongNormal(edge, roleEdgeCenter));
      expect(distanceAlongNormal(edge, cardOuterEdgeCenter)).toBeLessThan(
        distanceAlongNormal(edge, riverInnerEdgeCenter)
      );
    }

    for (let i = 0; i < zones.length; i += 1) {
      for (let j = i + 1; j < zones.length; j += 1) {
        const a = zones[i];
        const b = zones[j];

        if (a === undefined || b === undefined) {
          throw new Error("Expected all current trick zones");
        }

        const aRadius = Math.hypot(a.width, a.height) / 2;
        const bRadius = Math.hypot(b.width, b.height) / 2;

        expect(distanceBetween(a, b)).toBeGreaterThan(aRadius + bRadius);
      }
    }
  });

  it("keeps current trick geometry unchanged when river cell height ratio changes", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;
    const baseline = Object.fromEntries(
      seatIds.map((seatId) => [
        seatId,
        {
          card: createCurrentTrickCardPlane(tableDesignMockLayout, seatId),
          cardSize: createCurrentTrickCardSize(tableDesignMockLayout, seatId),
          zone: createCurrentTrickZoneGeometry(tableDesignMockLayout, seatId)
        }
      ])
    ) as Record<typeof seatIds[number], {
      card: ReturnType<typeof createCurrentTrickCardPlane>;
      cardSize: ReturnType<typeof createCurrentTrickCardSize>;
      zone: ReturnType<typeof createCurrentTrickZoneGeometry>;
    }>;

    for (const cellHeightRatio of [1, 1.5, 2]) {
      const layout = {
        ...tableDesignMockLayout,
        riverGrid: { ...tableDesignMockLayout.riverGrid, cellHeightRatio }
      };

      for (const seatId of seatIds) {
        expect(createCurrentTrickZoneGeometry(layout, seatId)).toEqual(baseline[seatId].zone);
        expect(createCurrentTrickCardSize(layout, seatId)).toEqual(baseline[seatId].cardSize);
        expect(createCurrentTrickCardPlane(layout, seatId)).toEqual(baseline[seatId].card);
      }
    }
  });

  it("generates role markers inside the five sectors between the outer and inner pentagons", () => {
    const expected = {
      "top-left": { x: 208.648, y: 168.619 },
      "top-right": { x: 421.352, y: 168.619 },
      right: { x: 487.081, y: 370.913 },
      self: { x: 315, y: 495.937 },
      left: { x: 142.919, y: 370.913 }
    } as const;

    for (const seatId of ["top-left", "top-right", "right", "self", "left"] as const) {
      const marker = createRoleMarkerGeometry(tableDesignMockLayout.center, seatId);
      const sector = createRoleSectorGeometry(tableDesignMockLayout.center, seatId);
      const outerMidpoint = midpointBetween(sector.outerStart, sector.outerEnd);
      const innerMidpoint = midpointBetween(sector.innerStart, sector.innerEnd);

      expect(marker.width).toBe(104.4);
      expect(marker.height).toBe(61.2);
      expect(marker.x).toBeCloseTo(expected[seatId].x);
      expect(marker.y).toBeCloseTo(expected[seatId].y);
      expect(marker.x).toBeCloseTo((outerMidpoint.x + innerMidpoint.x) / 2);
      expect(marker.y).toBeCloseTo((outerMidpoint.y + innerMidpoint.y) / 2);
      expect(marker.sector).toEqual(sector);
    }
  });
});

function distanceAlongNormal(edge: { normal: { x: number; y: number }; start: { x: number; y: number } }, point: {
  x: number;
  y: number;
}): number {
  return (point.x - edge.start.x) * edge.normal.x + (point.y - edge.start.y) * edge.normal.y;
}

function distanceBetween(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpointBetween(a: { x: number; y: number }, b: { x: number; y: number }): { x: number; y: number } {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2
  };
}

function projectedSectorAlignment(
  point: { x: number; y: number },
  sector: {
    center: { x: number; y: number };
    end: { x: number; y: number };
    start: { x: number; y: number };
  }
): number {
  const start = subtractPoints(sector.start, sector.center);
  const end = subtractPoints(sector.end, sector.center);
  const candidate = subtractPoints(point, sector.center);
  const direction = normalizeSectorVector({
    x: (start.x + end.x) / 2,
    y: (start.y + end.y) / 2
  });
  const candidateDirection = normalizeSectorVector(candidate);

  return direction.x * candidateDirection.x + direction.y * candidateDirection.y;
}

function subtractPoints(
  a: { x: number; y: number },
  b: { x: number; y: number }
): { x: number; y: number } {
  return { x: a.x - b.x, y: a.y - b.y };
}

function normalizeSectorVector(vector: { x: number; y: number }): { x: number; y: number } {
  const length = Math.hypot(vector.x, vector.y);

  return { x: vector.x / length, y: vector.y / length };
}

function boxFromCenter(box: {
  height: number;
  width: number;
  x: number;
  y: number;
}): { bottom: number; height: number; left: number; right: number; top: number; width: number; x: number; y: number } {
  return {
    bottom: box.y + box.height / 2,
    height: box.height,
    left: box.x - box.width / 2,
    right: box.x + box.width / 2,
    top: box.y - box.height / 2,
    width: box.width,
    x: box.x,
    y: box.y
  };
}

function boxFromTopLeft(box: {
  height: number;
  width: number;
  x: number;
  y: number;
}): { bottom: number; height: number; left: number; right: number; top: number; width: number; x: number; y: number } {
  return {
    bottom: box.y + box.height,
    height: box.height,
    left: box.x,
    right: box.x + box.width,
    top: box.y,
    width: box.width,
    x: box.x + box.width / 2,
    y: box.y + box.height / 2
  };
}

function boxesOverlap(
  a: { bottom: number; left: number; right: number; top: number },
  b: { bottom: number; left: number; right: number; top: number }
): boolean {
  return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
}

function distanceFromTestBox(
  point: { x: number; y: number },
  box: { bottom: number; left: number; right: number; top: number }
): number {
  const dx = Math.max(box.left - point.x, 0, point.x - box.right);
  const dy = Math.max(box.top - point.y, 0, point.y - box.bottom);

  return Math.hypot(dx, dy);
}

function boundingTestBox(points: readonly { x: number; y: number }[]): {
  bottom: number;
  height: number;
  left: number;
  right: number;
  top: number;
  width: number;
  x: number;
  y: number;
} {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);
  const width = right - left;
  const height = bottom - top;

  return {
    bottom,
    height,
    left,
    right,
    top,
    width,
    x: left + width / 2,
    y: top + height / 2
  };
}

function transformTestBox(
  box: { bottom: number; left: number; right: number; top: number },
  scale: number,
  translate: { x: number; y: number }
): {
  bottom: number;
  height: number;
  left: number;
  right: number;
  top: number;
  width: number;
  x: number;
  y: number;
} {
  const left = box.left * scale + translate.x;
  const right = box.right * scale + translate.x;
  const top = box.top * scale + translate.y;
  const bottom = box.bottom * scale + translate.y;
  const width = right - left;
  const height = bottom - top;

  return {
    bottom,
    height,
    left,
    right,
    top,
    width,
    x: left + width / 2,
    y: top + height / 2
  };
}

function riverCardsTestBox(seatId: "left" | "right" | "top-left" | "top-right", cardCount: number): {
  bottom: number;
  height: number;
  left: number;
  right: number;
  top: number;
  width: number;
  x: number;
  y: number;
} {
  const river = createRiverGeometry(tableDesignMockLayout, seatId);
  const placements = createRiverPlacements(cardCount, tableDesignMockLayout, seatId);

  return boundingTestBox(placements.flatMap((placement) => tableCardTestCorners({
    direction: river.direction,
    height: river.visibleCardSize.height,
    normal: river.normal,
    width: river.visibleCardSize.width,
    x: river.x + river.direction.x * placement.x + river.normal.x * placement.y,
    y: river.y + river.direction.y * placement.x + river.normal.y * placement.y
  })));
}

function projectedRiverCardsTestBox(seatId: "left" | "right" | "top-left" | "top-right", cardCount: number): {
  bottom: number;
  height: number;
  left: number;
  right: number;
  top: number;
  width: number;
  x: number;
  y: number;
} {
  const river = createRiverGeometry(tableDesignMockLayout, seatId);
  const placements = createRiverPlacements(cardCount, tableDesignMockLayout, seatId);

  return boundingTestBox(placements.flatMap((placement) => projectTableCard({
    direction: river.direction,
    height: river.visibleCardSize.height,
    normal: river.normal,
    width: river.visibleCardSize.width,
    x: river.x + river.direction.x * placement.x + river.normal.x * placement.y,
    y: river.y + river.direction.y * placement.x + river.normal.y * placement.y
  }, tableDesignMockLayout.camera, "top-left")));
}

function tableCardTestCorners(card: {
  direction: { x: number; y: number };
  height: number;
  normal: { x: number; y: number };
  width: number;
  x: number;
  y: number;
}): Array<{ x: number; y: number }> {
  return [
    { x: card.x, y: card.y },
    { x: card.x + card.direction.x * card.width, y: card.y + card.direction.y * card.width },
    {
      x: card.x + card.direction.x * card.width + card.normal.x * card.height,
      y: card.y + card.direction.y * card.width + card.normal.y * card.height
    },
    { x: card.x + card.normal.x * card.height, y: card.y + card.normal.y * card.height }
  ];
}

function normalizeTestVector(vector: { x: number; y: number }): { x: number; y: number } {
  const length = Math.hypot(vector.x, vector.y);

  if (length === 0) {
    return { x: 0, y: -1 };
  }

  return {
    x: vector.x / length,
    y: vector.y / length
  };
}
