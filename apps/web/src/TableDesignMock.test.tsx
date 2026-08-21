import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  TableDesignMock,
  createCurrentTrickZoneGeometry,
  createOpponentHandCardMetrics,
  createOpponentHandGeometry,
  createOpponentHandsGeometry,
  createProjectedBoardFit,
  createProjectedTableBoundingBox,
  createRiverGeometry,
  createRiverPlacements,
  createRoleMarkerGeometry,
  createRoleBoardEdgeGeometry,
  createRoleSectorGeometry,
  createSelfHandViewportLayout,
  createSelfHandViewportMetrics,
  createTableSurfaceEdgeGeometry,
  opponentHandWidth,
  projectVerticalCard,
  projectTablePoint,
  projectTablePolygon,
  regularPentagon,
  roleBoardSelfSideLength,
  selfHandWidth,
  selfRiverWidth,
  tableDesignMockLayout
} from "./TableDesignMock";

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
    expect(html).toContain("--mock-trick-card-width:212.4px");
    expect(html).toContain("--mock-trick-card-height:297.36px");
    expect(html).toContain("--mock-trick-zone-width:291.6px");
    expect(html).toContain("--mock-trick-zone-height:390.96px");
    expect(html).not.toContain("--mock-river-card-width:56px");
    expect(html).not.toContain("mock-player-label");
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
    expect(tableDesignMockLayout.riverGrid).toMatchObject({ maxColumns: 5, maxRows: 4 });
    expect((html.match(/mock-current-trick-zone/g) ?? [])).toHaveLength(10);
    expect((html.match(/mock-trick-card mock-playing-card/g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-self-hand-card/g) ?? [])).toHaveLength(13);
    expect((html.match(/mock-playing-card-svg/g) ?? [])).toHaveLength(32);
    expect((html.match(/mock-opponent-hand-world-card/g) ?? [])).toHaveLength(23);
    expect((html.match(/aria-label="B1"/g) ?? [])).toHaveLength(0);
    expect(html).not.toContain("mock-card-back-fan");
    expect((html.match(/class="role-marker /g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-role-board-sector-line/g) ?? [])).toHaveLength(5);
    expect(html).toContain("mock-role-board-inner-pentagon");
    expect(html).not.toContain("role-cell");
    expect(html).not.toContain("mock-card-corner");
    expect(html).not.toContain("mock-card-face");
    expect(html).toContain("aria-label=\"Sj\"");
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
    expect((html.match(/mock-projected-playing-card /g) ?? [])).toHaveLength(42);
    expect((html.match(/mock-projected-playing-card-opponent-hand/g) ?? [])).toHaveLength(23);
    expect((html.match(/matrix3d\(/g) ?? [])).toHaveLength(42);
    expect((html.match(/aria-label="B1"/g) ?? [])).toHaveLength(23);
    expect(html).toContain("mock-projected-card-layer");
    expect(html).not.toContain("mock-projected-card-corner");
    expect(html).not.toContain("mock-projected-card-face");
    expect(html).not.toContain("mock-current-trick-zone mock-current-trick-zone");
    expect(html).not.toContain("mock-point-river mock-point-river");
    expect(html).toContain("北西の裏向き手札");
    expect(html).toContain("自分の表向き手札");
    expect((html.match(/mock-self-hand-card/g) ?? [])).toHaveLength(13);
    expect(html).toContain('aria-label="自分 プレイヤー" class="mock-avatar mock-avatar-self" style="--mock-x:808px;--mock-y:1492px');
  });

  it("lays out the self hand as viewport-width 2D UI for up to thirteen cards", () => {
    const viewportWidth = tableDesignMockLayout.page.width;
    const metrics = createSelfHandViewportMetrics(viewportWidth);
    const fullHand = createSelfHandViewportLayout(tableDesignMockLayout, 13);
    const reducedHand = createSelfHandViewportLayout(tableDesignMockLayout, 5);

    expect(metrics.cardSize.width).toBeCloseTo((0.8 * viewportWidth) / 13);
    expect(metrics.cardSize.height).toBeCloseTo(metrics.cardSize.width * 7 / 5);
    expect(metrics.gap).toBeCloseTo((0.16 * viewportWidth) / 12);
    expect(selfHandWidth(13, metrics)).toBeCloseTo(0.96 * viewportWidth);
    expect(fullHand.handWidth).toBeCloseTo(0.96 * viewportWidth);
    expect(fullHand.left).toBeCloseTo(0.02 * viewportWidth);
    expect(tableDesignMockLayout.page.height - fullHand.bottom).toBe(tableDesignMockLayout.selfHandUi.bottomInset);
    expect(fullHand.top).toBeGreaterThan(0);
    expect(fullHand.bottom).toBeLessThanOrEqual(tableDesignMockLayout.page.height);

    expect(reducedHand.cardSize.width).toBe(fullHand.cardSize.width);
    expect(reducedHand.cardSize.height).toBe(fullHand.cardSize.height);
    expect(reducedHand.gap).toBe(fullHand.gap);
    expect(reducedHand.handWidth).toBeLessThan(fullHand.handWidth);
    expect(reducedHand.center.x).toBeCloseTo(viewportWidth / 2);
  });

  it("keeps the self hand viewport-width based when projected board fit changes", () => {
    const fullHdHand = createSelfHandViewportLayout(tableDesignMockLayout, 13, { width: 1920, height: 1080 });
    const qhdHand = createSelfHandViewportLayout(tableDesignMockLayout, 13, { width: 2560, height: 1440 });

    expect(fullHdHand.cardSize.width).toBeCloseTo((0.8 * 1920) / 13);
    expect(fullHdHand.cardSize.height).toBeCloseTo(fullHdHand.cardSize.width * 7 / 5);
    expect(fullHdHand.gap).toBeCloseTo((0.16 * 1920) / 12);
    expect(fullHdHand.handWidth).toBeCloseTo(0.96 * 1920);
    expect(fullHdHand.center.x).toBeCloseTo(960);
    expect(fullHdHand.bottom).toBe(1064);

    expect(qhdHand.cardSize.width).toBeCloseTo((0.8 * 2560) / 13);
    expect(qhdHand.handWidth).toBeCloseTo(0.96 * 2560);
    expect(qhdHand.center.x).toBeCloseTo(1280);
    expect(qhdHand.cardSize.width).toBeGreaterThan(fullHdHand.cardSize.width);
  });

  it("fits projected table scale from viewport height and keeps width growth from enlarging it", () => {
    const rawBox = createProjectedTableBoundingBox(tableDesignMockLayout);
    const fullHd = createProjectedBoardFit(tableDesignMockLayout, { width: 1920, height: 1080 });
    const widerSameHeight = createProjectedBoardFit(tableDesignMockLayout, { width: 2560, height: 1080 });
    const qhd = createProjectedBoardFit(tableDesignMockLayout, { width: 2560, height: 1440 });

    expect(rawBox.height).toBeGreaterThan(0);
    expect(fullHd.scale).toBeCloseTo((1080 * tableDesignMockLayout.projectedFit.tableHeightRatio) / rawBox.height);
    expect(fullHd.transformedTableBox.height).toBeCloseTo(1080 * tableDesignMockLayout.projectedFit.tableHeightRatio);
    expect(fullHd.transformedTableBox.top).toBeCloseTo(1080 * 0.015);
    expect(fullHd.transformedTableBox.x).toBeCloseTo(960);
    expect(fullHd.transformedTableBox.width).toBeLessThan(1920);

    expect(widerSameHeight.scale).toBeCloseTo(fullHd.scale);
    expect(widerSameHeight.transformedTableBox.height).toBeCloseTo(fullHd.transformedTableBox.height);
    expect(widerSameHeight.transformedTableBox.width).toBeCloseTo(fullHd.transformedTableBox.width);
    expect(widerSameHeight.transformedTableBox.x).toBeCloseTo(1280);

    expect(qhd.scale).toBeGreaterThan(fullHd.scale);
    expect(qhd.transformedTableBox.height).toBeCloseTo(1440 * tableDesignMockLayout.projectedFit.tableHeightRatio);
    expect(qhd.transformedTableBox.top).toBeCloseTo(1440 * 0.015);
    expect(qhd.transformedTableBox.x).toBeCloseTo(1280);
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

  it("defines all river geometry from the corresponding role-board edge", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;
    const expected = {
      "top-left": { d: 370.305, rotation: 144 },
      "top-right": { d: 370.305, rotation: -144 },
      right: { d: 370.305, rotation: -72 },
      self: { d: 370.305, rotation: 0 },
      left: { d: 370.305, rotation: 72 }
    } as const;
    const sideLengths = seatIds.map((seatId) => createRoleBoardEdgeGeometry(tableDesignMockLayout.center, seatId).d);

    for (const seatId of seatIds) {
      const edge = createRoleBoardEdgeGeometry(tableDesignMockLayout.center, seatId);
      const river = createRiverGeometry(tableDesignMockLayout, seatId);

      expect(river.d).toBeCloseTo(edge.d);
      expect(river.d).toBeCloseTo(expected[seatId].d);
      expect(river.rotation).toBeCloseTo(expected[seatId].rotation);
      expect(river.cardSize.width).toBeCloseTo(river.d * 0.5);
      expect(river.width).toBeCloseTo(river.d);
      expect(river.normal.x * edge.direction.x + river.normal.y * edge.direction.y).toBeCloseTo(0);
    }

    expect(new Set(sideLengths)).toEqual(new Set([370.305]));
    expect(selfRiverWidth(tableDesignMockLayout)).toBeCloseTo(roleBoardSelfSideLength(tableDesignMockLayout.center));
  });

  it("applies the same 5x4 river placement rule to every seat", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;

    for (const seatId of seatIds) {
      const river = createRiverGeometry(tableDesignMockLayout, seatId);
      const oneCard = createRiverPlacements(1, tableDesignMockLayout, seatId);
      const fiveCards = createRiverPlacements(5, tableDesignMockLayout, seatId);
      const sixCards = createRiverPlacements(6, tableDesignMockLayout, seatId);
      const maxCards = createRiverPlacements(21, tableDesignMockLayout, seatId);
      const columnOffset = river.d * 0.125;

      expect(oneCard).toHaveLength(1);
      expect(oneCard[0]).toMatchObject({ x: 0, y: 0, rotation: 0 });
      expect(fiveCards).toHaveLength(5);
      expect(fiveCards[1]?.x).toBeCloseTo(columnOffset);
      expect(fiveCards[4]?.x).toBeCloseTo(columnOffset * 4);
      expect((fiveCards[4]?.x ?? 0) + river.cardSize.width - (fiveCards[0]?.x ?? 0)).toBeCloseTo(
        river.d
      );
      expect(sixCards[5]?.x).toBeCloseTo(0);
      expect(sixCards[5]?.y).toBeCloseTo(river.rowPitch);
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

  it("generates every current trick zone from the same edge geometry outside the river", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;
    const riverCardCounts = {
      "top-left": 2,
      "top-right": 2,
      right: 3,
      self: 5,
      left: 2
    } as const;
    const expected = {
      "top-left": { rotation: 144, x: 654.278, y: 268.988 },
      "top-right": { rotation: -144, x: 1585.722, y: 268.988 },
      right: { rotation: -72, x: 1873.555, y: 1154.845 },
      self: { rotation: 0, x: 1120, y: 1702.334 },
      left: { rotation: 72, x: 366.445, y: 1154.845 }
    } as const;

    for (const seatId of seatIds) {
      const edge = createRoleBoardEdgeGeometry(tableDesignMockLayout.center, seatId);
      const river = createRiverGeometry(tableDesignMockLayout, seatId);
      const zone = createCurrentTrickZoneGeometry(tableDesignMockLayout, seatId, riverCardCounts[seatId]);

      expect(zone.rotation).toBeCloseTo(edge.rotation);
      expect(zone.rotation).toBeCloseTo(expected[seatId].rotation);
      expect(zone.width).toBeCloseTo(
        tableDesignMockLayout.cardSizes.trick.width + tableDesignMockLayout.currentTrickZone.paddingInline
      );
      expect(zone.height).toBeCloseTo(
        tableDesignMockLayout.cardSizes.trick.height + tableDesignMockLayout.currentTrickZone.paddingBlock
      );
      expect(zone.x).toBeCloseTo(expected[seatId].x);
      expect(zone.y).toBeCloseTo(expected[seatId].y);
      expect(distanceAlongNormal(edge, zone)).toBeGreaterThan(distanceAlongNormal(edge, river));
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
