import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  TableDesignMock,
  createCurrentTrickZoneGeometry,
  createRiverGeometry,
  createRiverPlacements,
  createRoleMarkerGeometry,
  createRoleBoardEdgeGeometry,
  createRoleSectorGeometry,
  projectTablePoint,
  projectTablePolygon,
  regularPentagon,
  roleBoardSelfSideLength,
  selfRiverWidth,
  tableDesignMockLayout
} from "./TableDesignMock";

describe("TableDesignMock", () => {
  it("renders the issue 330 world mock with unprojected tabletop geometry", () => {
    const html = renderToStaticMarkup(<TableDesignMock />);

    expect(tableDesignMockLayout.seats).toHaveLength(5);
    expect(html).toContain("Issue 330 table design world mock");
    expect(html).toContain("契約HUD");
    expect(html).toContain("中央役職表示");
    expect(html).toContain("自分の表向き手札");
    expect(html).toContain("北西の裏向き手札");
    expect(html).toContain("北西の現在トリック置き場");
    expect(html).toContain("自分のポイント札の河");
    expect(html).not.toContain("自席操作UI");
    expect(html).toContain("--mock-page-width:2200px");
    expect(html).toContain("--mock-self-card-height:240.8px");
    expect(html).toContain("--mock-trick-card-width:118px");
    expect(html).toContain("--mock-trick-card-height:165.2px");
    expect(html).toContain("--mock-trick-zone-width:162px");
    expect(html).toContain("--mock-trick-zone-height:217.2px");
    expect(html).not.toContain("--mock-river-card-width:56px");
    expect(html).not.toContain("mock-player-label");
    expect(html).toContain("mock-table-surface-world");
    expect(html).toContain("1120,210 1785.74,693.688 1531.45,1476.312 708.55,1476.312 454.26,693.688");
    expect(tableDesignMockLayout.center).toMatchObject({ height: 350, width: 350, x: 1120, y: 910 });
    expect(tableDesignMockLayout.tableSurface).toHaveLength(5);
    expect(tableDesignMockLayout.tableSurface).toEqual(regularPentagon({ x: 1120, y: 910 }, 700, -90));
    expect(tableDesignMockLayout.seats.find((seat) => seat.id === "self")).toMatchObject({ hand: { y: 1640 } });
    expect(tableDesignMockLayout.seats.some((seat) => "trickZone" in seat)).toBe(false);
    expect(tableDesignMockLayout.riverGrid).toMatchObject({ maxColumns: 5, maxRows: 4 });
    expect((html.match(/mock-current-trick-zone/g) ?? [])).toHaveLength(10);
    expect((html.match(/mock-trick-card mock-playing-card/g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-playing-card-svg/g) ?? [])).toHaveLength(33);
    expect((html.match(/aria-label="B1"/g) ?? [])).toHaveLength(12);
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

  it("renders the issue 330 projected mock from projected tabletop polygons", () => {
    const html = renderToStaticMarkup(<TableDesignMock variant="projected" />);

    expect(html).toContain("Issue 330 table design projected mock");
    expect(html).toContain("投影後の卓上Geometry");
    expect(html).toContain("mock-projected-tabletop");
    expect(html).toContain("mock-table-surface-polygon");
    expect((html.match(/mock-projected-current-trick-zone mock-projected-current-trick-zone-/g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-projected-role-marker mock-projected-role-marker-/g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-projected-playing-card /g) ?? [])).toHaveLength(19);
    expect((html.match(/matrix3d\(/g) ?? [])).toHaveLength(19);
    expect(html).toContain("mock-projected-card-layer");
    expect(html).not.toContain("mock-projected-card-corner");
    expect(html).not.toContain("mock-projected-card-face");
    expect(html).not.toContain("mock-current-trick-zone mock-current-trick-zone");
    expect(html).not.toContain("mock-point-river mock-point-river");
    expect(html).toContain("北西の裏向き手札");
    expect(html).toContain("自分の表向き手札");
  });

  it("projects table points through the shared camera so closer geometry is larger", () => {
    const projectedTable = projectTablePolygon(tableDesignMockLayout.tableSurface);
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
      focalLength: 2150,
      position: { x: 1120, y: 3150, z: 1720 },
      target: { x: 1120, y: 910, z: 0 }
    });
    expect(distanceBetween(nearSegment[0], nearSegment[1])).toBeGreaterThan(
      distanceBetween(farSegment[0], farSegment[1])
    );
  });

  it("defines all river geometry from the corresponding role-board edge", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;
    const expected = {
      "top-left": { d: 205.725, rotation: 144 },
      "top-right": { d: 205.725, rotation: -144 },
      right: { d: 205.725, rotation: -72 },
      self: { d: 205.725, rotation: 0 },
      left: { d: 205.725, rotation: 72 }
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

    expect(new Set(sideLengths)).toEqual(new Set([205.725]));
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
      "top-left": { rotation: 144, x: 861.266, y: 553.883 },
      "top-right": { rotation: -144, x: 1378.734, y: 553.883 },
      right: { rotation: -72, x: 1538.641, y: 1046.025 },
      self: { rotation: 0, x: 1120, y: 1350.185 },
      left: { rotation: 72, x: 701.359, y: 1046.025 }
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
      "top-left": { x: 115.916, y: 93.677 },
      "top-right": { x: 234.084, y: 93.677 },
      right: { x: 270.601, y: 206.063 },
      self: { x: 175, y: 275.52 },
      left: { x: 79.399, y: 206.063 }
    } as const;

    for (const seatId of ["top-left", "top-right", "right", "self", "left"] as const) {
      const marker = createRoleMarkerGeometry(tableDesignMockLayout.center, seatId);
      const sector = createRoleSectorGeometry(tableDesignMockLayout.center, seatId);
      const outerMidpoint = midpointBetween(sector.outerStart, sector.outerEnd);
      const innerMidpoint = midpointBetween(sector.innerStart, sector.innerEnd);

      expect(marker.width).toBe(58);
      expect(marker.height).toBe(34);
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
