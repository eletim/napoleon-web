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
  roleBoardSelfSideLength,
  selfRiverWidth,
  tableDesignMockLayout
} from "./TableDesignMock";

describe("TableDesignMock", () => {
  it("renders the issue 327 static table mock with compact role markers", () => {
    const html = renderToStaticMarkup(<TableDesignMock />);

    expect(tableDesignMockLayout.seats).toHaveLength(5);
    expect(html).toContain("Issue 327 table design mock");
    expect(html).toContain("契約HUD");
    expect(html).toContain("中央役職表示");
    expect(html).toContain("自分の表向き手札");
    expect(html).toContain("北西の裏向き手札");
    expect(html).toContain("北西の現在トリック置き場");
    expect(html).toContain("自分のポイント札の河");
    expect(html).not.toContain("自席操作UI");
    expect(html).toContain("--mock-page-width:2200px");
    expect(html).toContain("--mock-trick-card-width:118px");
    expect(html).toContain("--mock-trick-zone-width:162px");
    expect(html).toContain("--mock-trick-zone-height:211.121px");
    expect(html).not.toContain("--mock-river-card-width:56px");
    expect(html).not.toContain("mock-player-label");
    expect(tableDesignMockLayout.center).toMatchObject({ height: 303, width: 338, y: 890 });
    expect(tableDesignMockLayout.seats.find((seat) => seat.id === "self")).toMatchObject({ hand: { y: 1684 } });
    expect(tableDesignMockLayout.seats.some((seat) => "trickZone" in seat)).toBe(false);
    expect(tableDesignMockLayout.riverGrid).toMatchObject({ maxColumns: 5, maxRows: 4 });
    expect((html.match(/mock-current-trick-zone/g) ?? [])).toHaveLength(10);
    expect((html.match(/mock-trick-card mock-playing-card/g) ?? [])).toHaveLength(5);
    expect((html.match(/class="role-marker /g) ?? [])).toHaveLength(5);
    expect((html.match(/mock-role-board-sector-line/g) ?? [])).toHaveLength(5);
    expect(html).toContain("mock-role-board-inner-pentagon");
    expect(html).not.toContain("role-cell");
    expect(html).toContain("aria-label=\"J♠\"");
    expect(html).toContain(">ナポ</span>");
    expect(html).toContain(">副</span>");
  });

  it("defines all river geometry from the corresponding role-board edge", () => {
    const seatIds = ["top-left", "top-right", "right", "self", "left"] as const;
    const expected = {
      "top-left": { d: 204.495, rotation: 145.733 },
      "top-right": { d: 204.495, rotation: -145.733 },
      right: { d: 198.534, rotation: -71.127 },
      self: { d: 209.56, rotation: 0 },
      left: { d: 198.534, rotation: 71.127 }
    } as const;

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
      "top-left": { rotation: 145.733, x: 872.532, y: 556.869 },
      "top-right": { rotation: -145.733, x: 1367.468, y: 556.869 },
      right: { rotation: -71.127, x: 1526.966, y: 1039.896 },
      self: { rotation: 0, x: 1120, y: 1334.355 },
      left: { rotation: 71.127, x: 713.034, y: 1039.896 }
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
      "top-left": { x: 109.005, y: 84.81 },
      "top-right": { x: 228.995, y: 84.81 },
      right: { x: 266.192, y: 192.375 },
      self: { x: 169, y: 259.065 },
      left: { x: 71.808, y: 192.375 }
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

function midpointBetween(a: { x: number; y: number }, b: { x: number; y: number }): { x: number; y: number } {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2
  };
}
