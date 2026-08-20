import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  TableDesignMock,
  createRiverGeometry,
  createRiverPlacements,
  createRoleBoardEdgeGeometry,
  roleBoardSelfSideLength,
  selfRiverWidth,
  tableDesignMockLayout
} from "./TableDesignMock";

describe("TableDesignMock", () => {
  it("renders the issue 323 static table mock with separate trick zones and rivers", () => {
    const html = renderToStaticMarkup(<TableDesignMock />);

    expect(tableDesignMockLayout.seats).toHaveLength(5);
    expect(html).toContain("Issue 323 table design mock");
    expect(html).toContain("契約HUD");
    expect(html).toContain("中央役職表示");
    expect(html).toContain("自席操作UI");
    expect(html).toContain("自分の表向き手札");
    expect(html).toContain("北西の裏向き手札");
    expect(html).toContain("北西の現在トリック置き場");
    expect(html).toContain("自分のポイント札の河");
    expect(html).toContain("--mock-page-width:2200px");
    expect(html).toContain("--mock-trick-card-width:118px");
    expect(html).toContain("--mock-trick-zone-width:260px");
    expect(html).toContain("--mock-trick-zone-height:120px");
    expect(html).not.toContain("--mock-river-card-width:56px");
    expect(html).not.toContain("mock-player-label");
    expect(tableDesignMockLayout.center).toMatchObject({ height: 303, width: 338, y: 890 });
    expect(tableDesignMockLayout.action).toMatchObject({ height: 244, width: 200, y: 1470 });
    expect(tableDesignMockLayout.seats.find((seat) => seat.id === "self")).toMatchObject({
      hand: { y: 1684 },
      trickZone: { y: 1275, height: 120 }
    });
    expect(tableDesignMockLayout.riverGrid).toMatchObject({ maxColumns: 5, maxRows: 4 });
    expect((html.match(/mock-current-trick-zone/g) ?? [])).toHaveLength(10);
    expect((html.match(/mock-trick-card mock-playing-card/g) ?? [])).toHaveLength(4);
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
});
