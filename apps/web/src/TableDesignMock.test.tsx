import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  TableDesignMock,
  createRiverPlacements,
  roleBoardSelfSideLength,
  selfRiverWidth,
  tableDesignMockLayout
} from "./TableDesignMock";

describe("TableDesignMock", () => {
  it("renders the issue 308 static table mock from fixed data", () => {
    const html = renderToStaticMarkup(<TableDesignMock />);

    expect(tableDesignMockLayout.seats).toHaveLength(5);
    expect(html).toContain("Issue 308 table design mock");
    expect(html).toContain("契約HUD");
    expect(html).toContain("中央役職表示");
    expect(html).toContain("自席操作UI");
    expect(html).toContain("自分の表向き手札");
    expect(html).toContain("北西の裏向き手札");
    expect(html).toContain("--mock-page-width:2200px");
    expect(html).toContain("--mock-trick-card-width:118px");
    expect(html).toContain("--mock-river-card-width:104.78px");
    expect(html).not.toContain("mock-player-label");
    expect(tableDesignMockLayout.center).toMatchObject({ height: 303, width: 338, y: 890 });
    expect(tableDesignMockLayout.action).toMatchObject({ height: 274, width: 224, y: 1376 });
    expect(tableDesignMockLayout.cardSizes.river.self.width).toBeGreaterThan(
      tableDesignMockLayout.cardSizes.river.opponent.width
    );
    expect(tableDesignMockLayout.cardSizes.river.self.width).toBeLessThan(
      tableDesignMockLayout.cardSizes.trick.width
    );
    expect(tableDesignMockLayout.cardSizes.river.opponent.width).toBeLessThan(
      tableDesignMockLayout.cardSizes.trick.width
    );
    expect(tableDesignMockLayout.seats.find((seat) => seat.id === "self")).toMatchObject({
      hand: { y: 1684 },
      trick: { y: 1138 }
    });
    expect(tableDesignMockLayout.riverGrid).toMatchObject({ maxColumns: 5, maxRows: 4 });
    expect(html).toContain("--mock-self-river-height:141.294px");
  });

  it("lays out the self river from the role-board self edge length", () => {
    const d = selfRiverWidth(tableDesignMockLayout);
    const cardWidth = tableDesignMockLayout.cardSizes.river.self.width;
    const columnOffset = d * 0.125;
    const oneCard = createRiverPlacements(1, tableDesignMockLayout);
    const fiveCards = createRiverPlacements(5, tableDesignMockLayout);
    const sixCards = createRiverPlacements(6, tableDesignMockLayout);

    expect(d).toBeCloseTo(roleBoardSelfSideLength(tableDesignMockLayout.center));
    expect(d).toBeCloseTo(209.56);
    expect(cardWidth).toBeCloseTo(d * 0.5);
    expect(oneCard).toHaveLength(1);
    expect(oneCard[0]?.x).toBeCloseTo(0);
    expect(fiveCards).toHaveLength(5);
    expect(fiveCards[0]?.x).toBeCloseTo(0);
    expect(fiveCards[1]?.x).toBeCloseTo(columnOffset);
    expect(fiveCards[4]?.x).toBeCloseTo(columnOffset * 4);
    expect((fiveCards[4]?.x ?? 0) + cardWidth - (fiveCards[0]?.x ?? 0)).toBeCloseTo(d);
    expect(sixCards).toHaveLength(6);
    expect(sixCards[5]?.y).toBe(
      tableDesignMockLayout.cardSizes.river.self.height + tableDesignMockLayout.riverGrid.rowGap
    );
  });

  it("rotates left and right river axes by 90 degrees and keeps card spread mirrored", () => {
    const leftCards = createRiverPlacements(3, tableDesignMockLayout, "left");
    const rightCards = createRiverPlacements(3, tableDesignMockLayout, "right");

    expect(tableDesignMockLayout.seats.find((seat) => seat.id === "left")?.river.rotation).toBe(-102);
    expect(tableDesignMockLayout.seats.find((seat) => seat.id === "right")?.river.rotation).toBe(102);
    expect(leftCards.map((card) => card.rotation)).toEqual([-2, 0, 2]);
    expect(rightCards.map((card) => card.rotation)).toEqual([2, 0, -2]);
  });
});
