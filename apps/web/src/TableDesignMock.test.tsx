import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  TableDesignMock,
  createRiverPlacements,
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
    expect(html).toContain("--mock-trick-card-width:132px");
    expect(html).toContain("--mock-river-card-width:92px");
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
    expect(html).toContain("--mock-self-river-height:124px");
  });

  it("lays out the self river from the role-board self edge length", () => {
    const d = selfRiverWidth(tableDesignMockLayout);
    const oneCard = createRiverPlacements(1, tableDesignMockLayout);
    const fiveCards = createRiverPlacements(5, tableDesignMockLayout);
    const sixCards = createRiverPlacements(6, tableDesignMockLayout);

    expect(oneCard).toHaveLength(1);
    expect(oneCard[0]?.x).toBeCloseTo((d - tableDesignMockLayout.cardSizes.river.self.width) / 2);
    expect(fiveCards).toHaveLength(5);
    expect(fiveCards[0]?.x).toBeCloseTo(0);
    expect(fiveCards[4]?.x).toBeCloseTo(d - tableDesignMockLayout.cardSizes.river.self.width);
    expect(sixCards).toHaveLength(6);
    expect(sixCards[5]?.y).toBe(
      tableDesignMockLayout.cardSizes.river.self.height + tableDesignMockLayout.riverGrid.rowGap
    );
  });

  it("keeps left and right river card rotations mirrored instead of perpendicular", () => {
    const leftCards = createRiverPlacements(3, tableDesignMockLayout, "left");
    const rightCards = createRiverPlacements(3, tableDesignMockLayout, "right");

    expect(leftCards.map((card) => card.rotation)).toEqual([-2, 0, 2]);
    expect(rightCards.map((card) => card.rotation)).toEqual([2, 0, -2]);
  });
});
