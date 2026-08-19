import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TableDesignMock, tableDesignMockLayout } from "./TableDesignMock";

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
    expect(html).not.toContain("mock-player-label");
    expect(tableDesignMockLayout.center).toMatchObject({ height: 303, width: 338, y: 890 });
    expect(tableDesignMockLayout.action).toMatchObject({ height: 274, width: 224, y: 1376 });
    expect(tableDesignMockLayout.seats.find((seat) => seat.id === "self")).toMatchObject({
      hand: { y: 1684 },
      trick: { y: 1138 }
    });
  });
});
