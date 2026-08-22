import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import { App } from "./App";
import { cardDesignSuitOrder, cardDesignSuitSymbols } from "./CardDesignCard";
import { CardDesignMock } from "./CardDesignMock";
import { TableDesignMock } from "./TableDesignMock";
import { cardmeisterFourColorCsv, fourColorSuitColors } from "./cardSuitTheme";

const originalWindow = globalThis.window;

afterEach(() => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: originalWindow
  });
});

describe("CardDesignMock", () => {
  it("keeps the card design route as a cardmeister selected confirmation page", () => {
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        location: {
          pathname: "/mock/card-design"
        }
      }
    });

    const html = renderToStaticMarkup(<App />);

    expect(html).toContain("cardmeister selected card design mock");
    expect(html).toContain("/mock/card-design");
    expect(html).toContain("cardmeister 4-color");
    expect(html).not.toContain("Issue 348 table design");
  });

  it("renders the selected representative cards with the shared four-color cardmeister attributes", () => {
    const html = renderToStaticMarkup(<CardDesignMock />);

    expect(cardDesignSuitOrder).toEqual(["spades", "clubs", "hearts", "diamonds"]);
    expect(cardDesignSuitSymbols).toEqual({
      spades: "♠",
      hearts: "♥",
      diamonds: "♦",
      clubs: "♣"
    });
    expect(fourColorSuitColors).toEqual({
      spades: "#111827",
      hearts: "#dc2626",
      diamonds: "#2563eb",
      clubs: "#15803d"
    });
    expect(html).toContain("Selected card check");
    expect(html).toContain("CurrentTrick相当");
    expect(html).toContain("自分手札相当");
    expect(html).toContain('aria-label="A♠"');
    expect(html).toContain('aria-label="A♣"');
    expect(html).toContain('aria-label="A♥"');
    expect(html).toContain('aria-label="A♦"');
    expect(html).toContain('aria-label="10♠"');
    expect(html).toContain('aria-label="10♣"');
    expect(html).toContain('aria-label="10♥"');
    expect(html).toContain('aria-label="10♦"');
    expect(html).toContain('aria-label="J♠"');
    expect(html).toContain('aria-label="Q♥"');
    expect(html).toContain('aria-label="K♦"');
    expect(html).toContain("<playing-card");
    expect(html).toContain(`suitcolor="${cardmeisterFourColorCsv}"`);
    expect(html).toContain(`rankcolor="${cardmeisterFourColorCsv}"`);
  });

  it("does not keep the issue 392 comparison candidate references in the retained mock", () => {
    const html = renderToStaticMarkup(<CardDesignMock />);

    expect(html).not.toContain("SVGCards Vertical4");
    expect(html).not.toContain("SVGCards Horizontal4");
    expect(html).not.toContain("SVGCards Accessible");
    expect(html).not.toContain("Current @letele");
    expect(html).not.toContain("/vendor/card-design/svgcards");
    expect(html).not.toContain("@letele comparison");
  });

  it("shows the selected issue 396 river-only face in the table mock without removing retained table mocks", () => {
    const world = renderToStaticMarkup(<TableDesignMock />);
    const projected = renderToStaticMarkup(<TableDesignMock variant="projected" />);
    const bidding = renderToStaticMarkup(<TableDesignMock variant="bidding" />);

    expect(world).toContain("mock-river-card-face");
    expect(projected).toContain("mock-projected-river-card-face");
    expect(bidding).toContain("mock-bidding-overlay");
    expect(world).toContain("Issue 348 table design world mock");
    expect(projected).toContain("Issue 348 table design projected mock");
    expect(bidding).toContain("Issue 348 table design bidding mock");
  });
});
