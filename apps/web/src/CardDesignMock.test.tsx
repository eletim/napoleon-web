import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import { App } from "./App";
import {
  cardDesignCardHeight,
  cardDesignComparisonRanks,
  cardDesignConfig,
  cardDesignOverlapWidth,
  cardDesignSuitOrder,
  createCardDesignDeck
} from "./CardDesignCard";
import { CardDesignMock } from "./CardDesignMock";
import { TableDesignMock } from "./TableDesignMock";

const originalWindow = globalThis.window;

afterEach(() => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: originalWindow
  });
});

describe("CardDesignMock", () => {
  it("renders the issue 352 card design sandbox at /mock/card-design", () => {
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        location: {
          pathname: "/mock/card-design"
        }
      }
    });

    const html = renderToStaticMarkup(<App />);

    expect(html).toContain("Issue 352 card design mock");
    expect(html).toContain("/mock/card-design");
    expect(html).not.toContain("Issue 348 table design");
  });

  it("shows representative cards for all four suits at normal and small sizes", () => {
    const html = renderToStaticMarkup(<CardDesignMock />);

    expect(cardDesignSuitOrder).toEqual(["spades", "clubs", "hearts", "diamonds"]);
    expect(cardDesignComparisonRanks).toEqual(["A", "2", "5", "10", "J", "Q", "K"]);
    expect(html).toContain("Normal");
    expect(html).toContain("Small");
    expect(html).toContain("J / Q / K");
    expect(html).toContain('aria-label="A♠"');
    expect(html).toContain('aria-label="A♣"');
    expect(html).toContain('aria-label="A♥"');
    expect(html).toContain('aria-label="A♦"');
    expect(html).toContain('aria-label="10♠"');
    expect(html).toContain('aria-label="10♣"');
    expect(html).toContain('aria-label="10♥"');
    expect(html).toContain('aria-label="10♦"');
    expect(html).toContain('aria-label="J♠"');
    expect(html).toContain('aria-label="Q♦"');
    expect(html).toContain('aria-label="K♣"');
  });

  it("keeps card design settings centralized in config", () => {
    const ratio = cardDesignConfig.layout.leftIdentificationAreaRatio;

    expect(cardDesignConfig.colors).toEqual({
      spades: "#111827",
      hearts: "#dc2626",
      diamonds: "#2563eb",
      clubs: "#15803d"
    });
    expect(cardDesignConfig.layout).toMatchObject({
      rankFontRatio: 0.18,
      suitSymbolRatio: 0.2,
      cornerPaddingRatio: 0.055,
      rankSuitGapRatio: 0.165,
      centerSymbolRatio: 0.38,
      borderWidthRatio: 0.016,
      borderRadiusRatio: 0.065
    });
    expect(ratio).toBe(0.25);
    expect(cardDesignCardHeight(100)).toBe(140);
  });

  it("renders a 25 percent exposure strip for the four tens", () => {
    const html = renderToStaticMarkup(<CardDesignMock />);
    const width = cardDesignConfig.sizes.overlapWidth;

    expect(cardDesignOverlapWidth(4, width)).toBe(width + 3 * width * 0.25);
    expect(html).toContain("25% Exposure");
    expect(html).toContain("--card-design-overlap-step:25%");
    expect(html).toContain(`--card-design-overlap-width:${cardDesignOverlapWidth(4, width)}px`);
    expect(html).toContain("--card-design-overlap-left:0px");
    expect(html).toContain("--card-design-overlap-left:24px");
    expect(html).toContain("--card-design-overlap-left:48px");
    expect(html).toContain("--card-design-overlap-left:72px");
  });

  it("can render all 52 prototype cards and compare current letele cards", () => {
    const html = renderToStaticMarkup(<CardDesignMock />);

    expect(createCardDesignDeck()).toHaveLength(52);
    expect((html.match(/class="card-design-card card-design-card-deck"/g) ?? [])).toHaveLength(52);
    expect((html.match(/class="card-design-letele-card"/g) ?? [])).toHaveLength(8);
    expect(html).toContain("@letele comparison");
  });

  it("does not restore the reverted issue 350 river-only table mock design", () => {
    const world = renderToStaticMarkup(<TableDesignMock />);
    const projected = renderToStaticMarkup(<TableDesignMock variant="projected" />);

    expect(world).not.toContain("mock-river-card-face");
    expect(projected).not.toContain("mock-river-card-face");
    expect(world).toContain("Issue 348 table design world mock");
    expect(projected).toContain("Issue 348 table design projected mock");
  });
});
