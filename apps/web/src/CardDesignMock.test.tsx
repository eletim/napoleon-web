import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import { App } from "./App";
import {
  cardDesignCardHeight,
  cardDesignComparisonRanks,
  cardDesignConfig,
  cardDesignExposureLeft,
  cardDesignExposureOffset,
  cardDesignIdentificationGuideX,
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
  it("renders the issue 355 card design sandbox at /mock/card-design", () => {
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        location: {
          pathname: "/mock/card-design"
        }
      }
    });

    const html = renderToStaticMarkup(<App />);

    expect(html).toContain("Issue 355 card design mock");
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
    const identificationAreaRatio = cardDesignConfig.layout.identificationAreaRatio;
    const exposureOffsetRatio = cardDesignConfig.layout.exposureOffsetRatio;

    expect(cardDesignConfig.colors).toEqual({
      spades: "#111827",
      hearts: "#dc2626",
      diamonds: "#2563eb",
      clubs: "#15803d"
    });
    expect(cardDesignConfig.layout).toMatchObject({
      identificationAreaRatio: 0.25,
      exposureOffsetRatio: 0.25,
      rankFontRatio: 0.18,
      suitSymbolRatio: 0.2,
      cornerPaddingRatio: 0.055,
      rankSuitGapRatio: 0.165,
      centerSymbolRatio: 0.38,
      borderWidthRatio: 0.016,
      borderRadiusRatio: 0.065
    });
    expect(identificationAreaRatio).toBe(0.25);
    expect(exposureOffsetRatio).toBe(0.25);
    expect(identificationAreaRatio).toBe(exposureOffsetRatio);
    expect(cardDesignCardHeight(100)).toBe(140);
  });

  it("renders a 25 percent exposure strip for the four tens using the explicit offset ratio", () => {
    const html = renderToStaticMarkup(<CardDesignMock />);
    const width = cardDesignConfig.sizes.overlapWidth;
    const offset = width * cardDesignConfig.layout.exposureOffsetRatio;

    expect(cardDesignConfig.layout.identificationAreaRatio).toBe(cardDesignConfig.layout.exposureOffsetRatio);
    expect(cardDesignIdentificationGuideX(width)).toBe(offset);
    expect(cardDesignExposureOffset(width)).toBe(offset);
    expect(cardDesignExposureLeft(0, width)).toBe(0);
    expect(cardDesignExposureLeft(1, width)).toBe(offset);
    expect(cardDesignExposureLeft(2, width)).toBe(offset * 2);
    expect(cardDesignExposureLeft(3, width)).toBe(offset * 3);
    expect(cardDesignOverlapWidth(4, width)).toBe(width + 3 * offset);
    expect(html).toContain("25% Exposure");
    expect(html).toContain("guide 25% / offset 25%");
    expect(html).toContain("--card-design-identification-guide-x:40px");
    expect(html).toContain("--card-design-overlap-step:25%");
    expect(html).toContain("--card-design-exposure-offset:40px");
    expect(html).toContain(`--card-design-overlap-width:${cardDesignOverlapWidth(4, width)}px`);
    expect(html).toContain("--card-design-overlap-left:0px");
    expect(html).toContain("--card-design-overlap-left:40px");
    expect(html).toContain("--card-design-overlap-left:80px");
    expect(html).toContain("--card-design-overlap-left:120px");
    expect(html).toContain("--card-design-overlap-guide-left:40px");
    expect(html).toContain("--card-design-overlap-guide-left:80px");
    expect(html).toContain("--card-design-overlap-guide-left:120px");
    expect(html).toContain('aria-label="10♠"');
    expect(html).toContain('aria-label="10♣"');
    expect(html).toContain('aria-label="10♥"');
    expect(html).toContain('aria-label="10♦"');
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
