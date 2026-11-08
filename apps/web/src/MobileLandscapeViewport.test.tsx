// @vitest-environment happy-dom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { landscapeViewport, MobileLandscapeViewport, needsMobileViewport } from "./MobileLandscapeViewport";
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
afterEach(() => vi.unstubAllGlobals());
describe("mobile landscape viewport", () => {
  it("recognizes a phone even if the primary pointer/hover is reported as desktop", () => {
    expect(needsMobileViewport({ userAgent: "Mozilla Android", maxTouchPoints: 5 }, false)).toBe(true);
    expect(needsMobileViewport({ userAgent: "iPhone", maxTouchPoints: 5 }, false)).toBe(true);
    expect(needsMobileViewport({ userAgent: "Linux", maxTouchPoints: 5 }, true)).toBe(true);
    expect(needsMobileViewport({ userAgent: "Windows", maxTouchPoints: 0 }, false)).toBe(false);
    expect(needsMobileViewport({ userAgent: "Windows", maxTouchPoints: 5 }, false)).toBe(false);
  });
  it("recognizes desktop-site touch devices independently of a 980px or larger layout", () => {
    const desktopUaPhone = { userAgent: "X11; Linux x86_64", maxTouchPoints: 5 };
    expect(needsMobileViewport(desktopUaPhone, true, { screenShortSide: 407, hasHover: false })).toBe(true);
    expect(needsMobileViewport(desktopUaPhone, true, { screenShortSide: 1220, hasHover: false })).toBe(true);
    expect(needsMobileViewport(desktopUaPhone, false, { screenShortSide: 407, hasHover: true })).toBe(true);
    // A touch laptop's narrow window must not turn it into a phone.
    expect(needsMobileViewport({ userAgent: "Windows", maxTouchPoints: 10 }, true,
      { screenShortSide: 1080, hasHover: true })).toBe(false);
    expect(needsMobileViewport({ userAgent: "Windows", maxTouchPoints: 0 }, false,
      { screenShortSide: 1080, hasHover: true })).toBe(false);
  });
  it("undoes desktop-site shrink-to-fit for both orientations", () => {
    const scale = 407 / 980;
    expect(landscapeViewport(980, 904 / scale, scale)).toEqual({ width: 904, height: 407, rotated: true });
    expect(landscapeViewport(904 / scale, 980, scale)).toEqual({ width: 904, height: 407, rotated: false });
    const largeScale = 407 / 1220;
    expect(landscapeViewport(1220, 904 / largeScale, largeScale)).toEqual({ width: 904, height: 407, rotated: true });
  });
  it("uses the same landscape dimensions in either physical orientation", () => {
    expect(landscapeViewport(390, 844)).toEqual({ width: 844, height: 390, rotated: true });
    expect(landscapeViewport(844, 390)).toEqual({ width: 844, height: 390, rotated: false });
  });
  it("re-evaluates a prior desktop decision, without later discarding an active mobile game", async () => {
    vi.stubGlobal("navigator", { userAgent: "Linux", maxTouchPoints: 0 });
    const container = document.createElement("div"); const root = createRoot(container);
    const render = async () => { await act(async () => root.render(<MobileLandscapeViewport><button>game</button></MobileLandscapeViewport>)); };
    await render(); expect(container.querySelector("iframe")).toBeNull();
    vi.stubGlobal("navigator", { userAgent: "Android", maxTouchPoints: 5 });
    await render(); const frame = container.querySelector("iframe"); expect(frame).not.toBeNull();
    vi.stubGlobal("navigator", { userAgent: "Linux", maxTouchPoints: 0 });
    await render(); expect(container.querySelector("iframe")).toBe(frame);
    await act(async () => root.unmount());
  });
  it("responds to VisualViewport scale and rotation changes without replacing the game frame", async () => {
    vi.stubGlobal("navigator", { userAgent: "Android", maxTouchPoints: 5 });
    const viewport = Object.assign(new EventTarget(), {
      width: 980, height: 904 * 980 / 407, scale: 407 / 980, offsetLeft: 0, offsetTop: 0
    });
    vi.stubGlobal("visualViewport", viewport);
    const container = document.createElement("div"); const root = createRoot(container);
    await act(async () => root.render(<MobileLandscapeViewport>game</MobileLandscapeViewport>));
    const frame = container.querySelector("iframe")!;
    expect(frame.style.width).toBe("904px"); expect(frame.style.height).toBe("407px");
    expect(frame.style.transform).toContain("rotate(90deg)"); expect(frame.style.transform).toContain("scale(");
    viewport.width = 904 * 980 / 407; viewport.height = 980;
    await act(async () => viewport.dispatchEvent(new Event("resize")));
    expect(container.querySelector("iframe")).toBe(frame);
    expect(frame.style.width).toBe("904px"); expect(frame.style.height).toBe("407px");
    expect(frame.style.transform).not.toContain("rotate");
    await act(async () => root.unmount());
  });
  it("resizes and rotates the existing browsing context without reloading its source", async () => {
    vi.stubGlobal("navigator", { userAgent: "Android", maxTouchPoints: 5 });
    vi.stubGlobal("innerWidth", 390); vi.stubGlobal("innerHeight", 844);
    const container = document.createElement("div"); document.body.append(container); const root = createRoot(container);
    await act(async () => root.render(<MobileLandscapeViewport><button>desktop</button></MobileLandscapeViewport>));
    const frame = container.querySelector("iframe")!; const source = frame.src;
    expect(frame.style.width).toBe("844px"); expect(frame.style.height).toBe("390px"); expect(frame.style.transform).toContain("rotate(90deg)");
    vi.stubGlobal("innerWidth", 844); vi.stubGlobal("innerHeight", 390);
    await act(async () => window.dispatchEvent(new Event("resize")));
    expect(container.querySelector("iframe")).toBe(frame); expect(frame.src).toBe(source); expect(frame.style.transform).toBe("none");
    await act(async () => root.unmount()); container.remove();
  });
});
