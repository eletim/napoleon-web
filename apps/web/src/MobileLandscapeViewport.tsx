import { useEffect, useRef, useState, type ReactNode } from "react";
import "./MobileLandscapeViewport.css";

interface DeviceSignals {
  screenShortSide: number;
  hasHover: boolean;
}

export function needsMobileViewport(
  browser: Pick<Navigator, "userAgent" | "maxTouchPoints">,
  coarse: boolean,
  signals: DeviceSignals = { screenShortSide: Infinity, hasHover: !coarse }
): boolean {
  // Desktop-site mode can hide Android in the UA and expose a 980px (or
  // larger) layout viewport. Its width is not a device classification.
  return /Android|iPhone|iPod/i.test(browser.userAgent) ||
    (browser.maxTouchPoints > 0 && (
      (coarse && !signals.hasHover) || (signals.screenShortSide > 0 && signals.screenShortSide <= 600)
    ));
}

export function landscapeViewport(width: number, height: number, visualScale = 1) {
  return {
    width: Math.round(Math.max(width, height) * visualScale),
    height: Math.round(Math.min(width, height) * visualScale),
    rotated: height > width
  };
}

function readViewport() {
  const viewport = window.visualViewport;
  return {
    width: viewport?.width || window.innerWidth,
    height: viewport?.height || window.innerHeight,
    // Undo desktop-site shrink-to-fit for the nested logical viewport. The
    // inverse CSS scale maps it back into the outer layout's coordinates.
    scale: Math.min(1, viewport?.scale || 1),
    left: viewport?.offsetLeft || 0,
    top: viewport?.offsetTop || 0
  };
}

/** A real nested viewport keeps CSS media queries, fixed dialogs and DOM
 * coordinates in landscape space. The browser maps input through the outer
 * rotation, so card FLIP animations and hit testing need no special cases.
 * Keep the same iframe mounted on resize: rotation must never reload a game.
 */
export function MobileLandscapeViewport({ children }: { children: ReactNode }) {
  const detectedMobile = window.self === window.top && needsMobileViewport(
    navigator,
    window.matchMedia?.("(any-pointer: coarse)").matches ?? false,
    { screenShortSide: Math.min(window.screen.width, window.screen.height),
      hasHover: window.matchMedia?.("(any-hover: hover)").matches ?? false }
  );
  // A negative decision is re-evaluated on render (including a live update).
  // Once selected, keep the shell if a mouse is later connected to the phone.
  const mobileRef = useRef(false);
  mobileRef.current ||= detectedMobile;
  const mobile = mobileRef.current;
  const [size, setSize] = useState(readViewport);
  const [source] = useState(() => window.location.href);
  useEffect(() => {
    if (!mobile) return;
    const update = () => {
      const next = readViewport();
      setSize(current => current.width === next.width && current.height === next.height &&
        current.scale === next.scale && current.left === next.left && current.top === next.top
        ? current : next);
    };
    window.addEventListener("resize", update);
    window.visualViewport?.addEventListener("resize", update);
    window.visualViewport?.addEventListener("scroll", update);
    document.documentElement.classList.add("mobile-viewport-host");
    update();
    return () => {
      window.removeEventListener("resize", update);
      window.visualViewport?.removeEventListener("resize", update);
      window.visualViewport?.removeEventListener("scroll", update);
      document.documentElement.classList.remove("mobile-viewport-host");
    };
  }, [mobile]);
  if (!mobile) return children;
  const logical = landscapeViewport(size.width, size.height, size.scale);
  const scale = size.scale === 1 ? "" : ` scale(${1 / size.scale})`;
  const transform = logical.rotated ? `translateX(${size.width}px) rotate(90deg)${scale}` : scale.trim() || "none";
  return <iframe
    className="mobile-landscape-viewport"
    title="ナポレオン 横向きゲーム画面"
    src={source}
    data-rotated={logical.rotated}
    style={{ width: logical.width, height: logical.height, left: size.left, top: size.top, transform }}
  />;
}
