/// <reference types="node" />

import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  createPlayerInfoLayouts,
  createSelfHandViewportLayout,
  tableDesignMockLayout
} from "./TableDesignMock";

const cssPath = fileURLToPath(new URL("./TableDesignMock.css", import.meta.url));
const chromePath = findChromeExecutable();

interface BrowserHitTestResult {
  errors: string[];
  metrics: { inner: [number, number] };
  tapped: number[];
}

describe("exchange hand browser hit targets", () => {
  // The 13-card exchange hand is a single row (horizontal overlap is allowed by design when a
  // viewport is too narrow to fit every card at its nominal size, but is never required to
  // shrink cards or wrap to a second row). At these compact mobile-landscape viewports the
  // per-card size already scales down with viewport width, so every card renders with no
  // overlap and every corner stays independently exposed and tappable.
  it("keeps every 13-card exchange target fully exposed and tappable in a single row on compact viewports", () => {
    for (const viewport of [
      { width: 568, height: 320 },
      { width: 812, height: 341 },
      { width: 844, height: 390 }
    ]) {
      const result = runHitTest(viewport);

      expect(
        result.errors,
        `${viewport.width}x${viewport.height}: ${JSON.stringify(result.metrics)} ${result.errors.join(", ")}`
      ).toEqual([]);
      expect(result.tapped).toEqual(Array.from({ length: 13 }, (_, index) => index));
    }
  }, 20_000);
});

function findChromeExecutable(): string {
  const configuredPath = process.env.CHROME_BIN?.trim();

  if (configuredPath !== undefined && configuredPath.length > 0) {
    if (existsSync(configuredPath)) {
      return configuredPath;
    }

    throw new Error(`CHROME_BIN does not point to an executable: ${configuredPath}`);
  }

  const executableNames = process.platform === "win32"
    ? ["chrome.exe", "msedge.exe"]
    : ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"];
  const pathCandidates = (process.env.PATH ?? "")
    .split(delimiter)
    .filter((entry) => entry.length > 0)
    .flatMap((directory) => executableNames.map((name) => join(directory, name)));
  const platformCandidates = process.platform === "darwin"
    ? [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium"
      ]
    : process.platform === "win32"
      ? [
          join(process.env.PROGRAMFILES ?? "", "Google", "Chrome", "Application", "chrome.exe"),
          join(process.env["PROGRAMFILES(X86)"] ?? "", "Google", "Chrome", "Application", "chrome.exe"),
          join(process.env.LOCALAPPDATA ?? "", "Google", "Chrome", "Application", "chrome.exe"),
          join(process.env.PROGRAMFILES ?? "", "Microsoft", "Edge", "Application", "msedge.exe")
        ]
      : [
          "/usr/bin/google-chrome",
          "/usr/bin/google-chrome-stable",
          "/usr/bin/chromium",
          "/usr/bin/chromium-browser"
        ];
  const discoveredPath = [...pathCandidates, ...platformCandidates].find((candidate) =>
    candidate.length > 0 && existsSync(candidate)
  );

  if (discoveredPath === undefined) {
    throw new Error(
      "Chrome or Chromium was not found. Install one or set CHROME_BIN, then run pnpm --filter @napoleon/web test:browser."
    );
  }

  return discoveredPath;
}

function runHitTest(viewport: { height: number; width: number }): BrowserHitTestResult {
  const hand = createSelfHandViewportLayout(tableDesignMockLayout, 13, viewport);
  const selfInfo = createPlayerInfoLayouts(tableDesignMockLayout, viewport, true, 13)
    .find((info) => info.seatId === "self");
  expect(selfInfo).toBeDefined();
  const fixtureDirectory = mkdtempSync(join(tmpdir(), "napoleon-hand-hit-targets-"));
  const fixturePath = join(fixtureDirectory, "index.html");
  const style = [
    `--mock-self-card-gap:${hand.gap}px`,
    `--mock-self-card-height:${hand.cardSize.height}px`,
    `--mock-self-card-step:${hand.step}px`,
    `--mock-self-card-width:${hand.cardSize.width}px`,
    `--mock-self-content-left:${hand.contentLeftInset}px`,
    `--mock-self-hand-height:${hand.handHeight}px`,
    `--mock-self-hand-left:${hand.left}px`,
    `--mock-self-hand-top:${hand.top}px`,
    `--mock-self-hand-width:${hand.handWidth}px`
  ].join(";");
  // Every card sits in the single row at --mock-self-card-index * --mock-self-card-step; later
  // cards paint on top (via z-index), which keeps each card's own left edge tappable even if a
  // narrower viewport ever needed step to shrink far enough for cards to overlap.
  const cards = Array.from({ length: 13 }, (_, index) =>
    `<button class="mock-self-hand-card" data-card="${index}" style="--mock-self-card-index:${index}" type="button"><span class="card-face"></span></button>`
  ).join("");
  const playerStyle = [
    `--mock-player-avatar-size:${selfInfo?.avatarSize ?? 0}px`,
    `--mock-player-gap:${selfInfo?.gap ?? 0}px`,
    `--mock-player-height:${selfInfo?.height ?? 0}px`,
    `--mock-player-width:${selfInfo?.width ?? 0}px`,
    `--mock-x:${selfInfo?.x ?? 0}px`,
    `--mock-y:${selfInfo?.y ?? 0}px`
  ].join(";");
  const html = `<!doctype html>
<style>
html, body { height: 100%; margin: 0; overflow: hidden; width: 100%; }
.mock-self-hand-card { border: 0; box-sizing: border-box; padding: 0; }
.card-face { display: block; height: 100%; width: 100%; }
${readFileSync(cssPath, "utf8")}
</style>
<div class="mock-player-info" data-self-panel style="${playerStyle}"></div>
<div class="mock-self-hand" style="${style}">${cards}</div>
<pre id="result"></pre>
<script>
  const errors = [];
  const tapped = new Set();
  if (innerWidth !== ${viewport.width} || innerHeight !== ${viewport.height}) {
    errors.push('viewport@' + innerWidth + 'x' + innerHeight);
  }
  const cards = [...document.querySelectorAll('[data-card]')];
  const panelRect = document.querySelector('[data-self-panel]').getBoundingClientRect();
  for (const card of cards) {
    const index = Number(card.dataset.card);
    card.addEventListener('pointerup', () => tapped.add(index));
    const rect = card.getBoundingClientRect();
    if (rect.left < panelRect.right && rect.right > panelRect.left
      && rect.top < panelRect.bottom && rect.bottom > panelRect.top) {
      errors.push('panel-overlap@' + index);
    }
    const points = [
      [rect.left + 1, rect.top + 1],
      [rect.right - 1, rect.top + 1],
      [rect.left + 1, rect.bottom - 1],
      [rect.right - 1, rect.bottom - 1],
      [rect.left + rect.width / 2, rect.top + rect.height / 2]
    ];
    for (const [x, y] of points) {
      if (document.elementFromPoint(x, y)?.closest('[data-card]') !== card) {
        errors.push(index + '@' + x.toFixed(1) + ',' + y.toFixed(1));
      }
    }
    const [x, y] = points[4];
    const target = document.elementFromPoint(x, y);
    target?.dispatchEvent(new PointerEvent('pointerup', {
      bubbles: true,
      clientX: x,
      clientY: y,
      pointerType: 'touch'
    }));
  }
  document.querySelector('#result').textContent = JSON.stringify({
    errors,
    metrics: {
      inner: [innerWidth, innerHeight],
      rects: cards.map((card) => {
        const rect = card.getBoundingClientRect();
        return [rect.left, rect.top, rect.right, rect.bottom];
      })
    },
    tapped: [...tapped].sort((a, b) => a - b)
  });
</script>`;

  try {
    writeFileSync(fixturePath, html);
    const initial = runChromeFixture(fixturePath, viewport);
    const [innerWidth, innerHeight] = initial.metrics.inner;

    if (innerWidth === viewport.width && innerHeight === viewport.height) {
      return initial;
    }

    return runChromeFixture(fixturePath, {
      height: viewport.height + (viewport.height - innerHeight),
      width: viewport.width + (viewport.width - innerWidth)
    });
  } finally {
    rmSync(fixtureDirectory, { force: true, recursive: true });
  }
}

function runChromeFixture(
  fixturePath: string,
  windowSize: { height: number; width: number }
): BrowserHitTestResult {
  const chrome = spawnSync(chromePath, [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--force-device-scale-factor=1",
    `--window-size=${windowSize.width},${windowSize.height}`,
    "--virtual-time-budget=1000",
    "--dump-dom",
    `file://${fixturePath}`
  ], { encoding: "utf8" });

  expect(chrome.status, chrome.stderr).toBe(0);
  const serializedResult = chrome.stdout.match(/<pre id="result">([^<]+)<\/pre>/)?.[1];
  expect(serializedResult).toBeDefined();

  return JSON.parse(serializedResult ?? "{}") as BrowserHitTestResult;
}
