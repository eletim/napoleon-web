/// <reference types="node" />

import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  createSelfHandViewportLayout,
  tableDesignMockLayout
} from "./TableDesignMock";

const cssPath = fileURLToPath(new URL("./TableDesignMock.css", import.meta.url));
const chromePath = process.env.CHROME_BIN ?? "/usr/bin/google-chrome";

interface BrowserHitTestResult {
  errors: string[];
  metrics: { inner: [number, number] };
  tapped: number[];
}

describe("exchange hand browser hit targets", () => {
  it("keeps every 13-card exchange target fully exposed and tappable on compact viewports", () => {
    expect(existsSync(chromePath), `Chrome executable not found at ${chromePath}`).toBe(true);

    for (const viewport of [
      { width: 568, height: 320 },
      { width: 812, height: 341 }
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

function runHitTest(viewport: { height: number; width: number }): BrowserHitTestResult {
  const hand = createSelfHandViewportLayout(tableDesignMockLayout, 13, viewport);
  const fixtureDirectory = mkdtempSync(join(tmpdir(), "napoleon-hand-hit-targets-"));
  const fixturePath = join(fixtureDirectory, "index.html");
  const style = [
    `--mock-self-card-gap:${hand.gap}px`,
    `--mock-self-card-height:${hand.cardSize.height}px`,
    `--mock-self-card-width:${hand.cardSize.width}px`,
    `--mock-self-hand-columns:${hand.columnCount}`,
    `--mock-self-hand-height:${hand.handHeight}px`,
    `--mock-self-hand-left:${hand.left}px`,
    `--mock-self-hand-rows:${hand.rowCount}`,
    `--mock-self-hand-top:${hand.top}px`,
    `--mock-self-hand-width:${hand.handWidth}px`,
    `--mock-self-row-gap:${hand.rowGap}px`
  ].join(";");
  const cards = Array.from({ length: 13 }, (_, index) =>
    `<button class="mock-self-hand-card" data-card="${index}" type="button"><span class="card-face"></span></button>`
  ).join("");
  const html = `<!doctype html>
<style>
html, body { height: 100%; margin: 0; overflow: hidden; width: 100%; }
.mock-self-hand-card { border: 0; box-sizing: border-box; padding: 0; }
.card-face { display: block; height: 100%; width: 100%; }
${readFileSync(cssPath, "utf8")}
</style>
<div class="mock-self-hand" style="${style}">${cards}</div>
<pre id="result"></pre>
<script>
  const errors = [];
  const tapped = new Set();
  if (innerWidth !== ${viewport.width} || innerHeight !== ${viewport.height}) {
    errors.push('viewport@' + innerWidth + 'x' + innerHeight);
  }
  const cards = [...document.querySelectorAll('[data-card]')];
  for (const card of cards) {
    const index = Number(card.dataset.card);
    card.addEventListener('pointerup', () => tapped.add(index));
    const rect = card.getBoundingClientRect();
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
