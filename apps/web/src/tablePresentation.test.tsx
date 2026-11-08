// @vitest-environment happy-dom
import { act, StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PublicGameState, PublicStandardCard } from "@napoleon/protocol";
import { useTrickAnimation } from "./useTrickAnimation";
import { DEAL_DURATION_MS, useTablePresentation } from "./useTablePresentation";
import { ACTION_GAP_MS, TRICK_COLLECT_DURATION_MS, TRICK_RESULT_HOLD_MS } from "./presentationTiming";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const card = (rank: PublicStandardCard["rank"]): PublicStandardCard => ({ type: "standard", id: `clubs-${rank}`, suit: "clubs", rank });
const cards = [card("A"), card("K"), card("Q"), card("J"), card("10")];
function state(complete = false): PublicGameState {
  return {
    self: { id: "p0", hand: [card("2")], handCount: 1, capturedPointCards: complete ? cards : [] },
    opponents: [1, 2, 3, 4].map(i => ({ id: `p${i}`, handCount: 1, capturedPointCards: [] })),
    phase: "playing", trumpSuit: "hearts", contract: null, specialCards: { orumaCardId: "spades-A", yoromekiCardId: "hearts-Q", seiJackCardId: null, uraJackCardId: null },
    adjutant: null, latestEvent: null, result: null, bidding: null, exchange: null, adjutantChoice: null,
    currentPlayerId: "p0", currentTrick: complete ? cards.map((card, i) => ({ playerId: `p${i}`, card })) : [],
    completedTrickCount: complete ? 1 : 0, trickNumber: 1, isTrickComplete: complete, isGameOver: false, legalActions: []
  };
}
let animation: ReturnType<typeof useTrickAnimation>;
let presentation: ReturnType<typeof useTablePresentation>;
function Harness({ value, round = "one" }: { value: PublicGameState; round?: string }) {
  animation = useTrickAnimation({ state: value, selfPlayerId: "p0", roundKey: round });
  presentation = useTablePresentation(value, undefined, round, animation.isCollected);
  return <output>{presentation.state?.self.capturedPointCards.length}</output>;
}
const mounts: Array<() => void> = [];
afterEach(async () => { for (const dispose of mounts.splice(0)) await act(dispose); vi.useRealTimers(); vi.unstubAllGlobals(); });
async function mount(value = state()) {
  vi.useFakeTimers();
  const node = document.createElement("div"); document.body.append(node); const root = createRoot(node);
  mounts.push(() => { root.unmount(); node.remove(); });
  const render = async (value: PublicGameState, round = "one") => { await act(async () => root.render(<StrictMode><Harness value={value} round={round} /></StrictMode>)); };
  await render(value); return { render, node };
}
async function tick(ms: number) { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); }
async function reveal() { await tick(ACTION_GAP_MS * 4); await tick(TRICK_RESULT_HOLD_MS); }

describe("table presentation sequencing", () => {
  it.each([false, true])("credits points once, only after collection (final=%s), and clears the collected cards", async final => {
    const { render, node } = await mount();
    const complete = { ...state(true), isGameOver: final, phase: final ? "finished" as const : "playing" as const };
    await render(complete);
    expect(node.textContent).toBe("0"); await reveal(); expect(node.textContent).toBe("0");
    const next = vi.fn(async () => {});
    let first: Promise<unknown>;
    await act(async () => { first = animation.playCollectionBefore(next); void animation.playCollectionBefore(next); });
    expect(animation.collectingWinnerId).toBe("p0");
    await tick(TRICK_COLLECT_DURATION_MS - 1); expect(node.textContent).toBe("0"); expect(next).not.toHaveBeenCalled();
    await tick(1); await first!;
    expect(node.textContent).toBe("5"); expect(animation.displayedTrick).toHaveLength(0); expect(next).toHaveBeenCalledTimes(1);
    await render({ ...complete }); await act(async () => { await animation.playCollectionBefore(next); });
    expect(node.textContent).toBe("5"); expect(next).toHaveBeenCalledTimes(1);
    await tick(10000); expect(next).toHaveBeenCalledTimes(1);
  });

  it("cancels an old collection on a new round, even if trick number and cards repeat", async () => {
    const { render, node } = await mount(); await render(state(true)); await reveal();
    const old = vi.fn(async () => {}); let oldPromise: Promise<unknown>;
    await act(async () => { oldPromise = animation.playCollectionBefore(old); });
    await tick(200); await render(state(), "two"); await tick(1000); await oldPromise!;
    expect(old).not.toHaveBeenCalled(); expect(node.textContent).toBe("0");
    await render(state(true), "two"); await reveal(); const next = vi.fn(async () => {});
    await act(async () => { void animation.playCollectionBefore(next); }); await tick(TRICK_COLLECT_DURATION_MS);
    expect(next).toHaveBeenCalledTimes(1); expect(node.textContent).toBe("5");
  });

  it("holds all final-trick gains, including buried points, until collection", async () => {
    const { render, node } = await mount(); const complete = state(true);
    await render({ ...complete, self: { ...complete.self, capturedPointCards: [...cards, { ...card("K"), id: "hearts-K", suit: "hearts" }] } });
    expect(node.textContent).toBe("0"); await reveal();
    await act(async () => { void animation.playCollectionBefore(async () => {}); }); await tick(TRICK_COLLECT_DURATION_MS);
    expect(node.textContent).toBe("6");
  });

  it("does not dispatch a pending collection callback after unmount", async () => {
    await mount(state(true)); await reveal(); const next = vi.fn(async () => {});
    let pending: Promise<unknown>;
    await act(async () => { pending = animation.playCollectionBefore(next); });
    await act(mounts.pop()!); await tick(TRICK_COLLECT_DURATION_MS); await pending!;
    expect(next).not.toHaveBeenCalled();
  });

  it("locks the initial deal once per round and deals again next round", async () => {
    const bidding = { ...state(), phase: "bidding" as const }; const { render } = await mount(bidding);
    expect(presentation.isDealing).toBe(true); await tick(DEAL_DURATION_MS - 1); expect(presentation.isDealing).toBe(true);
    await tick(1); expect(presentation.isDealing).toBe(false); await render({ ...bidding }); expect(presentation.isDealing).toBe(false);
    await render(bidding, "two"); expect(presentation.isDealing).toBe(true);
  });

  it("supports reduced motion while preserving credit ordering", async () => {
    vi.stubGlobal("matchMedia", () => ({ matches: true, addEventListener() {}, removeEventListener() {} }));
    const { render, node } = await mount(); await render(state(true)); await tick(120);
    expect(node.textContent).toBe("0"); const next = vi.fn(async () => {});
    await act(async () => { await animation.playCollectionBefore(next); });
    expect(node.textContent).toBe("5"); expect(next).toHaveBeenCalledTimes(1);
  });
});
