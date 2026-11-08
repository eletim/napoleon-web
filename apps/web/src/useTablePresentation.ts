import { useEffect, useRef, useState } from "react";
import type { PublicGameState, PublicMatchState } from "@napoleon/protocol";
import { usePrefersReducedMotion } from "./presentationTiming";

export const DEAL_DURATION_MS = 1600;

/** Presentation snapshots only: the authoritative session is never changed. */
export function useTablePresentation(
  state: PublicGameState | undefined,
  match: PublicMatchState | undefined,
  roundKey: string,
  isCollected: boolean
) {
  const reduced = usePrefersReducedMotion();
  const [dealtRound, setDealtRound] = useState<string>();
  const previous = useRef<{ roundKey: string; state?: PublicGameState; match?: PublicMatchState } | undefined>(undefined);
  const isDealing = state?.phase === "bidding" && dealtRound !== roundKey && !reduced;
  useEffect(() => {
    if (!state || dealtRound === roundKey) return;
    if (state.phase !== "bidding" || reduced) {
      setDealtRound(roundKey);
      return;
    }
    const timer = setTimeout(() => setDealtRound(roundKey), DEAL_DURATION_MS);
    return () => clearTimeout(timer);
  }, [state?.phase, roundKey, reduced, dealtRound]);

  const pending = state?.isTrickComplete && !isCollected;
  const old = previous.current?.roundKey === roundKey ? previous.current : undefined;
  let displayedState = state;
  if (state && pending) {
    const currentIds = new Set(state.currentTrick.map(({ card }) => card.id));
    const beforePoints = (player: PublicGameState["self"] | PublicGameState["opponents"][number]) => {
      const before = old?.state && [old.state.self, ...old.state.opponents].find(p => p.id === player.id);
      return before?.capturedPointCards ?? player.capturedPointCards.filter(c => !currentIds.has(c.id));
    };
    displayedState = {
      ...state,
      self: { ...state.self, capturedPointCards: beforePoints(state.self) },
      opponents: state.opponents.map(p => ({ ...p, capturedPointCards: beforePoints(p) }))
    };
  }
  useEffect(() => {
    if (!pending) previous.current = { roundKey, state, match };
  }, [state, match, roundKey, pending]);
  return { isDealing, state: displayedState, match: pending ? old?.match : match };
}
