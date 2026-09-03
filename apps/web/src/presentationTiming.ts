import { useEffect, useState } from "react";

// Central presentation-layer timing constants for trick animation. These are
// intentionally kept out of game-core / server code: they only govern how
// fast the *UI* replays a trick that the server already resolved, never the
// underlying game rules or turn order. Tune these to change overall table
// tempo without hunting for magic numbers elsewhere.

/** How long a single card's hand -> table flight animation takes. */
export const CARD_PLAY_DURATION_MS = 420;

/** Pause before each COM action so plays don't feel instantaneous/robotic. */
export const ACTION_GAP_MS = 1000;

/** How long a completed trick's 5 cards stay visible before being collected. */
export const TRICK_RESULT_HOLD_MS = 3000;

/** How long the winner-ward collection flight/fade takes. */
export const TRICK_COLLECT_DURATION_MS = 650;

/** Short pulse near the end of the result hold that calls out the winner. */
export const TRICK_WINNER_PULSE_MS = 480;

// Reduced-motion viewers skip flights/holds almost entirely so the game
// still advances quickly, without disorienting motion.
export const REDUCED_MOTION_HOLD_MS = 120;

export function usePrefersReducedMotion(): boolean {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updatePreference = () => setPrefersReducedMotion(mediaQuery.matches);

    updatePreference();
    mediaQuery.addEventListener("change", updatePreference);

    return () => {
      mediaQuery.removeEventListener("change", updatePreference);
    };
  }, []);

  return prefersReducedMotion;
}
