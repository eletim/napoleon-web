import { useEffect, useMemo, useRef, useState } from "react";
import type { PublicGameState, PublicPlayedCard } from "@napoleon/protocol";
import { determineCurrentWinningPlayer } from "@napoleon/game-core";

const PLAY_INTERVAL_MS = 220;
const RESULT_EMPHASIS_DELAY_MS = 190;
const RESULT_HOLD_MS = 520;
const COLLECT_MS = 260;
const REDUCED_RESULT_HOLD_MS = 80;

interface UseTrickAnimationOptions {
  state: PublicGameState | undefined;
}

interface TrickAnimationState {
  collectingWinnerId: string | undefined;
  displayedTrick: readonly PublicPlayedCard[];
  isAnimating: boolean;
  isResultEmphasisActive: boolean;
  playCollectionBefore: <T>(work: () => Promise<T>) => Promise<T | undefined>;
}

export function useTrickAnimation({
  state
}: UseTrickAnimationOptions): TrickAnimationState {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [displayedTrick, setDisplayedTrick] = useState<readonly PublicPlayedCard[]>([]);
  const [isPlayingSequence, setIsPlayingSequence] = useState(false);
  const [isResultEmphasisActive, setIsResultEmphasisActive] = useState(false);
  const [collectingWinnerId, setCollectingWinnerId] = useState<string | undefined>();
  const displayedTrickRef = useRef<readonly PublicPlayedCard[]>([]);
  const displayedTrickNumberRef = useRef<number | undefined>(undefined);
  const collectionInProgressRef = useRef(false);
  const targetKeyRef = useRef("");
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    displayedTrickRef.current = displayedTrick;
  }, [displayedTrick]);

  useEffect(() => {
    const targetTrick = state?.currentTrick ?? [];
    const targetKey = createTrickKey(state);

    if (targetKey === targetKeyRef.current) {
      return;
    }

    targetKeyRef.current = targetKey;
    clearTimers(timersRef.current);
    setIsResultEmphasisActive(false);

    if (state === undefined || targetTrick.length === 0) {
      setDisplayedTrick([]);
      displayedTrickRef.current = [];
      displayedTrickNumberRef.current = state?.trickNumber;
      setIsPlayingSequence(false);
      return;
    }

    const hasMovedToNewTrick =
      displayedTrickNumberRef.current !== undefined &&
      displayedTrickNumberRef.current !== state.trickNumber;
    const currentDisplayedTrick = hasMovedToNewTrick ? [] : displayedTrickRef.current;

    if (hasMovedToNewTrick) {
      setDisplayedTrick([]);
      displayedTrickRef.current = [];
    }

    displayedTrickNumberRef.current = state.trickNumber;

    const currentLength = currentDisplayedTrick.length;
    const isSameTrickGrowth =
      currentLength <= targetTrick.length &&
      matchesPrefix(currentDisplayedTrick, targetTrick);

    if (!isSameTrickGrowth || prefersReducedMotion) {
      setDisplayedTrick(targetTrick);
      if (state.isTrickComplete && targetTrick.length === 5) {
        holdResultEmphasis(timersRef.current, setIsResultEmphasisActive, prefersReducedMotion);
      }
      setIsPlayingSequence(false);
      return;
    }

    const additions = targetTrick.slice(currentLength);
    if (additions.length === 0) {
      setIsPlayingSequence(false);
      return;
    }

    setIsPlayingSequence(true);
    additions.forEach((_, index) => {
      const timer = setTimeout(() => {
        const nextLength = currentLength + index + 1;
        setDisplayedTrick(targetTrick.slice(0, nextLength));

        if (index === additions.length - 1) {
          if (state.isTrickComplete && targetTrick.length === 5) {
            const resultTimer = setTimeout(
              () => {
                holdResultEmphasis(
                  timersRef.current,
                  setIsResultEmphasisActive,
                  prefersReducedMotion,
                  () => setIsPlayingSequence(false)
                );
              },
              prefersReducedMotion ? 0 : RESULT_EMPHASIS_DELAY_MS
            );
            timersRef.current.push(resultTimer);
            return;
          }

          setIsPlayingSequence(false);
        }
      }, index * PLAY_INTERVAL_MS);
      timersRef.current.push(timer);
    });
  }, [prefersReducedMotion, state]);

  useEffect(
    () => () => {
      clearTimers(timersRef.current);
    },
    []
  );

  const completedWinnerId = useMemo(() => {
    if (
      state === undefined ||
      !state.isTrickComplete ||
      state.currentTrick.length !== 5 ||
      state.trumpSuit === null
    ) {
      return undefined;
    }

    return determineCurrentWinningPlayer(
      state.currentTrick,
      { trumpSuit: state.trumpSuit },
      { trickNumber: state.trickNumber }
    );
  }, [state]);

  async function playCollectionBefore<T>(work: () => Promise<T>): Promise<T | undefined> {
    if (collectionInProgressRef.current) {
      return undefined;
    }

    if (state === undefined || !state.isTrickComplete || completedWinnerId === undefined) {
      return work();
    }

    collectionInProgressRef.current = true;
    setCollectingWinnerId(completedWinnerId);

    try {
      if (!prefersReducedMotion) {
        await delay(COLLECT_MS);
      }

      return await work();
    } finally {
      collectionInProgressRef.current = false;
      setCollectingWinnerId(undefined);
    }
  }

  const hasRenderedNewTrickBeforeReset =
    state !== undefined &&
    displayedTrickNumberRef.current !== undefined &&
    displayedTrickNumberRef.current !== state.trickNumber;

  return {
    collectingWinnerId,
    displayedTrick: hasRenderedNewTrickBeforeReset ? [] : displayedTrick,
    isAnimating: isPlayingSequence || isResultEmphasisActive || collectingWinnerId !== undefined,
    isResultEmphasisActive,
    playCollectionBefore
  };
}

function usePrefersReducedMotion(): boolean {
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

function createTrickKey(state: PublicGameState | undefined): string {
  if (state === undefined) {
    return "none";
  }

  return [
    state.trickNumber,
    state.isTrickComplete ? "complete" : "open",
    ...state.currentTrick.map((played) => `${played.playerId}:${played.card.id}`)
  ].join("|");
}

function matchesPrefix(
  current: readonly PublicPlayedCard[],
  target: readonly PublicPlayedCard[]
): boolean {
  return current.every((played, index) => {
    const targetPlayed = target[index];

    return (
      targetPlayed !== undefined &&
      targetPlayed.playerId === played.playerId &&
      targetPlayed.card.id === played.card.id
    );
  });
}

function holdResultEmphasis(
  timers: ReturnType<typeof setTimeout>[],
  setIsResultEmphasisActive: (isActive: boolean) => void,
  prefersReducedMotion: boolean,
  onComplete?: () => void
): void {
  setIsResultEmphasisActive(true);
  const timer = setTimeout(
    () => {
      setIsResultEmphasisActive(false);
      onComplete?.();
    },
    prefersReducedMotion ? REDUCED_RESULT_HOLD_MS : RESULT_HOLD_MS
  );
  timers.push(timer);
}

function clearTimers(timers: ReturnType<typeof setTimeout>[]): void {
  while (timers.length > 0) {
    const timer = timers.pop();

    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
