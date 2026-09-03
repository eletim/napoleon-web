import { useEffect, useMemo, useRef, useState } from "react";
import type { PublicGameState, PublicPlayedCard } from "@napoleon/protocol";
import { determineCurrentWinningPlayer } from "@napoleon/game-core";
import {
  ACTION_GAP_MS,
  REDUCED_MOTION_HOLD_MS,
  TRICK_COLLECT_DURATION_MS,
  TRICK_RESULT_HOLD_MS,
  usePrefersReducedMotion
} from "./presentationTiming";

interface UseTrickAnimationOptions {
  selfPlayerId: string | undefined;
  state: PublicGameState | undefined;
}

interface TrickAnimationState {
  collectingWinnerId: string | undefined;
  displayedTrick: readonly PublicPlayedCard[];
  isAnimating: boolean;
  isResultEmphasisActive: boolean;
  /**
   * Who presentation-wise "has the turn" right now. During the pause before
   * a COM's card is revealed this is that COM (even though the server has
   * already resolved the whole trick), so the table visibly hands the turn
   * off one seat at a time instead of jumping straight to the final state.
   * Falls back to the real `state.currentPlayerId` outside of that gap.
   */
  presentationCurrentPlayerId: string | undefined;
  /** Cards revealed via the animated deal, eligible for the hand->table flight. */
  isEntryAnimated: (playerId: string, cardId: string) => boolean;
  /**
   * The trick winner as soon as all 5 cards are visibly on the table, kept
   * defined through the result hold and the collection flight. Unlike the
   * optional "tentative winner" highlight toggle, this is core information
   * about who just won and is always available once the trick is settled.
   */
  resultWinnerId: string | undefined;
  playCollectionBefore: <T>(work: () => Promise<T>) => Promise<T | undefined>;
}

export function useTrickAnimation({
  selfPlayerId,
  state
}: UseTrickAnimationOptions): TrickAnimationState {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [displayedTrick, setDisplayedTrick] = useState<readonly PublicPlayedCard[]>([]);
  const [isPlayingSequence, setIsPlayingSequence] = useState(false);
  const [isResultEmphasisActive, setIsResultEmphasisActive] = useState(false);
  const [collectingWinnerId, setCollectingWinnerId] = useState<string | undefined>();
  const [pendingActorId, setPendingActorId] = useState<string | undefined>();
  const displayedTrickRef = useRef<readonly PublicPlayedCard[]>([]);
  const displayedTrickNumberRef = useRef<number | undefined>(undefined);
  const collectionInProgressRef = useRef(false);
  const targetKeyRef = useRef("");
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const animatedEntryKeysRef = useRef<Set<string>>(new Set());

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
    setPendingActorId(undefined);

    if (state === undefined || targetTrick.length === 0) {
      setDisplayedTrick([]);
      displayedTrickRef.current = [];
      displayedTrickNumberRef.current = state?.trickNumber;
      setIsPlayingSequence(false);
      if (state === undefined) {
        animatedEntryKeysRef.current.clear();
      }
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
      // A jump straight to the final state (resumed/mismatched state, or a
      // reduced-motion viewer): show it outright rather than animating cards
      // whose hand->table flight would have no meaningful starting point.
      setDisplayedTrick(targetTrick);
      if (state.isTrickComplete && targetTrick.length === 5) {
        holdResultEmphasis(
          timersRef.current,
          setIsResultEmphasisActive,
          prefersReducedMotion
        );
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

    let elapsed = 0;
    additions.forEach((addition, index) => {
      // The local player's own card is already visible the instant they act
      // (they just clicked it); only COM turns get the presentation pause,
      // so the table doesn't feel like it's waiting on the human.
      const gap = addition.playerId === selfPlayerId ? 0 : ACTION_GAP_MS;
      elapsed += gap;
      const revealAt = elapsed;

      if (gap > 0) {
        const actorTimer = setTimeout(() => {
          setPendingActorId(addition.playerId);
        }, revealAt - gap);
        timersRef.current.push(actorTimer);
      }

      const revealTimer = setTimeout(() => {
        const nextLength = currentLength + index + 1;
        animatedEntryKeysRef.current.add(entryKey(addition.playerId, addition.card.id));
        setDisplayedTrick(targetTrick.slice(0, nextLength));
        setPendingActorId(undefined);

        if (index === additions.length - 1) {
          if (state.isTrickComplete && targetTrick.length === 5) {
            holdResultEmphasis(
              timersRef.current,
              setIsResultEmphasisActive,
              prefersReducedMotion,
              () => setIsPlayingSequence(false)
            );
            return;
          }

          setIsPlayingSequence(false);
        }
      }, revealAt);
      timersRef.current.push(revealTimer);
    });
  }, [prefersReducedMotion, selfPlayerId, state]);

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
      state.trumpSuit === null ||
      displayedTrick.length !== 5
    ) {
      return undefined;
    }

    return determineCurrentWinningPlayer(
      state.currentTrick,
      { trumpSuit: state.trumpSuit },
      { trickNumber: state.trickNumber }
    );
  }, [displayedTrick.length, state]);

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
        await delay(TRICK_COLLECT_DURATION_MS);
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
    presentationCurrentPlayerId: pendingActorId ?? state?.currentPlayerId,
    isEntryAnimated: (playerId: string, cardId: string) =>
      animatedEntryKeysRef.current.has(entryKey(playerId, cardId)),
    resultWinnerId: completedWinnerId,
    playCollectionBefore
  };
}

function entryKey(playerId: string, cardId: string): string {
  return `${playerId}:${cardId}`;
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
    prefersReducedMotion ? REDUCED_MOTION_HOLD_MS : TRICK_RESULT_HOLD_MS
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
