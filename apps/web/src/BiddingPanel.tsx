import { useEffect, useMemo, useState } from "react";
import type {
  PublicBidAction,
  PublicBiddingHistoryEntry,
  PublicBiddingState,
  PublicSuit
} from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";
import {
  biddingSuitOptions,
  findBidAction,
  getBidStepperState,
  getBidTargetsForSuit,
  normalizeBidSelection,
  type BidSelection
} from "./biddingOptions";

interface BiddingPanelProps {
  bidding: PublicBiddingState | null;
  currentPlayerId: string;
  selfPlayerId: string;
  legalBidActions: readonly PublicBidAction[];
  canPass: boolean;
  isBusy: boolean;
  formatPlayerLabel: (playerId: string | null | undefined) => string;
  onBid: (action: PublicBidAction) => void;
  onPass: () => void;
}

export function BiddingPanel({
  bidding,
  currentPlayerId,
  selfPlayerId,
  legalBidActions,
  canPass,
  isBusy,
  formatPlayerLabel,
  onBid,
  onPass
}: BiddingPanelProps) {
  const [selection, setSelection] = useState<BidSelection | null>(() =>
    normalizeBidSelection(legalBidActions, null)
  );
  const isSelfTurn = currentPlayerId === selfPlayerId;

  useEffect(() => {
    setSelection((current) => normalizeBidSelection(legalBidActions, current));
  }, [legalBidActions]);

  const selectedAction = useMemo(
    () => findBidAction(legalBidActions, selection),
    [legalBidActions, selection]
  );
  const stepper = useMemo(
    () => getBidStepperState(legalBidActions, selection),
    [legalBidActions, selection]
  );
  const canOperate = isSelfTurn && !isBusy;
  const recentHistory = bidding?.history.slice(-5) ?? [];

  function handleSuitSelect(suit: PublicSuit): void {
    const targets = getBidTargetsForSuit(legalBidActions, suit);

    if (targets.length === 0) {
      return;
    }

    setSelection((current) => ({
      suit,
      targetPointCards:
        current !== null && targets.includes(current.targetPointCards)
          ? current.targetPointCards
          : targets[0]
    }));
  }

  function moveTarget(targetPointCards: number | null): void {
    if (selection === null || targetPointCards === null) {
      return;
    }

    setSelection({
      suit: selection.suit,
      targetPointCards
    });
  }

  return (
    <section className="bidding-panel" aria-label="競り">
      <div className="bidding-panel-header">
        <div>
          <h2>競り</h2>
          <p>
            {isSelfTurn
              ? "あなたの競り手番です"
              : `${formatPlayerLabel(currentPlayerId)}が競り中です`}
          </p>
        </div>
        <div className="bidding-pass-count">
          <span>連続パス</span>
          <strong>{bidding?.consecutivePassCount ?? 0}</strong>
        </div>
      </div>

      <div className="highest-bid-card">
        <span>現在の最高入札</span>
        <strong>{formatHighestBid(bidding?.highestBid ?? null, formatPlayerLabel)}</strong>
      </div>

      <div className="bid-control-group" aria-label="あなたの入札">
        <span className="control-label">スート</span>
        <div className="bid-suit-buttons">
          {biddingSuitOptions.map((suit) => {
            const isSelected = selection?.suit === suit;
            const isAvailable = getBidTargetsForSuit(legalBidActions, suit).length > 0;

            return (
              <button
                aria-label={`${formatSuitName(suit)}を選択`}
                aria-pressed={isSelected}
                className={[
                  "bid-suit-button",
                  isRedSuit(suit) ? "bid-suit-red" : "bid-suit-black",
                  isSelected ? "bid-suit-selected" : ""
                ]
                  .filter(Boolean)
                  .join(" ")}
                disabled={!canOperate || !isAvailable}
                key={suit}
                onClick={() => handleSuitSelect(suit)}
                type="button"
              >
                <span>{suitSymbols[suit]}</span>
                <small>{isSelected ? "選択中" : formatSuitName(suit)}</small>
              </button>
            );
          })}
        </div>
      </div>

      <div className="bid-control-group">
        <span className="control-label">宣言枚数</span>
        <div className="bid-stepper">
          <button
            aria-label="前の宣言枚数"
            className="stepper-button"
            disabled={!canOperate || stepper.previousTarget === null}
            onClick={() => moveTarget(stepper.previousTarget)}
            type="button"
          >
            -
          </button>
          <output className="bid-target-output">
            {selection === null ? "-" : `${selection.targetPointCards}枚`}
          </output>
          <button
            aria-label="次の宣言枚数"
            className="stepper-button"
            disabled={!canOperate || stepper.nextTarget === null}
            onClick={() => moveTarget(stepper.nextTarget)}
            type="button"
          >
            +
          </button>
        </div>
      </div>

      <div className="bid-submit-row">
        <button
          className="primary-button bid-submit-button"
          disabled={!canOperate || selectedAction === undefined}
          onClick={() => {
            if (selectedAction !== undefined) {
              onBid(selectedAction);
            }
          }}
          type="button"
        >
          {selectedAction === undefined
            ? "入札できません"
            : `${suitSymbols[selectedAction.suit]}${selectedAction.targetPointCards}枚で入札`}
        </button>
        <button
          className="secondary-button pass-button"
          disabled={!canOperate || !canPass}
          onClick={onPass}
          type="button"
        >
          パス
        </button>
      </div>

      <div className="compact-bidding-history" aria-label="競り履歴">
        <span className="control-label">競り履歴</span>
        <div className="history-strip">
          {recentHistory.length === 0 ? (
            <span className="history-empty">履歴はまだありません</span>
          ) : (
            recentHistory.map((entry, index) => (
              <span className="history-chip" key={`${entry.playerId}-${entry.type}-${index}`}>
                {formatHistoryEntry(entry, formatPlayerLabel)}
              </span>
            ))
          )}
        </div>
      </div>
    </section>
  );
}

function formatHighestBid(
  bid: PublicBiddingState["highestBid"],
  formatPlayerLabel: (playerId: string | null | undefined) => string
): string {
  if (bid === null) {
    return "なし";
  }

  return `${formatPlayerLabel(bid.playerId)} ${suitSymbols[bid.suit]}${bid.targetPointCards}枚`;
}

function formatHistoryEntry(
  entry: PublicBiddingHistoryEntry,
  formatPlayerLabel: (playerId: string | null | undefined) => string
): string {
  if (entry.type === "pass") {
    return `${formatPlayerLabel(entry.playerId)} パス`;
  }

  return `${formatPlayerLabel(entry.playerId)} ${suitSymbols[entry.suit]}${entry.targetPointCards}`;
}

function formatSuitName(suit: PublicSuit): string {
  switch (suit) {
    case "spades":
      return "スペード";
    case "clubs":
      return "クラブ";
    case "hearts":
      return "ハート";
    case "diamonds":
      return "ダイヤ";
  }
}
