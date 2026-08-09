import type { PublicGameState } from "@napoleon/protocol";
import {
  createLatestBiddingDeclarations,
  emptyBiddingDeclaration
} from "./biddingDeclarations";
import type { Seat, TablePlayer } from "./tableTypes";

export function createTablePlayers(state: PublicGameState | undefined): ReadonlyArray<TablePlayer> {
  const opponents = state?.opponents ?? [];
  const biddingDeclarations =
    state?.phase === "bidding" ? createLatestBiddingDeclarations(state.bidding) : null;
  const seatDefinitions: ReadonlyArray<{ seat: Seat; label: string }> = [
    { seat: "left", label: "左側AI" },
    { seat: "top-left", label: "奥左AI" },
    { seat: "top-right", label: "奥右AI" },
    { seat: "right", label: "右側AI" }
  ];
  const opponentPlayers = seatDefinitions.map((definition, index) => {
    const player = opponents[index];

    return {
      id: player?.id ?? `player-${index + 1}`,
      label: definition.label,
      seat: definition.seat,
      handCount: player?.handCount ?? 0,
      capturedPointCards: player?.capturedPointCards ?? [],
      isSelf: false,
      biddingDeclaration:
        biddingDeclarations === null
          ? undefined
          : (biddingDeclarations.get(player?.id ?? `player-${index + 1}`) ??
            emptyBiddingDeclaration)
    };
  });
  const selfId = state?.self.id ?? "player-0";

  return [
    ...opponentPlayers,
    {
      id: selfId,
      label: "自分",
      seat: "self",
      handCount: state?.self.handCount ?? 0,
      capturedPointCards: state?.self.capturedPointCards ?? [],
      isSelf: true,
      biddingDeclaration:
        biddingDeclarations === null
          ? undefined
          : (biddingDeclarations.get(selfId) ?? emptyBiddingDeclaration)
    }
  ];
}
