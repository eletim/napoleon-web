import type { PublicGameState } from "@napoleon/protocol";
import type { Seat, TablePlayer } from "./tableTypes";

export function createTablePlayers(state: PublicGameState | undefined): ReadonlyArray<TablePlayer> {
  const opponents = state?.opponents ?? [];
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
      isSelf: false
    };
  });

  return [
    ...opponentPlayers,
    {
      id: state?.self.id ?? "player-0",
      label: "自分",
      seat: "self",
      handCount: state?.self.handCount ?? 0,
      capturedPointCards: state?.self.capturedPointCards ?? [],
      isSelf: true
    }
  ];
}
