import type { PublicMatchState } from "@napoleon/protocol";

export function getMatchAdvanceLabel(match: PublicMatchState): string {
  return match.remainingRounds === 0 ? "試合結果へ" : "次局へ";
}
