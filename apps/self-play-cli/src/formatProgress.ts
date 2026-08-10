import type { DatasetGenerationProgress } from "@napoleon/training-data";

export function formatProgress(
  progress: DatasetGenerationProgress,
  options: { rolloutWorkers?: number; rolloutConcurrency?: number } = {}
): string {
  const parts = [
    `games: ${progress.completedGames} / ${progress.totalGames}`,
    `samples: ${progress.sampleCount}`,
    `shards: ${progress.completedShards}`,
    `seed: ${progress.currentSeed}`
  ];

  if (options.rolloutWorkers !== undefined) {
    parts.push(`workers: ${options.rolloutWorkers}`);
  }
  if (options.rolloutConcurrency !== undefined) {
    parts.push(`concurrency: ${options.rolloutConcurrency}`);
  }

  return parts.join(" | ");
}

export function createProgressReporter(
  totalGames: number,
  write: (text: string) => void,
  options: { rolloutWorkers?: number; rolloutConcurrency?: number } = {}
): (progress: DatasetGenerationProgress) => void {
  const interval = Math.max(1, Math.ceil(totalGames / 100));
  let lastCompletedShards = 0;

  return (progress) => {
    const shouldReport =
      progress.completedGames === progress.totalGames ||
      progress.completedGames % interval === 0 ||
      progress.completedShards !== lastCompletedShards;

    if (!shouldReport) {
      return;
    }

    lastCompletedShards = progress.completedShards;
    write(`${formatProgress(progress, options)}\n`);
  };
}
