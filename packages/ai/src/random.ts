export function selectRandom<T>(items: readonly T[], rng: () => number): T {
  if (items.length === 0) {
    throw new Error("Cannot select from an empty list.");
  }

  const index = Math.floor(rng() * items.length);
  return items[Math.min(index, items.length - 1)];
}
