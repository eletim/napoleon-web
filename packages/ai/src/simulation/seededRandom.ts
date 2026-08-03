export function createSeededRandom(seed: number): () => number {
  let state = normalizeSeed(seed);

  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function deriveSeed(baseSeed: number, streamId: string): number {
  let hash = 0x811c9dc5;
  const input = `${normalizeSeed(baseSeed)}:${streamId}`;

  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }

  return hash >>> 0;
}

export function normalizeSeed(seed: number): number {
  return seed >>> 0;
}
