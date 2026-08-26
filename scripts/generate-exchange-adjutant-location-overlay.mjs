#!/usr/bin/env node
// Issue #450 diagnostic only: recover hidden pre-exchange called-card locations.
import { createHash } from "node:crypto";
import { createReadStream, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { createInterface } from "node:readline";
import { createSeededRandom, deriveSeed } from "../packages/ai/dist/index.js";
import { createDeck, createInitialGame, shuffleDeck } from "../packages/game-core/dist/index.js";

const CLASS_NAMES = [
  "opponentSeat1", "opponentSeat2", "opponentSeat3", "opponentSeat4", "selfKittySolo"
];
const PLAYER_IDS = ["player-0", "player-1", "player-2", "player-3", "player-4"];
const args = parseArgs(process.argv.slice(2));
if (!args.output || args._.length === 0) usage();

const entries = {};
const datasetManifests = [];
for (const rawDirectory of args._) {
  const directory = resolve(rawDirectory);
  const manifestPath = resolve(directory, "manifest.json");
  const manifestBytes = readFileSync(manifestPath);
  const manifest = JSON.parse(manifestBytes);
  datasetManifests.push({
    path: manifestPath,
    sha256: createHash("sha256").update(manifestBytes).digest("hex"),
    sourceStateCount: manifest.sourceStateCount
  });
  for (const shard of manifest.shards) {
    const lines = createInterface({
      input: createReadStream(resolve(directory, shard.file)), crlfDelay: Infinity
    });
    for await (const line of lines) {
      const row = JSON.parse(line);
      if (row.candidateIndex !== 0) continue;
      if (entries[row.sourceStateKey]) throw new Error(`duplicate sourceStateKey ${row.sourceStateKey}`);
      const deal = manifest.pseudoFixedThirteen
        ? pseudoFixedDeal(manifest, row)
        : normalDeal(row);
      validateVisibleCards(deal, row);
      entries[row.sourceStateKey] = classify(deal, row);
    }
  }
}

const output = {
  artifactType: "issue450-exchange-training-location-overlay-v1",
  semantics: "actual pre-exchange called-card owner; Napoleon-relative clockwise seat; self includes original hand and kitty",
  classNames: CLASS_NAMES,
  sourceStateCount: Object.keys(entries).length,
  datasetManifests,
  entries
};
writeFileSync(resolve(args.output), `${JSON.stringify(output, null, 2)}\n`);
console.log(JSON.stringify({ output: resolve(args.output), sourceStateCount: output.sourceStateCount }));

function normalDeal(row) {
  return createInitialGame({
    playerIds: PLAYER_IDS,
    rng: createSeededRandom(deriveSeed(row.dealSeed, "game"))
  });
}

function pseudoFixedDeal(manifest, row) {
  const repeats = manifest.pseudoFixedThirteen.acceptedDealsPerFixedThirteenGroup;
  const groupIndex = Math.floor(row.sourceIndex / repeats);
  const shuffled = shuffleDeck(
    createDeck(),
    createSeededRandom(deriveSeed(manifest.startSeed, `fixed-thirteen-group:${groupIndex}`))
  );
  const original = shuffled.slice(0, 10);
  const kitty = shuffled.slice(10, 13);
  const fixed = new Set([...original, ...kitty].map((card) => card.id));
  const remaining = createDeck().filter((card) => !fixed.has(card.id));
  const hidden = shuffleDeck(
    remaining,
    createSeededRandom(deriveSeed(row.dealSeed, "pseudo-fixed-hidden-opponents"))
  );
  const candidateIndex = manifest.pseudoFixedThirteen.candidatePlayerIndex;
  const players = PLAYER_IDS.map((id, playerIndex) => {
    if (playerIndex === candidateIndex) return { id, hand: original };
    const offset = playerIndex < candidateIndex ? playerIndex : playerIndex - 1;
    return { id, hand: hidden.slice(offset * 10, offset * 10 + 10) };
  });
  return { players, unusedCards: kitty };
}

function classify(deal, row) {
  const cardId = row.calledAdjutantCardId;
  const ownerSeat = deal.players.findIndex((player) => player.hand.some((card) => card.id === cardId));
  const inKitty = deal.unusedCards.some((card) => card.id === cardId);
  if (ownerSeat < 0 && !inKitty) throw new Error(`called card ${cardId} not found`);
  const relative = ownerSeat < 0 ? 0 : (ownerSeat - row.napoleonSeatIndex + 5) % 5;
  const classIndex = relative === 0 ? 4 : relative - 1;
  const calledCardOrigin = row.originalHandCardIds.includes(cardId)
    ? "originalHand" : row.kittyPickupCardIds.includes(cardId) ? "kitty" : "opponentHand";
  return {
    classIndex,
    className: CLASS_NAMES[classIndex],
    calledCardOrigin,
    dealSeed: row.dealSeed,
    napoleonSeatIndex: row.napoleonSeatIndex,
    calledAdjutantCardId: cardId
  };
}

function validateVisibleCards(deal, row) {
  const napoleon = deal.players[row.napoleonSeatIndex].hand.map((card) => card.id);
  const kitty = deal.unusedCards.map((card) => card.id);
  if (!sameSet(napoleon, row.originalHandCardIds) || !sameSet(kitty, row.kittyPickupCardIds)) {
    throw new Error(`reconstructed deal mismatch for ${row.sourceStateKey}`);
  }
}
function sameSet(left, right) {
  return [...left].sort().join("|") === [...right].sort().join("|");
}
function parseArgs(values) {
  const result = { _: [] };
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] === "--output") result.output = values[++index];
    else result._.push(values[index]);
  }
  return result;
}
function usage() {
  console.error("usage: node scripts/generate-exchange-adjutant-location-overlay.mjs --output FILE DATASET...");
  process.exit(2);
}
