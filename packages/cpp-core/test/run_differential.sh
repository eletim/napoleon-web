#!/usr/bin/env bash
set -euo pipefail

node dist/test/differential.test.js

while IFS= read -r case_name; do
  if [[ -z "$case_name" ]]; then
    continue
  fi

  case_dir=".differential/$case_name"
  seed="$(cat "$case_dir/seed.txt")"
  ./build/napoleon_core_cli --snapshot --seed "$seed" \
    < "$case_dir/actions.txt" \
    > "$case_dir/actual.json"

  node -e '
    const assert = require("node:assert/strict");
    const fs = require("node:fs");
    const [expectedPath, actualPath, caseName] = process.argv.slice(1);
    const expected = JSON.parse(fs.readFileSync(expectedPath, "utf8"));
    const actual = JSON.parse(fs.readFileSync(actualPath, "utf8"));
    assert.deepStrictEqual(actual, expected);
    console.log(`differential ok: ${caseName}`);
  ' "$case_dir/expected.json" "$case_dir/actual.json" "$case_name"

  if [[ -f "$case_dir/rule_based_expected.json" ]]; then
    agent_seed="$(cat "$case_dir/rule_based_seed.txt")"
    ./build/napoleon_core_cli --select-rule-based-action --seed "$seed" --agent-seed "$agent_seed" \
      < "$case_dir/actions.txt" \
      > "$case_dir/rule_based_actual.json"

    node -e '
      const assert = require("node:assert/strict");
      const fs = require("node:fs");
      const [expectedPath, actualPath, snapshotPath, caseName] = process.argv.slice(1);
      const expected = JSON.parse(fs.readFileSync(expectedPath, "utf8"));
      const actual = JSON.parse(fs.readFileSync(actualPath, "utf8"));
      const snapshot = JSON.parse(fs.readFileSync(snapshotPath, "utf8"));
      assert.deepStrictEqual(actual, expected);
      assert(
        snapshot.currentPlayerLegalActions.some((action) => JSON.stringify(action) === JSON.stringify(actual)),
        "selected RuleBased action must be legal"
      );
      console.log(`rule-based parity ok: ${caseName}`);
    ' "$case_dir/rule_based_expected.json" "$case_dir/rule_based_actual.json" "$case_dir/actual.json" "$case_name"
  fi
done < .differential/cases.txt

echo "C++ core differential harness ok"
