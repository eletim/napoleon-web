"""Generate the diagnostic Issue #450 oracle overlay for the fixed #446 audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS
from napoleon_ml.exchange_value.full_gold_audit import ADJUTANT_COUNT, load_exchange_full_gold_audit
from napoleon_ml.exchange_value.oracle_location import ADJUTANT_LOCATION_CLASS_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_directory", type=Path)
    parser.add_argument("--core-cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = load_exchange_full_gold_audit(args.audit_directory)
    sources = audit.manifest["sourceDiagnostics"]
    request = "".join(f"{row['seed']} {row['napoleonSeatIndex']}\n" for row in sources)
    completed = subprocess.run(
        [str(args.core_cli), "--adjutant-location-oracle"],
        input=request,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != len(sources):
        raise ValueError("core oracle row count differs from fixed audit source count.")
    classes = np.asarray([
        [row["classIndices"][row["cardIds"].index(card_id)] for card_id in EXPECTED_CARD_IDS]
        for row in rows
    ], dtype=np.int64)
    if classes.shape != (len(sources), ADJUTANT_COUNT):
        raise ValueError("core oracle class matrix shape mismatch.")
    for source_index in range(len(sources)):
        state = np.asarray(audit.state_features[source_index * ADJUTANT_COUNT])
        self_side = set(np.flatnonzero(state[:53])) | set(np.flatnonzero(state[53:106]))
        if any(classes[source_index, card] != 4 for card in self_side):
            raise ValueError(f"source {source_index}: self/kitty class mismatch.")
        if any(classes[source_index, card] == 4 for card in set(range(53)) - self_side):
            raise ValueError(f"source {source_index}: opponent class mismatch.")
    output = {
        "artifactType": "issue450-fixed-full-gold-location-overlay-v1",
        "fixedHoldoutManifestSha256": audit.manifest["fixedHoldout"]["manifestSha256"],
        "sourceStateCount": len(sources),
        "groupCount": len(sources) * ADJUTANT_COUNT,
        "classNames": list(ADJUTANT_LOCATION_CLASS_NAMES),
        "semantics": (
            "actual pre-exchange owner; Napoleon-relative clockwise seat; "
            "original hand and kitty map to selfKittySolo"
        ),
        "sourceSeeds": [int(row["seed"]) for row in sources],
        "classIndices": classes.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sourceStateCount": len(sources)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
