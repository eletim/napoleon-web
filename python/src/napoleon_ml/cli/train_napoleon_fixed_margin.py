"""Train Issue #423 Napoleon-fixed empirical contract margin models."""

from __future__ import annotations

from collections.abc import Sequence

from napoleon_ml.cli.train_fixed_hand_bidding_margin import main as fixed_hand_main


def main(argv: Sequence[str] | None = None) -> int:
    return fixed_hand_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
