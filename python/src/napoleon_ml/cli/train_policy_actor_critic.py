"""Train a playing policy checkpoint with Actor-Critic v1 self-play trajectories."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from napoleon_ml.policy.actor_critic import ACTOR_CRITIC_ALGORITHM

from .train_policy_reinforce import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if not any(arg == "--algorithm" or arg.startswith("--algorithm=") for arg in args):
        args = [*args, "--algorithm", ACTOR_CRITIC_ALGORITHM]
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
