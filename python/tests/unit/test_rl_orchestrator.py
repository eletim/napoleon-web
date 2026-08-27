from __future__ import annotations

from napoleon_ml.rl_orchestrator import (
    _cpp_policy_win_by_seed_rotation,
    _policy_win_by_seed_rotation,
)


def test_policy_win_map_skips_typescript_all_pass_records() -> None:
    result: dict[str, object] = {
        "run": {
            "games": [
                {
                    "status": "completed",
                    "resultType": "all-pass",
                    "seed": 1,
                    "rotationOffset": 0,
                    "winner": None,
                    "seats": [
                        {
                            "sourceAgentIndex": 0,
                            "role": "starter",
                        }
                    ],
                },
                {
                    "status": "completed",
                    "resultType": "standard",
                    "seed": 2,
                    "rotationOffset": 0,
                    "winner": "alliance",
                    "seats": [
                        {
                            "sourceAgentIndex": 0,
                            "role": "alliance",
                        }
                    ],
                },
            ]
        }
    }

    assert _policy_win_by_seed_rotation(result) == {(2, 0): True}


def test_policy_win_map_skips_cpp_all_pass_records() -> None:
    result: dict[str, object] = {
        "games": [
            {
                "status": "completed",
                "resultType": "all-pass",
                "seed": 1,
                "rotationOffset": 0,
                "contract": None,
                "winner": None,
                "roster": {
                    "seats": [
                        {
                            "seatIndex": 0,
                            "agent": {"type": "current-policy"},
                        }
                    ]
                },
            },
            {
                "status": "completed",
                "resultType": "standard",
                "seed": 2,
                "rotationOffset": 0,
                "contract": {
                    "napoleonPlayerId": "player-1",
                    "adjutantPlayerId": None,
                },
                "winner": "alliance",
                "roster": {
                    "seats": [
                        {
                            "seatIndex": 0,
                            "agent": {"type": "current-policy"},
                        }
                    ]
                },
            },
        ]
    }

    assert _cpp_policy_win_by_seed_rotation(result) == {(2, 0): True}
