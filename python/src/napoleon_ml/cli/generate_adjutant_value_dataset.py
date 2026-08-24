"""Generate Issue #446 adjutant joint-value datasets with a compact396 scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

from napoleon_ml.exchange_value import load_exchange_value_checkpoint
from napoleon_ml.policy.device import resolve_torch_device

REQUEST_MAGIC = 0x3151544A
RESPONSE_MAGIC = 0x3153544A
DONE_MAGIC = 0x3044544A
ADJUTANT_COUNT = 53
DISCARD_COUNT = 286
EXCHANGE_FEATURE_COUNT = 396


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-cli", type=Path, required=True)
    parser.add_argument("--exchange-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("proposal", "full-gold"), required=True)
    parser.add_argument("--states", type=int, required=True)
    parser.add_argument("--start-seed", type=int, default=446000000)
    parser.add_argument("--max-deal-attempts", type=int)
    parser.add_argument("--proposal-top-k", type=int, default=16)
    parser.add_argument("--diversity-count", type=int, default=8)
    parser.add_argument("--scorer-top-k", type=int, default=64)
    parser.add_argument("--agent-seed", type=int, default=446)
    parser.add_argument("--bidding-policy-id", default="frozen-raise-v1")
    parser.add_argument(
        "--bidding-margin-onnx",
        type=Path,
        default=Path("benchmarks/bidding-margin-policies/frozen-raise-v1/margin.onnx"),
    )
    parser.add_argument("--playing-policy-id", default="ppo-separated-v1000")
    parser.add_argument(
        "--playing-policy-onnx",
        type=Path,
        default=Path("benchmarks/playing-policies/ppo-separated-v1000/policy.onnx"),
    )
    parser.add_argument(
        "--playing-critic-onnx",
        type=Path,
        default=Path("benchmarks/playing-policies/ppo-separated-v1000/critic.onnx"),
    )
    parser.add_argument("--policy-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--score-batch-size", type=int, default=8192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.states <= 0:
        raise ValueError("--states must be positive.")
    if args.scorer_top_k <= 0 or args.scorer_top_k > DISCARD_COUNT:
        raise ValueError("--scorer-top-k must be in [1,286].")
    if args.proposal_top_k <= 0 or args.proposal_top_k > args.scorer_top_k:
        raise ValueError("--proposal-top-k must be in [1,scorer-top-k].")
    if args.score_batch_size <= 0:
        raise ValueError("--score-batch-size must be positive.")

    model, checkpoint = load_exchange_value_checkpoint(args.exchange_checkpoint)
    device = resolve_torch_device(args.device)
    model.to(device.torch_device)
    model.eval()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.cpp_cli),
        "--mode",
        args.mode,
        "--output-directory",
        str(args.output_directory),
        "--states",
        str(args.states),
        "--start-seed",
        str(args.start_seed),
        "--proposal-top-k",
        str(args.proposal_top_k),
        "--diversity-count",
        str(args.diversity_count),
        "--scorer-top-k",
        str(args.scorer_top_k),
        "--agent-seed",
        str(args.agent_seed),
        "--bidding-policy-id",
        args.bidding_policy_id,
        "--bidding-margin-onnx",
        str(args.bidding_margin_onnx),
        "--playing-policy-id",
        args.playing_policy_id,
        "--playing-policy-onnx",
        str(args.playing_policy_onnx),
        "--playing-critic-onnx",
        str(args.playing_critic_onnx),
        "--policy-device",
        args.policy_device,
    ]
    if args.max_deal_attempts is not None:
        command.extend(["--max-deal-attempts", str(args.max_deal_attempts)])

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("failed to open C++ stream pipes.")

    scored_source_states = 0
    try:
        while True:
            magic_bytes = _read_exact(process.stdout, 4)
            magic = struct.unpack("<I", magic_bytes)[0]
            if magic == DONE_MAGIC:
                break
            if magic != REQUEST_MAGIC:
                raise RuntimeError(f"invalid request magic: {magic:#x}")
            source_index, adj_count, discard_count, feature_count = struct.unpack(
                "<IIII", _read_exact(process.stdout, 16)
            )
            if (
                adj_count != ADJUTANT_COUNT
                or discard_count != DISCARD_COUNT
                or feature_count != EXCHANGE_FEATURE_COUNT
            ):
                raise RuntimeError(
                    "unexpected scorer request shape: "
                    f"{adj_count=} {discard_count=} {feature_count=}"
                )
            item_count = adj_count * discard_count * feature_count
            matrix_bytes = _read_exact(process.stdout, item_count * 4)
            matrix = np.frombuffer(matrix_bytes, dtype="<f4").reshape(
                adj_count * discard_count, feature_count
            )
            top_indices = _score_topk(
                model,
                matrix,
                scorer_top_k=args.scorer_top_k,
                batch_size=args.score_batch_size,
                device=device.torch_device,
            )
            process.stdin.write(
                struct.pack(
                    "<IIII",
                    RESPONSE_MAGIC,
                    source_index,
                    adj_count,
                    args.scorer_top_k,
                )
            )
            process.stdin.write(top_indices.astype("<u4", copy=False).tobytes(order="C"))
            process.stdin.flush()
            scored_source_states += 1
    finally:
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"C++ stream teacher exited with code {return_code}.")

    report = {
        "mode": args.mode,
        "requestedStates": args.states,
        "scoredSourceStates": scored_source_states,
        "cppCommand": command,
        "exchangeCheckpoint": {
            "path": str(args.exchange_checkpoint),
            "sha256": _sha256(args.exchange_checkpoint),
            "trainingConfig": checkpoint.get("trainingConfig"),
            "targetStandardization": checkpoint.get("targetStandardization"),
        },
        "scorer": {
            "topK": args.scorer_top_k,
            "proposalTopK": args.proposal_top_k,
            "diversityCount": args.diversity_count,
            "device": device.to_metadata(),
            "batchSize": args.score_batch_size,
        },
        "datasetFiles": _dataset_file_hashes(args.output_directory),
    }
    (args.output_directory / "generation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _read_exact(stream: BinaryIO, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected EOF while reading {byte_count} bytes.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _score_topk(
    model: torch.nn.Module,
    matrix: np.ndarray,
    *,
    scorer_top_k: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            batch = torch.from_numpy(np.array(matrix[start : start + batch_size], copy=True)).to(
                device=device,
                dtype=torch.float32,
            )
            predictions.append(model(batch).detach().cpu())
    scores = torch.cat(predictions, dim=0).reshape(ADJUTANT_COUNT, DISCARD_COUNT)
    return torch.topk(scores, k=scorer_top_k, dim=1).indices.cpu().numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_file_hashes(directory: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for file_name in (
        "manifest.json",
        "features.f32",
        "contract-margin.f32",
        "relative-reward.f32",
        "state-index.u32",
        "candidate-card.u8",
    ):
        path = directory / file_name
        if path.exists():
            result[file_name] = {
                "byteLength": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
