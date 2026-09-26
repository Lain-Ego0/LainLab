#!/usr/bin/env python3
"""Refresh the browser policy artifacts.

Run this only when refreshing the checked-in browser demo artifacts.  The
runtime demo never needs PyTorch or the source Gym project; it consumes the
resulting files in ``public/policies`` through ONNX Runtime Web.  The
handstand artifact is copied from LainLab's own exported policy; the remaining
policies are converted from the reference Gym project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

POLICIES = {
    "go2-rear-stand.onnx": ("go2_handstand/exported/policies/policy_1.pt", 45),
    "go2-trot.onnx": ("go2_trot/exported/policies/policy_trot.pt", 470),
    "go2-jump.onnx": ("go2_jump/exported/policies/policy_jump.pt", 470),
    "go2-spring-jump.onnx": ("go2_spring_jump/exported/policies/policy_1.pt", 470),
    "go2-dreamwaq.onnx": ("go2_dreamwaq/exported/policies/policy_dwaq.pt", 270),
    "go2-amp-cts.onnx": ("go2_amp_cts/exported/policies/policy_cts.pt", 270),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="My_unitree_go2_gym-main path")
    parser.add_argument(
        "--handstand",
        type=Path,
        default=(
            Path(__file__).parents[2]
            / "logs/rsl_rl/go2_handstand/2026-09-07_12-05-41_migration_2048x1000"
            / "2026-09-07_12-05-41_migration_2048x1000.onnx"
        ),
        help="LainLab handstand ONNX export",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "public" / "policies")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    handstand_target = args.output / "go2-handstand.onnx"
    if not args.handstand.is_file():
        raise FileNotFoundError(args.handstand)
    handstand_target.write_bytes(args.handstand.read_bytes())
    print(f"copied {handstand_target.name}: LainLab handstand 48 -> 12")

    for output_name, (relative_source, input_size) in POLICIES.items():
        source = args.source / "logs" / relative_source
        if not source.is_file():
            raise FileNotFoundError(source)
        policy = torch.jit.load(source, map_location="cpu").eval()
        target = args.output / output_name
        torch.onnx.export(
            policy,
            torch.zeros(1, input_size, dtype=torch.float32),
            target,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        print(f"exported {target.name}: {input_size} -> 12")


if __name__ == "__main__":
    main()
