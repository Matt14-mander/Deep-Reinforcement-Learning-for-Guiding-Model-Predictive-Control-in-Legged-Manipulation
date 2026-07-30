#!/usr/bin/env python3
"""Export a normalized RSL-RL actor to ONNX and/or TorchScript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deployment_core import ISAAC_JOINT_ORDER, OBSERVATION_SLICES, PINOCCHIO_JOINT_ORDER
from policy_runtime import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="RSL-RL model_*.pt checkpoint")
    parser.add_argument("--output", required=True, help="Output base path or .onnx/.ts path")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--format", choices=("onnx", "torchscript", "both"), default="both")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obs_dim, action_dim = ((45, 12) if args.stage == 1 else (48, 15))
    policy = load_policy(
        args.checkpoint,
        expected_observation_dim=obs_dim,
        expected_action_dim=action_dim,
    )
    output = Path(args.output).expanduser().resolve()
    base = output.with_suffix("") if output.suffix in (".onnx", ".ts", ".pt") else output
    base.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, obs_dim, dtype=torch.float32)

    written: list[Path] = []
    if args.format in ("torchscript", "both"):
        torchscript_path = base.with_suffix(".ts")
        torch.jit.trace(policy.module.cpu(), example).save(str(torchscript_path))
        written.append(torchscript_path)

    if args.format in ("onnx", "both"):
        onnx_path = base.with_suffix(".onnx")
        torch.onnx.export(
            policy.module.cpu(),
            example,
            str(onnx_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )
        written.append(onnx_path)

    metadata = {
        "schema_version": 1,
        "stage": args.stage,
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": list(policy.info.hidden_dims),
        "normalizer_mean_key": policy.info.normalizer_mean_key,
        "normalizer_scale_key": policy.info.normalizer_scale_key,
        "normalizer_scale_kind": policy.info.normalizer_scale_kind,
        "checkpoint": str(policy.checkpoint_path),
        "isaac_joint_order": list(ISAAC_JOINT_ORDER),
        "pinocchio_joint_order": list(PINOCCHIO_JOINT_ORDER),
        "observation_slices": {key: [value.start, value.stop] for key, value in OBSERVATION_SLICES.items()},
        "action_layout": {
            "bezier_parameters": [0, 12],
            "gait_modifiers": [12, 15] if args.stage == 2 else None,
        },
        "note": "The exported graph includes observation normalization but not action denormalization or MPC.",
    }
    metadata_path = base.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written.append(metadata_path)
    print("Export complete:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()

