#!/usr/bin/env python3
"""Validate checkpoint dimensions and optional Isaac/MuJoCo policy I/O parity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deployment_core import OBSERVATION_SLICES, StageActionProcessor, build_policy_observation
from policy_runtime import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--sample",
        help="Optional .npz with 'observation' and optional 'action' exported from Isaac Lab",
    )
    parser.add_argument("--onnx", help="Optional exported ONNX graph to compare")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def default_observation(stage: int) -> np.ndarray:
    return build_policy_observation(
        root_position_w=(0.0, 0.0, 0.4),
        root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        root_linear_velocity_w=(0.0, 0.0, 0.0),
        root_angular_velocity_w=(0.0, 0.0, 0.0),
        joint_position_isaac=(0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5),
        joint_velocity_isaac=np.zeros(12),
        foot_contacts=np.zeros(4),
        target_position_w=(1.0, 0.0, 0.0),
        trajectory_phase=0.0,
        applied_gait_modifiers=np.ones(3) if stage == 2 else None,
    )


def main() -> None:
    args = parse_args()
    obs_dim, action_dim = ((45, 12) if args.stage == 1 else (48, 15))
    policy = load_policy(
        args.checkpoint,
        expected_observation_dim=obs_dim,
        expected_action_dim=action_dim,
    )

    expected_action = None
    if args.sample:
        sample = np.load(Path(args.sample).expanduser())
        if "observation" not in sample:
            raise KeyError("Sample must contain an 'observation' array")
        observation = np.asarray(sample["observation"], dtype=np.float32)
        expected_action = np.asarray(sample["action"], dtype=np.float32) if "action" in sample else None
    else:
        observation = default_observation(args.stage)

    if observation.ndim == 1:
        observation = observation[None, :]
    action = policy.act(observation)
    StageActionProcessor(stage=args.stage).process(action[0])
    print(
        f"Checkpoint: obs={policy.info.observation_dim}, "
        f"action={policy.info.action_dim}, hidden={policy.info.hidden_dims}"
    )
    print(
        f"Normalizer: mean={policy.info.normalizer_mean_key or 'identity'}, "
        f"scale={policy.info.normalizer_scale_key or 'identity'} ({policy.info.normalizer_scale_kind})"
    )
    print(f"Observation batch: {observation.shape}; actor output: {action.shape}; finite={np.all(np.isfinite(action))}")
    print("Observation contract:")
    for name, item in OBSERVATION_SLICES.items():
        if item.stop <= obs_dim:
            print(f"  {item.start:02d}:{item.stop:02d}  {name}")

    if expected_action is not None:
        expected_action = expected_action.reshape(action.shape)
        np.testing.assert_allclose(action, expected_action, atol=args.atol, rtol=args.rtol)
        print("Isaac checkpoint action parity: OK")

    if args.onnx:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("Install onnxruntime to use --onnx") from exc
        session = ort.InferenceSession(str(Path(args.onnx).expanduser()), providers=["CPUExecutionProvider"])
        onnx_action = session.run(None, {session.get_inputs()[0].name: observation.astype(np.float32)})[0]
        np.testing.assert_allclose(action, onnx_action, atol=args.atol, rtol=args.rtol)
        print(f"ONNX parity: OK (max abs error {np.max(np.abs(action - onnx_action)):.3e})")


if __name__ == "__main__":
    main()
