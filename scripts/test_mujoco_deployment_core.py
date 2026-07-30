#!/usr/bin/env python3
"""Dependency-light checks for the MuJoCo deployment data contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parent / "mujoco_deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from deployment_core import (  # noqa: E402
    ISAAC_JOINT_ORDER,
    PINOCCHIO_JOINT_ORDER,
    ComReferenceManager,
    StageActionProcessor,
    build_policy_observation,
    reorder_indices,
)


def main() -> None:
    pin_values = np.arange(12)
    pin_to_isaac = reorder_indices(PINOCCHIO_JOINT_ORDER, ISAAC_JOINT_ORDER)
    np.testing.assert_array_equal(pin_values[pin_to_isaac], [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])

    base_kwargs = dict(
        root_position_w=(0.0, 0.0, 0.4),
        root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        root_linear_velocity_w=np.zeros(3),
        root_angular_velocity_w=np.zeros(3),
        joint_position_isaac=np.zeros(12),
        joint_velocity_isaac=np.zeros(12),
        foot_contacts=np.zeros(4),
        target_position_w=(1.0, 0.0, 0.0),
        trajectory_phase=0.0,
    )
    assert build_policy_observation(**base_kwargs).shape == (45,)
    assert build_policy_observation(**base_kwargs, applied_gait_modifiers=np.ones(3)).shape == (48,)

    processor = StageActionProcessor(stage=2)
    first = processor.process(np.zeros(15))
    assert first.bezier_parameters.shape == (12,)
    np.testing.assert_allclose(first.gait_modifiers, [1.125, 1.1, 1.1])
    second = processor.process(np.ones(15))
    np.testing.assert_array_less(second.gait_delta, [0.151, 0.101, 0.101])

    manager = ComReferenceManager()
    reference = manager.update(first.bezier_parameters, (0.0, 0.0, 0.4))
    assert reference.shape == (25, 3)
    assert manager.normalized_phase == 1 / 150
    np.testing.assert_allclose(reference[0], [0.0, 0.0, 0.4])
    assert np.all(np.isfinite(reference))
    print("MuJoCo deployment core checks passed")


if __name__ == "__main__":
    main()
