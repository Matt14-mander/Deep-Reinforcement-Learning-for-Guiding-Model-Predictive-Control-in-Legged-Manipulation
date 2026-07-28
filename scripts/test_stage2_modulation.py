#!/usr/bin/env python3
"""Dependency-light tests for Stage 2 gait/swing action processing."""

import os
import sys
import types

import numpy as np


PACKAGE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "source", "RL_Bezier_MPC", "RL_Bezier_MPC"
))
root_package = types.ModuleType("RL_Bezier_MPC")
root_package.__path__ = [PACKAGE_ROOT]
sys.modules.setdefault("RL_Bezier_MPC", root_package)
gait_package = types.ModuleType("RL_Bezier_MPC.gait")
gait_package.__path__ = [os.path.join(PACKAGE_ROOT, "gait")]
sys.modules.setdefault("RL_Bezier_MPC.gait", gait_package)

from RL_Bezier_MPC.gait.stage2_modulation import (  # noqa: E402
    advance_reference_clocks,
    filter_gait_modulation,
    scale_foothold_step,
)


def main():
    previous = np.ones((2, 3))
    raw = np.array([[2.0, 0.5, 2.0], [0.5, 2.0, 0.5]])
    applied, delta = filter_gait_modulation(
        raw_actions=raw,
        previous=previous,
        update_mask=np.array([True, False]),
        lower=(0.5, 0.5, 0.5),
        upper=(2.0, 2.0, 2.0),
        max_delta=(0.15, 0.10, 0.10),
        smoothing=1.0,
    )
    assert np.allclose(applied[0], [1.15, 0.90, 1.10])
    assert np.allclose(delta[0], [0.15, -0.10, 0.10])
    assert np.allclose(applied[1], previous[1])
    assert np.allclose(delta[1], 0.0)
    print("[1/4] policy-rate hold, bounds and per-update rate limit OK")

    smooth, smooth_delta = filter_gait_modulation(
        raw_actions=np.array([[1.2, 0.8, 1.4]]),
        previous=np.ones((1, 3)),
        update_mask=np.array([True]),
        lower=(0.5, 0.5, 0.5),
        upper=(2.0, 2.0, 2.0),
        max_delta=(1.0, 1.0, 1.0),
        smoothing=0.5,
    )
    assert np.allclose(smooth, [[1.1, 0.9, 1.2]])
    assert np.allclose(smooth_delta, [[0.1, -0.1, 0.2]])
    print("[2/4] exponential smoothing OK")

    finite, _ = filter_gait_modulation(
        raw_actions=np.array([[np.nan, np.inf, -np.inf]]),
        previous=np.ones((1, 3)),
        update_mask=np.array([True]),
        lower=(0.5, 0.5, 0.5),
        upper=(2.0, 2.0, 2.0),
        max_delta=(0.1, 0.1, 0.1),
        smoothing=1.0,
    )
    assert np.all(np.isfinite(finite)) and np.allclose(finite, 1.0)

    landing = scale_foothold_step(
        start_pos=np.array([0.0, 0.0, 0.02]),
        nominal_landing_pos=np.array([0.4, -0.2, 0.035]),
        step_length_scale=0.5,
    )
    assert np.allclose(landing, [0.2, -0.1, 0.035])
    print("[3/4] step-length modulation changes XY and preserves terrain Z")

    counters, phases = advance_reference_clocks(
        np.array([0, 9, 15]), np.array([0, 149, 150]), max_phase=150
    )
    assert np.array_equal(counters, [1, 10, 16])
    assert np.array_equal(phases, [1, 150, 150])
    print("[4/4] MPC/reference clocks advance and trajectory phase saturates")
    print("\nALL STAGE 2 MODULATION TESTS PASSED")


if __name__ == "__main__":
    main()
