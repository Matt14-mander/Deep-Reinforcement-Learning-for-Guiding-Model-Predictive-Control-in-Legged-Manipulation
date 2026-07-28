# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure NumPy utilities for safe Stage 2 gait modulation."""

from typing import Sequence, Tuple

import numpy as np


def filter_gait_modulation(
    raw_actions: np.ndarray,
    previous: np.ndarray,
    update_mask: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
    max_delta: Sequence[float],
    smoothing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp, smooth and rate-limit gait modifiers at policy updates.

    Non-update environments keep their previous modifiers exactly. This makes
    the 5 Hz policy / 50 Hz MPC frequency hierarchy explicit and prevents a
    single policy action from abruptly changing contact timing.
    """
    raw = np.asarray(raw_actions, dtype=np.float64)
    prev = np.asarray(previous, dtype=np.float64)
    mask = np.asarray(update_mask, dtype=bool).reshape(-1)
    if raw.shape != prev.shape or raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"Expected raw/previous shape (E, 3), got {raw.shape}/{prev.shape}")
    if mask.shape != (raw.shape[0],):
        raise ValueError(f"Expected update_mask shape ({raw.shape[0]},), got {mask.shape}")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must be in (0, 1]")

    lo = np.asarray(lower, dtype=np.float64).reshape(3)
    hi = np.asarray(upper, dtype=np.float64).reshape(3)
    limit = np.asarray(max_delta, dtype=np.float64).reshape(3)
    if np.any(lo >= hi) or np.any(limit <= 0.0):
        raise ValueError("Invalid gait bounds or max_delta")

    finite_raw = np.where(np.isfinite(raw), raw, prev)
    target = np.clip(finite_raw, lo, hi)
    smoothed_delta = smoothing * (target - prev)
    delta = np.clip(smoothed_delta, -limit, limit)
    delta[~mask] = 0.0
    applied = np.clip(prev + delta, lo, hi)
    return applied, delta


def scale_foothold_step(
    start_pos: np.ndarray,
    nominal_landing_pos: np.ndarray,
    step_length_scale: float,
) -> np.ndarray:
    """Scale horizontal swing displacement while preserving terrain height."""
    start = np.asarray(start_pos, dtype=np.float64).reshape(3)
    nominal = np.asarray(nominal_landing_pos, dtype=np.float64).reshape(3)
    scale = float(np.clip(step_length_scale, 0.25, 2.5))
    landing = nominal.copy()
    landing[:2] = start[:2] + scale * (nominal[:2] - start[:2])
    return landing


def advance_reference_clocks(
    step_counter: np.ndarray,
    trajectory_phase: np.ndarray,
    max_phase: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance MPC and reference counters with a saturated trajectory phase."""
    counters = np.asarray(step_counter, dtype=np.int64) + 1
    phases = np.minimum(
        np.asarray(trajectory_phase, dtype=np.int64) + 1, max(0, int(max_phase))
    )
    return counters.astype(np.int32), phases.astype(np.int32)
