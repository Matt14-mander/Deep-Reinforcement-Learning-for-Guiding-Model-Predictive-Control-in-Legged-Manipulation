"""Canonical Stage-1 data contracts for the simulator/MPC boundary.

The classes in this module deliberately depend only on NumPy.  They describe
the semantic protocol; :mod:`RL_Bezier_MPC.mpc_cluster` is one transport for it.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _array(name: str, value, shape, dtype=None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} has shape {result.shape}, expected {shape}")
    return result


@dataclass(frozen=True)
class RobotStateBatch:
    """State sampled from one IsaacLab physics step.

    ``q_pin`` and ``v_pin`` use Pinocchio floating-base ordering.  Contact
    fields are part of the canonical contract even though the current
    Crocoddyl adapter does not consume them yet.
    """

    q_pin: np.ndarray
    v_pin: np.ndarray
    foot_pos_w: np.ndarray
    physics_step_id: np.ndarray
    reset_generation: np.ndarray
    timestamp: Optional[np.ndarray] = None
    foot_vel_w: Optional[np.ndarray] = None
    foot_contact: Optional[np.ndarray] = None
    foot_force_w: Optional[np.ndarray] = None

    def __post_init__(self):
        e = np.asarray(self.q_pin).shape[0]
        object.__setattr__(self, "q_pin", _array("q_pin", self.q_pin, (e, 19), np.float64))
        object.__setattr__(self, "v_pin", _array("v_pin", self.v_pin, (e, 18), np.float64))
        object.__setattr__(self, "foot_pos_w", _array("foot_pos_w", self.foot_pos_w, (e, 4, 3), np.float64))
        object.__setattr__(self, "physics_step_id", _array("physics_step_id", self.physics_step_id, (e,), np.int64))
        object.__setattr__(self, "reset_generation", _array("reset_generation", self.reset_generation, (e,), np.int64))
        for name, shape, dtype in (
            ("timestamp", (e,), np.float64),
            ("foot_vel_w", (e, 4, 3), np.float64),
            ("foot_contact", (e, 4), bool),
            ("foot_force_w", (e, 4, 3), np.float64),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _array(name, value, shape, dtype))

    @property
    def pin_state(self) -> np.ndarray:
        return np.concatenate((self.q_pin, self.v_pin), axis=1)


@dataclass(frozen=True)
class MPCCommandBatch:
    """References and gait modulation paired with a robot-state batch."""

    com_reference: np.ndarray
    gait: np.ndarray
    solve_mask: np.ndarray

    def __post_init__(self):
        reference = np.asarray(self.com_reference, dtype=np.float64)
        if reference.ndim != 3 or reference.shape[2] != 3:
            raise ValueError(
                "com_reference must have shape (num_envs, horizon_steps, 3), "
                f"got {reference.shape}"
            )
        e = reference.shape[0]
        object.__setattr__(self, "com_reference", reference)
        object.__setattr__(self, "gait", _array("gait", self.gait, (e, 3), np.float64))
        object.__setattr__(self, "solve_mask", _array("solve_mask", self.solve_mask, (e,), bool))


@dataclass(frozen=True)
class MPCOutputBatch:
    """A solver response whose provenance can be checked before actuation."""

    tau_ff: np.ndarray
    q_ref: np.ndarray
    dq_ref: np.ndarray
    cost: np.ndarray
    converged: np.ndarray
    status: np.ndarray
    solve_time: np.ndarray
    iterations: np.ndarray
    dynamics_gap: np.ndarray
    constraint_violation: np.ndarray
    source_state_id: np.ndarray
    solution_id: np.ndarray
    reset_generation: np.ndarray
    source_timestamp: np.ndarray
    solution_age: np.ndarray
    fresh: np.ndarray

    def __post_init__(self):
        e = np.asarray(self.tau_ff).shape[0]
        for name, shape, dtype in (
            ("tau_ff", (e, 12), np.float64),
            ("q_ref", (e, 12), np.float64),
            ("dq_ref", (e, 12), np.float64),
            ("cost", (e,), np.float64),
            ("converged", (e,), bool),
            ("status", (e,), np.float64),
            ("solve_time", (e,), np.float64),
            ("iterations", (e,), np.float64),
            ("dynamics_gap", (e,), np.float64),
            ("constraint_violation", (e,), np.float64),
            ("source_state_id", (e,), np.int64),
            ("solution_id", (e,), np.int64),
            ("reset_generation", (e,), np.int64),
            ("source_timestamp", (e,), np.float64),
            ("solution_age", (e,), np.float64),
            ("fresh", (e,), bool),
        ):
            object.__setattr__(self, name, _array(name, getattr(self, name), shape, dtype))
