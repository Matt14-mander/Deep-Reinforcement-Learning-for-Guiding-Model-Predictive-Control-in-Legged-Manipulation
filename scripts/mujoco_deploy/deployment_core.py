"""Pure NumPy data contracts shared by the MuJoCo deployment tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


FOOT_ORDER = ("LF", "RF", "LH", "RH")

# Isaac Lab exposes the USD articulation in this order in the current Go2 task.
ISAAC_JOINT_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)

# Pinocchio and the Unitree URDF use leg-major order.
PINOCCHIO_JOINT_ORDER = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

STANDING_JOINTS_PIN = np.asarray(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5],
    dtype=np.float64,
)

OBSERVATION_SLICES = {
    "root_position_w": slice(0, 3),
    "root_quaternion_wxyz": slice(3, 7),
    "root_linear_velocity_w": slice(7, 10),
    "root_angular_velocity_w": slice(10, 13),
    "joint_position_isaac": slice(13, 25),
    "joint_velocity_isaac": slice(25, 37),
    "foot_contacts": slice(37, 41),
    "target_position_w": slice(41, 44),
    "trajectory_phase": slice(44, 45),
    "applied_gait_modifiers": slice(45, 48),
}


def canonical_joint_name(name: str) -> str:
    """Normalize common MuJoCo/URDF actuator and joint naming variants."""
    value = name.strip().lower().replace("-", "_")
    for suffix in ("_joint", "_motor", "_actuator"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def reorder_indices(source_names: Iterable[str], target_names: Iterable[str]) -> np.ndarray:
    """Return indices that reorder a source vector into target-name order."""
    source = list(source_names)
    target = list(target_names)
    canonical = {}
    for index, name in enumerate(source):
        key = canonical_joint_name(name)
        if key in canonical:
            raise ValueError(f"Duplicate canonical joint name {key!r} in {source}")
        canonical[key] = index
    missing = [name for name in target if canonical_joint_name(name) not in canonical]
    if missing:
        raise ValueError(f"Missing joints {missing}; available names are {source}")
    return np.asarray([canonical[canonical_joint_name(name)] for name in target], dtype=np.int32)


def reorder(values: Sequence[float], source_names: Iterable[str], target_names: Iterable[str]) -> np.ndarray:
    values_array = np.asarray(values)
    source = list(source_names)
    indices = reorder_indices(source, target_names)
    if values_array.shape[-1] != len(source):
        raise ValueError(f"Expected final dimension {len(source)}, got {values_array.shape}")
    return values_array[..., indices]


def build_policy_observation(
    *,
    root_position_w: Sequence[float],
    root_quaternion_wxyz: Sequence[float],
    root_linear_velocity_w: Sequence[float],
    root_angular_velocity_w: Sequence[float],
    joint_position_isaac: Sequence[float],
    joint_velocity_isaac: Sequence[float],
    foot_contacts: Sequence[float],
    target_position_w: Sequence[float],
    trajectory_phase: float,
    applied_gait_modifiers: Sequence[float] | None = None,
) -> np.ndarray:
    """Build the exact 45D Stage 1 or 48D Stage 2 policy observation."""
    parts = (
        np.asarray(root_position_w, dtype=np.float32).reshape(3),
        np.asarray(root_quaternion_wxyz, dtype=np.float32).reshape(4),
        np.asarray(root_linear_velocity_w, dtype=np.float32).reshape(3),
        np.asarray(root_angular_velocity_w, dtype=np.float32).reshape(3),
        np.asarray(joint_position_isaac, dtype=np.float32).reshape(12),
        np.asarray(joint_velocity_isaac, dtype=np.float32).reshape(12),
        np.asarray(foot_contacts, dtype=np.float32).reshape(4),
        np.asarray(target_position_w, dtype=np.float32).reshape(3),
        np.asarray([trajectory_phase], dtype=np.float32),
    )
    observation = np.concatenate(parts)
    if applied_gait_modifiers is not None:
        observation = np.concatenate(
            (observation, np.asarray(applied_gait_modifiers, dtype=np.float32).reshape(3))
        )
    expected = 48 if applied_gait_modifiers is not None else 45
    if observation.shape != (expected,) or not np.all(np.isfinite(observation)):
        raise ValueError(
            f"Invalid policy observation: shape={observation.shape}, "
            f"finite={np.all(np.isfinite(observation))}"
        )
    return observation


def training_compatible_contacts(root_height: float, standing_height: float = 0.4) -> np.ndarray:
    """Reproduce the current Isaac Lab task's temporary contact observation."""
    return np.full(4, float(root_height < standing_height * 0.5), dtype=np.float32)


def quaternion_wxyz_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a normalized wxyz quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def bezier_parameter_bounds(max_displacement: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``BezierTrajectoryGenerator.get_param_bounds`` without SciPy."""
    low = np.full(12, -max_displacement, dtype=np.float64)
    high = np.full(12, max_displacement, dtype=np.float64)
    low[:3] = 0.0
    high[:3] = 0.0
    for index in (5, 8, 11):
        low[index] = -0.05
        high[index] = 0.10
    return low, high


def cubic_bezier_waypoints(
    params: Sequence[float], start_position: Sequence[float], dt: float = 0.02, horizon: float = 3.0
) -> np.ndarray:
    """Evaluate the training task's cubic CoM Bezier curve."""
    offsets = np.asarray(params, dtype=np.float64).reshape(4, 3)
    control_points = np.asarray(start_position, dtype=np.float64).reshape(1, 3) + offsets
    count = int(horizon / dt) + 1
    t = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    omt = 1.0 - t
    return (
        omt**3 * control_points[0]
        + 3 * omt**2 * t * control_points[1]
        + 3 * omt * t**2 * control_points[2]
        + t**3 * control_points[3]
    )


def blend_trajectories(old_trajectory: np.ndarray, new_trajectory: np.ndarray, blend_steps: int = 5) -> np.ndarray:
    """Linearly blend the same leading samples as the Isaac Lab task."""
    old = np.asarray(old_trajectory, dtype=np.float64)
    result = np.asarray(new_trajectory, dtype=np.float64).copy()
    count = min(max(0, int(blend_steps)), len(old), len(result))
    if count:
        weights = np.linspace(0.0, 1.0, count)[:, None]
        result[:count] = (1.0 - weights) * old[:count] + weights * result[:count]
    return result


@dataclass
class ProcessedAction:
    bezier_parameters: np.ndarray
    gait_modifiers: np.ndarray
    gait_delta: np.ndarray


class StageActionProcessor:
    """Stateful action denormalization matching Stage 1/2 training semantics."""

    def __init__(
        self,
        stage: int = 2,
        max_displacement: float = 0.5,
        bezier_horizon: float = 3.0,
        forward_velocity_bias: float = 0.15,
        gait_lower: Sequence[float] = (0.5, 0.5, 0.5),
        gait_upper: Sequence[float] = (2.0, 2.0, 2.0),
        gait_max_delta: Sequence[float] = (0.15, 0.10, 0.10),
        gait_smoothing: float = 0.5,
    ):
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        self.stage = stage
        self.bezier_low, self.bezier_high = bezier_parameter_bounds(max_displacement)
        self.gait_lower = np.asarray(gait_lower, dtype=np.float64)
        self.gait_upper = np.asarray(gait_upper, dtype=np.float64)
        self.gait_max_delta = np.asarray(gait_max_delta, dtype=np.float64)
        self.gait_smoothing = float(gait_smoothing)
        self.forward_bias = np.asarray(
            [
                0.0, 0.0, 0.0,
                forward_velocity_bias * bezier_horizon / 3.0, 0.0, 0.0,
                forward_velocity_bias * bezier_horizon * 2.0 / 3.0, 0.0, 0.0,
                forward_velocity_bias * bezier_horizon, 0.0, 0.0,
            ],
            dtype=np.float64,
        )
        self.reset()

    def reset(self) -> None:
        self.previous_gait = np.ones(3, dtype=np.float64)

    def process(self, raw_action: Sequence[float]) -> ProcessedAction:
        action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        expected = 12 if self.stage == 1 else 15
        if action.shape != (expected,):
            raise ValueError(f"Stage {self.stage} expects {expected} actions, got {action.shape}")
        action = np.clip(np.where(np.isfinite(action), action, 0.0), -1.0, 1.0)
        bezier = 0.5 * (action[:12] + 1.0) * (self.bezier_high - self.bezier_low) + self.bezier_low
        bezier += self.forward_bias

        if self.stage == 1:
            applied = np.ones(3, dtype=np.float64)
            delta = np.zeros(3, dtype=np.float64)
        else:
            requested = 0.5 * (action[12:] + 1.0) * (self.gait_upper - self.gait_lower) + self.gait_lower
            target = np.clip(requested, self.gait_lower, self.gait_upper)
            delta = np.clip(
                self.gait_smoothing * (target - self.previous_gait),
                -self.gait_max_delta,
                self.gait_max_delta,
            )
            applied = np.clip(self.previous_gait + delta, self.gait_lower, self.gait_upper)
            self.previous_gait = applied.copy()
        return ProcessedAction(bezier, applied, delta)


class ComReferenceManager:
    """Regenerate, blend and slice the rolling CoM reference."""

    def __init__(self, dt: float = 0.02, horizon: float = 3.0, mpc_steps: int = 25, blend_steps: int = 5):
        self.dt = float(dt)
        self.horizon = float(horizon)
        self.mpc_steps = int(mpc_steps)
        self.blend_steps = int(blend_steps)
        self.reset()

    @property
    def normalized_phase(self) -> float:
        if self.trajectory is None:
            return 0.0
        return float(self.phase / max(len(self.trajectory) - 1, 1))

    def reset(self) -> None:
        self.trajectory: np.ndarray | None = None
        self.phase = 0

    def update(self, bezier_parameters: Sequence[float], root_position: Sequence[float]) -> np.ndarray:
        new_trajectory = cubic_bezier_waypoints(bezier_parameters, root_position, self.dt, self.horizon)
        if self.trajectory is not None and self.phase > 0:
            old_shifted = np.roll(self.trajectory, -self.phase, axis=0)
            self.trajectory = blend_trajectories(old_shifted, new_trajectory, self.blend_steps)
        else:
            self.trajectory = new_trajectory
        self.phase = 0
        reference = self.current_reference()
        self.advance()
        return reference

    def advance(self) -> None:
        if self.trajectory is not None:
            self.phase = min(self.phase + 1, len(self.trajectory) - 1)

    def current_reference(self) -> np.ndarray:
        if self.trajectory is None:
            raise RuntimeError("Reference manager has not received an action")
        reference = self.trajectory[self.phase : self.phase + self.mpc_steps]
        if len(reference) < self.mpc_steps:
            reference = np.concatenate(
                (reference, np.repeat(reference[-1:], self.mpc_steps - len(reference), axis=0)), axis=0
            )
        return reference
