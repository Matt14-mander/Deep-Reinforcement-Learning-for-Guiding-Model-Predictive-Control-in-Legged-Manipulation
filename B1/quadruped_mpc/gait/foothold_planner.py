# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foothold planner for computing foot landing positions from CoM trajectory.

This is the key module that enables CURVE WALKING. By sampling the CoM
trajectory at each foot-landing moment and using the local heading (tangent
direction) plus hip offsets, it automatically produces correct inner/outer
leg step length differences during turns.

Why the Crocoddyl demo only walks straight:
In quadruped.py line 546-547, foot displacement `dp` only has an x-component
(`stepLength * (k+1)/numKnots, 0.0, ...`). The y-component is always 0.
Our FootholdPlanner replaces this by deriving landing positions from the
CoM curve geometry.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from ..trajectory.bezier_foot_trajectory import BezierFootTrajectory
from ..utils.math_utils import heading_from_tangent, rotation_matrix_z
from .contact_sequence import ContactSequence


@dataclass
class FootholdPlan:
    """Plan for a single foot swing event.

    Attributes:
        foot_name: Name of the foot ("LF", "RF", "LH", "RH").
        start_pos: Foot position at lift-off, shape (3,).
        end_pos: Computed landing position, shape (3,).
        swing_start_time: Time when swing begins (seconds).
        swing_end_time: Time when swing ends (seconds).
        trajectory: Swing trajectory from BezierFootTrajectory, shape (N, 3).
    """

    foot_name: str
    start_pos: np.ndarray
    end_pos: np.ndarray
    swing_start_time: float
    swing_end_time: float
    trajectory: Optional[np.ndarray] = None


class FootholdPlanner:
    """Compute foot landing positions from CoM trajectory.

    For each swing phase in the ContactSequence:
    1. Find the CoM position and heading at the moment the foot should land
    2. Compute hip position = CoM + R(yaw) * hip_offset
    3. Project hip to ground → landing position
    4. Generate swing trajectory from current foot pos to landing pos

    This naturally produces:
    - Straight walking when CoM Bezier is a straight line
    - Curve walking when CoM Bezier curves (inner legs take shorter steps)
    - Turning in place when CoM Bezier has pure heading change

    Attributes:
        hip_offsets: Dict mapping foot name to hip offset in body frame.
        default_ground_height: Default ground height when no terrain function.
        foot_trajectory_gen: BezierFootTrajectory for generating swing arcs.
    """

    # Standard hip offset directions (relative to CoM, in body frame)
    # +x = forward, +y = left, +z = up
    DEFAULT_HIP_OFFSETS = {
        "LF": np.array([+0.2, +0.1, 0.0]),  # front-left
        "RF": np.array([+0.2, -0.1, 0.0]),  # front-right
        "LH": np.array([-0.2, +0.1, 0.0]),  # hind-left
        "RH": np.array([-0.2, -0.1, 0.0]),  # hind-right
    }

    def __init__(
        self,
        hip_offsets: Optional[Dict[str, np.ndarray]] = None,
        default_ground_height: float = 0.0,
        step_height: float = 0.05,
    ):
        """Initialize the foothold planner.

        Args:
            hip_offsets: Dict mapping foot name to hip offset in body frame.
                If None, uses DEFAULT_HIP_OFFSETS.
            default_ground_height: Ground height when no terrain function.
            step_height: Default foot swing height for trajectory generation.
        """
        if hip_offsets is None:
            self.hip_offsets = {k: v.copy() for k, v in self.DEFAULT_HIP_OFFSETS.items()}
        else:
            self.hip_offsets = {k: np.asarray(v) for k, v in hip_offsets.items()}

        self.default_ground_height = default_ground_height
        self.foot_trajectory_gen = BezierFootTrajectory(step_height=step_height)

    def plan_footholds(
        self,
        com_trajectory: np.ndarray,
        contact_sequence: ContactSequence,
        current_foot_positions: Dict[str, np.ndarray],
        dt: float,
        terrain_height_fn: Optional[Callable[[float, float], float]] = None,
        step_height: Optional[float] = None,
    ) -> Dict[str, List[FootholdPlan]]:
        """Compute foot landing positions for each swing phase.

        For each swing phase, computes where each swinging foot should land
        based on the CoM trajectory heading at landing time.

        Args:
            com_trajectory: Dense CoM waypoints from Bezier, shape (T, 3).
            contact_sequence: ContactSequence defining swing/support phases.
            current_foot_positions: Current position of each foot, Dict[str, (3,)].
            dt: Trajectory sampling timestep in seconds.
            terrain_height_fn: Optional function terrain_height(x, y) → z.
            step_height: Override step height for swing trajectories.

        Returns:
            Dict mapping foot name → list of FootholdPlan for each swing event.
        """
        # Initialize output and tracking
        foothold_plans: Dict[str, List[FootholdPlan]] = {
            foot: [] for foot in self.hip_offsets.keys()
        }

        # Track current foot positions (updated after each swing)
        foot_positions = {k: np.asarray(v).copy() for k, v in current_foot_positions.items()}

        # Track time through the contact sequence
        cumulative_time = 0.0

        for phase in contact_sequence.phases:
            phase_duration = phase.duration
            phase_end_time = cumulative_time + phase_duration

            # Convert time to trajectory index
            phase_start_idx = int(cumulative_time / dt)
            phase_end_idx = int(phase_end_time / dt)

            # Clamp to valid range
            phase_start_idx = min(phase_start_idx, len(com_trajectory) - 1)
            phase_end_idx = min(phase_end_idx, len(com_trajectory) - 1)

            # For each swinging foot, compute landing position
            for foot_name in phase.swing_feet:
                if foot_name not in self.hip_offsets:
                    continue

                # Get CoM state at landing time (end of swing)
                com_at_landing = com_trajectory[phase_end_idx]

                # Compute heading from trajectory tangent at landing
                heading = self._compute_heading_at_time(
                    com_trajectory, phase_end_idx, dt
                )

                # Compute landing position
                landing_pos = self._hip_to_foothold(
                    com_position=com_at_landing,
                    heading=heading,
                    hip_offset=self.hip_offsets[foot_name],
                    terrain_height_fn=terrain_height_fn,
                )

                # Number of trajectory samples for this swing
                num_swing_samples = max(2, phase_end_idx - phase_start_idx + 1)

                # Generate swing trajectory
                swing_trajectory = self.foot_trajectory_gen.generate(
                    start_pos=foot_positions[foot_name],
                    end_pos=landing_pos,
                    num_samples=num_swing_samples,
                    step_height=step_height,
                )

                # Create foothold plan
                plan = FootholdPlan(
                    foot_name=foot_name,
                    start_pos=foot_positions[foot_name].copy(),
                    end_pos=landing_pos,
                    swing_start_time=cumulative_time,
                    swing_end_time=phase_end_time,
                    trajectory=swing_trajectory,
                )

                foothold_plans[foot_name].append(plan)

                # Update foot position for next iteration
                foot_positions[foot_name] = landing_pos.copy()

            # Advance time
            cumulative_time = phase_end_time

        return foothold_plans

    def _compute_heading_at_time(
        self,
        com_trajectory: np.ndarray,
        t_index: int,
        dt: float,
    ) -> float:
        """Compute yaw heading from CoM trajectory tangent at given time index.

        Uses finite differences to estimate velocity direction.
        Falls back to previous heading if velocity is near zero.

        Args:
            com_trajectory: CoM trajectory, shape (T, 3).
            t_index: Index in trajectory.
            dt: Trajectory timestep.

        Returns:
            Yaw angle in radians.
        """
        T = len(com_trajectory)

        # Use central differences where possible, one-sided at boundaries
        if t_index == 0:
            # Forward difference
            if T > 1:
                tangent = (com_trajectory[1] - com_trajectory[0]) / dt
            else:
                return 0.0
        elif t_index >= T - 1:
            # Backward difference
            tangent = (com_trajectory[T - 1] - com_trajectory[T - 2]) / dt
        else:
            # Central difference
            tangent = (com_trajectory[t_index + 1] - com_trajectory[t_index - 1]) / (2 * dt)

        return heading_from_tangent(tangent[:2])

    def _hip_to_foothold(
        self,
        com_position: np.ndarray,
        heading: float,
        hip_offset: np.ndarray,
        terrain_height_fn: Optional[Callable[[float, float], float]] = None,
    ) -> np.ndarray:
        """Compute foot landing position from CoM position and heading.

        foot_pos = com_position + Rz(heading) @ hip_offset
        foot_pos[2] = terrain_height(foot_pos[0], foot_pos[1]) or default

        Args:
            com_position: CoM position at landing time, shape (3,).
            heading: Yaw angle in radians.
            hip_offset: Hip offset in body frame, shape (3,).
            terrain_height_fn: Optional terrain height function.

        Returns:
            Foot landing position, shape (3,).
        """
        # Rotate hip offset by heading
        R = rotation_matrix_z(heading)
        rotated_offset = R @ hip_offset

        # Compute world-frame foot position
        foot_pos = com_position + rotated_offset

        # Get ground height
        if terrain_height_fn is not None:
            foot_pos[2] = terrain_height_fn(foot_pos[0], foot_pos[1])
        else:
            foot_pos[2] = self.default_ground_height

        return foot_pos

    def get_footholds_at_time(
        self,
        com_position: np.ndarray,
        heading: float,
        terrain_height_fn: Optional[Callable[[float, float], float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute nominal foot positions for all feet at a given CoM pose.

        Useful for computing initial standing foot positions.

        Args:
            com_position: CoM position, shape (3,).
            heading: Yaw angle in radians.
            terrain_height_fn: Optional terrain height function.

        Returns:
            Dict mapping foot name to position, each shape (3,).
        """
        positions = {}
        for foot_name, hip_offset in self.hip_offsets.items():
            positions[foot_name] = self._hip_to_foothold(
                com_position=com_position,
                heading=heading,
                hip_offset=hip_offset,
                terrain_height_fn=terrain_height_fn,
            )
        return positions

    def compute_step_lengths(
        self,
        foothold_plans: Dict[str, List[FootholdPlan]],
    ) -> Dict[str, List[float]]:
        """Compute step lengths from foothold plans.

        Useful for analysis and debugging curve walking behavior.

        Args:
            foothold_plans: Output from plan_footholds.

        Returns:
            Dict mapping foot name to list of step lengths.
        """
        step_lengths = {}
        for foot_name, plans in foothold_plans.items():
            lengths = []
            for plan in plans:
                length = np.linalg.norm(plan.end_pos - plan.start_pos)
                lengths.append(length)
            step_lengths[foot_name] = lengths
        return step_lengths

    def visualize_footholds(
        self,
        com_trajectory: np.ndarray,
        foothold_plans: Dict[str, List[FootholdPlan]],
    ) -> Dict:
        """Prepare data for visualizing footholds.

        Returns data suitable for matplotlib or other plotting.

        Args:
            com_trajectory: CoM trajectory, shape (T, 3).
            foothold_plans: Output from plan_footholds.

        Returns:
            Dict with:
                - "com": CoM trajectory array
                - "footholds": Dict[foot_name, List[(start, end, trajectory)]]
        """
        viz_data = {
            "com": com_trajectory,
            "footholds": {},
        }

        for foot_name, plans in foothold_plans.items():
            foot_data = []
            for plan in plans:
                foot_data.append({
                    "start": plan.start_pos,
                    "end": plan.end_pos,
                    "trajectory": plan.trajectory,
                    "swing_start_time": plan.swing_start_time,
                    "swing_end_time": plan.swing_end_time,
                })
            viz_data["footholds"][foot_name] = foot_data

        return viz_data
