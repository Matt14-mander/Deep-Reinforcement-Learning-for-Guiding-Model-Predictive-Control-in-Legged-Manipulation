# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bezier curve foot swing trajectory generator for quadruped locomotion.

This module generates smooth foot swing trajectories using Bezier curves.
The trajectory is rule-based (NOT learned by RL) - given a start position,
landing position, and step height, it produces a smooth arc for the swing phase.

This replaces the hardcoded triangular foot trajectory in Crocoddyl's demo
(quadruped.py lines 543-558) which uses piecewise linear: rise → peak → descend.
"""

from typing import Optional

import numpy as np

from ..utils.math_utils import bezier_curve


class BezierFootTrajectory:
    """Generate foot swing trajectory as a Bezier curve.

    Given step start/end positions and step height, computes 4 control points
    that produce a smooth arc: lift → cruise at height → land.

    Default control points layout (side view):
              P1────P2
             /        \\
            /          \\
    P0 ──/              \\── P3
    ─────                  ─────
    ground                 ground

    The intermediate control points (P1, P2) are placed at:
    - P1: lift_ratio along the step, at height_ratio * step_height
    - P2: land_ratio along the step, at height_ratio * step_height

    This produces a smooth bell-shaped curve with a flat cruise section.

    Attributes:
        step_height: Default swing height in meters.
        lift_ratio: P1 horizontal position as fraction of step length.
        land_ratio: P2 horizontal position as fraction of step length.
        height_ratio: P1/P2 height as fraction of step_height.
    """

    def __init__(
        self,
        step_height: float = 0.05,
        lift_ratio: float = 0.25,
        land_ratio: float = 0.75,
        height_ratio: float = 0.8,
    ):
        """Initialize foot trajectory generator.

        Args:
            step_height: Default swing height in meters. Default 0.05 (5cm).
            lift_ratio: P1 at this fraction of step length. Default 0.25.
            land_ratio: P2 at this fraction of step length. Default 0.75.
            height_ratio: P1,P2 height as fraction of step_height. Default 0.8.
        """
        self.step_height = step_height
        self.lift_ratio = lift_ratio
        self.land_ratio = land_ratio
        self.height_ratio = height_ratio

    def generate(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        num_samples: int,
        step_height: Optional[float] = None,
    ) -> np.ndarray:
        """Generate swing trajectory from start to end position.

        Control point computation:
            P0 = start_pos
            P1 = start_pos + (end_pos - start_pos) * lift_ratio + [0, 0, step_height * height_ratio]
            P2 = start_pos + (end_pos - start_pos) * land_ratio + [0, 0, step_height * height_ratio]
            P3 = end_pos

        Args:
            start_pos: Foot position at lift-off, shape (3,).
            end_pos: Foot position at landing (from FootholdPlanner), shape (3,).
            num_samples: Number of trajectory points (= number of knots in swing phase).
            step_height: Override default step height. If None, uses self.step_height.

        Returns:
            Foot swing trajectory with shape (num_samples, 3).
        """
        start_pos = np.asarray(start_pos)
        end_pos = np.asarray(end_pos)

        if step_height is None:
            step_height = self.step_height

        # Direction vector from start to end
        step_vector = end_pos - start_pos

        # Height offset vector (always in z direction)
        height_offset = np.array([0.0, 0.0, step_height * self.height_ratio])

        # Compute control points
        P0 = start_pos
        P1 = start_pos + step_vector * self.lift_ratio + height_offset
        P2 = start_pos + step_vector * self.land_ratio + height_offset
        P3 = end_pos

        control_points = np.array([P0, P1, P2, P3])

        # Evaluate Bezier curve
        trajectory = bezier_curve(control_points, num_samples)

        return trajectory

    def generate_with_apex(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        num_samples: int,
        apex_height: float,
        apex_ratio: float = 0.5,
    ) -> np.ndarray:
        """Generate swing trajectory with specified apex height and position.

        Alternative generation method where user specifies the maximum
        height point rather than control point heights.

        Args:
            start_pos: Foot position at lift-off, shape (3,).
            end_pos: Foot position at landing, shape (3,).
            num_samples: Number of trajectory points.
            apex_height: Maximum height above ground during swing.
            apex_ratio: Horizontal position of apex as fraction of step (0.5 = midpoint).

        Returns:
            Foot swing trajectory with shape (num_samples, 3).
        """
        start_pos = np.asarray(start_pos)
        end_pos = np.asarray(end_pos)

        # For a Bezier curve to have its maximum at apex_ratio,
        # we need to set the control point heights appropriately.
        # For cubic Bezier, the max height occurs near the middle
        # when P1 and P2 are at equal heights above the endpoints.

        # Approximate: set control points at ~1.3x apex height to achieve apex_height at middle
        # This is an empirical factor for cubic Bezier curves
        control_height_factor = 1.3

        step_vector = end_pos - start_pos
        height_offset = np.array([0.0, 0.0, apex_height * control_height_factor])

        # Adjust lift/land ratios to shift apex position
        adjusted_lift = apex_ratio * 0.5  # Half of apex_ratio
        adjusted_land = 1.0 - (1.0 - apex_ratio) * 0.5  # 1 - half of remaining

        P0 = start_pos
        P1 = start_pos + step_vector * adjusted_lift + height_offset
        P2 = start_pos + step_vector * adjusted_land + height_offset
        P3 = end_pos

        control_points = np.array([P0, P1, P2, P3])
        trajectory = bezier_curve(control_points, num_samples)

        return trajectory

    def generate_from_params(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        bezier_offsets: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        """Generate swing trajectory with RL-provided control point offsets.

        This is a future extension for Phase 4 where RL can modulate the
        default Bezier control points to adapt foot trajectories.

        Args:
            start_pos: Foot position at lift-off, shape (3,).
            end_pos: Foot position at landing, shape (3,).
            bezier_offsets: RL-provided offsets for P1 and P2, shape (6,).
                Layout: [P1_offset_x, P1_offset_y, P1_offset_z,
                         P2_offset_x, P2_offset_y, P2_offset_z]
            num_samples: Number of trajectory points.

        Returns:
            Foot swing trajectory with shape (num_samples, 3).
        """
        start_pos = np.asarray(start_pos)
        end_pos = np.asarray(end_pos)
        bezier_offsets = np.asarray(bezier_offsets)

        # Default control points
        step_vector = end_pos - start_pos
        height_offset = np.array([0.0, 0.0, self.step_height * self.height_ratio])

        P0 = start_pos
        P1_default = start_pos + step_vector * self.lift_ratio + height_offset
        P2_default = start_pos + step_vector * self.land_ratio + height_offset
        P3 = end_pos

        # Apply RL offsets
        P1 = P1_default + bezier_offsets[:3]
        P2 = P2_default + bezier_offsets[3:6]

        control_points = np.array([P0, P1, P2, P3])
        trajectory = bezier_curve(control_points, num_samples)

        return trajectory

    def get_control_points(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        step_height: Optional[float] = None,
    ) -> np.ndarray:
        """Get the Bezier control points for a swing trajectory.

        Useful for visualization and debugging.

        Args:
            start_pos: Foot position at lift-off, shape (3,).
            end_pos: Foot position at landing, shape (3,).
            step_height: Override default step height.

        Returns:
            Control points array with shape (4, 3).
        """
        start_pos = np.asarray(start_pos)
        end_pos = np.asarray(end_pos)

        if step_height is None:
            step_height = self.step_height

        step_vector = end_pos - start_pos
        height_offset = np.array([0.0, 0.0, step_height * self.height_ratio])

        P0 = start_pos
        P1 = start_pos + step_vector * self.lift_ratio + height_offset
        P2 = start_pos + step_vector * self.land_ratio + height_offset
        P3 = end_pos

        return np.array([P0, P1, P2, P3])

    def get_velocity(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        num_samples: int,
        step_height: Optional[float] = None,
        swing_duration: float = 0.15,
    ) -> np.ndarray:
        """Compute foot velocity at each point along the swing trajectory.

        Args:
            start_pos: Foot position at lift-off, shape (3,).
            end_pos: Foot position at landing, shape (3,).
            num_samples: Number of trajectory points.
            step_height: Override default step height.
            swing_duration: Duration of swing phase in seconds.

        Returns:
            Velocity vectors with shape (num_samples, 3).
        """
        from ..utils.math_utils import bezier_tangent

        control_points = self.get_control_points(start_pos, end_pos, step_height)

        # Get tangent vectors (derivative of position w.r.t. parameter t)
        t_values = np.linspace(0, 1, num_samples)
        tangents = np.array([bezier_tangent(control_points, t) for t in t_values])

        # Scale by dt/ds = 1/swing_duration to get velocity in world units
        velocities = tangents / swing_duration

        return velocities
