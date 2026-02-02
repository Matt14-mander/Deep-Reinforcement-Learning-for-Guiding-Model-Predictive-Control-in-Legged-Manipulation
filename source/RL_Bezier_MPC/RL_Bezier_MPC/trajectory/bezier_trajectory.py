# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bezier curve trajectory generator for 3D position trajectories.

This module implements cubic Bezier curve trajectory generation for quadrotor
position control. The RL policy outputs 4 control point offsets (12D total for 3D),
and this generator converts them to dense position waypoints for MPC tracking.

Bezier Curve Formula (cubic):
    B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃,  t ∈ [0, 1]
"""

from typing import Tuple

import numpy as np
from scipy.special import comb

from .base_trajectory import BaseTrajectoryGenerator


class BezierTrajectoryGenerator(BaseTrajectoryGenerator):
    """Bezier curve trajectory generator for 3D position control.

    Generates position trajectories using Bezier curves parameterized by
    control point offsets from the current position. This provides a smooth,
    continuous trajectory representation suitable for MPC tracking.

    The parameter vector layout for cubic 3D Bezier (default):
        params[0:3]  = P0 offset (typically [0,0,0] - start at current position)
        params[3:6]  = P1 offset (first intermediate control point)
        params[6:9]  = P2 offset (second intermediate control point)
        params[9:12] = P3 offset (end point / target displacement)

    Attributes:
        degree: Bezier curve degree (3 for cubic).
        state_dim: Dimension of each waypoint (3 for 3D position).
        max_displacement: Maximum offset magnitude per control point (meters).
    """

    def __init__(
        self,
        degree: int = 3,
        state_dim: int = 3,
        max_displacement: float = 2.0,
    ):
        """Initialize Bezier trajectory generator.

        Args:
            degree: Bezier curve degree. Default 3 (cubic) with 4 control points.
            state_dim: Dimension of position space. Default 3 (x, y, z).
            max_displacement: Maximum offset from start position for each
                control point, in meters. Default 2.0m.
        """
        self.degree = degree
        self.state_dim = state_dim
        self.max_displacement = max_displacement
        self.num_control_points = degree + 1

        # Precompute binomial coefficients for Bernstein polynomials
        self._binomial_coeffs = np.array(
            [comb(degree, i, exact=True) for i in range(self.num_control_points)]
        )

    def params_to_waypoints(
        self,
        params: np.ndarray,
        dt: float,
        horizon: float,
        start_position: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert Bezier control point offsets to dense trajectory waypoints.

        Args:
            params: Control point offsets with shape (param_dim,).
                Layout: [P0_offset(3), P1_offset(3), P2_offset(3), P3_offset(3)]
                for cubic 3D Bezier. P0_offset is typically zeros.
            dt: Sampling timestep in seconds (e.g., 0.02 for 50Hz MPC).
            horizon: Trajectory duration in seconds (e.g., 1.5s).
            start_position: Current position to offset from, shape (3,).
                If None, uses origin [0, 0, 0].

        Returns:
            Dense waypoint sequence with shape (num_steps, 3) where
            num_steps = int(horizon / dt) + 1.
        """
        if start_position is None:
            start_position = np.zeros(self.state_dim)

        # Reshape parameters to control points: (num_control_points, state_dim)
        control_point_offsets = params.reshape(self.num_control_points, self.state_dim)

        # Convert offsets to absolute control points
        control_points = start_position + control_point_offsets

        # Generate parameter values t ∈ [0, 1]
        num_steps = self.get_num_waypoints(dt, horizon)
        t_values = np.linspace(0, 1, num_steps)

        # Evaluate Bezier curve at each t value
        waypoints = self._evaluate_bezier(control_points, t_values)

        return waypoints

    def _evaluate_bezier(
        self,
        control_points: np.ndarray,
        t_values: np.ndarray,
    ) -> np.ndarray:
        """Evaluate Bezier curve at given parameter values.

        Uses the Bernstein polynomial formulation:
            B(t) = Σᵢ bᵢ,ₙ(t) Pᵢ
        where bᵢ,ₙ(t) = C(n,i) tⁱ (1-t)^(n-i) is the Bernstein basis polynomial.

        Args:
            control_points: Control points with shape (num_control_points, state_dim).
            t_values: Parameter values with shape (num_steps,), each in [0, 1].

        Returns:
            Waypoints with shape (num_steps, state_dim).
        """
        n = self.degree
        num_steps = len(t_values)

        # Compute Bernstein basis polynomials for all t values
        # Shape: (num_steps, num_control_points)
        basis = np.zeros((num_steps, self.num_control_points))

        for i in range(self.num_control_points):
            # bᵢ,ₙ(t) = C(n,i) * t^i * (1-t)^(n-i)
            basis[:, i] = (
                self._binomial_coeffs[i]
                * (t_values ** i)
                * ((1 - t_values) ** (n - i))
            )

        # Compute Bezier curve: B(t) = Σᵢ bᵢ,ₙ(t) Pᵢ
        # Shape: (num_steps, state_dim)
        waypoints = basis @ control_points

        return waypoints

    def get_param_dim(self) -> int:
        """Return dimension of Bezier parameter vector.

        For cubic 3D Bezier: 4 control points × 3D = 12 parameters.

        Returns:
            Total parameter dimension.
        """
        return self.num_control_points * self.state_dim

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounds for Bezier control point offsets.

        P0 offset is constrained to [0, 0, 0] (start from current position).
        Other control points can offset by ±max_displacement in each dimension.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each with shape (param_dim,).
        """
        param_dim = self.get_param_dim()
        low = -self.max_displacement * np.ones(param_dim)
        high = self.max_displacement * np.ones(param_dim)

        # P0 offset should be zero (start from current position)
        low[: self.state_dim] = 0.0
        high[: self.state_dim] = 0.0

        return low, high

    def get_state_dim(self) -> int:
        """Return dimension of each waypoint.

        Returns:
            State dimension (3 for 3D position).
        """
        return self.state_dim

    def get_derivatives(
        self,
        params: np.ndarray,
        dt: float,
        horizon: float,
        start_position: np.ndarray | None = None,
        order: int = 1,
    ) -> np.ndarray:
        """Compute trajectory derivatives (velocity, acceleration, etc.).

        The derivative of a degree-n Bezier curve is a degree-(n-1) Bezier curve
        with control points Q_i = n * (P_{i+1} - P_i).

        Args:
            params: Control point offsets, shape (param_dim,).
            dt: Sampling timestep in seconds.
            horizon: Trajectory duration in seconds.
            start_position: Current position, shape (3,). If None, uses origin.
            order: Derivative order (1=velocity, 2=acceleration).

        Returns:
            Derivative values with shape (num_steps, state_dim).
        """
        if start_position is None:
            start_position = np.zeros(self.state_dim)

        # Get control points
        control_point_offsets = params.reshape(self.num_control_points, self.state_dim)
        control_points = start_position + control_point_offsets

        # Compute derivative control points
        derivative_cps = control_points.copy()
        current_degree = self.degree

        for _ in range(order):
            if current_degree < 1:
                # Derivative of constant is zero
                num_steps = self.get_num_waypoints(dt, horizon)
                return np.zeros((num_steps, self.state_dim))

            # Q_i = n * (P_{i+1} - P_i)
            new_cps = current_degree * np.diff(derivative_cps, axis=0)
            derivative_cps = new_cps
            current_degree -= 1

        # Scale by (1/horizon)^order to convert from curve parameter to time
        time_scale = (1.0 / horizon) ** order

        # Evaluate the derivative Bezier curve
        num_steps = self.get_num_waypoints(dt, horizon)
        t_values = np.linspace(0, 1, num_steps)

        # Update binomial coefficients for lower degree
        derivative_binomial = np.array(
            [comb(current_degree, i, exact=True) for i in range(current_degree + 1)]
        )

        # Evaluate derivative curve
        basis = np.zeros((num_steps, current_degree + 1))
        for i in range(current_degree + 1):
            basis[:, i] = (
                derivative_binomial[i]
                * (t_values ** i)
                * ((1 - t_values) ** (current_degree - i))
            )

        derivatives = time_scale * (basis @ derivative_cps)

        return derivatives

    def get_velocity(
        self,
        params: np.ndarray,
        dt: float,
        horizon: float,
        start_position: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute trajectory velocity at each waypoint.

        Convenience wrapper for get_derivatives with order=1.

        Args:
            params: Control point offsets, shape (param_dim,).
            dt: Sampling timestep in seconds.
            horizon: Trajectory duration in seconds.
            start_position: Current position, shape (3,).

        Returns:
            Velocity values with shape (num_steps, state_dim).
        """
        return self.get_derivatives(params, dt, horizon, start_position, order=1)

    def get_acceleration(
        self,
        params: np.ndarray,
        dt: float,
        horizon: float,
        start_position: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute trajectory acceleration at each waypoint.

        Convenience wrapper for get_derivatives with order=2.

        Args:
            params: Control point offsets, shape (param_dim,).
            dt: Sampling timestep in seconds.
            horizon: Trajectory duration in seconds.
            start_position: Current position, shape (3,).

        Returns:
            Acceleration values with shape (num_steps, state_dim).
        """
        return self.get_derivatives(params, dt, horizon, start_position, order=2)

    def sample_random_params(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample random valid Bezier parameters.

        Useful for random exploration or initialization.

        Args:
            rng: Numpy random generator. If None, uses default.

        Returns:
            Random parameter vector within bounds.
        """
        if rng is None:
            rng = np.random.default_rng()

        low, high = self.get_param_bounds()
        return rng.uniform(low, high)

    def interpolate_params(
        self,
        params_start: np.ndarray,
        params_end: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Linearly interpolate between two parameter sets.

        Useful for trajectory blending during RL policy updates.

        Args:
            params_start: Starting parameters.
            params_end: Ending parameters.
            alpha: Interpolation factor in [0, 1]. 0 = start, 1 = end.

        Returns:
            Interpolated parameter vector.
        """
        alpha = np.clip(alpha, 0.0, 1.0)
        return (1 - alpha) * params_start + alpha * params_end
