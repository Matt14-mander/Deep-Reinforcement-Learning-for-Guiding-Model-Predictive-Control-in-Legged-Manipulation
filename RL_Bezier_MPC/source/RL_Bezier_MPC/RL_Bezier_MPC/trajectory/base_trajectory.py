# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract interface for trajectory generators.

This module defines the base class for all trajectory generators in the RL+MPC system.
Trajectory generators convert compact parameter representations (e.g., Bezier control points)
into dense waypoint sequences that can be tracked by MPC controllers.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseTrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators.

    Trajectory generators convert low-dimensional parameter vectors into dense
    sequences of waypoints. This enables RL policies to output compact actions
    (trajectory parameters) while MPC controllers track high-frequency waypoints.

    Subclasses must implement:
        - params_to_waypoints: Convert parameters to waypoint sequence
        - get_param_dim: Return dimension of parameter vector
        - get_param_bounds: Return parameter bounds for action space
        - get_state_dim: Return dimension of each waypoint
    """

    @abstractmethod
    def params_to_waypoints(
        self,
        params: np.ndarray,
        dt: float,
        horizon: float,
        start_position: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert trajectory parameters to dense waypoints.

        Args:
            params: Parameter vector defining the trajectory shape.
                Shape: (param_dim,)
            dt: Sampling timestep in seconds (e.g., 0.02 for 50Hz).
            horizon: Total trajectory duration in seconds.
            start_position: Optional starting position for the trajectory.
                Shape: (state_dim,). If None, use origin or default.

        Returns:
            Waypoint sequence with shape (num_steps, state_dim) where
            num_steps = int(horizon / dt).
        """
        pass

    @abstractmethod
    def get_param_dim(self) -> int:
        """Return dimension of the parameter vector.

        This defines the RL action space size when the trajectory generator
        is used as the action parameterization.

        Returns:
            Integer dimension of parameter vector.
        """
        pass

    @abstractmethod
    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounds for trajectory parameters.

        Used to define the RL action space bounds and for parameter normalization.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each with shape (param_dim,).
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return dimension of each waypoint.

        For position-only trajectories, this is typically 3 (x, y, z).
        Could also include velocity, orientation, etc.

        Returns:
            Integer dimension of each waypoint state.
        """
        pass

    def get_num_waypoints(self, dt: float, horizon: float) -> int:
        """Compute number of waypoints for given timestep and horizon.

        Args:
            dt: Sampling timestep in seconds.
            horizon: Trajectory duration in seconds.

        Returns:
            Number of waypoints in the trajectory.
        """
        return int(horizon / dt) + 1

    def validate_params(self, params: np.ndarray) -> bool:
        """Check if parameters are within valid bounds.

        Args:
            params: Parameter vector to validate.

        Returns:
            True if all parameters are within bounds, False otherwise.
        """
        low, high = self.get_param_bounds()
        return bool(np.all(params >= low) and np.all(params <= high))

    def clip_params(self, params: np.ndarray) -> np.ndarray:
        """Clip parameters to valid bounds.

        Args:
            params: Parameter vector to clip.

        Returns:
            Clipped parameter vector.
        """
        low, high = self.get_param_bounds()
        return np.clip(params, low, high)
