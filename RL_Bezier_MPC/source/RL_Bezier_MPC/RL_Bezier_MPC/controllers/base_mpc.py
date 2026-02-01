# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract interface for Model Predictive Control (MPC) controllers.

This module defines the base class for MPC controllers used in the RL+MPC system.
MPC controllers take a reference trajectory and current state, then compute optimal
control inputs to track the trajectory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MPCSolution:
    """Container for MPC solver output.

    Attributes:
        control: First optimal control action to apply.
            Shape: (control_dim,)
        predicted_states: Predicted state trajectory over horizon.
            Shape: (horizon_steps, state_dim)
        predicted_controls: Predicted control trajectory over horizon.
            Shape: (horizon_steps, control_dim)
        solve_time: Wall-clock solver time in seconds.
        converged: Whether the solver converged to a solution.
        cost: Optimal cost value achieved.
        iterations: Number of solver iterations.
    """

    control: np.ndarray
    predicted_states: np.ndarray
    predicted_controls: np.ndarray
    solve_time: float
    converged: bool
    cost: float
    iterations: int = 0


class BaseMPC(ABC):
    """Abstract base class for MPC controllers.

    MPC controllers solve optimal control problems (OCPs) to track reference
    trajectories. The controller computes the optimal control sequence and
    returns the first control action for immediate execution.

    The typical MPC workflow:
        1. Receive current robot state and reference trajectory
        2. Formulate and solve the OCP
        3. Extract first optimal control
        4. Apply control and repeat at next timestep

    Subclasses must implement:
        - solve: Solve OCP and return optimal control
        - get_control_dim: Return control output dimension
        - get_state_dim: Return expected state dimension
    """

    @abstractmethod
    def solve(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
        warm_start: bool = True,
    ) -> MPCSolution:
        """Solve MPC problem and return optimal control.

        Args:
            current_state: Current robot state.
                Shape: (state_dim,)
            reference_trajectory: Reference positions/states to track.
                Shape: (horizon_steps, ref_dim)
            warm_start: If True, initialize solver with shifted previous
                solution. Significantly speeds up convergence.

        Returns:
            MPCSolution containing optimal control and solver info.
        """
        pass

    @abstractmethod
    def get_control_dim(self) -> int:
        """Return dimension of control output.

        For quadrotor: 4D [thrust, torque_x, torque_y, torque_z]

        Returns:
            Integer control dimension.
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return expected state dimension.

        For quadrotor: 13D [pos(3), quat(4), lin_vel(3), ang_vel(3)]

        Returns:
            Integer state dimension.
        """
        pass

    def get_horizon_steps(self) -> int:
        """Return MPC horizon length in timesteps.

        Returns:
            Number of steps in MPC prediction horizon.
        """
        return getattr(self, "horizon_steps", 25)

    def get_dt(self) -> float:
        """Return MPC timestep in seconds.

        Returns:
            MPC discretization timestep.
        """
        return getattr(self, "dt", 0.02)

    def reset(self):
        """Reset controller state.

        Called at episode start to clear warm-start buffers, etc.
        Default implementation does nothing; override if needed.
        """
        pass

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return control input bounds.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (control_dim,).
        """
        control_dim = self.get_control_dim()
        return (
            -np.inf * np.ones(control_dim),
            np.inf * np.ones(control_dim),
        )

    def set_cost_weights(self, **weights):
        """Update cost function weights.

        Allows runtime adjustment of tracking vs. regularization trade-off.
        Default implementation does nothing; override to enable.

        Args:
            **weights: Named weight values (e.g., position_weight=100.0).
        """
        pass
