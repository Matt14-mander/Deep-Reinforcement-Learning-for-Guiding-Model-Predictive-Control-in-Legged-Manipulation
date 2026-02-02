# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crocoddyl-based MPC controller for quadrotor trajectory tracking.

This module implements Model Predictive Control for quadrotor position tracking
using the Crocoddyl optimal control library. The controller tracks 3D position
references while respecting thrust and torque constraints.

Quadrotor State (13D):
    [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    - Position: x, y, z
    - Orientation: quaternion (w, x, y, z)
    - Linear velocity: vx, vy, vz
    - Angular velocity: wx, wy, wz

Control (4D):
    [thrust, tau_x, tau_y, tau_z]
    - thrust: Collective thrust (N)
    - tau_x, tau_y, tau_z: Body torques (N.m)
"""

import time
from typing import Tuple

import numpy as np

from .base_mpc import BaseMPC, MPCSolution

# Try to import crocoddyl - it may not be available in all environments
try:
    import crocoddyl
    import pinocchio

    CROCODDYL_AVAILABLE = True
except ImportError:
    CROCODDYL_AVAILABLE = False
    crocoddyl = None
    pinocchio = None


class CrocoddylQuadrotorMPC(BaseMPC):
    """Crocoddyl-based MPC for quadrotor trajectory tracking.

    Uses the FDDP (Feasibility-driven DDP) solver from Crocoddyl to solve
    the trajectory tracking optimal control problem. The quadrotor is modeled
    as a free-flying rigid body with thrust and torque inputs.

    The OCP minimizes:
        J = Σ (position_error² + velocity_penalty + control_penalty) + terminal_cost

    Subject to:
        - Quadrotor dynamics (rigid body + thrust/torque actuation)
        - Thrust bounds: [0, max_thrust]
        - Torque bounds: [-max_torque, max_torque]

    Attributes:
        dt: MPC discretization timestep (seconds).
        horizon_steps: Number of steps in prediction horizon.
        mass: Quadrotor mass (kg).
        inertia: 3x3 inertia matrix (kg.m²).
    """

    # State indices
    POS_IDX = slice(0, 3)
    QUAT_IDX = slice(3, 7)
    LINVEL_IDX = slice(7, 10)
    ANGVEL_IDX = slice(10, 13)

    # Control indices
    THRUST_IDX = 0
    TORQUE_IDX = slice(1, 4)

    def __init__(
        self,
        dt: float = 0.02,
        horizon_steps: int = 25,
        # Quadrotor physical parameters (Crazyflie-like)
        mass: float = 0.027,
        inertia: np.ndarray | None = None,
        arm_length: float = 0.046,
        # Cost weights
        position_weight: float = 100.0,
        velocity_weight: float = 10.0,
        orientation_weight: float = 10.0,
        angular_velocity_weight: float = 1.0,
        control_weight: float = 0.1,
        terminal_weight_factor: float = 10.0,
        # Control limits
        thrust_min: float = 0.0,
        thrust_max: float = 0.6,
        torque_max: float = 0.01,
        # Solver settings
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
        verbose: bool = False,
    ):
        """Initialize Crocoddyl MPC for quadrotor.

        Args:
            dt: MPC timestep in seconds. Should match control rate (e.g., 0.02 for 50Hz).
            horizon_steps: Number of MPC lookahead steps. 25 steps @ 50Hz = 0.5s horizon.
            mass: Quadrotor mass in kg.
            inertia: 3x3 inertia matrix. If None, uses Crazyflie-like defaults.
            arm_length: Motor arm length in meters.
            position_weight: Cost weight for position tracking error.
            velocity_weight: Cost weight for velocity regularization.
            orientation_weight: Cost weight for orientation (upright) tracking.
            angular_velocity_weight: Cost weight for angular velocity regularization.
            control_weight: Cost weight for control effort.
            terminal_weight_factor: Multiplier for terminal cost weights.
            thrust_min: Minimum thrust in Newtons (usually 0).
            thrust_max: Maximum thrust in Newtons.
            torque_max: Maximum torque magnitude in N.m.
            max_iterations: Maximum FDDP solver iterations.
            convergence_threshold: Solver convergence criterion.
            verbose: If True, print solver info.
        """
        if not CROCODDYL_AVAILABLE:
            raise ImportError(
                "Crocoddyl is required for CrocoddylQuadrotorMPC. "
                "Install with: pip install crocoddyl"
            )

        self.dt = dt
        self.horizon_steps = horizon_steps

        # Physical parameters
        self.mass = mass
        self.arm_length = arm_length
        if inertia is None:
            # Crazyflie-like inertia (diagonal approximation)
            ixx = 1.4e-5  # kg.m²
            iyy = 1.4e-5
            izz = 2.17e-5
            self.inertia = np.diag([ixx, iyy, izz])
        else:
            self.inertia = np.array(inertia)

        # Cost weights
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.orientation_weight = orientation_weight
        self.angular_velocity_weight = angular_velocity_weight
        self.control_weight = control_weight
        self.terminal_weight_factor = terminal_weight_factor

        # Control bounds
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.torque_max = torque_max

        # Solver settings
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose

        # Gravity
        self.gravity = np.array([0.0, 0.0, -9.81])

        # Hover thrust (for initialization)
        self.hover_thrust = self.mass * 9.81

        # Build the optimal control problem
        self._build_ocp()

        # Warm-start buffers
        self._prev_xs = None
        self._prev_us = None

    def _build_ocp(self):
        """Construct Crocoddyl shooting problem for quadrotor tracking.

        Creates a differential action model for quadrotor dynamics with
        position tracking cost and control regularization.
        """
        # State: SE(3) manifold with velocities
        # We use a simplified state-space representation
        state_dim = 13
        control_dim = 4

        # Create state manifold (vector space for simplicity)
        # In practice, could use StateMultibody with a floating base model
        self.state = crocoddyl.StateVector(state_dim)

        # Create running cost model
        running_cost_model = crocoddyl.CostModelSum(self.state)

        # Position tracking cost (will be updated each solve with reference)
        # Using a residual model for state tracking
        self.position_ref = np.zeros(3)  # Will be updated in solve()
        position_residual = crocoddyl.ResidualModelState(
            self.state, np.zeros(state_dim), state_dim
        )
        position_activation = crocoddyl.ActivationModelWeightedQuad(
            np.array([self.position_weight] * 3 + [0.0] * 10)  # Only penalize position
        )
        position_cost = crocoddyl.CostModelResidual(
            self.state, position_activation, position_residual
        )
        running_cost_model.addCost("position", position_cost, 1.0)

        # Velocity regularization
        velocity_activation = crocoddyl.ActivationModelWeightedQuad(
            np.array([0.0] * 7 + [self.velocity_weight] * 3 + [0.0] * 3)
        )
        velocity_residual = crocoddyl.ResidualModelState(
            self.state, np.zeros(state_dim), state_dim
        )
        velocity_cost = crocoddyl.CostModelResidual(
            self.state, velocity_activation, velocity_residual
        )
        running_cost_model.addCost("velocity", velocity_cost, 1.0)

        # Control regularization
        control_residual = crocoddyl.ResidualModelControl(self.state, control_dim)
        control_activation = crocoddyl.ActivationModelWeightedQuad(
            np.array([self.control_weight] * control_dim)
        )
        control_cost = crocoddyl.CostModelResidual(
            self.state, control_activation, control_residual
        )
        running_cost_model.addCost("control", control_cost, 1.0)

        # Terminal cost model (higher weights)
        terminal_cost_model = crocoddyl.CostModelSum(self.state)

        terminal_position_activation = crocoddyl.ActivationModelWeightedQuad(
            np.array(
                [self.position_weight * self.terminal_weight_factor] * 3 + [0.0] * 10
            )
        )
        terminal_position_cost = crocoddyl.CostModelResidual(
            self.state, terminal_position_activation, position_residual
        )
        terminal_cost_model.addCost("terminal_position", terminal_position_cost, 1.0)

        terminal_velocity_activation = crocoddyl.ActivationModelWeightedQuad(
            np.array(
                [0.0] * 7
                + [self.velocity_weight * self.terminal_weight_factor] * 3
                + [0.0] * 3
            )
        )
        terminal_velocity_cost = crocoddyl.CostModelResidual(
            self.state, terminal_velocity_activation, velocity_residual
        )
        terminal_cost_model.addCost("terminal_velocity", terminal_velocity_cost, 1.0)

        # Create action models with quadrotor dynamics
        self.running_models = []
        for _ in range(self.horizon_steps):
            # Create differential model with quadrotor dynamics
            diff_model = _QuadrotorDifferentialActionModel(
                self.state,
                running_cost_model,
                mass=self.mass,
                inertia=self.inertia,
                gravity=self.gravity,
            )

            # Integrate to get discrete-time model
            model = crocoddyl.IntegratedActionModelEuler(diff_model, self.dt)

            # Add control bounds
            model.u_lb = np.array(
                [self.thrust_min, -self.torque_max, -self.torque_max, -self.torque_max]
            )
            model.u_ub = np.array(
                [self.thrust_max, self.torque_max, self.torque_max, self.torque_max]
            )

            self.running_models.append(model)

        # Terminal model (no control, just cost)
        terminal_diff_model = _QuadrotorDifferentialActionModel(
            self.state,
            terminal_cost_model,
            mass=self.mass,
            inertia=self.inertia,
            gravity=self.gravity,
        )
        self.terminal_model = crocoddyl.IntegratedActionModelEuler(
            terminal_diff_model, 0.0
        )

        # Create shooting problem
        x0 = np.zeros(13)
        x0[3] = 1.0  # qw = 1 (identity quaternion)
        self.problem = crocoddyl.ShootingProblem(
            x0, self.running_models, self.terminal_model
        )

        # Create solver
        self.solver = crocoddyl.SolverBoxFDDP(self.problem)
        self.solver.setCallbacks([crocoddyl.CallbackVerbose()] if self.verbose else [])

    def solve(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
        warm_start: bool = True,
    ) -> MPCSolution:
        """Solve trajectory tracking OCP.

        Args:
            current_state: Current quadrotor state (13D).
                [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            reference_trajectory: Reference positions (T x 3).
                Only position is tracked; velocities assumed zero.
            warm_start: Use shifted previous solution for initialization.

        Returns:
            MPCSolution with optimal control and solver info.
        """
        start_time = time.time()

        # Ensure state is 13D
        state = self._ensure_state_format(current_state)

        # Update initial state
        self.problem.x0 = state

        # Update reference trajectory in cost functions
        self._update_reference(reference_trajectory)

        # Prepare initial guess
        if warm_start and self._prev_xs is not None and self._prev_us is not None:
            # Shift previous solution
            xs = self._shift_trajectory(self._prev_xs, state)
            us = self._shift_controls(self._prev_us)
        else:
            # Cold start: hover at current position
            xs = [state.copy() for _ in range(self.horizon_steps + 1)]
            us = [
                np.array([self.hover_thrust, 0.0, 0.0, 0.0])
                for _ in range(self.horizon_steps)
            ]

        # Set initial guess
        self.solver.setCandidate(xs, us)

        # Solve
        converged = self.solver.solve(
            xs, us, self.max_iterations, False, self.convergence_threshold
        )

        solve_time = time.time() - start_time

        # Extract solution
        optimal_xs = list(self.solver.xs)
        optimal_us = list(self.solver.us)

        # Store for warm-starting
        self._prev_xs = optimal_xs
        self._prev_us = optimal_us

        # Get first control
        control = optimal_us[0].copy()

        # Clip control to bounds (safety)
        control = self._clip_control(control)

        return MPCSolution(
            control=control,
            predicted_states=np.array(optimal_xs),
            predicted_controls=np.array(optimal_us),
            solve_time=solve_time,
            converged=converged,
            cost=self.solver.cost,
            iterations=self.solver.iter,
        )

    def _ensure_state_format(self, state: np.ndarray) -> np.ndarray:
        """Ensure state is in correct 13D format."""
        state = np.asarray(state).flatten()

        if len(state) == 12:
            # Assume [pos(3), euler(3), linvel(3), angvel(3)]
            pos = state[:3]
            euler = state[3:6]
            linvel = state[6:9]
            angvel = state[9:12]
            quat = self._euler_to_quat(euler)
            return np.concatenate([pos, quat, linvel, angvel])

        elif len(state) == 13:
            result = state.copy()
            quat = result[3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                result[3:7] = quat / quat_norm
            else:
                result[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            return result

        else:
            raise ValueError(f"Expected 12D or 13D state, got {len(state)}D")

    def _euler_to_quat(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
        roll, pitch, yaw = euler
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])

    def _update_reference(self, reference_trajectory: np.ndarray):
        """Update reference positions in cost functions."""
        ref = np.atleast_2d(reference_trajectory)
        num_refs = len(ref)

        for i, model in enumerate(self.running_models):
            ref_idx = min(i, num_refs - 1)
            ref_pos = ref[ref_idx]
            ref_state = np.zeros(13)
            ref_state[:3] = ref_pos
            ref_state[3] = 1.0

            diff_model = model.differential
            if hasattr(diff_model, "costs"):
                for name, cost_item in diff_model.costs.costs.items():
                    if "position" in name:
                        cost_item.cost.residual.reference = ref_state

        final_ref_pos = ref[-1]
        terminal_ref_state = np.zeros(13)
        terminal_ref_state[:3] = final_ref_pos
        terminal_ref_state[3] = 1.0

        terminal_diff = self.terminal_model.differential
        if hasattr(terminal_diff, "costs"):
            for name, cost_item in terminal_diff.costs.costs.items():
                if "position" in name:
                    cost_item.cost.residual.reference = terminal_ref_state

    def _shift_trajectory(self, prev_xs: list, current_state: np.ndarray) -> list:
        """Shift previous state trajectory for warm-starting."""
        xs = [current_state.copy()]
        for i in range(1, self.horizon_steps):
            if i < len(prev_xs):
                xs.append(prev_xs[i].copy())
            else:
                xs.append(prev_xs[-1].copy())
        xs.append(prev_xs[-1].copy())
        return xs

    def _shift_controls(self, prev_us: list) -> list:
        """Shift previous control trajectory for warm-starting."""
        us = []
        for i in range(self.horizon_steps):
            if i + 1 < len(prev_us):
                us.append(prev_us[i + 1].copy())
            else:
                us.append(prev_us[-1].copy())
        return us

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        """Clip control to valid bounds."""
        control = control.copy()
        control[0] = np.clip(control[0], self.thrust_min, self.thrust_max)
        control[1:] = np.clip(control[1:], -self.torque_max, self.torque_max)
        return control

    def get_control_dim(self) -> int:
        return 4

    def get_state_dim(self) -> int:
        return 13

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.thrust_min, -self.torque_max, -self.torque_max, -self.torque_max])
        upper = np.array([self.thrust_max, self.torque_max, self.torque_max, self.torque_max])
        return lower, upper

    def reset(self):
        self._prev_xs = None
        self._prev_us = None

    def set_cost_weights(self, **weights):
        for key, value in weights.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._build_ocp()


# =============================================================================
# Crocoddyl-dependent class (only defined when Crocoddyl is available)
# =============================================================================

if CROCODDYL_AVAILABLE:

    class _QuadrotorDifferentialActionModel(crocoddyl.DifferentialActionModelAbstract):
        """Custom differential action model for quadrotor dynamics.

        Implements the continuous-time quadrotor equations of motion:
            m * v̇ = R @ [0, 0, thrust]ᵀ + m * g
            I * ω̇ = τ - ω × (I @ ω)
            ṗ = v
            q̇ = 0.5 * Ω(ω) @ q
        """

        def __init__(self, state, cost_model, mass: float, inertia: np.ndarray, gravity: np.ndarray):
            nu = 4
            crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, cost_model.nr)
            self.costs = cost_model
            self.mass = mass
            self.inertia = inertia
            self.inertia_inv = np.linalg.inv(inertia)
            self.gravity = gravity

        def calc(self, data, x, u=None):
            if u is None:
                u = np.zeros(4)

            quat = x[3:7]
            vel = x[7:10]
            omega = x[10:13]
            thrust = u[0]
            torque = u[1:4]

            R = self._quat_to_rot(quat)
            thrust_body = np.array([0.0, 0.0, thrust])
            thrust_world = R @ thrust_body
            acc = thrust_world / self.mass + self.gravity

            omega_cross_I_omega = np.cross(omega, self.inertia @ omega)
            angular_acc = self.inertia_inv @ (torque - omega_cross_I_omega)

            omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
            quat_dot = 0.5 * self._quat_mult(omega_quat, quat)

            data.xout = np.concatenate([vel, quat_dot, acc, angular_acc])
            self.costs.calc(data.costs, x, u)
            data.cost = data.costs.cost

        def calcDiff(self, data, x, u=None):
            if u is None:
                u = np.zeros(4)

            self.costs.calcDiff(data.costs, x, u)

            eps = 1e-6
            nx = self.state.nx
            nu = self.nu

            data.Fx = np.zeros((nx, nx))
            for i in range(nx):
                x_plus, x_minus = x.copy(), x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                data_plus, data_minus = self.createData(), self.createData()
                self.calc(data_plus, x_plus, u)
                self.calc(data_minus, x_minus, u)
                data.Fx[:, i] = (data_plus.xout - data_minus.xout) / (2 * eps)

            data.Fu = np.zeros((nx, nu))
            for i in range(nu):
                u_plus, u_minus = u.copy(), u.copy()
                u_plus[i] += eps
                u_minus[i] -= eps
                data_plus, data_minus = self.createData(), self.createData()
                self.calc(data_plus, x, u_plus)
                self.calc(data_minus, x, u_minus)
                data.Fu[:, i] = (data_plus.xout - data_minus.xout) / (2 * eps)

            data.Lx = data.costs.Lx
            data.Lu = data.costs.Lu
            data.Lxx = data.costs.Lxx
            data.Luu = data.costs.Luu
            data.Lxu = data.costs.Lxu

        def createData(self):
            data = crocoddyl.DifferentialActionDataAbstract(self)
            data.costs = self.costs.createData(crocoddyl.DataCollectorAbstract())
            return data

        @staticmethod
        def _quat_to_rot(q: np.ndarray) -> np.ndarray:
            w, x, y, z = q
            return np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
            ])

        @staticmethod
        def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

else:
    # Placeholder when Crocoddyl is not available
    _QuadrotorDifferentialActionModel = None
