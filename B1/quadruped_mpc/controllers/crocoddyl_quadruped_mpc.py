# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crocoddyl MPC controller for quadruped locomotion.

This controller integrates all the quadruped-specific components:
- GaitScheduler → contact timing
- FootholdPlanner → landing positions
- BezierFootTrajectory → swing trajectories
- OCPFactory → Crocoddyl OCP construction
- SolverFDDP → trajectory optimization

State representation: (nq + nv) dimensional
    For a 12-DOF quadruped:
        q = [x, y, z, qw, qx, qy, qz, joint1...joint12] → nq = 19
        v = [vx, vy, vz, ωx, ωy, ωz, dq1...dq12]       → nv = 18
        Full state: 37D

Control: 12D joint torques (floating base is unactuated)
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..gait.contact_sequence import ContactSequence
from ..gait.foothold_planner import FootholdPlan, FootholdPlanner
from ..gait.gait_scheduler import GaitScheduler
from ..gait.ocp_factory import OCPFactory
from ..trajectory.bezier_foot_trajectory import BezierFootTrajectory
from ..utils.math_utils import heading_from_tangent
from .base_mpc import BaseMPC, MPCSolution

# Try to import Crocoddyl
try:
    import crocoddyl
    import pinocchio

    CROCODDYL_AVAILABLE = True
except ImportError:
    CROCODDYL_AVAILABLE = False
    crocoddyl = None
    pinocchio = None


class CrocoddylQuadrupedMPC(BaseMPC):
    """MPC controller for quadruped locomotion.

    Integrates:
    - GaitScheduler → contact timing
    - FootholdPlanner → landing positions
    - BezierFootTrajectory → swing trajectories
    - OCPFactory → Crocoddyl OCP construction
    - SolverFDDP → trajectory optimization

    Attributes:
        rmodel: Pinocchio robot model.
        rdata: Pinocchio robot data.
        state: Crocoddyl state model.
        actuation: Crocoddyl actuation model.
        gait_scheduler: GaitScheduler for contact sequence generation.
        foothold_planner: FootholdPlanner for computing landing positions.
        foot_trajectory_gen: BezierFootTrajectory for swing arcs.
        ocp_factory: OCPFactory for building OCP nodes.
    """

    def __init__(
        self,
        rmodel: "pinocchio.Model",
        foot_frame_names: Dict[str, str],
        hip_offsets: Optional[Dict[str, np.ndarray]] = None,
        gait_type: str = "trot",
        dt: float = 0.02,
        horizon_steps: int = 25,
        step_duration: float = 0.15,
        support_duration: float = 0.05,
        step_height: float = 0.05,
        mu: float = 0.7,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
    ):
        """Initialize MPC with all sub-components.

        Args:
            rmodel: Pinocchio robot model.
            foot_frame_names: Dict mapping foot name to Pinocchio frame name.
                Example: {"LF": "LF_FOOT", "RF": "RF_FOOT", ...}
            hip_offsets: Dict mapping foot name to hip offset in body frame.
                If None, uses FootholdPlanner defaults.
            gait_type: Type of gait ("trot", "walk", "pace", "bound").
            dt: MPC timestep in seconds (50 Hz default).
            horizon_steps: Number of MPC prediction horizon steps.
            step_duration: Duration of each swing phase in seconds.
            support_duration: Double-support duration between swings.
            step_height: Default foot swing height.
            mu: Friction coefficient.
            max_iterations: Maximum FDDP solver iterations.
            convergence_threshold: Solver convergence threshold.
        """
        if not CROCODDYL_AVAILABLE:
            raise ImportError(
                "Crocoddyl is not available. Please install crocoddyl."
            )

        self.rmodel = rmodel
        self.rdata = rmodel.createData()

        # Store parameters
        self.dt = dt
        self.horizon_steps = horizon_steps
        self.gait_type = gait_type
        self.step_duration = step_duration
        self.support_duration = support_duration
        self.step_height = step_height
        self.mu = mu
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Get frame IDs from names
        self.foot_frame_names = foot_frame_names
        self.foot_frame_ids = {}
        for foot_name, frame_name in foot_frame_names.items():
            try:
                frame_id = rmodel.getFrameId(frame_name)
                self.foot_frame_ids[foot_name] = frame_id
            except Exception as e:
                raise ValueError(f"Frame '{frame_name}' not found in model: {e}")

        # Initialize sub-components
        self.gait_scheduler = GaitScheduler()
        self.foothold_planner = FootholdPlanner(
            hip_offsets=hip_offsets,
            step_height=step_height,
        )
        self.foot_trajectory_gen = BezierFootTrajectory(step_height=step_height)
        self.ocp_factory = OCPFactory(
            rmodel=rmodel,
            foot_frame_ids=self.foot_frame_ids,
            mu=mu,
        )

        # State and actuation from factory
        self.state = self.ocp_factory.state
        self.actuation = self.ocp_factory.actuation

        # Warm-start storage
        self._prev_xs: Optional[List[np.ndarray]] = None
        self._prev_us: Optional[List[np.ndarray]] = None
        self._solver: Optional[Any] = None

        # Contact sequence cache
        self._cached_contact_sequence: Optional[ContactSequence] = None

    def solve(
        self,
        current_state: np.ndarray,
        com_reference: np.ndarray,
        current_foot_positions: Optional[Dict[str, np.ndarray]] = None,
        gait_params: Optional[Dict[str, float]] = None,
        warm_start: bool = True,
    ) -> MPCSolution:
        """Solve MPC and return optimal control.

        Full MPC pipeline:
        1. Generate ContactSequence from GaitScheduler
        2. Compute footholds from FootholdPlanner using com_reference
        3. Generate foot swing trajectories from BezierFootTrajectory
        4. Build OCP from OCPFactory
        5. Warm-start from previous solution (shifted by one step)
        6. Solve with FDDP
        7. Return first control action (12D joint torques)

        Args:
            current_state: Current robot state (nq + nv).
            com_reference: CoM trajectory from Bezier, shape (T, 3).
            current_foot_positions: Current position of each foot.
                If None, computed from current_state via FK.
            gait_params: Optional RL-provided gait modulation:
                - "step_length": modifier for step length
                - "step_height": modifier for step height
                - "step_frequency": modifier for step frequency
            warm_start: If True, use shifted previous solution.

        Returns:
            MPCSolution with optimal control and solver info.
        """
        start_time = time.time()

        # Parse gait parameters
        step_frequency_mod = 1.0
        step_height_mod = 1.0
        if gait_params is not None:
            step_frequency_mod = gait_params.get("step_frequency", 1.0)
            step_height_mod = gait_params.get("step_height", 1.0)

        # Compute current foot positions from FK if not provided
        if current_foot_positions is None:
            current_foot_positions = self._compute_foot_positions(current_state)

        # Compute current heading from state
        current_heading = self._extract_heading(current_state)

        # Generate contact sequence
        step_duration = self.step_duration / step_frequency_mod
        support_duration = self.support_duration / step_frequency_mod

        # Determine number of gait cycles to fill horizon
        cycle_duration = self._get_cycle_duration(step_duration, support_duration)
        num_cycles = max(1, int(np.ceil(self.horizon_steps * self.dt / cycle_duration)))

        contact_sequence = self.gait_scheduler.generate(
            gait_type=self.gait_type,
            step_duration=step_duration,
            support_duration=support_duration,
            num_cycles=num_cycles,
        )

        # Compute heading trajectory from CoM reference tangent
        heading_trajectory = self._compute_heading_trajectory(com_reference, self.dt)

        # Plan footholds
        step_height = self.step_height * step_height_mod
        foothold_plans = self.foothold_planner.plan_footholds(
            com_trajectory=com_reference,
            contact_sequence=contact_sequence,
            current_foot_positions=current_foot_positions,
            dt=self.dt,
            step_height=step_height,
        )

        # Build OCP
        problem = self.ocp_factory.build_problem(
            x0=current_state,
            contact_sequence=contact_sequence,
            com_trajectory=com_reference,
            foot_trajectories=foothold_plans,
            dt=self.dt,
            heading_trajectory=heading_trajectory,
        )

        # Create solver
        solver = crocoddyl.SolverFDDP(problem)
        solver.th_stop = self.convergence_threshold

        # Warm-start
        if warm_start and self._prev_xs is not None and self._prev_us is not None:
            # Shift previous solution by one step
            xs_init = self._shift_trajectory(self._prev_xs, current_state)
            us_init = self._shift_controls(self._prev_us)

            # Ensure correct lengths
            T = len(problem.runningModels)
            xs_init = self._adjust_length(xs_init, T + 1, current_state)
            us_init = self._adjust_length(us_init, T, np.zeros(self.actuation.nu))

            solver.setCandidate(xs_init, us_init, False)

        # Solve
        converged = solver.solve(
            [], [],  # Initial guess (use candidate if set)
            self.max_iterations,
            False,  # regInit
            0.0,  # stopTh (use th_stop)
        )

        solve_time = time.time() - start_time

        # Extract solution
        xs = list(solver.xs)
        us = list(solver.us)

        # Store for warm-start
        self._prev_xs = xs
        self._prev_us = us

        # First control action
        control = us[0] if len(us) > 0 else np.zeros(self.actuation.nu)

        # Predicted trajectory
        predicted_states = np.array(xs)
        predicted_controls = np.array(us) if len(us) > 0 else np.zeros((0, self.actuation.nu))

        return MPCSolution(
            control=control,
            predicted_states=predicted_states,
            predicted_controls=predicted_controls,
            solve_time=solve_time,
            converged=bool(converged),
            cost=float(solver.cost),
            iterations=int(solver.iter),
        )

    def _compute_foot_positions(
        self, state: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute current foot positions from state using FK.

        Args:
            state: Robot state (nq + nv).

        Returns:
            Dict mapping foot name to position (3,).
        """
        q = state[: self.rmodel.nq]
        pinocchio.framesForwardKinematics(self.rmodel, self.rdata, q)

        positions = {}
        for foot_name, frame_id in self.foot_frame_ids.items():
            oMf = self.rdata.oMf[frame_id]
            positions[foot_name] = oMf.translation.copy()

        return positions

    def _extract_heading(self, state: np.ndarray) -> float:
        """Extract body yaw angle from state.

        Args:
            state: Robot state.

        Returns:
            Yaw angle in radians.
        """
        # Quaternion is at indices 3:7 (x, y, z, w) or depends on model convention
        # For typical floating base: [x, y, z, qx, qy, qz, qw, joints...]
        # Pinocchio uses (qx, qy, qz, qw) convention internally
        q = state[: self.rmodel.nq]

        # Get base orientation (assume first 7 elements are floating base)
        quat_xyzw = q[3:7]  # Pinocchio convention: (qx, qy, qz, qw)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        from ..utils.math_utils import yaw_from_quaternion
        return yaw_from_quaternion(quat_wxyz)

    def _compute_heading_trajectory(
        self, com_trajectory: np.ndarray, dt: float
    ) -> np.ndarray:
        """Compute heading trajectory from CoM trajectory tangent.

        Args:
            com_trajectory: CoM waypoints, shape (T, 3).
            dt: Timestep.

        Returns:
            Heading angles, shape (T,).
        """
        T = len(com_trajectory)
        headings = np.zeros(T)

        for i in range(T):
            # Use central differences where possible
            if i == 0 and T > 1:
                tangent = (com_trajectory[1] - com_trajectory[0]) / dt
            elif i >= T - 1:
                tangent = (com_trajectory[-1] - com_trajectory[-2]) / dt
            else:
                tangent = (com_trajectory[i + 1] - com_trajectory[i - 1]) / (2 * dt)

            headings[i] = heading_from_tangent(tangent[:2])

        return headings

    def _get_cycle_duration(
        self, step_duration: float, support_duration: float
    ) -> float:
        """Compute duration of one full gait cycle.

        Args:
            step_duration: Swing phase duration.
            support_duration: Support phase duration.

        Returns:
            Cycle duration in seconds.
        """
        pattern = GaitScheduler.GAIT_PATTERNS.get(self.gait_type)
        if pattern is None:
            return 2 * step_duration + 2 * support_duration  # Default

        num_swing_groups = len(pattern["swing_groups"])
        return num_swing_groups * (step_duration + support_duration)

    def _shift_trajectory(
        self, xs: List[np.ndarray], x0: np.ndarray
    ) -> List[np.ndarray]:
        """Shift state trajectory by one step for warm-starting.

        Args:
            xs: Previous state trajectory.
            x0: New initial state.

        Returns:
            Shifted trajectory.
        """
        if len(xs) <= 1:
            return [x0]

        shifted = [x0] + xs[2:]  # Skip first two, use x0 as new start
        # Pad with last element if needed
        while len(shifted) < len(xs):
            shifted.append(xs[-1].copy())

        return shifted

    def _shift_controls(self, us: List[np.ndarray]) -> List[np.ndarray]:
        """Shift control trajectory by one step for warm-starting.

        Args:
            us: Previous control trajectory.

        Returns:
            Shifted trajectory.
        """
        if len(us) <= 1:
            return us

        shifted = us[1:]  # Skip first control
        # Pad with last element
        shifted.append(us[-1].copy())

        return shifted

    def _adjust_length(
        self,
        trajectory: List[np.ndarray],
        target_length: int,
        padding_value: np.ndarray,
    ) -> List[np.ndarray]:
        """Adjust trajectory length by padding or truncating.

        Args:
            trajectory: Trajectory to adjust.
            target_length: Desired length.
            padding_value: Value to use for padding.

        Returns:
            Adjusted trajectory.
        """
        while len(trajectory) < target_length:
            trajectory.append(padding_value.copy())
        while len(trajectory) > target_length:
            trajectory.pop()

        return trajectory

    def get_control_dim(self) -> int:
        """Return control dimension (12 for typical quadruped)."""
        return self.actuation.nu

    def get_state_dim(self) -> int:
        """Return state dimension (nq + nv)."""
        return self.state.nx

    def get_horizon_steps(self) -> int:
        """Return MPC horizon length."""
        return self.horizon_steps

    def get_dt(self) -> float:
        """Return MPC timestep."""
        return self.dt

    def reset(self):
        """Reset controller state (clear warm-start buffers)."""
        self._prev_xs = None
        self._prev_us = None
        self._solver = None
        self._cached_contact_sequence = None

    def set_gait_type(self, gait_type: str):
        """Change the gait type.

        Args:
            gait_type: New gait type ("trot", "walk", "pace", "bound").
        """
        if gait_type not in GaitScheduler.GAIT_PATTERNS:
            raise ValueError(f"Unknown gait type: {gait_type}")
        self.gait_type = gait_type
        self._cached_contact_sequence = None
