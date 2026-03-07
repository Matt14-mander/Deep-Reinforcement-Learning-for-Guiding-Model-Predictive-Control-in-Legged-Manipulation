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
    For a 12-DOF quadruped (Pinocchio convention):
        q = [x, y, z, qx, qy, qz, qw, joint1...joint12] → nq = 19
        v = [vx, vy, vz, ωx, ωy, ωz, dq1...dq12]       → nv = 18
        Full state: 37D

    IMPORTANT: Pinocchio's FreeFlyer base velocity v[0:6] is in the LOCAL
    (body) frame, NOT the world frame. The env must rotate Isaac Lab's
    world-frame velocities before passing them here.

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
        step_duration: float = 0.25,
        support_duration: float = 0.10,
        step_height: float = 0.05,
        foot_radius: float = 0.02,
        mu: float = 0.7,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-4,
        verbose: bool = False,
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
            foot_radius: Foot sphere radius (m). Used as default_ground_height
                so MPC targets the foot sphere center correctly above ground.
            mu: Friction coefficient.
            max_iterations: Maximum FDDP solver iterations.
            convergence_threshold: Solver convergence threshold.
            verbose: If True, print detailed solver info for debugging.
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
        self.verbose = verbose
        self._solve_count = 0  # Track solve calls for selective verbose

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
            default_ground_height=foot_radius,  # sphere center sits above ground
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

        # Gait phase tracking: tracks elapsed time within the current gait cycle.
        # Advances by dt each solve so the OCP always starts from the CORRECT
        # contact phase (support vs swing) rather than always restarting at t=0.
        # Without this, the OCP always plans "wait in support, THEN swing" and
        # us[0] is always a support-phase gravity-comp torque → robot never walks.
        self._gait_phase: float = 0.0

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

        # Parse gait parameters (clamp to safe ranges)
        step_frequency_mod = 1.0
        step_height_mod = 1.0
        if gait_params is not None:
            step_frequency_mod = gait_params.get("step_frequency", 1.0)
            step_height_mod = gait_params.get("step_height", 1.0)

        # Safety clamp: frequency must be positive to avoid negative durations
        step_frequency_mod = max(0.3, min(abs(step_frequency_mod), 3.0))
        step_height_mod = max(0.1, min(abs(step_height_mod), 3.0))

        # Compute current foot positions from FK if not provided
        if current_foot_positions is None:
            current_foot_positions = self._compute_foot_positions(current_state)

        # Compute current heading from state
        current_heading = self._extract_heading(current_state)

        # Generate contact sequence
        step_duration = self.step_duration / step_frequency_mod
        support_duration = self.support_duration / step_frequency_mod

        # Determine number of gait cycles to fill horizon.
        # Use +2 to ensure enough phases remain after phase_offset trimming.
        cycle_duration = self._get_cycle_duration(step_duration, support_duration)
        num_cycles = max(2, int(np.ceil(self.horizon_steps * self.dt / cycle_duration)) + 1)

        # Generate contact sequence starting from current gait phase.
        # phase_offset causes the sequence to start mid-cycle so the OCP
        # correctly models whether feet are currently in support or swing.
        # first_step_fraction=1.0: full steps (no ramping; phase tracking handles timing).
        # include_final_support=False: avoid double-counting the support between cycles.
        contact_sequence = self.gait_scheduler.generate(
            gait_type=self.gait_type,
            step_duration=step_duration,
            support_duration=support_duration,
            num_cycles=num_cycles,
            first_step_fraction=1.0,
            include_initial_support=True,
            include_final_support=False,
            phase_offset=self._gait_phase,
        )

        # Advance gait phase for next solve (mod cycle_duration to wrap around)
        self._gait_phase = (self._gait_phase + self.dt) % cycle_duration

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

        # Verbose logging for debugging
        self._solve_count += 1
        is_verbose_call = self.verbose and self._solve_count <= 5
        T = len(problem.runningModels)

        if is_verbose_call:
            import sys
            solver.setCallbacks([crocoddyl.CallbackVerbose()])
            print(f"\n[MPC Debug] Solve #{self._solve_count}", flush=True)
            print(f"  Problem: {T} running models, nu={self.actuation.nu}, nx={self.state.nx}", flush=True)
            print(f"  x0 pos: [{current_state[0]:.3f}, {current_state[1]:.3f}, {current_state[2]:.3f}]", flush=True)
            print(f"  x0 quat(xyzw): [{current_state[3]:.4f}, {current_state[4]:.4f}, {current_state[5]:.4f}, {current_state[6]:.4f}]", flush=True)
            quat_norm = np.linalg.norm(current_state[3:7])
            print(f"  x0 quat norm: {quat_norm:.6f} (should be 1.0)", flush=True)
            print(f"  x0 vel (body): [{current_state[self.rmodel.nq]:.3f}, {current_state[self.rmodel.nq+1]:.3f}, {current_state[self.rmodel.nq+2]:.3f}]", flush=True)
            print(f"  x0 joints[:6]: {current_state[7:13]}", flush=True)
            print(f"  CoM ref[0]: {com_reference[0]}", flush=True)
            print(f"  CoM ref[-1]: {com_reference[-1]}", flush=True)
            print(f"  OCP: {T} running nodes → terminal uses ref[{min(T, len(com_reference)-1)}] (NOT ref[-1]={com_reference[-1]})", flush=True)
            # gait_phase was already advanced before this log (see below), subtract dt to show current
            prev_phase = (self._gait_phase - self.dt) % cycle_duration
            phase0 = contact_sequence.phases[0] if contact_sequence.phases else None
            phase0_str = f"{phase0.phase_type}(feet={phase0.support_feet}, dur={phase0.duration:.3f}s)" if phase0 else "N/A"
            print(f"  Gait phase: {prev_phase:.3f}s / {cycle_duration:.3f}s, first OCP phase: {phase0_str}", flush=True)
            if current_foot_positions:
                for fname, fpos in current_foot_positions.items():
                    print(f"  Foot {fname}: [{fpos[0]:.3f}, {fpos[1]:.3f}, {fpos[2]:.3f}]", flush=True)
            sys.stdout.flush()

        # Warm-start or gravity-compensation cold-start
        if warm_start and self._prev_xs is not None and self._prev_us is not None:
            # Shift previous solution by one step
            xs_init = self._shift_trajectory(self._prev_xs, current_state)
            us_init = self._shift_controls(self._prev_us)

            # Ensure correct lengths
            xs_init = self._adjust_length(xs_init, T + 1, current_state)
            us_init = self._adjust_length(us_init, T, np.zeros(self.actuation.nu))

            solver.setCandidate(xs_init, us_init, False)
            if is_verbose_call:
                print(f"  Warm-start: YES (shifted prev solution)", flush=True)
        else:
            # COLD START: Use gravity compensation as initial control guess
            # Without this, the solver starts from zero controls (= robot falls)
            # which is a terrible initial trajectory for FDDP to recover from.
            u_grav = self._compute_gravity_compensation(current_state)
            xs_init = [current_state.copy() for _ in range(T + 1)]
            us_init = [u_grav.copy() for _ in range(T)]
            solver.setCandidate(xs_init, us_init, False)
            if is_verbose_call:
                print(f"  Cold-start: gravity compensation |u|={np.linalg.norm(u_grav):.3f}", flush=True)
                print(f"  u_grav: [{', '.join(f'{v:.2f}' for v in u_grav)}]", flush=True)

        # Solve.
        # regInit=1e-9: provide a small non-zero initial regularization so
        # FDDP's backward pass stays well-conditioned on the contact Hessian.
        # With regInit=0 the solver may stall because preg/dreg never increase
        # when the descent direction is poor but the Hessian is nominally PD.
        converged = solver.solve(
            [], [],  # Initial guess (use candidate if set)
            self.max_iterations,
            False,  # isFeasible (warm-start trajectory is NOT dynamically feasible)
            1e-9,   # regInit: small initial regularization for numerical stability
        )

        if is_verbose_call:
            print(f"  Result: converged={converged}, iters={solver.iter}, cost={solver.cost:.2f}", flush=True)
            if len(solver.us) > 0:
                u0 = solver.us[0]
                print(f"  u[0]: [{', '.join(f'{v:.2f}' for v in u0)}]", flush=True)
                print(f"  |u[0]| = {np.linalg.norm(u0):.3f}", flush=True)
            import sys
            sys.stdout.flush()

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

    def _compute_gravity_compensation(self, state: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques for the current configuration.

        Uses Pinocchio's RNEA (Recursive Newton-Euler Algorithm) with zero
        velocity and zero acceleration to find the joint torques needed to
        hold the robot in a static pose against gravity.

        This provides a MUCH better initial guess for the FDDP solver than
        zero controls (which would cause a free-fall trajectory).

        Args:
            state: Robot state (nq + nv).

        Returns:
            Joint torques (nu = nv-6 = 12D) for gravity compensation.
        """
        q = state[:self.rmodel.nq].copy()
        v = np.zeros(self.rmodel.nv)
        a = np.zeros(self.rmodel.nv)

        # RNEA: tau = M(q)*a + C(q,v)*v + g(q)
        # With v=0 and a=0, this gives tau = g(q) (gravity torques)
        tau = pinocchio.rnea(self.rmodel, self.rdata, q, v, a)

        # tau is (nv,) = 18D: [base(6), joints(12)]
        # For ActuationModelFloatingBase, control is only joints (12D)
        u_grav = tau[6:]  # Skip unactuated floating base

        return u_grav

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

    def reset(self):
        """Reset MPC state for a new episode.

        Clears warm-start cache and resets gait phase to 0 so the new episode
        starts from the beginning of the gait cycle (full-support phase).
        Call this from the environment's episode reset.
        """
        self._prev_xs = None
        self._prev_us = None
        self._gait_phase = 0.0
        self._solve_count = 0

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

        # BUG FIX: xs[2:] skipped xs[1] (2-step shift), inflating ||ffeas||.
        # Correct 1-step shift: replace old x0 with new x0, keep x1 onward.
        shifted = [x0] + list(xs[1:])
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
