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
        step_duration: float = 0.15,
        support_duration: float = 0.05,
        step_height: float = 0.05,
        mu: float = 0.7,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-4,
        verbose: bool = False,
        force_standing_contacts: bool = False,
        use_demo_stabilization_weights: bool = False,
        enable_warm_start: bool = True,
        reference_is_root_position: bool = False,
        return_quasi_static_control: bool = False,
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
            verbose: If True, print detailed solver info for debugging.
            reference_is_root_position: Interpret incoming Bezier positions as
                floating-base/root positions and translate them to true model
                CoM positions before constructing Crocoddyl costs.
            return_quasi_static_control: Diagnostic mode that bypasses FDDP and
                returns the contact-consistent quasi-static control directly.
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
        self.force_standing_contacts = force_standing_contacts
        self.use_demo_stabilization_weights = use_demo_stabilization_weights
        self.enable_warm_start = enable_warm_start
        self.reference_is_root_position = reference_is_root_position
        self.return_quasi_static_control = return_quasi_static_control
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
            step_height=step_height,
            default_ground_height=0.02,  # foot sphere radius; updated dynamically in solve()
        )
        self.foot_trajectory_gen = BezierFootTrajectory(step_height=step_height)
        self.ocp_factory = OCPFactory(
            rmodel=rmodel,
            foot_frame_ids=self.foot_frame_ids,
            mu=mu,
            use_demo_stabilization_weights=use_demo_stabilization_weights,
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

        # Gait phase clock: tracks elapsed time in the gait cycle across MPC calls.
        # Incremented by dt after each solve so that successive calls generate contact
        # sequences starting from the correct phase (not always from the beginning).
        # Without this, every solve starts with "initial support phase" and the robot
        # never reaches the swing phases — the "Groundhog Day" bug.
        self._gait_clock: float = 0.0


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

        # QuadrupedMPCEnv anchors its Bezier curve at Isaac's floating-base/root
        # position. ResidualModelCoMPosition tracks the model's true centre of
        # mass, so using the root height directly commands an artificial upward
        # displacement. Preserve the desired root displacement while anchoring
        # the OCP trajectory at Pinocchio's current CoM.
        raw_reference = np.asarray(com_reference, dtype=float)
        ocp_com_reference = raw_reference
        current_model_com = None
        root_to_com_offset = None
        if self.reference_is_root_position:
            q = np.asarray(current_state[: self.rmodel.nq], dtype=float)
            current_model_com = np.asarray(
                pinocchio.centerOfMass(self.rmodel, self.rdata, q), dtype=float
            ).copy()
            root_to_com_offset = current_model_com - np.asarray(
                current_state[:3], dtype=float
            )
            ocp_com_reference = raw_reference + root_to_com_offset[None, :]

        # Compute current heading from state
        current_heading = self._extract_heading(current_state)

        # Generate contact sequence
        step_duration = self.step_duration / step_frequency_mod
        support_duration = self.support_duration / step_frequency_mod

        if self.force_standing_contacts:
            # Isolation groups 1 and 2 keep exactly the same four-foot support
            # topology at every node and solve. This removes gait-clock, swing,
            # impact and topology-transition effects from the closed-loop test.
            contact_sequence = self.gait_scheduler.generate_standing(
                duration=max(self.horizon_steps * self.dt, self.dt)
            )
        else:
            # Determine number of gait cycles to fill horizon
            cycle_duration = self._get_cycle_duration(step_duration, support_duration)
            num_cycles = max(
                1, int(np.ceil(self.horizon_steps * self.dt / cycle_duration))
            )

            # Start the horizon at the current gait phase.
            phase_offset = self._gait_clock % cycle_duration
            contact_sequence = self.gait_scheduler.generate_from_phase_offset(
                gait_type=self.gait_type,
                step_duration=step_duration,
                support_duration=support_duration,
                num_cycles=num_cycles,
                phase_offset=phase_offset,
            )

        # Compute heading trajectory from CoM reference tangent
        heading_trajectory = self._compute_heading_trajectory(ocp_com_reference, self.dt)

        # Update ground height estimate from current foot positions (Fix: eliminates
        # foot_track cost explosion when feet are above ground at initialization)
        if current_foot_positions:
            ground_z = min(pos[2] for pos in current_foot_positions.values())
            self.foothold_planner.default_ground_height = 0.02

        # Plan footholds
        step_height = self.step_height * step_height_mod
        foothold_plans = self.foothold_planner.plan_footholds(
            com_trajectory=ocp_com_reference,
            contact_sequence=contact_sequence,
            current_foot_positions=current_foot_positions,
            dt=self.dt,
            step_height=step_height,
        )

        # Build OCP (cap at horizon_steps to prevent node overflow from long contact sequences)
        problem = self.ocp_factory.build_problem(
            x0=current_state,
            contact_sequence=contact_sequence,
            com_trajectory=ocp_com_reference,
            foot_trajectories=foothold_plans,
            current_foot_positions=current_foot_positions,
            dt=self.dt,
            heading_trajectory=heading_trajectory,
            max_nodes=self.horizon_steps,
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
            if self.reference_is_root_position:
                print(f"  Root ref[0]: {raw_reference[0]}", flush=True)
                print(f"  Model CoM now: {current_model_com}", flush=True)
                print(f"  Root->CoM offset: {root_to_com_offset}", flush=True)
            print(f"  CoM ref[0]: {ocp_com_reference[0]}", flush=True)
            print(f"  CoM ref[-1]: {ocp_com_reference[-1]}", flush=True)
            print(
                "  Isolation: "
                f"fixed_contacts={self.force_standing_contacts}, "
                f"warm_start_enabled={self.enable_warm_start}, "
                f"root_reference={self.reference_is_root_position}, "
                f"demo_weights={self.use_demo_stabilization_weights}",
                flush=True,
            )
            print(
                "  OCP weights: "
                f"com={self.ocp_factory.weights['com_track']:.1e}, "
                f"state={self.ocp_factory.weights['state_reg']:.1e}, "
                f"control={self.ocp_factory.weights['ctrl_reg']:.1e}",
                flush=True,
            )
            print(
                "  Contact phases: "
                + ", ".join(
                    f"{phase.phase_type}:{'/'.join(phase.support_feet)}"
                    for phase in contact_sequence.phases
                ),
                flush=True,
            )
            if current_foot_positions:
                for fname, fpos in current_foot_positions.items():
                    print(f"  Foot {fname}: [{fpos[0]:.3f}, {fpos[1]:.3f}, {fpos[2]:.3f}]", flush=True)
            sys.stdout.flush()

        # Warm-start from the shifted previous solution, or cold-start from a
        # rollout of gravity-compensation controls. SolverFDDP.solve() calls
        # setCandidate() internally, so these trajectories must be passed to
        # solve() rather than installed before a later solve([], []) call.
        use_warm_start = False
        rollout_is_finite = False
        if (
            warm_start
            and self.enable_warm_start
            and self._prev_xs is not None
            and self._prev_us is not None
        ):
            # Shift previous solution by one step (xs stays near current state)
            xs_init = self._shift_trajectory(self._prev_xs, current_state)
            us_init = self._shift_controls(self._prev_us)
            xs_init = self._adjust_length(xs_init, T + 1, current_state)
            us_init = self._adjust_length(us_init, T, np.zeros(self.actuation.nu))
            use_warm_start = True
            if is_verbose_call:
                print(f"  Warm-start: YES (shifted prev solution)", flush=True)
        else:
            # Compute contact-consistent equilibrium controls. Taking only the
            # actuated tail of Pinocchio's free-body gravity vector ignores the
            # contact force distribution and is not a standing equilibrium.
            x_static = np.asarray(current_state, dtype=float).copy()
            x_static[self.rmodel.nq :] = 0.0
            us_init = list(problem.quasiStatic([x_static.copy() for _ in range(T)]))
            xs_init = list(problem.rollout(us_init))
            rollout_array = np.asarray(xs_init)
            rollout_is_finite = bool(np.all(np.isfinite(rollout_array)))
            if is_verbose_call:
                u0_static = np.asarray(us_init[0])
                print(
                    "  Cold-start: contact quasi-static "
                    f"|u[0]|={np.linalg.norm(u0_static):.3f}",
                    flush=True,
                )
                print(
                    f"  u_static[0]: [{', '.join(f'{v:.2f}' for v in u0_static)}]",
                    flush=True,
                )
                print(f"  Rollout finite: {rollout_is_finite}", flush=True)
                if not rollout_is_finite:
                    bad_index = np.argwhere(~np.isfinite(rollout_array))[0]
                    print(
                        "  First non-finite rollout entry: "
                        f"node={int(bad_index[0])}, state_index={int(bad_index[1])}",
                        flush=True,
                    )
            if not rollout_is_finite:
                # Keep FDDP inputs finite so it can handle the dynamics gaps,
                # while the diagnostic above still exposes the failed rollout.
                xs_init = [
                    np.asarray(current_state, dtype=float).copy()
                    for _ in range(T + 1)
                ]

            if self.return_quasi_static_control:
                if is_verbose_call:
                    print(
                        "  Diagnostic bypass: returning quasi-static control "
                        "without FDDP",
                        flush=True,
                    )
                solve_time = time.time() - start_time
                return MPCSolution(
                    control=np.asarray(us_init[0]).copy(),
                    predicted_states=np.asarray(xs_init),
                    predicted_controls=np.asarray(us_init),
                    solve_time=solve_time,
                    converged=rollout_is_finite,
                    # Keep the environment guard out of this controlled test;
                    # rollout validity is reported separately above.
                    cost=0.0 if rollout_is_finite else float("inf"),
                    iterations=0,
                )

        # Solve.
        # A shifted warm-start can contain dynamics gaps after replacing x0,
        # while the cold rollout is feasible by construction.
        COLD_START_ITERS = 100
        n_iters = self.max_iterations if use_warm_start else COLD_START_ITERS
        initial_guess_feasible = not use_warm_start and rollout_is_finite
        if is_verbose_call:
            print(
                f"  Initial guess: xs={len(xs_init)}, us={len(us_init)}, "
                f"is_feasible={initial_guess_feasible}",
                flush=True,
            )
        converged = solver.solve(
            xs_init,
            us_init,
            n_iters,
            initial_guess_feasible,
            1e-9,  # regInit: small initial regularization for a good guess
        )

        if is_verbose_call:
            mode = "warm" if use_warm_start else "cold"
            print(f"  Result [{mode}-start, max={n_iters}]: converged={converged}, iters={solver.iter}, cost={solver.cost:.2f}", flush=True)
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

        # Only store xs/us for next warm-start when this solve did NOT diverge.
        # Diverged xs (cost >> 1e4) corrupt the next warm-start → cascade failure.
        STORE_COST_THRESHOLD = 1e4
        if solver.cost < STORE_COST_THRESHOLD:
            self._prev_xs = xs
            self._prev_us = us
        # else: keep previous _prev_xs intact (last good solution)

        # Advance the gait phase clock so the next solve starts from the correct
        # position in the gait cycle (fixes the "Groundhog Day" bug).
        if not self.force_standing_contacts:
            self._gait_clock += self.dt

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

        # One MPC interval has already been executed: the measured ``x0`` now
        # replaces the old predicted x1. The next predicted state must therefore
        # start at old x2, matching the control shift u1, u2, ... below. Keeping
        # old x1 here pairs it with old u1 and creates a one-tick dynamics gap.
        shifted = [x0.copy()] + [x.copy() for x in xs[2:]]
        # Preserve the original T+1 state count by extending the terminal state.
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
        """Reset controller state (clear warm-start buffers and gait clock)."""
        self._prev_xs = None
        self._prev_us = None
        self._solver = None
        self._cached_contact_sequence = None
        self._gait_clock = 0.0

    def set_gait_type(self, gait_type: str):
        """Change the gait type.

        Args:
            gait_type: New gait type ("trot", "walk", "pace", "bound").
        """
        if gait_type not in GaitScheduler.GAIT_PATTERNS:
            raise ValueError(f"Unknown gait type: {gait_type}")
        self.gait_type = gait_type
        self._cached_contact_sequence = None
