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
from ..utils.math_utils import heading_from_tangent, rotation_matrix_z
from .base_mpc import BaseMPC, MPCSolution, solver_residual_norm

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
        friction_cone_weight: Optional[float] = None,
        use_pseudo_impulse: bool = False,
        initial_full_support_duration: float = 0.0,
        use_feasible_cold_start_rollout: bool = False,
        enable_warm_start: bool = True,
        reference_is_root_position: bool = False,
        return_quasi_static_control: bool = False,
        touchdown_hold_steps: int = 0,
        swing_landing_height_ratio: float = 0.8,
        touchdown_gate_height_tolerance: float = 0.0,
        touchdown_gate_max_steps: int = 0,
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
            friction_cone_weight: Optional override for the weighted friction-cone
                barrier. This is exposed for controlled diagnostics; ``None``
                preserves the factory default.
            use_pseudo_impulse: Insert the legacy zero-time touchdown node.
                Exposed so diagnostics can isolate its timing effect.
            reference_is_root_position: Interpret incoming Bezier positions as
                floating-base/root positions and translate them to true model
                CoM positions before constructing Crocoddyl costs.
            return_quasi_static_control: Diagnostic mode that bypasses FDDP and
                returns the contact-consistent quasi-static control directly.
            touchdown_hold_steps: Extra final swing nodes held at the planned
                landing position before switching that foot to support.
            swing_landing_height_ratio: P2 vertical control-point height as a
                fraction of swing height; lower values begin descent earlier.
            touchdown_gate_height_tolerance: At the final swing node, delay
                contact-phase activation while a foot is more than this many
                metres above its locked landing target. Zero disables gating.
            touchdown_gate_max_steps: Maximum extra MPC ticks allowed for the
                physical foot to reach the landing target.
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
        self.friction_cone_weight = friction_cone_weight
        self.use_pseudo_impulse = use_pseudo_impulse
        self.initial_full_support_duration = max(
            0.0, float(initial_full_support_duration)
        )
        self.use_feasible_cold_start_rollout = use_feasible_cold_start_rollout
        self.enable_warm_start = enable_warm_start
        self.reference_is_root_position = reference_is_root_position
        self.return_quasi_static_control = return_quasi_static_control
        self.touchdown_hold_steps = max(0, int(touchdown_hold_steps))
        self.swing_landing_height_ratio = float(swing_landing_height_ratio)
        self.touchdown_gate_height_tolerance = max(
            0.0, float(touchdown_gate_height_tolerance)
        )
        self.touchdown_gate_max_steps = max(0, int(touchdown_gate_max_steps))
        self._touchdown_gate_steps = 0
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
            touchdown_hold_steps=self.touchdown_hold_steps,
            swing_landing_height_ratio=self.swing_landing_height_ratio,
        )
        self.foot_trajectory_gen = BezierFootTrajectory(
            step_height=step_height,
            landing_height_ratio=self.swing_landing_height_ratio,
        )
        self.ocp_factory = OCPFactory(
            rmodel=rmodel,
            foot_frame_ids=self.foot_frame_ids,
            mu=mu,
            weights=(
                {"friction_cone": float(friction_cone_weight)}
                if friction_cone_weight is not None
                else None
            ),
            use_demo_stabilization_weights=use_demo_stabilization_weights,
            use_pseudo_impulse=use_pseudo_impulse,
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
        self._active_swing_start_positions: Dict[str, np.ndarray] = {}
        self._active_swing_end_positions: Dict[str, np.ndarray] = {}
        self._nominal_foot_offsets: Optional[Dict[str, np.ndarray]] = None


    def solve(
        self,
        current_state: np.ndarray,
        com_reference: np.ndarray,
        current_foot_positions: Optional[Dict[str, np.ndarray]] = None,
        gait_params: Optional[Dict[str, float]] = None,
        warm_start: bool = True,
        current_foot_velocities: Optional[Dict[str, np.ndarray]] = None,
        current_foot_contacts: Optional[Dict[str, bool]] = None,
        current_foot_forces: Optional[Dict[str, np.ndarray]] = None,
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
            current_foot_velocities: Measured world-frame foot velocities.
            current_foot_contacts: Measured physical contact flags. During
                Stage 1 these are diagnostic and do not replace scheduled OCP
                contacts.
            current_foot_forces: Measured world-frame net contact forces.
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
        q = np.asarray(current_state[: self.rmodel.nq], dtype=float)
        current_model_com = np.asarray(
            pinocchio.centerOfMass(self.rmodel, self.rdata, q), dtype=float
        ).copy()
        root_to_com_offset = current_model_com - np.asarray(
            current_state[:3], dtype=float
        )
        if self.reference_is_root_position:
            ocp_com_reference = raw_reference + root_to_com_offset[None, :]

        # Compute current heading from state
        current_heading = self._extract_heading(current_state)

        # The configured offsets locate Go2's hip joints, not its nominal foot
        # contacts.  Calibrate the support footprint once from the simulator's
        # actual standing feet so foothold planning preserves the robot's true
        # fore/aft and lateral stance instead of pulling every foot under a hip.
        if self._nominal_foot_offsets is None and current_foot_positions:
            world_to_heading = rotation_matrix_z(current_heading).T
            self._nominal_foot_offsets = {}
            for foot_name, foot_position in current_foot_positions.items():
                offset = world_to_heading @ (
                    np.asarray(foot_position, dtype=float) - current_model_com
                )
                offset[2] = 0.0
                self._nominal_foot_offsets[foot_name] = offset
            self.foothold_planner.hip_offsets = {
                foot: offset.copy()
                for foot, offset in self._nominal_foot_offsets.items()
            }

        # Generate contact sequence
        requested_step_duration = self.step_duration / step_frequency_mod
        requested_support_duration = self.support_duration / step_frequency_mod

        # Contact phases must live on the same integer time grid as the OCP.
        # For example, 0.25 / 0.02 = 12.5; repeatedly rounding the remaining
        # duration can remove two nodes on one MPC call while warm-start shifts
        # by exactly one. Quantize once before scheduling so every receding-
        # horizon update advances one model for one 20 ms control tick.
        step_knots = max(
            1, int(np.floor(requested_step_duration / self.dt + 0.5))
        )
        support_knots = max(
            1, int(np.floor(requested_support_duration / self.dt + 0.5))
        )
        step_duration = step_knots * self.dt
        support_duration = support_knots * self.dt

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

            # Let the simulated robot establish real four-foot contact before
            # entering the first two-foot swing phase. During this startup window
            # the horizon still contains the upcoming gait, so the solver can
            # anticipate liftoff instead of repeatedly solving a standing OCP.
            gait_elapsed = max(
                0.0, self._gait_clock - self.initial_full_support_duration
            )
            phase_offset = gait_elapsed % cycle_duration
            gait_sequence = self.gait_scheduler.generate_from_phase_offset(
                gait_type=self.gait_type,
                step_duration=step_duration,
                support_duration=support_duration,
                num_cycles=num_cycles,
                phase_offset=phase_offset,
            )
            startup_remaining = self.initial_full_support_duration - self._gait_clock
            if startup_remaining > 1e-9:
                startup_sequence = self.gait_scheduler.generate_standing(
                    duration=startup_remaining
                )
                contact_sequence = ContactSequence(
                    phases=startup_sequence.phases + gait_sequence.phases
                )
            else:
                contact_sequence = gait_sequence

        # Compute heading trajectory from CoM reference tangent
        heading_trajectory = self._compute_heading_trajectory(ocp_com_reference, self.dt)

        current_phase = (
            contact_sequence.phases[0] if contact_sequence.phases else None
        )

        # Estimate the collision-foot centre ground height from the lowest
        # finite scheduled support foot.  Using the median of two support feet
        # is unsafe: if one is still rebounding, the two-value median is their
        # average and can lift the next touchdown target several centimetres
        # above ground.  Excluding swing feet also prevents a descending or
        # penetrating swing measurement from lowering the estimate.
        if current_foot_positions:
            ground_candidate_feet = (
                current_phase.support_feet
                if current_phase is not None and current_phase.support_feet
                else list(current_foot_positions.keys())
            )
            finite_support_heights = [
                float(current_foot_positions[foot][2])
                for foot in ground_candidate_feet
                if foot in current_foot_positions
                and np.isfinite(current_foot_positions[foot][2])
            ]
            if finite_support_heights:
                self.foothold_planner.default_ground_height = float(
                    np.min(finite_support_heights)
                )

        # Plan footholds
        step_height = self.step_height * step_height_mod
        if current_phase is not None and current_phase.phase_type == "swing":
            active_feet = set(current_phase.swing_feet)
            self._active_swing_start_positions = {
                foot: start
                for foot, start in self._active_swing_start_positions.items()
                if foot in active_feet
            }
            self._active_swing_end_positions = {
                foot: end
                for foot, end in self._active_swing_end_positions.items()
                if foot in active_feet
            }
            for foot in active_feet:
                if foot not in self._active_swing_start_positions:
                    self._active_swing_start_positions[foot] = np.asarray(
                        current_foot_positions[foot], dtype=float
                    ).copy()
        else:
            had_locked_swing = bool(self._active_swing_end_positions)
            self._active_swing_start_positions.clear()
            self._active_swing_end_positions.clear()
            if self.verbose and had_locked_swing:
                print(
                    f"[MPC Swing Lock] solve={self._solve_count + 1} released at support",
                    flush=True,
                )

        foothold_plans = self.foothold_planner.plan_footholds(
            com_trajectory=ocp_com_reference,
            contact_sequence=contact_sequence,
            current_foot_positions=current_foot_positions,
            dt=self.dt,
            step_height=step_height,
            active_swing_start_positions=self._active_swing_start_positions,
            active_swing_end_positions=self._active_swing_end_positions,
        )
        if current_phase is not None and current_phase.phase_type == "swing":
            newly_locked_targets = {}
            for foot in current_phase.swing_feet:
                plans = foothold_plans.get(foot, [])
                if foot not in self._active_swing_end_positions and plans:
                    self._active_swing_end_positions[foot] = np.asarray(
                        plans[0].end_pos, dtype=float
                    ).copy()
                    newly_locked_targets[foot] = self._active_swing_end_positions[foot]
            if self.verbose and newly_locked_targets:
                targets_text = ", ".join(
                    f"{foot}=[{target[0]:.3f},{target[1]:.3f},{target[2]:.3f}]"
                    for foot, target in newly_locked_targets.items()
                )
                print(
                    f"[MPC Swing Lock] solve={self._solve_count + 1} locked {targets_text}",
                    flush=True,
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
            if current_foot_contacts is not None:
                contact_text = "".join(
                    "1" if current_foot_contacts.get(name, False) else "0"
                    for name in ("LF", "RF", "LH", "RH")
                )
                force_z_text = "-"
                if current_foot_forces is not None:
                    force_z_text = ",".join(
                        f"{float(current_foot_forces[name][2]):.1f}"
                        for name in ("LF", "RF", "LH", "RH")
                    )
                print(
                    f"  Physical contacts LF/RF/LH/RH={contact_text} "
                    f"force_z=[{force_z_text}]",
                    flush=True,
                )
            print(
                "  Isolation: "
                f"fixed_contacts={self.force_standing_contacts}, "
                f"warm_start_enabled={self.enable_warm_start}, "
                f"root_reference={self.reference_is_root_position}, "
                f"demo_weights={self.use_demo_stabilization_weights}, "
                f"pseudo_impulse={self.use_pseudo_impulse}, "
                f"initial_support={self.initial_full_support_duration:.3f}s, "
                f"feasible_cold_start={self.use_feasible_cold_start_rollout}",
                flush=True,
            )
            print(
                "  OCP weights: "
                f"com={self.ocp_factory.weights['com_track']:.1e}, "
                f"foot={self.ocp_factory.weights['foot_track']:.1e}, "
                f"state={self.ocp_factory.weights['state_reg']:.1e}, "
                f"control={self.ocp_factory.weights['ctrl_reg']:.1e}, "
                f"friction={self.ocp_factory.weights['friction_cone']:.1e}",
                flush=True,
            )
            print(
                "  Effective gait: "
                f"swing={step_duration:.3f}s/{step_knots}ticks "
                f"(requested={requested_step_duration:.3f}s), "
                f"support={support_duration:.3f}s/{support_knots}ticks "
                f"(requested={requested_support_duration:.3f}s), "
                f"step_height={step_height:.3f}m, "
                f"ground_z={self.foothold_planner.default_ground_height:.4f}m, "
                f"phase_clock={self._gait_clock:.3f}s, "
                f"contact_gains={self.ocp_factory.CONTACT_GAINS.tolist()}",
                flush=True,
            )
            print(
                "  Contact phases: "
                + ", ".join(
                    f"{phase.phase_type}[support="
                    f"{'/'.join(phase.support_feet) or '-'},swing="
                    f"{'/'.join(phase.swing_feet) or '-'},"
                    f"elapsed={phase.elapsed:.3f}]"
                    for phase in contact_sequence.phases
                ),
                flush=True,
            )
            if current_foot_positions:
                for fname, fpos in current_foot_positions.items():
                    print(f"  Foot {fname}: [{fpos[0]:.3f}, {fpos[1]:.3f}, {fpos[2]:.3f}]", flush=True)
            if self._nominal_foot_offsets:
                print(
                    "  Nominal foot offsets: "
                    + ", ".join(
                        f"{foot}=[{offset[0]:.3f},{offset[1]:.3f}]"
                        for foot, offset in self._nominal_foot_offsets.items()
                    ),
                    flush=True,
                )
            first_landings = {
                foot: plans[0].end_pos
                for foot, plans in foothold_plans.items()
                if plans
            }
            if first_landings:
                print(
                    "  First landing targets: "
                    + ", ".join(
                        f"{foot}=[{target[0]:.3f},{target[1]:.3f},{target[2]:.3f}]"
                        for foot, target in first_landings.items()
                    ),
                    flush=True,
                )
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
            candidate_xs = self._shift_trajectory(self._prev_xs, current_state)
            candidate_us = self._shift_controls(self._prev_us)
            candidate_xs = self._adjust_length(
                candidate_xs, T + 1, current_state
            )
            candidate_us = self._adjust_length(
                candidate_us, T, np.zeros(self.actuation.nu)
            )
            control_dims_match = all(
                np.asarray(control).size == int(model.nu)
                for control, model in zip(candidate_us, problem.runningModels)
            )
            if len(candidate_us) == T and control_dims_match:
                xs_init = candidate_xs
                us_init = candidate_us
                use_warm_start = True
                if is_verbose_call:
                    print("  Warm-start: YES (shifted prev solution)", flush=True)
            elif self.verbose:
                print(
                    "[MPC Warm-start] topology changed; using cold-start for "
                    f"solve {self._solve_count}",
                    flush=True,
                )

        if not use_warm_start:
            # Compute contact-consistent equilibrium controls. Taking only the
            # actuated tail of Pinocchio's free-body gravity vector ignores the
            # contact force distribution and is not a standing equilibrium.
            x_static = np.asarray(current_state, dtype=float).copy()
            x_static[self.rmodel.nq :] = 0.0
            us_init = list(problem.quasiStatic([x_static.copy() for _ in range(T)]))
            rollout_xs = list(problem.rollout(us_init))
            rollout_array = np.asarray(rollout_xs)
            rollout_is_finite = bool(np.all(np.isfinite(rollout_array)))
            if self.use_feasible_cold_start_rollout and rollout_is_finite:
                xs_init = rollout_xs
                cold_start_is_feasible = True
                cold_start_mode = "feasible rollout"
            else:
                # Crocoddyl's locomotion examples initialize switching-contact
                # problems with repeated states and quasi-static controls, then
                # let FDDP close the dynamics gaps. A feasible rollout can fall
                # far away from the task before the first touchdown and create
                # a catastrophically poor initial candidate.
                xs_init = [
                    np.asarray(current_state, dtype=float).copy()
                    for _ in range(T + 1)
                ]
                cold_start_is_feasible = False
                cold_start_mode = "repeated state (FDDP infeasible guess)"
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
                print(f"  Cold-start state guess: {cold_start_mode}", flush=True)
                if not rollout_is_finite:
                    bad_index = np.argwhere(~np.isfinite(rollout_array))[0]
                    print(
                        "  First non-finite rollout entry: "
                        f"node={int(bad_index[0])}, state_index={int(bad_index[1])}",
                        flush=True,
                    )
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
        initial_guess_feasible = (
            False if use_warm_start else cold_start_is_feasible
        )
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

        if self.verbose and solver.cost >= 5e3:
            self._print_cost_breakdown(problem, solver)

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
        # position in the gait cycle (fixes the "Groundhog Day" bug). At the
        # last swing node, however, Crocoddyl must not activate a contact that
        # is still physically above its landing target in Isaac. Hold the clock
        # for a bounded number of ticks so the endpoint remains a swing-foot
        # tracking target instead of becoming a fictitious support contact.
        if not self.force_standing_contacts:
            hold_gait_clock = False
            at_swing_boundary = (
                self.touchdown_gate_height_tolerance > 0.0
                and self.touchdown_gate_max_steps > 0
                and current_phase is not None
                and current_phase.phase_type == "swing"
                and current_phase.duration <= self.dt + 1e-9
            )
            if at_swing_boundary:
                delayed_feet = []
                for foot in current_phase.swing_feet:
                    target = self._active_swing_end_positions.get(foot)
                    measured = current_foot_positions.get(foot)
                    if target is None or measured is None:
                        continue
                    height_error = float(measured[2] - target[2])
                    if height_error > self.touchdown_gate_height_tolerance:
                        delayed_feet.append((foot, height_error))

                if (
                    delayed_feet
                    and self._touchdown_gate_steps < self.touchdown_gate_max_steps
                ):
                    self._touchdown_gate_steps += 1
                    hold_gait_clock = True
                    if self.verbose:
                        delayed_text = ", ".join(
                            f"{foot}=+{error * 1000.0:.1f}mm"
                            for foot, error in delayed_feet
                        )
                        print(
                            f"[MPC Touchdown Gate] solve={self._solve_count} "
                            f"hold={self._touchdown_gate_steps}/"
                            f"{self.touchdown_gate_max_steps} {delayed_text}",
                            flush=True,
                        )
                else:
                    if delayed_feet and self.verbose:
                        print(
                            f"[MPC Touchdown Gate] solve={self._solve_count} "
                            "maximum hold reached; advancing contact schedule",
                            flush=True,
                        )
                    self._touchdown_gate_steps = 0
            else:
                self._touchdown_gate_steps = 0

            if not hold_gait_clock:
                self._gait_clock += self.dt

        # First control action
        if len(us) > 0 and np.asarray(us[0]).size == self.actuation.nu:
            control = us[0]
        else:
            # A prediction may contain nu=0 impulse controls, but the current
            # real-time command must always stay in the actuated 12-D space.
            control = np.zeros(self.actuation.nu)

        # Predicted trajectory
        predicted_states = np.array(xs)
        # Impulse nodes have nu=0.  IPC does not transmit the predicted control
        # trajectory, but keep MPCSolution's public shape homogeneous by
        # padding those zero-time controls to the actuated dimension.
        predicted_controls = np.zeros((len(us), self.actuation.nu))
        for index, predicted_control in enumerate(us):
            predicted_control = np.asarray(predicted_control, dtype=float)
            if predicted_control.size == self.actuation.nu:
                predicted_controls[index] = predicted_control

        dynamics_gap = solver_residual_norm(solver, "ffeas")
        constraint_terms = [
            value for value in (
                solver_residual_norm(solver, "gfeas"),
                solver_residual_norm(solver, "hfeas"),
            )
            if np.isfinite(value)
        ]
        constraint_violation = (
            max(constraint_terms) if constraint_terms else float("nan")
        )

        return MPCSolution(
            control=control,
            predicted_states=predicted_states,
            predicted_controls=predicted_controls,
            solve_time=solve_time,
            converged=bool(converged),
            cost=float(solver.cost),
            iterations=int(solver.iter),
            dynamics_gap=dynamics_gap,
            constraint_violation=constraint_violation,
        )

    def _print_cost_breakdown(self, problem: Any, solver: Any) -> None:
        """Print weighted residual contributions for an abnormally costly solve.

        This diagnostic is guarded against Crocoddyl API differences so a
        failed inspection can never interrupt MPC control.
        """
        try:
            problem.calc(solver.xs, solver.us)
            totals: Dict[str, float] = {}
            node_costs: List[float] = []

            def map_entries(mapping: Any) -> List[Any]:
                """Normalize Boost.Python std::map wrappers across versions."""
                if hasattr(mapping, "todict"):
                    return list(mapping.todict().items())
                if hasattr(mapping, "keys"):
                    return [(key, mapping[key]) for key in mapping.keys()]
                return [(key, mapping[key]) for key in mapping]

            def accumulate(model: Any, data: Any, prefix: str = "") -> None:
                if hasattr(model, "differential"):
                    model_costs = model.differential.costs.costs
                    data_costs = data.differential.costs.costs
                else:
                    model_costs = model.costs.costs
                    data_costs = data.costs.costs
                raw_contributions = []
                for name, item in map_entries(model_costs):
                    try:
                        cost_data = data_costs[name]
                    except (KeyError, TypeError):
                        continue
                    value = float(item.weight) * float(cost_data.cost)
                    raw_contributions.append((str(name), value))

                raw_total = sum(value for _, value in raw_contributions)
                integrated_total = float(data.cost)
                scale = integrated_total / raw_total if raw_total > 0.0 else 0.0
                for name, value in raw_contributions:
                    key = f"{prefix}{name}"
                    totals[key] = totals.get(key, 0.0) + value * scale

            for model, data in zip(problem.runningModels, problem.runningDatas):
                node_costs.append(float(data.cost))
                accumulate(model, data)

            terminal_cost = float(problem.terminalData.cost)
            accumulate(problem.terminalModel, problem.terminalData, "terminal/")
            ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
            max_node = int(np.argmax(node_costs)) if node_costs else -1
            max_node_cost = node_costs[max_node] if node_costs else 0.0

            print(
                f"[MPC Cost Breakdown] solve={self._solve_count} "
                f"total={float(solver.cost):.1f} max_running_node={max_node} "
                f"node_cost={max_node_cost:.1f} terminal={terminal_cost:.1f}",
                flush=True,
            )
            print(
                "  "
                + ", ".join(
                    f"{name}={value:.1f}" for name, value in ranked[:10]
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                f"[MPC Cost Breakdown] unavailable: {type(exc).__name__}: {exc}",
                flush=True,
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
        self._active_swing_start_positions.clear()
        self._active_swing_end_positions.clear()
        self._touchdown_gate_steps = 0
        self._nominal_foot_offsets = None

    def set_gait_type(self, gait_type: str):
        """Change the gait type.

        Args:
            gait_type: New gait type ("trot", "walk", "pace", "bound").
        """
        if gait_type not in GaitScheduler.GAIT_PATTERNS:
            raise ValueError(f"Unknown gait type: {gait_type}")
        self.gait_type = gait_type
        self._cached_contact_sequence = None
        self._active_swing_start_positions.clear()
        self._active_swing_end_positions.clear()
