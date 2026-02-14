# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for building Crocoddyl optimal control problems for quadruped locomotion.

This module builds Crocoddyl OCP nodes (action models) for the quadruped.
It replaces `createSwingFootModel`, `createFootSwitchModel`, and `createImpulseModel`
from Crocoddyl's demo while using only Crocoddyl's public API.

Key difference from demo:
The demo hardcodes foot trajectories inside OCP construction. This factory
receives pre-computed trajectories from the trajectory pipeline and only
handles OCP assembly.

Reference Crocoddyl API classes used:
    - crocoddyl.StateMultibody(pinocchio_model)
    - crocoddyl.ActuationModelFloatingBase(state)
    - crocoddyl.ContactModelMultiple / ContactModel3D
    - crocoddyl.CostModelSum / CostModelResidual
    - crocoddyl.ResidualModelCoMPosition      → CoM tracking
    - crocoddyl.ResidualModelFrameTranslation  → foot tracking
    - crocoddyl.ResidualModelState             → state regularization
    - crocoddyl.ResidualModelControl           → control regularization
    - crocoddyl.ResidualModelContactFrictionCone → friction cone
    - crocoddyl.ResidualModelFrameVelocity     → impact velocity
    - crocoddyl.ActivationModelWeightedQuad    → weighted state cost
    - crocoddyl.ActivationModelQuadraticBarrier → state bounds, friction cone
    - crocoddyl.DifferentialActionModelContactFwdDynamics
    - crocoddyl.IntegratedActionModelEuler
    - crocoddyl.ImpulseModelMultiple / ImpulseModel3D
    - crocoddyl.ActionModelImpulseFwdDynamics
    - crocoddyl.ShootingProblem
    - crocoddyl.SolverFDDP
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contact_sequence import ContactSequence
from .foothold_planner import FootholdPlan

# Try to import Crocoddyl
try:
    import crocoddyl
    import pinocchio

    CROCODDYL_AVAILABLE = True
except ImportError:
    CROCODDYL_AVAILABLE = False
    crocoddyl = None
    pinocchio = None


class OCPFactory:
    """Factory for building Crocoddyl optimal control problems for quadruped locomotion.

    Replaces the functionality of Crocoddyl's SimpleQuadrupedalGaitProblem example
    but uses clean interfaces and Crocoddyl's public API only.

    Attributes:
        rmodel: Pinocchio robot model.
        state: Crocoddyl StateMultibody.
        actuation: Crocoddyl ActuationModelFloatingBase.
        foot_frame_ids: Dict mapping foot name to Pinocchio frame ID.
    """

    # Default cost weights (from Crocoddyl demo)
    DEFAULT_WEIGHTS = {
        "com_track": 1e6,
        "foot_track": 1e6,
        "state_reg": 1e1,
        "ctrl_reg": 1e0,           # Increased: reduce aggressive torques in turns
        "friction_cone": 1e3,      # Increased from 1e1: must dominate to prevent
                                    # downward GRF, especially on hind feet during curves
        "state_bounds": 1e3,
        "orientation_track": 1e3,  # Reduced from 1e4: avoid conflict with friction cone
                                    # during curve walking (yaw tracking less critical
                                    # than physical constraint satisfaction)
    }

    def __init__(
        self,
        rmodel: "pinocchio.Model",
        foot_frame_ids: Dict[str, int],
        mu: float = 0.7,
        integrator: str = "euler",
        fwddyn: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize factory with robot model and cost weights.

        Args:
            rmodel: Pinocchio robot model.
            foot_frame_ids: Dict mapping foot name to frame ID.
                Example: {"LF": 12, "RF": 18, "LH": 24, "RH": 30}
            mu: Friction coefficient for friction cone constraints.
            integrator: Integration scheme, "euler" or "rk4".
            fwddyn: If True, use forward dynamics. Otherwise inverse dynamics.
            weights: Dict of cost weights. Uses DEFAULT_WEIGHTS for missing keys.
        """
        if not CROCODDYL_AVAILABLE:
            raise ImportError(
                "Crocoddyl is not available. Please install crocoddyl to use OCPFactory."
            )

        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.foot_frame_ids = foot_frame_ids
        self.mu = mu
        self.integrator = integrator
        self.fwddyn = fwddyn

        # Merge weights with defaults
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if weights is not None:
            self.weights.update(weights)

        # Create Crocoddyl state and actuation models
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        # Dimensions
        self.nq = self.rmodel.nq  # configuration dimension
        self.nv = self.rmodel.nv  # velocity dimension
        self.nx = self.state.nx  # state dimension (nq + nv)
        self.nu = self.actuation.nu  # control dimension (actuated joints only)

        # Default state for regularization (standing pose)
        self.x0 = self.rmodel.defaultState if hasattr(self.rmodel, 'defaultState') else self._compute_default_state()

        # Precompute state weights (from Crocoddyl demo structure)
        self._compute_state_weights()

    def _compute_default_state(self) -> np.ndarray:
        """Compute a default standing state for regularization.

        Returns:
            Default state vector (nq + nv).
        """
        x0 = np.zeros(self.nx)
        # Default configuration: all joints at zero
        x0[:self.nq] = pinocchio.neutral(self.rmodel)
        return x0

    def _compute_state_weights(self):
        """Compute state regularization weights.

        From Crocoddyl demo (lines 642-648):
        - Base position (x,y,z): 0.0 (free to track CoM)
        - Base orientation (roll,pitch,yaw): 500.0 (keep upright)
        - Joint positions: 0.01 (light regularization)
        - Base velocity: 10.0 (moderate damping)
        - Joint velocities: 1.0 (light damping)
        """
        # Configuration part (nq dimensions)
        # Structure: [base_pos(3), base_quat(4), joints(nq-7)]
        state_weights_q = np.array(
            [0.0] * 3  # base position - free
            + [500.0] * 3  # base orientation (we use 3D since Crocoddyl uses tangent)
            + [0.01] * (self.nv - 6)  # joints
        )

        # Velocity part (nv dimensions)
        state_weights_v = np.array(
            [10.0] * 6  # base velocity
            + [1.0] * (self.nv - 6)  # joint velocities
        )

        self.state_weights = np.concatenate([state_weights_q, state_weights_v])

    def build_swing_node(
        self,
        dt: float,
        support_foot_ids: List[int],
        com_target: Optional[np.ndarray] = None,
        swing_foot_targets: Optional[List[Tuple[int, np.ndarray]]] = None,
        body_yaw_target: Optional[float] = None,
    ) -> Any:
        """Build a single OCP node for a swing phase timestep.

        Equivalent to demo's createSwingFootModel() but with:
        - External foot targets (from FootholdPlanner) instead of hardcoded
        - Optional body orientation cost for curve walking
        - com_target from CoM Bezier instead of straight-line assumption

        Costs included:
        - comTrack: CoM position tracking (if com_target provided)
        - footTrack: swing foot position tracking (if swing_foot_targets provided)
        - frictionCone: for each support foot
        - stateReg: weighted state regularization
        - ctrlReg: control regularization
        - stateBounds: joint limit enforcement
        - bodyOrientation: yaw tracking (if body_yaw_target provided)

        Args:
            dt: Timestep in seconds.
            support_foot_ids: List of Pinocchio frame IDs for feet in contact.
            com_target: Target CoM position, shape (3,). Optional.
            swing_foot_targets: List of (frame_id, target_pos) for swing feet.
            body_yaw_target: Target body yaw angle in radians. Optional.

        Returns:
            Crocoddyl IntegratedActionModel for this node.
        """
        # Create contact model
        contact_model = crocoddyl.ContactModelMultiple(self.state, self.nu)

        for foot_id in support_foot_ids:
            contact = crocoddyl.ContactModel3D(
                self.state,
                foot_id,
                np.array([0.0, 0.0, 0.0]),  # contact point offset
                pinocchio.LOCAL_WORLD_ALIGNED,
                self.nu,
                np.array([0.0, 0.0]),  # gains
            )
            contact_model.addContact(f"contact_{foot_id}", contact)

        # Create cost model
        cost_model = crocoddyl.CostModelSum(self.state, self.nu)

        # CoM tracking cost
        if com_target is not None:
            com_residual = crocoddyl.ResidualModelCoMPosition(
                self.state, com_target, self.nu
            )
            com_cost = crocoddyl.CostModelResidual(self.state, com_residual)
            cost_model.addCost("comTrack", com_cost, self.weights["com_track"])

        # Swing foot tracking costs
        if swing_foot_targets is not None:
            for frame_id, target_pos in swing_foot_targets:
                foot_residual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, frame_id, target_pos, self.nu
                )
                foot_cost = crocoddyl.CostModelResidual(self.state, foot_residual)
                cost_model.addCost(
                    f"footTrack_{frame_id}", foot_cost, self.weights["foot_track"]
                )

        # State regularization cost (weighted)
        state_activation = crocoddyl.ActivationModelWeightedQuad(self.state_weights)
        state_residual = crocoddyl.ResidualModelState(self.state, self.x0, self.nu)
        state_cost = crocoddyl.CostModelResidual(
            self.state, state_activation, state_residual
        )
        cost_model.addCost("stateReg", state_cost, self.weights["state_reg"])

        # Control regularization cost
        ctrl_residual = crocoddyl.ResidualModelControl(self.state, self.nu)
        ctrl_cost = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        cost_model.addCost("ctrlReg", ctrl_cost, self.weights["ctrl_reg"])

        # Friction cone constraints for each support foot
        for foot_id in support_foot_ids:
            # Create friction cone
            # FrictionCone expects a 3x3 rotation matrix (not a normal vector).
            # np.eye(3) means the cone normal is aligned with world z-axis (upward).
            cone = crocoddyl.FrictionCone(
                np.eye(3),     # rotation matrix (identity = z-up normal)
                self.mu,       # friction coefficient
                4,             # number of cone faces
                False,         # inner approximation
            )

            friction_residual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, foot_id, cone, self.nu
            )
            friction_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            friction_cost = crocoddyl.CostModelResidual(
                self.state, friction_activation, friction_residual
            )
            cost_model.addCost(
                f"frictionCone_{foot_id}", friction_cost, self.weights["friction_cone"]
            )

        # State bounds (joint limits)
        # state.lb/ub have dimension nx = nq + nv, but ResidualModelState
        # outputs in tangent space with dimension ndx = 2*nv.
        # For floating base: skip the first element (quaternion normalization)
        # and take nv elements for position part, then nv for velocity part.
        # Reference: Crocoddyl whole_body_manipulation example
        state_lb = np.concatenate([
            self.state.lb[1:self.nv + 1],   # position bounds in tangent space
            self.state.lb[-self.nv:]         # velocity bounds
        ])
        state_ub = np.concatenate([
            self.state.ub[1:self.nv + 1],   # position bounds in tangent space
            self.state.ub[-self.nv:]         # velocity bounds
        ])

        bounds_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(state_lb, state_ub)
        )
        bounds_residual = crocoddyl.ResidualModelState(self.state, self.x0, self.nu)
        bounds_cost = crocoddyl.CostModelResidual(
            self.state, bounds_activation, bounds_residual
        )
        cost_model.addCost("stateBounds", bounds_cost, self.weights["state_bounds"])

        # Body orientation cost for curve walking (if yaw target provided)
        if body_yaw_target is not None:
            # Create target rotation (only yaw matters for curve walking)
            target_rotation = pinocchio.utils.rpyToMatrix(0.0, 0.0, body_yaw_target)

            # Find base body frame - try common names, fall back to first BODY frame
            base_frame_id = None
            for name in ["base_link", "base", "trunk", "body"]:
                if self.rmodel.existFrame(name):
                    base_frame_id = self.rmodel.getFrameId(name)
                    break
            if base_frame_id is None:
                # Fall back: find first BODY-type frame (index 0 is universe)
                for i, frame in enumerate(self.rmodel.frames):
                    if frame.type == pinocchio.FrameType.BODY and i > 0:
                        base_frame_id = i
                        break

            if base_frame_id is not None:
                orientation_residual = crocoddyl.ResidualModelFrameRotation(
                    self.state,
                    base_frame_id,
                    target_rotation,
                    self.nu,
                )
                orientation_cost = crocoddyl.CostModelResidual(
                    self.state, orientation_residual
                )
                cost_model.addCost(
                    "bodyOrientation", orientation_cost, self.weights["orientation_track"]
                )

        # Create differential action model
        if self.fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contact_model, cost_model
            )
        else:
            raise NotImplementedError("Inverse dynamics not implemented yet")

        # Integrate to get discrete-time action model
        if self.integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)
        elif self.integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK4(dmodel, dt)
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

        return model

    def build_impulse_node(
        self,
        new_support_foot_ids: List[int],
        swing_foot_targets: Optional[List[Tuple[int, np.ndarray]]] = None,
    ) -> Any:
        """Build an impulse node for foot touchdown.

        Equivalent to demo's createImpulseModel(). Handles instantaneous
        contact changes and impact dynamics.

        Args:
            new_support_foot_ids: Feet that are making contact at this instant.
            swing_foot_targets: Optional targets for feet that just landed.

        Returns:
            Crocoddyl ActionModelImpulseFwdDynamics.
        """
        # Create impulse model
        impulse_model = crocoddyl.ImpulseModelMultiple(self.state)

        for foot_id in new_support_foot_ids:
            impulse = crocoddyl.ImpulseModel3D(
                self.state,
                foot_id,
                pinocchio.LOCAL_WORLD_ALIGNED,
            )
            impulse_model.addImpulse(f"impulse_{foot_id}", impulse)

        # Create cost model for impulse
        cost_model = crocoddyl.CostModelSum(self.state, 0)  # nu=0 for impulse

        # Foot velocity at impact (want zero velocity at touchdown)
        for foot_id in new_support_foot_ids:
            velocity_residual = crocoddyl.ResidualModelFrameVelocity(
                self.state,
                foot_id,
                pinocchio.Motion.Zero(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                0,  # nu=0 for impulse
            )
            velocity_cost = crocoddyl.CostModelResidual(self.state, velocity_residual)
            cost_model.addCost(f"impactVel_{foot_id}", velocity_cost, 1e4)

        # State regularization
        state_residual = crocoddyl.ResidualModelState(self.state, self.x0, 0)
        state_cost = crocoddyl.CostModelResidual(self.state, state_residual)
        cost_model.addCost("stateReg", state_cost, self.weights["state_reg"])

        # Create impulse action model
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulse_model, cost_model
        )

        return model

    def build_terminal_node(
        self,
        com_target: Optional[np.ndarray] = None,
        body_yaw_target: Optional[float] = None,
    ) -> Any:
        """Build terminal cost node for the OCP.

        Args:
            com_target: Target CoM position at end of horizon.
            body_yaw_target: Target body yaw at end of horizon.

        Returns:
            Crocoddyl IntegratedActionModel with zero time step.
        """
        # All feet in contact for terminal state
        all_foot_ids = list(self.foot_frame_ids.values())

        contact_model = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for foot_id in all_foot_ids:
            contact = crocoddyl.ContactModel3D(
                self.state,
                foot_id,
                np.array([0.0, 0.0, 0.0]),
                pinocchio.LOCAL_WORLD_ALIGNED,
                self.nu,
                np.array([0.0, 0.0]),
            )
            contact_model.addContact(f"contact_{foot_id}", contact)

        # Terminal costs
        cost_model = crocoddyl.CostModelSum(self.state, self.nu)

        # CoM tracking
        if com_target is not None:
            com_residual = crocoddyl.ResidualModelCoMPosition(
                self.state, com_target, self.nu
            )
            com_cost = crocoddyl.CostModelResidual(self.state, com_residual)
            cost_model.addCost("comTrack", com_cost, self.weights["com_track"] * 10)

        # State regularization (heavier for terminal)
        state_activation = crocoddyl.ActivationModelWeightedQuad(self.state_weights)
        state_residual = crocoddyl.ResidualModelState(self.state, self.x0, self.nu)
        state_cost = crocoddyl.CostModelResidual(
            self.state, state_activation, state_residual
        )
        cost_model.addCost("stateReg", state_cost, self.weights["state_reg"] * 10)

        # Velocity damping (want zero velocity at terminal)
        # ResidualModelState outputs in tangent space (ndx = 2*nv)
        ndx = 2 * self.nv
        velocity_weights = np.zeros(ndx)
        velocity_weights[self.nv:] = 100.0  # High weight on velocity part
        velocity_activation = crocoddyl.ActivationModelWeightedQuad(velocity_weights)
        velocity_residual = crocoddyl.ResidualModelState(self.state, self.x0, self.nu)
        velocity_cost = crocoddyl.CostModelResidual(
            self.state, velocity_activation, velocity_residual
        )
        cost_model.addCost("velocityDamping", velocity_cost, 1e2)

        # Body orientation
        if body_yaw_target is not None:
            target_rotation = pinocchio.utils.rpyToMatrix(0.0, 0.0, body_yaw_target)

            # Find base body frame - try common names, fall back to first BODY frame
            base_frame_id = None
            for name in ["base_link", "base", "trunk", "body"]:
                if self.rmodel.existFrame(name):
                    base_frame_id = self.rmodel.getFrameId(name)
                    break
            if base_frame_id is None:
                for i, frame in enumerate(self.rmodel.frames):
                    if frame.type == pinocchio.FrameType.BODY and i > 0:
                        base_frame_id = i
                        break

            if base_frame_id is not None:
                orientation_residual = crocoddyl.ResidualModelFrameRotation(
                    self.state, base_frame_id, target_rotation, self.nu
                )
                orientation_cost = crocoddyl.CostModelResidual(
                    self.state, orientation_residual
                )
                cost_model.addCost(
                    "bodyOrientation", orientation_cost, self.weights["orientation_track"]
                )

        # Differential model
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contact_model, cost_model
        )

        # Zero timestep for terminal
        model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)

        return model

    def build_problem(
        self,
        x0: np.ndarray,
        contact_sequence: ContactSequence,
        com_trajectory: np.ndarray,
        foot_trajectories: Dict[str, List[FootholdPlan]],
        dt: float,
        heading_trajectory: Optional[np.ndarray] = None,
    ) -> "crocoddyl.ShootingProblem":
        """Assemble a complete ShootingProblem from all components.

        Algorithm:
        1. Iterate through ContactSequence phases
        2. For each phase, discretize into knots
        3. For each knot:
            a. Get CoM target from com_trajectory
            b. Get foot targets from foot_trajectories
            c. Get heading from heading_trajectory (if curve walking)
            d. Call build_swing_node or build_impulse_node
        4. Wrap all nodes into ShootingProblem

        Args:
            x0: Initial state, shape (nx,).
            contact_sequence: ContactSequence defining the gait.
            com_trajectory: Dense CoM waypoints, shape (T, 3).
            foot_trajectories: Dict from FootholdPlanner.
            dt: OCP timestep in seconds.
            heading_trajectory: Target yaw angles, shape (T,). Optional.

        Returns:
            Crocoddyl ShootingProblem ready for solving.
        """
        running_models = []
        knot_index = 0

        # Track which swing is current for each foot
        foot_swing_indices = {foot: 0 for foot in self.foot_frame_ids.keys()}

        # Iterate through contact sequence phases
        for phase_idx, phase in enumerate(contact_sequence.phases):
            # Discretize phase into knots
            num_knots = max(1, round(phase.duration / dt))

            # Get support foot frame IDs for this phase
            support_foot_ids = [
                self.foot_frame_ids[foot]
                for foot in phase.support_feet
                if foot in self.foot_frame_ids
            ]

            for knot in range(num_knots):
                # Get CoM target at this timestep
                traj_idx = min(knot_index, len(com_trajectory) - 1)
                com_target = com_trajectory[traj_idx]

                # Get heading target if provided
                body_yaw_target = None
                if heading_trajectory is not None:
                    heading_idx = min(knot_index, len(heading_trajectory) - 1)
                    body_yaw_target = heading_trajectory[heading_idx]

                # Get swing foot targets
                swing_foot_targets = []
                for foot_name in phase.swing_feet:
                    if foot_name not in self.foot_frame_ids:
                        continue

                    frame_id = self.foot_frame_ids[foot_name]
                    swing_idx = foot_swing_indices[foot_name]

                    if swing_idx < len(foot_trajectories.get(foot_name, [])):
                        plan = foot_trajectories[foot_name][swing_idx]

                        # Interpolate within swing trajectory
                        if plan.trajectory is not None and len(plan.trajectory) > 0:
                            traj_knot = min(knot, len(plan.trajectory) - 1)
                            target_pos = plan.trajectory[traj_knot]
                        else:
                            # Linear interpolation fallback
                            alpha = knot / max(1, num_knots - 1)
                            target_pos = (1 - alpha) * plan.start_pos + alpha * plan.end_pos

                        swing_foot_targets.append((frame_id, target_pos))

                # Build the node
                model = self.build_swing_node(
                    dt=dt,
                    support_foot_ids=support_foot_ids,
                    com_target=com_target,
                    swing_foot_targets=swing_foot_targets if swing_foot_targets else None,
                    body_yaw_target=body_yaw_target,
                )
                running_models.append(model)
                knot_index += 1

            # After each swing phase, increment the swing index for those feet
            if phase.phase_type == "swing":
                for foot_name in phase.swing_feet:
                    if foot_name in foot_swing_indices:
                        foot_swing_indices[foot_name] += 1

        # Build terminal model
        terminal_com = com_trajectory[-1] if len(com_trajectory) > 0 else None
        terminal_yaw = heading_trajectory[-1] if heading_trajectory is not None else None

        terminal_model = self.build_terminal_node(
            com_target=terminal_com,
            body_yaw_target=terminal_yaw,
        )

        # Create shooting problem
        problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

        return problem

    def create_solver(
        self,
        problem: "crocoddyl.ShootingProblem",
        max_iterations: int = 100,
        th_stop: float = 1e-4,
    ) -> "crocoddyl.SolverFDDP":
        """Create a FDDP solver for the problem.

        Args:
            problem: ShootingProblem to solve.
            max_iterations: Maximum solver iterations.
            th_stop: Convergence threshold.

        Returns:
            Configured FDDP solver.
        """
        solver = crocoddyl.SolverFDDP(problem)
        solver.th_stop = th_stop
        return solver
