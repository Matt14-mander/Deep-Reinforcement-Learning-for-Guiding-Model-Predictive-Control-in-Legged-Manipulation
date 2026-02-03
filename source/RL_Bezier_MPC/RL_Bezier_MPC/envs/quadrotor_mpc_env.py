# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadrotor MPC environment for RL+MPC Bezier trajectory control.

This module implements the main DirectRL environment that integrates:
1. RL policy outputting Bezier control points
2. Trajectory generator converting params to waypoints
3. MPC controller tracking the trajectory
4. IsaacLab physics simulation

Data Flow:
    RL Policy (Bezier params) → Trajectory Generator → MPC → Physics

The environment handles the frequency mismatch between RL (5-10 Hz),
MPC (50 Hz), and physics (200 Hz) through appropriate buffering and
trajectory management.
"""

from typing import Dict, Tuple

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene

from ..controllers import CrocoddylQuadrotorMPC
from ..controllers.base_mpc import MPCSolution
from ..trajectory import BezierTrajectoryGenerator
from ..utils.math_utils import blend_trajectories, quat_to_rotation_matrix
from .quadrotor_mpc_env_cfg import QuadrotorMPCEnvCfg


class QuadrotorMPCEnv(DirectRLEnv):
    """DirectRL environment for quadrotor trajectory tracking with MPC.

    This environment combines reinforcement learning with model predictive
    control for quadrotor trajectory tracking. The RL policy outputs Bezier
    curve parameters, which are converted to dense trajectories and tracked
    by an MPC controller.

    The environment supports parallel simulation but MPC runs on CPU,
    creating a CPU-GPU synchronization point at each control step.

    Attributes:
        cfg: Environment configuration.
        trajectory_generator: Bezier curve trajectory generator.
        mpc_controllers: List of MPC controllers (one per environment).
        current_trajectories: Current reference trajectories per environment.
        trajectory_phases: Current timestep within trajectory per environment.
        mpc_step_counter: Counter for RL policy update timing.
    """

    cfg: QuadrotorMPCEnvCfg

    def __init__(self, cfg: QuadrotorMPCEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the quadrotor MPC environment.

        Args:
            cfg: Environment configuration.
            render_mode: Rendering mode (e.g., "human", "rgb_array").
            **kwargs: Additional arguments passed to parent.
        """
        # Store configuration before parent init
        self._env_cfg = cfg

        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize trajectory generator
        self.trajectory_generator = BezierTrajectoryGenerator(
            degree=3,
            state_dim=3,
            max_displacement=cfg.max_bezier_displacement,
        )

        # Initialize MPC controllers (one per environment for parallel solving)
        # Note: MPC runs on CPU, so we create separate instances
        self.mpc_controllers = []
        for _ in range(self.num_envs):
            mpc = CrocoddylQuadrotorMPC(
                dt=cfg.mpc_dt,
                horizon_steps=cfg.mpc_horizon_steps,
                mass=cfg.quadrotor_mass,
                thrust_min=cfg.thrust_min,
                thrust_max=cfg.thrust_max,
                torque_max=cfg.torque_max,
                max_iterations=10,
                verbose=False,
            )
            self.mpc_controllers.append(mpc)

        # Initialize trajectory buffers
        # Shape: (num_envs, num_waypoints, 3)
        self.current_trajectories = np.zeros(
            (self.num_envs, cfg.num_bezier_waypoints, 3)
        )

        # Trajectory phase (current timestep within trajectory)
        self.trajectory_phases = np.zeros(self.num_envs, dtype=np.int32)

        # MPC step counter for RL policy update timing
        self.mpc_step_counter = np.zeros(self.num_envs, dtype=np.int32)

        # Target positions for each environment
        self.target_positions = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        # Previous Bezier parameters for trajectory blending
        self.prev_bezier_params = np.zeros(
            (self.num_envs, self.trajectory_generator.get_param_dim())
        )

        # Last MPC solution for debugging/visualization
        self.last_mpc_solutions: list[MPCSolution | None] = [None] * self.num_envs

        # Action bounds for normalization
        low, high = self.trajectory_generator.get_param_bounds()
        self.action_low = torch.tensor(low, device=self.device, dtype=torch.float32)
        self.action_high = torch.tensor(high, device=self.device, dtype=torch.float32)

        # Pending control buffers for batch application
        self._pending_forces = np.zeros((self.num_envs, 3))
        self._pending_torques = np.zeros((self.num_envs, 3))

    def _setup_scene(self):
        """Set up the simulation scene with quadrotor and ground plane."""
        # Create the quadrotor rigid object from config
        self.quadrotor = RigidObject(self.cfg.robot_cfg)

        # Spawn ground plane
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Add rigid object to scene
        self.scene.rigid_objects["quadrotor"] = self.quadrotor

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process RL actions before physics stepping.

        This method handles the RL → Trajectory → MPC → Physics pipeline.

        Args:
            actions: Normalized Bezier parameters from RL policy.
                Shape: (num_envs, num_actions)
        """
        # Denormalize actions to Bezier parameter space
        bezier_params = self._denormalize_actions(actions)

        # Convert to numpy for trajectory generation and MPC
        bezier_params_np = bezier_params.cpu().numpy()

        # Get current robot states
        robot_states = self._get_robot_states_numpy()

        # Process each environment
        for env_idx in range(self.num_envs):
            # Check if time to update trajectory
            if self.mpc_step_counter[env_idx] % self.cfg.rl_policy_period == 0:
                # Generate new trajectory from Bezier parameters
                current_pos = robot_states[env_idx, :3]
                new_trajectory = self.trajectory_generator.params_to_waypoints(
                    params=bezier_params_np[env_idx],
                    dt=self.cfg.mpc_dt,
                    horizon=self.cfg.bezier_horizon,
                    start_position=current_pos,
                )

                # Blend with previous trajectory for smoothness
                if self.trajectory_phases[env_idx] > 0:
                    old_traj = self.current_trajectories[env_idx]
                    old_phase = self.trajectory_phases[env_idx]
                    # Shift old trajectory to align with current phase
                    old_traj_shifted = np.roll(old_traj, -old_phase, axis=0)
                    self.current_trajectories[env_idx] = blend_trajectories(
                        old_trajectory=old_traj_shifted,
                        new_trajectory=new_trajectory,
                        blend_steps=self.cfg.trajectory_blend_steps,
                    )
                else:
                    self.current_trajectories[env_idx] = new_trajectory

                # Reset phase counter
                self.trajectory_phases[env_idx] = 0

                # Store for next blending
                self.prev_bezier_params[env_idx] = bezier_params_np[env_idx]

            # Get current reference slice for MPC
            phase = self.trajectory_phases[env_idx]
            end_phase = min(
                phase + self.cfg.mpc_horizon_steps,
                len(self.current_trajectories[env_idx])
            )
            reference = self.current_trajectories[env_idx, phase:end_phase]

            # Pad if necessary
            if len(reference) < self.cfg.mpc_horizon_steps:
                pad_length = self.cfg.mpc_horizon_steps - len(reference)
                last_point = reference[-1:] if len(reference) > 0 else np.zeros((1, 3))
                reference = np.concatenate([
                    reference,
                    np.repeat(last_point, pad_length, axis=0)
                ], axis=0)

            # Solve MPC
            solution = self.mpc_controllers[env_idx].solve(
                current_state=robot_states[env_idx],
                reference_trajectory=reference,
                warm_start=True,
            )

            self.last_mpc_solutions[env_idx] = solution

            # Store control for batch application
            control = solution.control
            self._apply_control_single(env_idx, control)

            # Increment counters
            self.trajectory_phases[env_idx] += 1
            self.mpc_step_counter[env_idx] += 1

        # Apply all controls in batch after loop
        self._apply_all_controls()

    def _apply_control_single(self, env_idx: int, control: np.ndarray):
        """Store control for a single environment (called during MPC loop).

        Args:
            env_idx: Environment index.
            control: Control vector [thrust, tau_x, tau_y, tau_z].
        """
        # Extract control components
        thrust = control[0]
        torque = control[1:4]

        # Get robot orientation to transform thrust to world frame
        robot_data = self.quadrotor.data
        quat = robot_data.root_quat_w[env_idx].cpu().numpy()  # (w, x, y, z)

        # Rotation matrix from body to world
        R = quat_to_rotation_matrix(quat)

        # Thrust in body frame is along z-axis
        thrust_body = np.array([0.0, 0.0, thrust])
        thrust_world = R @ thrust_body

        # Store in buffers (will be applied in batch later)
        self._pending_forces[env_idx] = thrust_world
        self._pending_torques[env_idx] = torque

    def _apply_all_controls(self):
        """Apply all pending controls to the simulation in batch."""
        # Convert to tensors with proper shape: (num_envs, 1, 3) for single body
        forces = torch.tensor(
            self._pending_forces, device=self.device, dtype=torch.float32
        ).unsqueeze(1)  # (num_envs, 1, 3)

        torques = torch.tensor(
            self._pending_torques, device=self.device, dtype=torch.float32
        ).unsqueeze(1)  # (num_envs, 1, 3)

        # Apply using permanent wrench composer (new API)
        self.quadrotor.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=slice(None),  # All bodies (just 1 for quadrotor)
            env_ids=slice(None),   # All environments
        )

    def _apply_action(self) -> None:
        """Apply the computed controls to the simulation.

        This method is called at each physics step. The controls were already
        computed and stored in _pre_physics_step, so we just need to ensure
        they are written to the simulation.
        """
        # The permanent wrench composer automatically applies forces each step
        # We just need to call write_data_to_sim on the asset
        pass  # Forces are applied via permanent_wrench_composer set in _pre_physics_step

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for RL policy.

        Returns:
            Dictionary with 'policy' key containing observation tensor.
            Shape: (num_envs, num_observations)

        Observation structure (17D):
            - Position (3D): x, y, z
            - Orientation quaternion (4D): w, x, y, z
            - Linear velocity (3D): vx, vy, vz
            - Angular velocity (3D): wx, wy, wz
            - Target position (3D): current waypoint from trajectory
            - Trajectory phase (1D): normalized progress [0, 1]
        """
        # Get robot data
        robot_data = self.quadrotor.data

        # Robot state components
        position = robot_data.root_pos_w  # (num_envs, 3)
        orientation = robot_data.root_quat_w  # (num_envs, 4)
        linear_vel = robot_data.root_lin_vel_w  # (num_envs, 3)
        angular_vel = robot_data.root_ang_vel_w  # (num_envs, 3)

        # Get current target positions from trajectories
        target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        phases = torch.zeros((self.num_envs, 1), device=self.device)

        for env_idx in range(self.num_envs):
            phase = self.trajectory_phases[env_idx]
            traj = self.current_trajectories[env_idx]

            # Get current target (clamp to trajectory length)
            target_idx = min(phase, len(traj) - 1)
            target_positions[env_idx] = torch.tensor(
                traj[target_idx], device=self.device
            )

            # Normalized phase [0, 1]
            phases[env_idx] = phase / max(len(traj) - 1, 1)

        # Concatenate observations
        obs = torch.cat([
            position,           # 3D
            orientation,        # 4D
            linear_vel,         # 3D
            angular_vel,        # 3D
            target_positions,   # 3D
            phases,             # 1D
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for trajectory tracking.

        Returns:
            Reward tensor. Shape: (num_envs,)
        """
        robot_data = self.quadrotor.data
        position = robot_data.root_pos_w
        linear_vel = robot_data.root_lin_vel_w
        angular_vel = robot_data.root_ang_vel_w

        # Initialize rewards
        rewards = torch.zeros(self.num_envs, device=self.device)

        for env_idx in range(self.num_envs):
            # Position tracking error
            phase = self.trajectory_phases[env_idx]
            traj = self.current_trajectories[env_idx]
            target_idx = min(phase, len(traj) - 1)
            target_pos = torch.tensor(
                traj[target_idx], device=self.device, dtype=torch.float32
            )

            pos_error = torch.norm(position[env_idx] - target_pos)
            pos_reward = self.cfg.reward_position_tracking * torch.exp(-pos_error)

            # Velocity penalty (prefer smooth motion)
            vel_magnitude = torch.norm(linear_vel[env_idx])
            vel_penalty = self.cfg.reward_velocity_penalty * vel_magnitude

            # Angular velocity penalty
            angvel_magnitude = torch.norm(angular_vel[env_idx])
            angvel_penalty = self.cfg.reward_velocity_penalty * 0.5 * angvel_magnitude

            # Control penalty (if MPC solution available)
            control_penalty = 0.0
            if self.last_mpc_solutions[env_idx] is not None:
                control = self.last_mpc_solutions[env_idx].control
                control_magnitude = np.linalg.norm(control)
                control_penalty = self.cfg.reward_control_penalty * control_magnitude

            # Alive bonus
            alive_bonus = self.cfg.reward_alive

            # Target reached bonus
            target_bonus = 0.0
            if pos_error < self.cfg.position_threshold:
                target_bonus = self.cfg.reward_reached_target

            # Total reward
            rewards[env_idx] = (
                pos_reward
                + vel_penalty
                + angvel_penalty
                + control_penalty
                + alive_bonus
                + target_bonus
            )

        return rewards

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination conditions.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
        """
        robot_data = self.quadrotor.data
        position = robot_data.root_pos_w
        orientation = robot_data.root_quat_w

        # Initialize tensors
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Check position bounds
        pos_deviation = torch.norm(position[:, :2], dim=1)  # XY distance from origin
        out_of_bounds = pos_deviation > self.cfg.max_position_deviation

        # Check height limits
        too_low = position[:, 2] < self.cfg.min_height
        too_high = position[:, 2] > self.cfg.max_height

        # Check orientation (upright constraint)
        # Compute dot product of robot's up vector with world z-axis
        # For quaternion (w, x, y, z), the body z-axis in world frame is:
        # R @ [0, 0, 1] where R is rotation matrix
        w = orientation[:, 0]
        x = orientation[:, 1]
        y = orientation[:, 2]
        z = orientation[:, 3]

        # Body z-axis in world frame (from rotation matrix)
        up_world_z = 1 - 2 * (x * x + y * y)
        flipped = up_world_z < self.cfg.min_up_dot_product

        # Termination conditions
        terminated = out_of_bounds | too_low | too_high | flipped

        # Truncation (timeout)
        time_out = self.episode_length_buf >= self.max_episode_length
        truncated = time_out & ~terminated

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments.

        Args:
            env_ids: Indices of environments to reset.
        """
        num_resets = len(env_ids)
        if num_resets == 0:
            return

        env_ids_np = env_ids.cpu().numpy()

        # Reset robot poses with randomization
        pos_range = self.cfg.initial_pos_range
        random_pos = torch.zeros((num_resets, 3), device=self.device)
        random_pos[:, 0] = torch.rand(num_resets, device=self.device) * (
            pos_range[1] - pos_range[0]
        ) + pos_range[0]
        random_pos[:, 1] = torch.rand(num_resets, device=self.device) * (
            pos_range[3] - pos_range[2]
        ) + pos_range[2]
        random_pos[:, 2] = torch.rand(num_resets, device=self.device) * (
            pos_range[5] - pos_range[4]
        ) + pos_range[4]

        # Identity quaternion
        identity_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * num_resets, device=self.device
        )

        # Zero velocities
        zero_vel = torch.zeros((num_resets, 3), device=self.device)

        # Reset robot state
        self.quadrotor.write_root_pose_to_sim(
            torch.cat([random_pos, identity_quat], dim=-1), env_ids
        )
        self.quadrotor.write_root_velocity_to_sim(
            torch.cat([zero_vel, zero_vel], dim=-1), env_ids
        )

        # Generate new target positions
        target_range = self.cfg.target_pos_range
        for idx, env_idx in enumerate(env_ids_np):
            target = np.array([
                np.random.uniform(target_range[0], target_range[1]),
                np.random.uniform(target_range[2], target_range[3]),
                np.random.uniform(target_range[4], target_range[5]),
            ])
            self.target_positions[env_idx] = torch.tensor(
                target, device=self.device
            )

            # Initialize trajectory towards target
            start_pos = random_pos[idx].cpu().numpy()
            initial_params = self._generate_initial_bezier_params(
                start_pos, target
            )

            trajectory = self.trajectory_generator.params_to_waypoints(
                params=initial_params,
                dt=self.cfg.mpc_dt,
                horizon=self.cfg.bezier_horizon,
                start_position=start_pos,
            )
            self.current_trajectories[env_idx] = trajectory

            # Reset counters
            self.trajectory_phases[env_idx] = 0
            self.mpc_step_counter[env_idx] = 0

            # Reset MPC controller
            self.mpc_controllers[env_idx].reset()

            # Clear stored params
            self.prev_bezier_params[env_idx] = initial_params

        # Reset episode buffers (handled by parent)
        super()._reset_idx(env_ids)

    def _generate_initial_bezier_params(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        """Generate initial Bezier parameters for straight-line trajectory.

        Args:
            start_pos: Starting position.
            target_pos: Target position.

        Returns:
            Bezier parameter vector (12D).
        """
        # For a straight line, control points should be along the line
        direction = target_pos - start_pos
        distance = np.linalg.norm(direction)

        # Clamp to max displacement
        max_disp = self.cfg.max_bezier_displacement
        if distance > max_disp:
            direction = direction / distance * max_disp

        # Control points as fractions along the line
        # P0 = start (offset = 0)
        # P1 = 1/3 along
        # P2 = 2/3 along
        # P3 = end
        params = np.zeros(12)
        params[0:3] = 0.0  # P0 offset (must be zero)
        params[3:6] = direction / 3  # P1 offset
        params[6:9] = 2 * direction / 3  # P2 offset
        params[9:12] = direction  # P3 offset (target)

        return params

    def _get_robot_states_numpy(self) -> np.ndarray:
        """Get robot states as numpy array for MPC.

        Returns:
            Array of shape (num_envs, 13) with state vectors.
        """
        robot_data = self.quadrotor.data

        states = np.zeros((self.num_envs, 13))
        states[:, 0:3] = robot_data.root_pos_w.cpu().numpy()
        states[:, 3:7] = robot_data.root_quat_w.cpu().numpy()
        states[:, 7:10] = robot_data.root_lin_vel_w.cpu().numpy()
        states[:, 10:13] = robot_data.root_ang_vel_w.cpu().numpy()

        return states

    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions [-1, 1] to Bezier parameter space.

        Args:
            actions: Normalized actions from RL policy.

        Returns:
            Denormalized Bezier parameters.
        """
        # Scale from [-1, 1] to [low, high]
        return 0.5 * (actions + 1.0) * (
            self.action_high - self.action_low
        ) + self.action_low

    def close(self):
        """Clean up resources."""
        # Clear MPC controllers
        self.mpc_controllers.clear()
        super().close()
