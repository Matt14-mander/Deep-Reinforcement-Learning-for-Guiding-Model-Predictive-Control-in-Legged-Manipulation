# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadruped MPC environment for RL+MPC Bezier trajectory control.

This module implements the main DirectRL environment that integrates:
1. RL policy outputting CoM Bezier control points and gait modulation
2. Gait pipeline: ContactSequence → FootholdPlanner → swing trajectories
3. Crocoddyl MPC computing joint torques
4. IsaacLab physics simulation

Data Flow:
    RL Policy → CoM Bezier + Gait Params
        → BezierTrajectory → CoM waypoints
        → GaitScheduler → ContactSequence
        → FootholdPlanner → foot landings
        → BezierFootTrajectory → swing arcs
        → CrocoddylQuadrupedMPC → joint torques
        → IsaacLab Physics → next state

The environment handles the frequency mismatch between RL (5 Hz),
MPC (50 Hz), and physics (200 Hz) through appropriate buffering.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene

from ..controllers.crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC, CROCODDYL_AVAILABLE
from ..controllers.base_mpc import MPCSolution
from ..robots.quadruped_cfg import (
    DEFAULT_QUADRUPED_CFG,
    QuadrupedCfg,
    load_pinocchio_model,
    get_foot_frame_ids,
)
from ..trajectory import BezierTrajectoryGenerator
from ..utils.math_utils import blend_trajectories, quat_to_euler, euler_to_quat
from .quadruped_mpc_env_cfg import QuadrupedMPCEnvCfg


class QuadrupedMPCEnv(DirectRLEnv):
    """DirectRL environment for quadruped trajectory tracking with MPC.

    This environment combines reinforcement learning with model predictive
    control for quadruped locomotion. The RL policy outputs CoM Bezier
    curve parameters and gait modulation, which are processed through the
    full gait pipeline and tracked by a Crocoddyl MPC controller.

    The environment supports parallel simulation but MPC runs on CPU,
    creating a CPU-GPU synchronization point at each control step.

    Attributes:
        cfg: Environment configuration.
        trajectory_generator: CoM Bezier curve trajectory generator.
        mpc_controllers: List of MPC controllers (one per environment).
        current_com_trajectories: Current CoM reference trajectories.
        trajectory_phases: Current timestep within trajectory per environment.
        mpc_step_counter: Counter for RL policy update timing.
    """

    cfg: QuadrupedMPCEnvCfg

    def __init__(self, cfg: QuadrupedMPCEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the quadruped MPC environment.

        Args:
            cfg: Environment configuration.
            render_mode: Rendering mode (e.g., "human", "rgb_array").
            **kwargs: Additional arguments passed to parent.
        """
        # Store configuration before parent init
        self._env_cfg = cfg

        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize CoM trajectory generator
        self.trajectory_generator = BezierTrajectoryGenerator(
            degree=3,
            state_dim=3,
            max_displacement=cfg.max_bezier_displacement,
        )

        # Load Pinocchio model for MPC (if Crocoddyl available)
        self._pinocchio_model = None
        self._foot_frame_ids = None

        if CROCODDYL_AVAILABLE:
            try:
                self._pinocchio_model, _ = load_pinocchio_model(
                    robot_name=cfg.robot_name
                )
                self._foot_frame_ids = get_foot_frame_ids(
                    self._pinocchio_model,
                    cfg.foot_frame_names,
                )
            except Exception as e:
                print(f"Warning: Could not load Pinocchio model: {e}")
                print("MPC will run in dummy mode.")

        # Initialize MPC controllers (one per environment for parallel solving)
        self.mpc_controllers = []
        for _ in range(self.num_envs):
            if self._pinocchio_model is not None:
                mpc = CrocoddylQuadrupedMPC(
                    rmodel=self._pinocchio_model,
                    foot_frame_names=cfg.foot_frame_names,
                    hip_offsets=cfg.hip_offsets,
                    gait_type=cfg.gait_type,
                    dt=cfg.mpc_dt,
                    horizon_steps=cfg.mpc_horizon_steps,
                    step_duration=cfg.default_step_duration,
                    support_duration=cfg.default_support_duration,
                    step_height=cfg.default_step_height,
                    mu=cfg.friction_coefficient,
                    max_iterations=10,
                )
            else:
                mpc = None  # Dummy mode
            self.mpc_controllers.append(mpc)

        # Initialize CoM trajectory buffers
        # Shape: (num_envs, num_waypoints, 3)
        self.current_com_trajectories = np.zeros(
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
            (self.num_envs, cfg.num_bezier_actions)
        )

        # Previous gait modulation parameters
        self.prev_gait_mods = np.ones((self.num_envs, cfg.num_gait_mod_actions))

        # Last MPC solution for debugging/visualization
        self.last_mpc_solutions: list[MPCSolution | None] = [None] * self.num_envs

        # Action bounds for normalization
        bezier_low, bezier_high = self.trajectory_generator.get_param_bounds()
        gait_mod_low = np.array([
            cfg.step_length_mod_range[0],
            cfg.step_height_mod_range[0],
            cfg.step_frequency_mod_range[0],
        ])
        gait_mod_high = np.array([
            cfg.step_length_mod_range[1],
            cfg.step_height_mod_range[1],
            cfg.step_frequency_mod_range[1],
        ])

        self.action_low = torch.tensor(
            np.concatenate([bezier_low, gait_mod_low]),
            device=self.device,
            dtype=torch.float32
        )
        self.action_high = torch.tensor(
            np.concatenate([bezier_high, gait_mod_high]),
            device=self.device,
            dtype=torch.float32
        )

        # Foot contact tracking
        self.foot_contacts = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )

        # Pending force/torque buffers for batch application
        # Shape: (num_envs, 1, 3) - 1 body per env
        self._pending_forces = torch.zeros(
            (self.num_envs, 1, 3), device=self.device, dtype=torch.float32
        )
        self._pending_torques = torch.zeros(
            (self.num_envs, 1, 3), device=self.device, dtype=torch.float32
        )

    def _setup_scene(self):
        """Set up the simulation scene with quadruped and ground plane."""
        # Spawn ground plane
        ground_cfg = sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
        )
        ground_cfg.func("/World/ground", ground_cfg)

        # Spawn quadrupeds as articulated robots
        # Using a placeholder box until proper URDF is available
        quadruped_spawn = sim_utils.CuboidCfg(
            size=(0.4, 0.2, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=5.0,
                max_angular_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=self.cfg.robot_mass,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.4, 0.2),  # Brown color
            ),
        )

        # For now, use RigidObject as placeholder
        # In production, use ArticulationCfg with proper URDF
        from isaaclab.assets import RigidObject, RigidObjectCfg

        robot_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=quadruped_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.cfg.standing_height),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Create the robot asset and add to scene
        self.robot = RigidObject(robot_cfg)
        self.scene.rigid_objects["robot"] = self.robot

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(1.0, 1.0, 1.0),
        )
        light_cfg.func("/World/light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process RL actions before physics stepping.

        This method handles the full gait pipeline:
        RL → CoM Bezier → Gait → Foothold → MPC → Physics

        Args:
            actions: Normalized actions from RL policy.
                Shape: (num_envs, num_actions)
                Layout: [bezier_params(12), gait_mod(3)]
        """
        # Denormalize actions
        actions_denorm = self._denormalize_actions(actions)
        actions_np = actions_denorm.cpu().numpy()

        # Parse action components
        bezier_params = actions_np[:, :self.cfg.num_bezier_actions]
        gait_mods = actions_np[:, self.cfg.num_bezier_actions:]

        # Get current robot states
        robot_states = self._get_robot_states_numpy()

        # Process each environment
        for env_idx in range(self.num_envs):
            # Check if time to update trajectory
            if self.mpc_step_counter[env_idx] % self.cfg.rl_policy_period == 0:
                # Generate new CoM trajectory from Bezier parameters
                current_pos = robot_states[env_idx, :3]
                new_trajectory = self.trajectory_generator.params_to_waypoints(
                    params=bezier_params[env_idx],
                    dt=self.cfg.mpc_dt,
                    horizon=self.cfg.bezier_horizon,
                    start_position=current_pos,
                )

                # Blend with previous trajectory for smoothness
                if self.trajectory_phases[env_idx] > 0:
                    old_traj = self.current_com_trajectories[env_idx]
                    old_phase = self.trajectory_phases[env_idx]
                    old_traj_shifted = np.roll(old_traj, -old_phase, axis=0)
                    self.current_com_trajectories[env_idx] = blend_trajectories(
                        old_trajectory=old_traj_shifted,
                        new_trajectory=new_trajectory,
                        blend_steps=self.cfg.trajectory_blend_steps,
                    )
                else:
                    self.current_com_trajectories[env_idx] = new_trajectory

                # Reset phase counter
                self.trajectory_phases[env_idx] = 0

                # Store for next blending
                self.prev_bezier_params[env_idx] = bezier_params[env_idx]
                self.prev_gait_mods[env_idx] = gait_mods[env_idx]

            # Get current reference slice for MPC
            phase = self.trajectory_phases[env_idx]
            end_phase = min(
                phase + self.cfg.mpc_horizon_steps,
                len(self.current_com_trajectories[env_idx])
            )
            com_reference = self.current_com_trajectories[env_idx, phase:end_phase]

            # Pad if necessary
            if len(com_reference) < self.cfg.mpc_horizon_steps:
                pad_length = self.cfg.mpc_horizon_steps - len(com_reference)
                last_point = com_reference[-1:] if len(com_reference) > 0 else np.zeros((1, 3))
                com_reference = np.concatenate([
                    com_reference,
                    np.repeat(last_point, pad_length, axis=0)
                ], axis=0)

            # Parse gait modulation parameters
            gait_params = {
                "step_length": gait_mods[env_idx, 0],
                "step_height": gait_mods[env_idx, 1],
                "step_frequency": gait_mods[env_idx, 2],
            }

            # Get current foot positions (dummy for now)
            foot_positions = self._get_foot_positions_numpy(env_idx, robot_states[env_idx])

            # Solve MPC
            if self.mpc_controllers[env_idx] is not None:
                solution = self.mpc_controllers[env_idx].solve(
                    current_state=robot_states[env_idx],
                    com_reference=com_reference,
                    current_foot_positions=foot_positions,
                    gait_params=gait_params,
                    warm_start=True,
                )
                self.last_mpc_solutions[env_idx] = solution
                joint_torques = solution.control
            else:
                # Dummy mode: zero control
                joint_torques = np.zeros(self.cfg.num_joints)
                self.last_mpc_solutions[env_idx] = None

            # Apply control to robot
            self._apply_control(env_idx, joint_torques)

            # Increment counters
            self.trajectory_phases[env_idx] += 1
            self.mpc_step_counter[env_idx] += 1

    def _apply_action(self):
        """Apply accumulated forces/torques to the robot bodies."""
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._pending_forces,
            torques=self._pending_torques,
            body_ids=slice(None),
        )

    def _apply_control(self, env_idx: int, joint_torques: np.ndarray):
        """Apply joint torque control to quadruped.

        For the placeholder rigid body, we apply equivalent body forces.
        In production with actual articulation, this would set joint efforts.

        Args:
            env_idx: Environment index.
            joint_torques: Joint torque vector (12D for quadruped).
        """
        # For rigid body placeholder, compute equivalent body force/torque
        # This is a simplification - real implementation would use articulation

        # Sum of vertical forces from all legs (simplified)
        vertical_force = np.sum(np.abs(joint_torques)) * 0.1  # Scale factor
        vertical_force = min(vertical_force, self.cfg.robot_mass * 9.81 * 2)

        # Simple torque from joint imbalance
        torque_x = (joint_torques[0] + joint_torques[3] - joint_torques[6] - joint_torques[9]) * 0.01
        torque_y = (joint_torques[1] + joint_torques[4] - joint_torques[7] - joint_torques[10]) * 0.01
        torque_z = 0.0

        # Store pending forces/torques for batch application
        self._pending_forces[env_idx, 0, :] = torch.tensor(
            [0.0, 0.0, vertical_force], device=self.device, dtype=torch.float32
        )
        self._pending_torques[env_idx, 0, :] = torch.tensor(
            [torque_x, torque_y, torque_z], device=self.device, dtype=torch.float32
        )

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for RL policy.

        Returns:
            Dictionary with 'policy' key containing observation tensor.
            Shape: (num_envs, num_observations)

        Observation structure (45D):
            - Base state (13D): position(3), quaternion(4), linear_vel(3), angular_vel(3)
            - Joint state (24D): joint_pos(12), joint_vel(12)
            - Foot contacts (4D): binary contact for each foot
            - Target position (3D): goal position
            - Gait phase (1D): normalized progress [0, 1]
        """
        robot_data = self.robot.data

        # Base state components
        position = robot_data.root_pos_w  # (num_envs, 3)
        orientation = robot_data.root_quat_w  # (num_envs, 4)
        linear_vel = robot_data.root_lin_vel_w  # (num_envs, 3)
        angular_vel = robot_data.root_ang_vel_w  # (num_envs, 3)

        # Joint state (dummy zeros for rigid body placeholder)
        joint_pos = torch.zeros((self.num_envs, 12), device=self.device)
        joint_vel = torch.zeros((self.num_envs, 12), device=self.device)

        # Foot contacts (estimate from height for placeholder)
        foot_contacts = (position[:, 2:3] < self.cfg.standing_height * 0.5).float()
        foot_contacts = foot_contacts.expand(-1, 4)

        # Target positions and phases
        phases = torch.zeros((self.num_envs, 1), device=self.device)

        for env_idx in range(self.num_envs):
            phase = self.trajectory_phases[env_idx]
            traj_len = len(self.current_com_trajectories[env_idx])
            phases[env_idx] = phase / max(traj_len - 1, 1)

        # Concatenate observations
        obs = torch.cat([
            position,           # 3D
            orientation,        # 4D
            linear_vel,         # 3D
            angular_vel,        # 3D
            joint_pos,          # 12D
            joint_vel,          # 12D
            foot_contacts,      # 4D
            self.target_positions,  # 3D
            phases,             # 1D
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for locomotion task.

        Returns:
            Reward tensor. Shape: (num_envs,)
        """
        robot_data = self.robot.data
        position = robot_data.root_pos_w
        orientation = robot_data.root_quat_w
        linear_vel = robot_data.root_lin_vel_w
        angular_vel = robot_data.root_ang_vel_w

        rewards = torch.zeros(self.num_envs, device=self.device)

        for env_idx in range(self.num_envs):
            # CoM tracking error
            phase = self.trajectory_phases[env_idx]
            traj = self.current_com_trajectories[env_idx]
            target_idx = min(phase, len(traj) - 1)
            target_com = torch.tensor(
                traj[target_idx], device=self.device, dtype=torch.float32
            )

            com_error = torch.norm(position[env_idx] - target_com)
            com_reward = self.cfg.reward_com_tracking * torch.exp(-com_error * 2)

            # Body orientation (keep upright)
            # Compute pitch and roll from quaternion
            quat = orientation[env_idx].cpu().numpy()
            euler = quat_to_euler(quat)
            pitch, roll = euler[1], euler[0]

            orientation_penalty = self.cfg.reward_body_orientation * (
                np.exp(-abs(pitch) * 5) + np.exp(-abs(roll) * 5)
            ) / 2

            # Forward velocity (towards target)
            target_dir = self.target_positions[env_idx] - position[env_idx]
            target_dist = torch.norm(target_dir[:2])
            if target_dist > 0.1:
                target_dir_norm = target_dir / (target_dist + 1e-6)
                forward_vel = torch.dot(linear_vel[env_idx, :2], target_dir_norm[:2])
                velocity_reward = self.cfg.reward_velocity_tracking * torch.clamp(forward_vel, -0.5, 1.0)
            else:
                velocity_reward = 0.0

            # Control penalty
            torque_penalty = 0.0
            if self.last_mpc_solutions[env_idx] is not None:
                torques = self.last_mpc_solutions[env_idx].control
                torque_penalty = self.cfg.reward_torque_penalty * np.sum(torques ** 2)

            # Alive bonus
            alive_bonus = self.cfg.reward_alive

            # Target reached bonus
            target_bonus = 0.0
            if target_dist < self.cfg.target_threshold:
                target_bonus = self.cfg.reward_reached_target

            # Total reward
            rewards[env_idx] = (
                com_reward
                + orientation_penalty
                + velocity_reward
                + torque_penalty
                + alive_bonus
                + target_bonus
            )

        return rewards

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination conditions.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
        """
        robot_data = self.robot.data
        position = robot_data.root_pos_w
        orientation = robot_data.root_quat_w

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Height check (fallen)
        too_low = position[:, 2] < self.cfg.min_body_height
        too_high = position[:, 2] > self.cfg.max_body_height

        # Orientation check (flipped)
        w = orientation[:, 0]
        x = orientation[:, 1]
        y = orientation[:, 2]
        z = orientation[:, 3]

        # Body z-axis in world frame
        up_world_z = 1 - 2 * (x * x + y * y)
        flipped = up_world_z < 0.5  # ~60 degrees from upright

        # Distance from start
        distance = torch.norm(position[:, :2], dim=1)
        out_of_bounds = distance > self.cfg.max_distance_from_start

        # Termination
        terminated = too_low | too_high | flipped | out_of_bounds

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

        # Random yaw orientation
        yaw_range = self.cfg.initial_yaw_range
        random_yaw = torch.rand(num_resets, device=self.device) * (
            yaw_range[1] - yaw_range[0]
        ) + yaw_range[0]

        # Convert to quaternion
        random_quat = torch.zeros((num_resets, 4), device=self.device)
        for i in range(num_resets):
            yaw = random_yaw[i].cpu().numpy()
            quat = euler_to_quat(np.array([0.0, 0.0, yaw]))
            random_quat[i] = torch.tensor(quat, device=self.device)

        # Zero velocities
        zero_vel = torch.zeros((num_resets, 3), device=self.device)

        # Reset robot state
        self.robot.write_root_pose_to_sim(
            torch.cat([random_pos, random_quat], dim=-1), env_ids
        )
        self.robot.write_root_velocity_to_sim(
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
            self.current_com_trajectories[env_idx] = trajectory

            # Reset counters
            self.trajectory_phases[env_idx] = 0
            self.mpc_step_counter[env_idx] = 0

            # Reset MPC controller
            if self.mpc_controllers[env_idx] is not None:
                self.mpc_controllers[env_idx].reset()

            # Reset stored params
            self.prev_bezier_params[env_idx] = initial_params
            self.prev_gait_mods[env_idx] = np.ones(self.cfg.num_gait_mod_actions)

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
        direction = target_pos - start_pos
        distance = np.linalg.norm(direction)

        # Clamp to max displacement
        max_disp = self.cfg.max_bezier_displacement
        if distance > max_disp:
            direction = direction / distance * max_disp

        # Control points as fractions along the line
        params = np.zeros(12)
        params[0:3] = 0.0  # P0 offset
        params[3:6] = direction / 3  # P1 offset
        params[6:9] = 2 * direction / 3  # P2 offset
        params[9:12] = direction  # P3 offset

        return params

    def _get_robot_states_numpy(self) -> np.ndarray:
        """Get robot states as numpy array for MPC.

        For full quadruped, state is (nq + nv) dimensional.
        For placeholder, we return simplified state.

        Returns:
            Array of shape (num_envs, state_dim).
        """
        robot_data = self.robot.data

        # Simplified state for rigid body placeholder
        # Full articulation would include joint positions/velocities
        state_dim = 13  # Simplified: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)

        states = np.zeros((self.num_envs, state_dim))
        states[:, 0:3] = robot_data.root_pos_w.cpu().numpy()
        states[:, 3:7] = robot_data.root_quat_w.cpu().numpy()
        states[:, 7:10] = robot_data.root_lin_vel_w.cpu().numpy()
        states[:, 10:13] = robot_data.root_ang_vel_w.cpu().numpy()

        return states

    def _get_foot_positions_numpy(
        self, env_idx: int, state: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Get current foot positions for an environment.

        For placeholder, compute nominal positions from body state.

        Args:
            env_idx: Environment index.
            state: Robot state.

        Returns:
            Dict mapping foot name to position (3,).
        """
        # For rigid body placeholder, compute nominal foot positions
        # based on body position and orientation

        from ..utils.math_utils import rotation_matrix_z, quat_to_euler

        pos = state[:3]
        quat = state[3:7]  # (w, x, y, z)

        # Get yaw from quaternion
        euler = quat_to_euler(quat)
        yaw = euler[2]

        R = rotation_matrix_z(yaw)

        foot_positions = {}
        for foot_name, hip_offset in self.cfg.hip_offsets.items():
            # Transform hip offset to world frame
            world_offset = R @ hip_offset
            foot_pos = pos + world_offset
            foot_pos[2] = 0.0  # On ground
            foot_positions[foot_name] = foot_pos

        return foot_positions

    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions [-1, 1] to parameter space.

        Args:
            actions: Normalized actions from RL policy.

        Returns:
            Denormalized action parameters.
        """
        return 0.5 * (actions + 1.0) * (
            self.action_high - self.action_low
        ) + self.action_low

    def close(self):
        """Clean up resources."""
        self.mpc_controllers.clear()
        super().close()
