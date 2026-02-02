# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Quadruped MPC environment.

This module defines the configuration for the DirectRL environment that
integrates RL policy (CoM Bezier + gait modulation) with quadruped MPC.

Frequency Hierarchy:
    Physics simulation: 200 Hz (sim.dt = 0.005s)
    MPC control rate:   50 Hz  (decimation = 4)
    RL policy rate:    5 Hz   (every 10 MPC cycles, slower for quadruped)
    Bezier horizon:     1.5 s  (75 waypoints at 50 Hz)

Action Space:
    - CoM Bezier parameters (12D): 4 control points × 3D
    - Gait modulation (3D): step_length, step_height, step_frequency

Observation Space:
    - Base state (13D): position(3), quaternion(4), linear_vel(3), angular_vel(3)
    - Joint state (24D): joint_pos(12), joint_vel(12)
    - Foot contacts (4D): binary contact for each foot
    - Target position (3D): goal position
    - Gait phase (1D): normalized progress in current Bezier horizon
    Total: 45D
"""

from dataclasses import field
from typing import Dict, Tuple

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


@configclass
class QuadrupedMPCEnvCfg(DirectRLEnvCfg):
    """Configuration for Quadruped MPC trajectory tracking environment.

    The RL policy outputs CoM Bezier curve parameters and gait modulation
    parameters. These are processed through the quadruped gait pipeline:
        1. CoM Bezier → dense CoM waypoints
        2. GaitScheduler → contact sequence
        3. FootholdPlanner → foot landing positions
        4. BezierFootTrajectory → swing trajectories
        5. Crocoddyl MPC → joint torques

    Attributes:
        decimation: Ratio of simulation steps per control step (MPC rate).
        episode_length_s: Maximum episode duration in seconds.
        num_observations: Dimension of observation space.
        num_actions: Dimension of action space.
    """

    # ==========================================================================
    # Simulation Settings
    # ==========================================================================

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,  # 200 Hz physics simulation
        render_interval=4,  # Render at MPC rate (50 Hz)
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            enable_stabilization=True,
        ),
        gravity=(0.0, 0.0, -9.81),
    )

    # Control decimation: 200 Hz / 4 = 50 Hz MPC rate
    decimation: int = 4

    # Episode settings (longer for quadruped locomotion)
    episode_length_s: float = 15.0

    # ==========================================================================
    # Scene Configuration
    # ==========================================================================

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,  # Fewer envs due to heavier MPC computation
        env_spacing=4.0,  # Larger spacing for locomotion
        replicate_physics=True,
    )

    # ==========================================================================
    # RL Spaces
    # ==========================================================================

    # Observation space:
    #   - Base state (13D): position(3), quaternion(4), linear_vel(3), angular_vel(3)
    #   - Joint state (24D): joint_pos(12), joint_vel(12)
    #   - Foot contacts (4D): binary contact for each foot
    #   - Target position (3D): goal position
    #   - Gait phase (1D): normalized progress [0, 1]
    # Total: 45D
    num_observations: int = 45

    # Action space:
    #   - CoM Bezier control point offsets (12D): 4 points × 3D
    #   - Gait modulation (3D): step_length, step_height, step_frequency modifiers
    # Total: 15D
    num_actions: int = 15

    # Action sub-dimensions for parsing
    num_bezier_actions: int = 12
    num_gait_mod_actions: int = 3

    # ==========================================================================
    # MPC Configuration
    # ==========================================================================

    # MPC timestep (should match control rate)
    mpc_dt: float = 0.02  # 50 Hz

    # MPC prediction horizon
    mpc_horizon_steps: int = 25  # 0.5s lookahead

    # ==========================================================================
    # Trajectory Configuration
    # ==========================================================================

    # Bezier trajectory duration
    bezier_horizon: float = 1.5  # seconds

    # RL policy update period (in MPC steps)
    # 10 MPC steps @ 50Hz = 5 Hz policy rate (slower for quadruped)
    rl_policy_period: int = 10

    # Maximum displacement for CoM Bezier control points
    max_bezier_displacement: float = 1.5  # meters (less than quadrotor)

    # Trajectory blending steps when updating
    trajectory_blend_steps: int = 5

    # ==========================================================================
    # Gait Configuration
    # ==========================================================================

    # Default gait type
    gait_type: str = "trot"

    # Gait timing defaults
    default_step_duration: float = 0.15  # seconds per swing phase
    default_support_duration: float = 0.05  # double support duration
    default_step_height: float = 0.05  # meters

    # Gait modulation bounds (multipliers around 1.0)
    step_length_mod_range: Tuple[float, float] = (0.5, 2.0)
    step_height_mod_range: Tuple[float, float] = (0.5, 2.0)
    step_frequency_mod_range: Tuple[float, float] = (0.5, 2.0)

    # ==========================================================================
    # Robot Configuration
    # ==========================================================================

    # Robot type identifier
    robot_name: str = "go1"

    # Physical parameters (will be overridden by robot config)
    robot_mass: float = 12.0  # kg
    standing_height: float = 0.35  # meters

    # Joint configuration
    num_joints: int = 12
    max_joint_torque: float = 23.0  # N.m

    # Friction coefficient for MPC
    friction_coefficient: float = 0.7

    # Hip offsets (meters, in body frame)
    # +x = forward, +y = left, +z = up
    hip_offsets: Dict[str, np.ndarray] = field(default_factory=lambda: {
        "LF": np.array([+0.183, +0.047, 0.0]),
        "RF": np.array([+0.183, -0.047, 0.0]),
        "LH": np.array([-0.183, +0.047, 0.0]),
        "RH": np.array([-0.183, -0.047, 0.0]),
    })

    # Foot frame names (as in URDF)
    foot_frame_names: Dict[str, str] = field(default_factory=lambda: {
        "LF": "LF_FOOT",
        "RF": "RF_FOOT",
        "LH": "LH_FOOT",
        "RH": "RH_FOOT",
    })

    # ==========================================================================
    # Task Configuration
    # ==========================================================================

    # Initial position randomization
    initial_pos_range: Tuple[float, float, float, float, float, float] = (
        -0.2, 0.2,  # x_min, x_max
        -0.2, 0.2,  # y_min, y_max
        0.30, 0.40,  # z_min, z_max (near standing height)
    )

    # Initial orientation randomization (yaw only)
    initial_yaw_range: Tuple[float, float] = (-0.3, 0.3)  # radians

    # Target generation (goal positions)
    target_pos_range: Tuple[float, float, float, float, float, float] = (
        2.0, 5.0,   # x_min, x_max (forward)
        -2.0, 2.0,  # y_min, y_max
        0.0, 0.0,   # z_min, z_max (ground level)
    )

    # Success threshold for reaching target
    target_threshold: float = 0.5  # meters

    # ==========================================================================
    # Reward Weights
    # ==========================================================================

    # Primary rewards
    reward_com_tracking: float = 1.0
    reward_body_orientation: float = 0.5
    reward_velocity_tracking: float = 0.3
    reward_reached_target: float = 10.0

    # Penalties
    reward_torque_penalty: float = -0.01
    reward_joint_velocity_penalty: float = -0.01
    reward_foot_slip_penalty: float = -0.5
    reward_body_collision_penalty: float = -5.0
    reward_fall_penalty: float = -10.0

    # Regularization
    reward_action_rate_penalty: float = -0.05
    reward_alive: float = 0.1

    # ==========================================================================
    # Termination Conditions
    # ==========================================================================

    # Body height limits (fraction of standing height)
    min_body_height_ratio: float = 0.3  # Too low = fallen
    max_body_height_ratio: float = 1.5  # Too high = jumping

    # Orientation limits (degrees from upright)
    max_pitch_deg: float = 45.0
    max_roll_deg: float = 45.0

    # Position bounds
    max_distance_from_start: float = 10.0  # meters

    # ==========================================================================
    # Viewer Settings
    # ==========================================================================

    viewer_eye: Tuple[float, float, float] = (5.0, 5.0, 3.0)
    viewer_lookat: Tuple[float, float, float] = (0.0, 0.0, 0.3)

    # ==========================================================================
    # Debug/Visualization
    # ==========================================================================

    # Whether to visualize trajectories
    visualize_trajectories: bool = False

    # Whether to visualize contact forces
    visualize_contacts: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        # Compute derived values
        self.mpc_steps_per_episode = int(
            self.episode_length_s / self.mpc_dt
        )

        # Number of waypoints in Bezier trajectory
        self.num_bezier_waypoints = int(
            self.bezier_horizon / self.mpc_dt
        ) + 1

        # Compute height limits from ratios
        self.min_body_height = self.standing_height * self.min_body_height_ratio
        self.max_body_height = self.standing_height * self.max_body_height_ratio

        # Convert orientation limits to radians
        self.max_pitch_rad = np.deg2rad(self.max_pitch_deg)
        self.max_roll_rad = np.deg2rad(self.max_roll_deg)

        # Validate configuration
        assert self.decimation > 0, "Decimation must be positive"
        assert self.mpc_horizon_steps > 0, "MPC horizon must be positive"
        assert self.rl_policy_period > 0, "RL policy period must be positive"
        assert self.num_actions == self.num_bezier_actions + self.num_gait_mod_actions, (
            "Action dimensions must sum correctly"
        )
        assert self.num_bezier_waypoints > self.mpc_horizon_steps, (
            "Bezier horizon should be longer than MPC horizon"
        )


# Alias for convenience
QuadrupedBezierMPCEnvCfg = QuadrupedMPCEnvCfg
