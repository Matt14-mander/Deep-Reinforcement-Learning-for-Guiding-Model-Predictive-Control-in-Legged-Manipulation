# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Quadrotor MPC environment.

This module defines the configuration for the DirectRL environment that
integrates RL policy (Bezier parameters) with MPC trajectory tracking.

Frequency Hierarchy:
    Physics simulation: 200 Hz (sim.dt = 0.005s)
    MPC control rate:   50 Hz  (decimation = 4)
    RL policy rate:    5-10 Hz (every 5-10 MPC cycles)
    Bezier horizon:     1.5 s  (75 waypoints at 50 Hz)
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


@configclass
class QuadrotorMPCEnvCfg(DirectRLEnvCfg):
    """Configuration for Quadrotor MPC trajectory tracking environment.

    The RL policy outputs Bezier curve parameters that are converted to
    a dense position trajectory, which is then tracked by an MPC controller
    running at a higher frequency.

    Attributes:
        decimation: Ratio of simulation steps per control step (MPC rate).
        episode_length_s: Maximum episode duration in seconds.
        num_observations: Dimension of observation space.
        num_actions: Dimension of action space (Bezier parameters).
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
            # Enable GPU dynamics for parallel environments
            enable_stabilization=True,
        ),
        gravity=(0.0, 0.0, -9.81),
    )

    # Control decimation: 200 Hz / 4 = 50 Hz MPC rate
    decimation: int = 4

    # Episode settings
    episode_length_s: float = 10.0

    # ==========================================================================
    # Scene Configuration
    # ==========================================================================

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,  # Limited by CPU MPC solver
        env_spacing=3.0,  # Meters between environments
        replicate_physics=True,
    )

    # ==========================================================================
    # RL Spaces
    # ==========================================================================

    # Observation space:
    #   - Robot state (13D): position(3), quaternion(4), linear_vel(3), angular_vel(3)
    #   - Target position (3D): next goal position from trajectory
    #   - Trajectory phase (1D): normalized progress [0, 1]
    # Total: 17D
    num_observations: int = 17

    # Action space:
    #   - Bezier control point offsets (12D): 4 points × 3D
    num_actions: int = 12

    # ==========================================================================
    # MPC Configuration
    # ==========================================================================

    # MPC timestep (should match control rate)
    mpc_dt: float = 0.02  # 50 Hz

    # MPC prediction horizon (shorter than Bezier horizon for efficiency)
    mpc_horizon_steps: int = 25  # 0.5s lookahead

    # ==========================================================================
    # Trajectory Configuration
    # ==========================================================================

    # Bezier trajectory duration
    bezier_horizon: float = 1.5  # seconds

    # RL policy update period (in MPC steps)
    # 5 MPC steps @ 50Hz = 10 Hz policy rate
    rl_policy_period: int = 5

    # Maximum displacement for Bezier control points
    max_bezier_displacement: float = 2.0  # meters

    # Trajectory blending steps when updating
    trajectory_blend_steps: int = 3

    # ==========================================================================
    # Quadrotor Physical Parameters
    # ==========================================================================

    quadrotor_mass: float = 0.027  # kg
    quadrotor_arm_length: float = 0.046  # m

    # Control limits
    thrust_min: float = 0.0
    thrust_max: float = 0.6  # N (~2x hover thrust)
    torque_max: float = 0.01  # N.m

    # ==========================================================================
    # Task Configuration
    # ==========================================================================

    # Initial position randomization
    initial_pos_range: tuple = (-0.5, 0.5, -0.5, 0.5, 0.8, 1.2)  # (x_min, x_max, y_min, y_max, z_min, z_max)

    # Target generation
    target_pos_range: tuple = (-2.0, 2.0, -2.0, 2.0, 0.5, 2.0)

    # Success threshold
    position_threshold: float = 0.1  # meters

    # ==========================================================================
    # Reward Weights
    # ==========================================================================

    reward_position_tracking: float = 1.0
    reward_velocity_penalty: float = -0.1
    reward_control_penalty: float = -0.01
    reward_reached_target: float = 10.0
    reward_alive: float = 0.1
    reward_crash_penalty: float = -10.0

    # ==========================================================================
    # Termination Conditions
    # ==========================================================================

    # Position bounds for termination
    max_position_deviation: float = 5.0  # meters from origin

    # Orientation limit (dot product of up vector with world z)
    min_up_dot_product: float = 0.5  # ~60 degrees from upright

    # Height limits
    min_height: float = 0.1
    max_height: float = 3.0

    # ==========================================================================
    # Viewer Settings
    # ==========================================================================

    viewer_eye: tuple = (5.0, 5.0, 3.0)
    viewer_lookat: tuple = (0.0, 0.0, 1.0)

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

        # Validate configuration
        assert self.decimation > 0, "Decimation must be positive"
        assert self.mpc_horizon_steps > 0, "MPC horizon must be positive"
        assert self.rl_policy_period > 0, "RL policy period must be positive"
        assert self.num_bezier_waypoints > self.mpc_horizon_steps, (
            "Bezier horizon should be longer than MPC horizon"
        )


# Alias for convenience
QuadrotorBezierMPCEnvCfg = QuadrotorMPCEnvCfg
