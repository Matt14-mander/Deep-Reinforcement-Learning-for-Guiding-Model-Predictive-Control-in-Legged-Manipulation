#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation/play script for trained Quadruped MPC policies.

This script loads a trained RL policy and runs it in the simulation
for visualization and evaluation purposes. Supports trajectory logging
and matplotlib-based visualization for headless cloud servers.

Usage:
    # Random policy (for testing)
    python scripts/play_quadruped_mpc.py --num_envs 1 --headless

    # Load trained checkpoint + save trajectory plot
    python scripts/play_quadruped_mpc.py --checkpoint logs/quadruped_mpc/.../model_2999.pt --num_envs 1 --headless

    # Livestream (real-time viewing via browser)
    python scripts/play_quadruped_mpc.py --checkpoint model.pt --num_envs 1 --headless --livestream 2
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Play Quadruped MPC Policy")

# Environment settings
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of parallel environments (default: 1)",
)
parser.add_argument(
    "--gait", type=str, default="trot",
    choices=["trot", "walk", "pace", "bound"],
    help="Gait type (default: trot)",
)

# Policy settings
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to trained checkpoint",
)
parser.add_argument(
    "--random", action="store_true",
    help="Use random actions instead of policy",
)

# Episode settings
parser.add_argument(
    "--num_episodes", type=int, default=3,
    help="Number of episodes to run (default: 3)",
)
parser.add_argument(
    "--max_steps", type=int, default=300,
    help="Maximum steps per episode (default: 300)",
)

# Output settings
parser.add_argument(
    "--plot_dir", type=str, default="plots",
    help="Directory to save trajectory plots",
)

# AppLauncher arguments (adds --headless, --device, --livestream, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports after app launch
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch

# Add source to path
SOURCE_DIR = Path(__file__).parent.parent / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(SOURCE_DIR))

from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg


def plot_episode_trajectory(positions, orientations, target_pos, rewards,
                            episode_idx, save_dir):
    """Plot the robot's CoM trajectory for one episode.

    Generates a 2x2 figure:
      - Top-left: XY trajectory (bird's eye view) with start/target markers
      - Top-right: Height (Z) over time
      - Bottom-left: Cumulative reward over time
      - Bottom-right: Orientation (roll/pitch/yaw) over time

    Args:
        positions: (T, 3) numpy array of CoM positions.
        orientations: (T, 4) numpy array of quaternions (w, x, y, z).
        target_pos: (3,) target position.
        rewards: (T,) reward per step.
        episode_idx: Episode number for title.
        save_dir: Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    T = len(positions)
    time_steps = np.arange(T) * 0.02  # MPC dt = 0.02s

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Episode {episode_idx + 1}  |  {T} steps  |  "
                 f"Total reward: {np.sum(rewards):.2f}", fontsize=14)

    # --- Top-left: XY trajectory ---
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], "b-", linewidth=1.5, label="CoM path")
    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="Start")
    ax.plot(positions[-1, 0], positions[-1, 1], "rs", markersize=10, label="End")
    ax.plot(target_pos[0], target_pos[1], "r*", markersize=15, label="Target")

    # Draw arrow for direction at several points
    step = max(1, T // 10)
    for i in range(0, T - 1, step):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        if abs(dx) + abs(dy) > 1e-5:
            ax.annotate("", xy=(positions[i, 0] + dx * 3, positions[i, 1] + dy * 3),
                        xytext=(positions[i, 0], positions[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="blue", alpha=0.4))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory (Bird's Eye View)")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- Top-right: Height over time ---
    ax = axes[0, 1]
    ax.plot(time_steps, positions[:, 2], "b-", linewidth=1.5)
    ax.axhline(y=0.4, color="g", linestyle="--", alpha=0.5, label="Standing height (0.4m)")
    ax.axhline(y=0.12, color="r", linestyle="--", alpha=0.5, label="Fall threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("CoM Height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: Cumulative reward ---
    ax = axes[1, 0]
    cum_reward = np.cumsum(rewards)
    ax.plot(time_steps, cum_reward, "g-", linewidth=1.5)
    ax.plot(time_steps, rewards, "b-", alpha=0.3, linewidth=0.8, label="Per-step reward")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title("Reward (green=cumulative, blue=per-step)")
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: Orientation ---
    ax = axes[1, 1]
    # Convert quaternion (w,x,y,z) to roll/pitch/yaw
    rolls, pitches, yaws = [], [], []
    for q in orientations:
        w, x, y, z = q
        # Roll
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)
        # Pitch
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)

        rolls.append(np.degrees(roll))
        pitches.append(np.degrees(pitch))
        yaws.append(np.degrees(yaw))

    ax.plot(time_steps, rolls, "r-", linewidth=1, label="Roll")
    ax.plot(time_steps, pitches, "g-", linewidth=1, label="Pitch")
    ax.plot(time_steps, yaws, "b-", linewidth=1, label="Yaw")
    ax.axhline(y=45, color="k", linestyle="--", alpha=0.3)
    ax.axhline(y=-45, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Body Orientation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"episode_{episode_idx + 1}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {save_path}")


def main():
    """Main play function."""
    print("=" * 60)
    print("Quadruped MPC Bezier Trajectory - Play")
    print("=" * 60)

    # Create environment configuration
    env_cfg = QuadrupedMPCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.gait_type = args_cli.gait

    print(f"  Num envs: {env_cfg.scene.num_envs}")
    print(f"  Gait type: {env_cfg.gait_type}")
    print(f"  Episode length: {env_cfg.episode_length_s}s")

    # Create environment (no render_mode needed for trajectory logging)
    print("\nCreating environment...")
    env = QuadrupedMPCEnv(cfg=env_cfg)
    device = env.device

    # Load policy if checkpoint provided
    policy = None
    if args_cli.checkpoint and not args_cli.random:
        try:
            from rsl_rl.modules import ActorCritic

            # Create actor-critic network
            policy = ActorCritic(
                num_actor_obs=env_cfg.observation_space,
                num_critic_obs=env_cfg.observation_space,
                num_actions=env_cfg.action_space,
                actor_hidden_dims=[256, 256, 128],
                critic_hidden_dims=[256, 256, 128],
                activation="elu",
            ).to(device)

            # Load weights
            checkpoint = torch.load(
                args_cli.checkpoint, map_location=device, weights_only=False,
            )
            policy.load_state_dict(checkpoint["model_state_dict"])
            policy.eval()
            print(f"Loaded policy from: {args_cli.checkpoint}")

        except Exception as e:
            print(f"Warning: Could not load policy: {e}")
            print("Using random actions instead.")
            policy = None

    # Run episodes
    print(f"\nRunning {args_cli.num_episodes} episodes...")
    print("=" * 60)

    episode_rewards_all = []
    episode_lengths_all = []

    for episode in range(args_cli.num_episodes):
        print(f"\nEpisode {episode + 1}/{args_cli.num_episodes}")

        # Reset environment
        obs, _ = env.reset()
        obs_tensor = obs["policy"]

        # Trajectory logging buffers
        positions_log = []
        orientations_log = []
        rewards_log = []

        episode_reward = torch.zeros(args_cli.num_envs, device=device)
        episode_length = 0
        done = False

        max_steps = args_cli.max_steps

        while not done and episode_length < max_steps:
            # Get actions
            if policy is not None:
                with torch.no_grad():
                    actions = policy.act_inference(obs_tensor)
            else:
                # Random actions
                actions = torch.rand(
                    args_cli.num_envs, env_cfg.action_space, device=device,
                ) * 2 - 1

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            obs_tensor = obs["policy"]

            episode_reward += rewards
            episode_length += 1

            # Log trajectory data (env 0)
            robot_data = env.robot.data
            pos = robot_data.root_pos_w[0].cpu().numpy().copy()
            quat = robot_data.root_quat_w[0].cpu().numpy().copy()
            positions_log.append(pos)
            orientations_log.append(quat)
            rewards_log.append(rewards[0].cpu().item())

            # Check if any environment is done
            dones = terminated | truncated
            if dones.any():
                done = True

        # Episode summary
        avg_reward = episode_reward.mean().item()
        episode_rewards_all.append(avg_reward)
        episode_lengths_all.append(episode_length)

        print(f"  Reward: {avg_reward:.2f}")
        print(f"  Length: {episode_length} steps ({episode_length * 0.02:.1f}s)")

        # Get target position for plot
        target_pos = env.target_positions[0].cpu().numpy()

        # Generate trajectory plot
        positions_arr = np.array(positions_log)
        orientations_arr = np.array(orientations_log)
        rewards_arr = np.array(rewards_log)

        plot_episode_trajectory(
            positions_arr, orientations_arr, target_pos, rewards_arr,
            episode, args_cli.plot_dir,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Episodes: {len(episode_rewards_all)}")
    print(f"Average reward: {np.mean(episode_rewards_all):.2f} +/- {np.std(episode_rewards_all):.2f}")
    print(f"Average length: {np.mean(episode_lengths_all):.1f} steps")
    print(f"\nTrajectory plots saved to: {args_cli.plot_dir}/")

    # Cleanup
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
