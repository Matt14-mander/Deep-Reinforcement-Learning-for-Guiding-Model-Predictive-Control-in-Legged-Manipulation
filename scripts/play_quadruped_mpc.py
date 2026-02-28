#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation/play script for trained Quadruped MPC policies.

This script loads a trained RL policy and runs it in the simulation
for visualization and evaluation purposes.

Usage:
    # Random policy (for testing)
    python scripts/play_quadruped_mpc.py --num_envs 4 --headless

    # Load trained checkpoint
    python scripts/play_quadruped_mpc.py --checkpoint logs/quadruped_mpc/model_final.pt

    # Record video (headless cloud server)
    python scripts/play_quadruped_mpc.py --checkpoint model.pt --video --video_length 200 --headless

Requirements:
    - Isaac Lab (Isaac Sim 4.5+)
    - Trained checkpoint (optional, uses random policy otherwise)
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Play Quadruped MPC Policy")

# Environment settings
parser.add_argument(
    "--num_envs", type=int, default=4,
    help="Number of parallel environments (default: 4)",
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
    "--num_episodes", type=int, default=5,
    help="Number of episodes to run (default: 5)",
)

# Video settings
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
parser.add_argument(
    "--video_dir", type=str, default="videos",
    help="Directory to save videos",
)

# Hardware settings
parser.add_argument(
    "--device", type=str, default="cuda:0",
    help="Device for inference (default: cuda:0)",
)

# AppLauncher arguments (adds --headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports after app launch
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Add source to path
SOURCE_DIR = Path(__file__).parent.parent / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(SOURCE_DIR))

from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg


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

    # Create environment
    print("\nCreating environment...")
    env = QuadrupedMPCEnv(
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Wrap with video recorder if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(args_cli.video_dir, "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"Recording video to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

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
            ).to(args_cli.device)

            # Load weights
            checkpoint = torch.load(
                args_cli.checkpoint, map_location=args_cli.device, weights_only=False,
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

    episode_rewards = []
    episode_lengths = []

    for episode in range(args_cli.num_episodes):
        print(f"\nEpisode {episode + 1}/{args_cli.num_episodes}")

        # Reset environment
        obs, _ = env.reset()
        obs_tensor = obs["policy"]

        episode_reward = torch.zeros(args_cli.num_envs, device=args_cli.device)
        episode_length = 0
        done = False

        # Run episode
        max_steps = args_cli.video_length if args_cli.video else int(
            env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)
        )

        while not done and episode_length < max_steps:
            # Get actions
            if policy is not None:
                with torch.no_grad():
                    actions = policy.act_inference(obs_tensor)
            else:
                # Random actions
                actions = torch.rand(
                    args_cli.num_envs, env_cfg.action_space, device=args_cli.device,
                ) * 2 - 1

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            obs_tensor = obs["policy"]

            episode_reward += rewards
            episode_length += 1

            # Check if any environment is done
            dones = terminated | truncated
            if dones.any():
                done = True

        # Episode summary
        avg_reward = episode_reward.mean().item()
        episode_rewards.append(avg_reward)
        episode_lengths.append(episode_length)

        print(f"  Reward: {avg_reward:.2f}")
        print(f"  Length: {episode_length} steps")

        # For video mode, only record first episode then exit
        if args_cli.video:
            print(f"\nVideo recorded for episode 1.")
            break

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} steps")

    # Cleanup
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
