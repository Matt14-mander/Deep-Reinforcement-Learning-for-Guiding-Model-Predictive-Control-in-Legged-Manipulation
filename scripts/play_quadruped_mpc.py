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
    python scripts/play_quadruped_mpc.py --num_envs 4

    # Load trained checkpoint
    python scripts/play_quadruped_mpc.py --checkpoint logs/quadruped_mpc/model_final.pt

    # Record video
    python scripts/play_quadruped_mpc.py --checkpoint model.pt --video --video_length 10

Requirements:
    - Isaac Lab (Isaac Sim 4.5+)
    - Trained checkpoint (optional, uses random policy otherwise)
"""

import argparse
import os
import sys
from pathlib import Path

# Add source to path before imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "..", "source", "RL_Bezier_MPC")
sys.path.insert(0, SOURCE_DIR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Play Quadruped MPC Policy")

    # Environment settings
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--gait",
        type=str,
        default="trot",
        choices=["trot", "walk", "pace", "bound"],
        help="Gait type (default: trot)",
    )

    # Policy settings
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random actions instead of policy",
    )

    # Episode settings
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )

    # Video settings
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record video",
    )
    parser.add_argument(
        "--video_length",
        type=float,
        default=10.0,
        help="Video length in seconds (default: 10)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Directory to save videos",
    )

    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (default: cuda:0)",
    )

    return parser.parse_args()


def main():
    """Main play function."""
    args = parse_args()

    # Import Isaac Lab
    print("Initializing Isaac Lab...")

    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(
            headless=False,  # Always show viewer for play
            enable_cameras=args.video,
        )
        simulation_app = app_launcher.app

    except ImportError as e:
        print(f"Error: Could not import Isaac Lab: {e}")
        sys.exit(1)

    # Import after app launcher
    import torch
    import numpy as np

    from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

    # Create video directory if needed
    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)

    # Create environment configuration
    env_cfg = QuadrupedMPCEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.gait_type = args.gait

    print(f"\nEnvironment configuration:")
    print(f"  Num envs: {env_cfg.scene.num_envs}")
    print(f"  Gait type: {env_cfg.gait_type}")
    print(f"  Episode length: {env_cfg.episode_length_s}s")

    # Create environment
    print("\nCreating environment...")
    env = QuadrupedMPCEnv(cfg=env_cfg)

    # Load policy if checkpoint provided
    policy = None
    if args.checkpoint and not args.random:
        try:
            from rsl_rl.modules import ActorCritic

            # Create actor-critic network
            policy = ActorCritic(
                num_actor_obs=env.cfg.num_observations,
                num_critic_obs=env.cfg.num_observations,
                num_actions=env.cfg.num_actions,
                actor_hidden_dims=[256, 256, 128],
                critic_hidden_dims=[256, 256, 128],
                activation="elu",
            ).to(env.device)

            # Load weights
            checkpoint = torch.load(args.checkpoint, map_location=env.device)
            policy.load_state_dict(checkpoint["actor_critic_state_dict"])
            policy.eval()

            print(f"Loaded policy from: {args.checkpoint}")

        except Exception as e:
            print(f"Warning: Could not load policy: {e}")
            print("Using random actions instead.")
            policy = None

    # Run episodes
    print(f"\nRunning {args.num_episodes} episodes...")
    print("=" * 60)

    episode_rewards = []
    episode_lengths = []

    for episode in range(args.num_episodes):
        print(f"\nEpisode {episode + 1}/{args.num_episodes}")

        # Reset environment
        obs = env.reset()
        obs_tensor = obs["policy"]

        episode_reward = torch.zeros(env.num_envs, device=env.device)
        episode_length = 0
        done = False

        # Run episode
        while not done:
            # Get actions
            if policy is not None:
                with torch.no_grad():
                    actions = policy.act_inference(obs_tensor)
            else:
                # Random actions
                actions = torch.rand(
                    env.num_envs, env.cfg.num_actions, device=env.device
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

            # Video recording
            if args.video and episode == 0:  # Record first episode
                # Video recording handled by Isaac Lab viewer

        # Episode summary
        avg_reward = episode_reward.mean().item()
        episode_rewards.append(avg_reward)
        episode_lengths.append(episode_length)

        print(f"  Reward: {avg_reward:.2f}")
        print(f"  Length: {episode_length} steps")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Episodes: {args.num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} steps")

    # Keep viewer open for inspection
    print("\nViewer is open. Close window to exit.")
    while simulation_app.is_running():
        simulation_app.update()

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
