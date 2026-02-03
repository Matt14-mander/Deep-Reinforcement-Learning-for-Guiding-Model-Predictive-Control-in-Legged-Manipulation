#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified evaluation script for trained Quadrotor MPC agent.

This script directly loads a trained RSL-RL model checkpoint and evaluates
the policy in the quadrotor MPC environment.

Usage:
    # Evaluate with specific checkpoint
    python scripts/evaluate_quadrotor.py --checkpoint logs/quadrotor_mpc/2026-02-03_20-11-03/model_499.pt

    # Evaluate with random policy (for testing)
    python scripts/evaluate_quadrotor.py --random

    # Evaluate with more environments
    python scripts/evaluate_quadrotor.py --checkpoint path/to/model.pt --num_envs 8
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Evaluate Quadrotor MPC agent")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to trained model checkpoint (.pt file)"
)
parser.add_argument(
    "--num_envs", type=int, default=4,
    help="Number of parallel environments"
)
parser.add_argument(
    "--num_steps", type=int, default=500,
    help="Number of evaluation steps"
)
parser.add_argument("--random", action="store_true", help="Use random policy")
parser.add_argument("--video", action="store_true", help="Record video")

# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for rendering
args_cli.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports after app launch
import os
import time
from datetime import datetime

import numpy as np
import torch

# Add source to path
SOURCE_DIR = Path(__file__).parent.parent / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(SOURCE_DIR))

# Import environment
from RL_Bezier_MPC.envs import QuadrotorMPCEnv, QuadrotorMPCEnvCfg

# Import RSL-RL
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def create_runner_config():
    """Create minimal RSL-RL runner configuration matching training."""
    return {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_steps_per_env": 24,
        "max_iterations": 1,  # Not training, just need structure
        "empirical_normalization": True,  # Must match training!
        "obs_groups": {},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "save_interval": 100,
        "log_interval": 10,
        "experiment_name": "quadrotor_mpc_eval",
        "run_name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Quadrotor MPC Bezier Trajectory Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment configuration
    env_cfg = QuadrotorMPCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    print(f"Number of environments: {env_cfg.scene.num_envs}")
    print(f"Episode length: {env_cfg.episode_length_s}s")

    # Create environment
    env = QuadrotorMPCEnv(
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Wrap environment for RSL-RL
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Create policy
    if args_cli.random:
        print("\nUsing random policy")
        policy = lambda obs: torch.rand(
            (args_cli.num_envs, env.action_space.shape[0]),
            device=device
        ) * 2 - 1
    elif args_cli.checkpoint:
        print(f"\nLoading checkpoint: {args_cli.checkpoint}")

        if not os.path.exists(args_cli.checkpoint):
            print(f"ERROR: Checkpoint file not found: {args_cli.checkpoint}")
            env.close()
            return

        # Create runner with matching config
        runner_cfg = create_runner_config()
        runner = OnPolicyRunner(
            env_wrapped,
            runner_cfg,
            log_dir=None,
            device=device
        )

        # Load checkpoint
        runner.load(args_cli.checkpoint)
        print("Checkpoint loaded successfully!")

        # Get inference policy
        policy = runner.get_inference_policy(device=device)
        print("Policy ready for inference")
    else:
        print("\nUsing straight-line policy (no checkpoint specified)")
        # Simple policy that moves towards target
        def straight_line_policy(obs):
            batch_size = obs.shape[0]
            actions = torch.zeros((batch_size, 12), device=device)

            # Extract position and target from observation
            current_pos = obs[:, 0:3]
            target_pos = obs[:, 13:16]

            # Direction to target
            direction = target_pos - current_pos
            distance = torch.norm(direction, dim=1, keepdim=True)
            direction_normalized = direction / (distance + 1e-6)
            scale = torch.clamp(distance / 2.0, 0, 1)

            # Set control points along straight line
            actions[:, 0:3] = 0.0  # P0 offset
            actions[:, 3:6] = direction_normalized * scale / 3  # P1
            actions[:, 6:9] = direction_normalized * scale * 2 / 3  # P2
            actions[:, 9:12] = direction_normalized * scale  # P3

            return actions

        policy = straight_line_policy

    # Run evaluation
    print(f"\nRunning evaluation for {args_cli.num_steps} steps...")
    print("-" * 40)

    # Get initial observations using the wrapper's method
    obs = env_wrapped.get_observations()

    total_rewards = np.zeros(args_cli.num_envs)
    episode_rewards = []
    episode_lengths = []
    current_episode_length = np.zeros(args_cli.num_envs)

    start_time = time.time()

    for step in range(args_cli.num_steps):
        # Get actions from policy
        with torch.inference_mode():
            actions = policy(obs)

        # Step environment
        obs, rewards, dones, infos = env_wrapped.step(actions)

        # Accumulate rewards
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy()
        else:
            rewards_np = rewards

        total_rewards += rewards_np
        current_episode_length += 1

        # Check for episode completions
        if isinstance(dones, torch.Tensor):
            dones_np = dones.cpu().numpy()
        else:
            dones_np = dones

        for i in range(args_cli.num_envs):
            if dones_np[i]:
                episode_rewards.append(total_rewards[i])
                episode_lengths.append(current_episode_length[i])
                total_rewards[i] = 0
                current_episode_length[i] = 0

        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps = (step + 1) * args_cli.num_envs / elapsed
            print(f"Step {step + 1}/{args_cli.num_steps}, "
                  f"Episodes: {len(episode_rewards)}, "
                  f"FPS: {fps:.1f}")

    # Print final statistics
    print("-" * 40)
    print("Evaluation Results:")

    if episode_rewards:
        print(f"  Completed episodes: {len(episode_rewards)}")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
        print(f"  Min reward: {np.min(episode_rewards):.2f}")
        print(f"  Max reward: {np.max(episode_rewards):.2f}")
        print(f"  Mean episode length: {np.mean(episode_lengths):.1f}")
    else:
        print("  No episodes completed during evaluation")
        print(f"  Accumulated rewards: {total_rewards.mean():.2f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Average FPS: {args_cli.num_steps * args_cli.num_envs / elapsed:.1f}")

    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
