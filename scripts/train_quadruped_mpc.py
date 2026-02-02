#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for Quadruped MPC with RL.

This script trains an RL policy to output CoM Bezier trajectory parameters
and gait modulation for quadruped locomotion. The policy is combined with
a Crocoddyl MPC controller that handles the low-level joint torque control.

Usage:
    # Basic training
    python scripts/train_quadruped_mpc.py --num_envs 32 --max_iterations 500

    # Resume from checkpoint
    python scripts/train_quadruped_mpc.py --resume --checkpoint logs/quadruped_mpc/model_500.pt

    # With video recording
    python scripts/train_quadruped_mpc.py --video --video_interval 50

Requirements:
    - Isaac Lab (Isaac Sim 4.5+)
    - RSL-RL library for PPO training
    - Crocoddyl for MPC (optional, runs in dummy mode without)
"""

import argparse
import os
import sys
from datetime import datetime

# Add source to path before imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "..", "source", "RL_Bezier_MPC")
sys.path.insert(0, SOURCE_DIR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Quadruped MPC with RL")

    # Environment settings
    parser.add_argument(
        "--num_envs",
        type=int,
        default=32,
        help="Number of parallel environments (default: 32)",
    )
    parser.add_argument(
        "--gait",
        type=str,
        default="trot",
        choices=["trot", "walk", "pace", "bound"],
        help="Gait type (default: trot)",
    )

    # Training settings
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=500,
        help="Maximum training iterations (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Checkpoint settings
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for resuming",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Log directory (default: logs/quadruped_mpc_TIMESTAMP)",
    )

    # Video settings
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable video recording",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=100,
        help="Record video every N iterations",
    )

    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no viewer)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Import Isaac Lab AFTER parsing args to avoid slow startup for --help
    print("Initializing Isaac Lab...")

    try:
        from isaaclab.app import AppLauncher

        # Configure app launcher
        app_launcher = AppLauncher(
            headless=args.headless,
            enable_cameras=args.video,
        )
        simulation_app = app_launcher.app

    except ImportError as e:
        print(f"Error: Could not import Isaac Lab: {e}")
        print("\nPlease ensure Isaac Lab is properly installed.")
        print("See: https://isaac-sim.github.io/IsaacLab/")
        sys.exit(1)

    # Now import environment and training components
    import torch
    import numpy as np

    from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

    # Try to import RSL-RL
    try:
        from rsl_rl.runners import OnPolicyRunner
        from rsl_rl.algorithms import PPO
        from rsl_rl.modules import ActorCritic

        HAS_RSL_RL = True
    except ImportError:
        print("Warning: RSL-RL not available. Using simple training loop.")
        HAS_RSL_RL = False

    # Set up logging directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = os.path.join("logs", f"quadruped_mpc_{timestamp}")

    os.makedirs(args.log_dir, exist_ok=True)
    print(f"Logging to: {args.log_dir}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment configuration
    env_cfg = QuadrupedMPCEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.gait_type = args.gait

    # Disable rendering for faster training
    if args.headless:
        env_cfg.sim.render_interval = 0

    print(f"\nEnvironment configuration:")
    print(f"  Num envs: {env_cfg.scene.num_envs}")
    print(f"  Gait type: {env_cfg.gait_type}")
    print(f"  Episode length: {env_cfg.episode_length_s}s")
    print(f"  Observation dim: {env_cfg.num_observations}")
    print(f"  Action dim: {env_cfg.num_actions}")

    # Create environment
    print("\nCreating environment...")
    env = QuadrupedMPCEnv(cfg=env_cfg)

    print(f"Environment created:")
    print(f"  Num envs: {env.num_envs}")
    print(f"  Device: {env.device}")

    if HAS_RSL_RL:
        # Use RSL-RL PPO training
        train_with_rsl_rl(env, args)
    else:
        # Use simple training loop
        train_simple(env, args)

    # Cleanup
    env.close()
    simulation_app.close()


def train_with_rsl_rl(env, args):
    """Train using RSL-RL library.

    Args:
        env: QuadrupedMPCEnv instance.
        args: Command line arguments.
    """
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic

    print("\n" + "=" * 60)
    print("Training with RSL-RL PPO")
    print("=" * 60)

    # Create actor-critic network
    actor_critic = ActorCritic(
        num_actor_obs=env.cfg.num_observations,
        num_critic_obs=env.cfg.num_observations,
        num_actions=env.cfg.num_actions,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    ).to(env.device)

    # Create PPO algorithm
    ppo = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=3e-4,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        device=env.device,
    )

    # Create runner
    runner = OnPolicyRunner(
        env=env,
        algorithm=ppo,
        num_steps_per_env=24,
        save_interval=50,
        log_dir=args.log_dir,
    )

    # Resume from checkpoint if specified
    if args.resume and args.checkpoint:
        runner.load(args.checkpoint)
        print(f"Resumed from: {args.checkpoint}")

    # Training loop
    print(f"\nStarting training for {args.max_iterations} iterations...")

    for i in range(args.max_iterations):
        runner.learn(num_learning_iterations=1)

        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{args.max_iterations}")

        # Video recording
        if args.video and (i + 1) % args.video_interval == 0:
            runner.record_video(f"iteration_{i + 1}")

    # Save final model
    final_path = os.path.join(args.log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"\nFinal model saved to: {final_path}")


def train_simple(env, args):
    """Simple training loop without RSL-RL.

    Uses random actions for testing the environment.

    Args:
        env: QuadrupedMPCEnv instance.
        args: Command line arguments.
    """
    import torch

    print("\n" + "=" * 60)
    print("Simple Training Loop (No RSL-RL)")
    print("=" * 60)

    # Reset environment
    obs = env.reset()

    total_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    episode_count = 0

    print(f"\nRunning for {args.max_iterations * 100} steps...")

    for iteration in range(args.max_iterations):
        for step in range(100):  # 100 steps per iteration
            # Random actions (normalized to [-1, 1])
            actions = torch.rand(
                env.num_envs, env.cfg.num_actions, device=env.device
            ) * 2 - 1

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            total_rewards += rewards
            episode_lengths += 1

            # Handle episode resets
            dones = terminated | truncated
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

                for idx in done_indices:
                    episode_count += 1
                    print(
                        f"Episode {episode_count}: "
                        f"reward={total_rewards[idx].item():.2f}, "
                        f"length={episode_lengths[idx].item():.0f}"
                    )

                total_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

        # Print iteration summary
        if (iteration + 1) % 10 == 0:
            print(f"\nIteration {iteration + 1}/{args.max_iterations}")
            print(f"  Episodes completed: {episode_count}")

    print(f"\nTraining complete!")
    print(f"Total episodes: {episode_count}")


if __name__ == "__main__":
    main()
