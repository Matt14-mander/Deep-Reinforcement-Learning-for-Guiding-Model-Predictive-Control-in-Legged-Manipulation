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
    python scripts/train_quadruped_mpc.py --resume --load_run <run_name>
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Train Quadruped MPC agent")
parser.add_argument(
    "--num_envs", type=int, default=32,
    help="Number of parallel environments (limited by CPU MPC)"
)
parser.add_argument(
    "--max_iterations", type=int, default=500,
    help="Maximum training iterations"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--video", action="store_true", help="Record training videos")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
parser.add_argument("--video_interval", type=int, default=500, help="Steps between videos")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Run directory to load")
parser.add_argument("--load_checkpoint", type=str, default="model_*.pt", help="Checkpoint pattern")
parser.add_argument(
    "--gait", type=str, default="trot",
    choices=["trot", "walk", "pace", "bound"],
    help="Gait type (default: trot)",
)

# AppLauncher arguments
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
import time
from datetime import datetime
from pathlib import Path

import torch

# Add source to path
SOURCE_DIR = Path(__file__).parent.parent / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(SOURCE_DIR))

# Import environment
from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

# Import RSL-RL
try:
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    RSL_RL_AVAILABLE = True
except ImportError:
    RSL_RL_AVAILABLE = False
    print("WARNING: rsl-rl not available. Install with: pip install rsl-rl-lib")


def create_ppo_config(env_cfg: QuadrupedMPCEnvCfg):
    """Create PPO algorithm configuration.

    Args:
        env_cfg: Environment configuration.

    Returns:
        Dictionary with PPO configuration.
    """
    return {
        "seed": args_cli.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_steps_per_env": 24,  # Rollout length
        "max_iterations": args_cli.max_iterations,
        "empirical_normalization": True,
        # Observation groups (required by new RSL-RL)
        "obs_groups": {},
        # PPO algorithm parameters
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
        # Logging
        "save_interval": 100,
        "log_interval": 10,
        "experiment_name": "quadruped_mpc_bezier",
        "run_name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }


def train_with_rsl_rl(env_cfg: QuadrupedMPCEnvCfg, log_dir: str):
    """Train using RSL-RL library.

    Args:
        env_cfg: Environment configuration.
        log_dir: Directory for logging.
    """
    # Create environment
    env = QuadrupedMPCEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Create PPO configuration
    ppo_cfg = create_ppo_config(env_cfg)

    # Create runner
    runner = OnPolicyRunner(
        env,
        ppo_cfg,
        log_dir=log_dir,
        device=ppo_cfg["device"],
    )

    # Resume if requested
    if args_cli.resume and args_cli.load_run:
        checkpoint_path = os.path.join(
            log_dir, "..", args_cli.load_run, args_cli.load_checkpoint
        )
        import glob
        checkpoints = glob.glob(checkpoint_path)
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            print(f"Resuming from: {latest}")
            runner.load(latest)

    # Train
    print(f"Starting training for {args_cli.max_iterations} iterations...")
    start_time = time.time()

    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    env.close()


def main():
    """Main training function."""
    print("=" * 60)
    print("Quadruped MPC Bezier Trajectory Training")
    print("=" * 60)

    # Create environment configuration
    env_cfg = QuadrupedMPCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.gait_type = args_cli.gait

    print(f"Number of environments: {env_cfg.scene.num_envs}")
    print(f"Gait type: {env_cfg.gait_type}")
    print(f"Episode length: {env_cfg.episode_length_s}s")
    print(f"Observation dim: {env_cfg.observation_space}")
    print(f"Action dim: {env_cfg.action_space}")
    print(f"MPC rate: {1.0 / env_cfg.mpc_dt:.0f} Hz")
    print(f"RL policy rate: {1.0 / (env_cfg.mpc_dt * env_cfg.rl_policy_period):.0f} Hz")

    # Setup logging directory
    log_root = os.path.join(os.path.dirname(__file__), "..", "logs", "quadruped_mpc")
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    # Train
    if RSL_RL_AVAILABLE:
        train_with_rsl_rl(env_cfg, log_dir)
    else:
        print("ERROR: RSL-RL is required for training. Install with: pip install rsl-rl-lib")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
