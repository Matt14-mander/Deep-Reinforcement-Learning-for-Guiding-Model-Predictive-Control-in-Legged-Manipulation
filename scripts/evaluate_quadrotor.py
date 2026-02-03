#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified evaluation script for trained Quadrotor MPC agent.

Usage:
    python scripts/evaluate_quadrotor.py --checkpoint logs/quadrotor_mpc/2026-02-03_20-11-03/model_499.pt
    python scripts/evaluate_quadrotor.py --random
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
parser.add_argument("--video_length", type=int, default=300, help="Video length in steps")

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
        "max_iterations": 1,
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


def load_policy_direct(checkpoint_path: str, obs_dim: int, action_dim: int, device):
    """Load policy directly from checkpoint without using OnPolicyRunner."""
    print(f"Loading checkpoint directly: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Checkpoint keys: {checkpoint.keys()}")

    state_dict = checkpoint["model_state_dict"]

    # Build actor network manually (simpler, avoids RSL-RL initialization issues)
    print("Building actor network manually...")
    import torch.nn as nn

    actor = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, action_dim),
    ).to(device)

    # Load actor weights
    actor_state = {
        "0.weight": state_dict["actor.0.weight"],
        "0.bias": state_dict["actor.0.bias"],
        "2.weight": state_dict["actor.2.weight"],
        "2.bias": state_dict["actor.2.bias"],
        "4.weight": state_dict["actor.4.weight"],
        "4.bias": state_dict["actor.4.bias"],
        "6.weight": state_dict["actor.6.weight"],
        "6.bias": state_dict["actor.6.bias"],
    }
    actor.load_state_dict(actor_state)
    print("Actor weights loaded")

    # Load normalizer stats
    obs_mean = state_dict["actor_obs_normalizer._mean"].to(device)
    obs_std = state_dict["actor_obs_normalizer._std"].to(device)
    print(f"Normalizer loaded: mean shape {obs_mean.shape}, std shape {obs_std.shape}")

    actor.eval()
    print("Policy ready for inference")

    # Create inference function with normalization
    def inference_fn(obs):
        with torch.inference_mode():
            # Normalize observations
            obs_normalized = (obs - obs_mean) / (obs_std + 1e-8)
            # Get actions
            actions = actor(obs_normalized)
            return actions

    return inference_fn, actor


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
    print("Creating environment...")
    env = QuadrotorMPCEnv(
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    obs_dim = env.observation_space.shape[-1]  # 17
    action_dim = env.action_space.shape[-1]  # 12

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")

    # Create policy
    policy_nn = None

    if args_cli.random:
        print("\nUsing random policy")

        def policy(obs):
            batch_size = obs.shape[0]
            return torch.rand((batch_size, action_dim), device=device) * 2 - 1

    elif args_cli.checkpoint:
        if not os.path.exists(args_cli.checkpoint):
            print(f"ERROR: Checkpoint file not found: {args_cli.checkpoint}")
            env.close()
            return

        # Load policy directly (avoid OnPolicyRunner initialization issues)
        policy, policy_nn = load_policy_direct(
            args_cli.checkpoint, obs_dim, action_dim, device
        )
        print("Policy loaded successfully!")

    else:
        print("\nUsing straight-line policy (no checkpoint specified)")

        def policy(obs):
            batch_size = obs.shape[0]
            actions = torch.zeros((batch_size, 12), device=device)
            current_pos = obs[:, 0:3]
            target_pos = obs[:, 13:16]
            direction = target_pos - current_pos
            distance = torch.norm(direction, dim=1, keepdim=True)
            direction_normalized = direction / (distance + 1e-6)
            scale = torch.clamp(distance / 2.0, 0, 1)
            actions[:, 0:3] = 0.0
            actions[:, 3:6] = direction_normalized * scale / 3
            actions[:, 6:9] = direction_normalized * scale * 2 / 3
            actions[:, 9:12] = direction_normalized * scale
            return actions

    # Run evaluation
    print(f"\nRunning evaluation for {args_cli.num_steps} steps...")
    print("-" * 40)

    # Reset environment and get initial observations
    print("Resetting environment...")
    obs_dict, info = env.reset()
    obs = obs_dict["policy"]  # Get observation tensor
    print(f"Initial obs shape: {obs.shape}")

    total_rewards = np.zeros(args_cli.num_envs)
    episode_rewards = []
    episode_lengths = []
    current_episode_length = np.zeros(args_cli.num_envs)

    # Trajectory logging for visualization
    trajectory_data = []

    start_time = time.time()

    for step in range(args_cli.num_steps):
        # Log trajectory data for later visualization
        if args_cli.video:
            trajectory_data.append({
                "step": step,
                "obs": obs.cpu().numpy().copy(),
            })

        # Get actions from policy
        with torch.inference_mode():
            actions = policy(obs)

        # Step environment
        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"]
        dones = terminated | truncated

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
            print(
                f"Step {step + 1}/{args_cli.num_steps}, "
                f"Episodes: {len(episode_rewards)}, "
                f"FPS: {fps:.1f}"
            )

    # Save trajectory plot as video
    if args_cli.video and trajectory_data:
        print(f"\nGenerating trajectory visualization with {len(trajectory_data)} frames...")
        video_dir = os.path.join("logs", "quadrotor_mpc", "videos")
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

            # Extract positions and targets for env 0
            positions = np.array([d["obs"][0, 0:3] for d in trajectory_data])  # x,y,z
            targets = np.array([d["obs"][0, 13:16] for d in trajectory_data])  # target x,y,z

            # Create 2D trajectory plot animation
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle("Quadrotor MPC - Trajectory Evaluation", fontsize=14)

            labels = [("X", "Y", "XY Plane"), ("X", "Z", "XZ Plane"), ("Y", "Z", "YZ Plane")]
            idx_pairs = [(0, 1), (0, 2), (1, 2)]

            lines = []
            target_dots = []
            current_dots = []

            for ax, (xlabel, ylabel, title), (ix, iy) in zip(axes, labels, idx_pairs):
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

                # Set axis limits
                all_vals = np.concatenate([positions[:, [ix, iy]], targets[:, [ix, iy]]])
                margin = 0.5
                ax.set_xlim(all_vals[:, 0].min() - margin, all_vals[:, 0].max() + margin)
                ax.set_ylim(all_vals[:, 1].min() - margin, all_vals[:, 1].max() + margin)

                line, = ax.plot([], [], "b-", linewidth=1, alpha=0.7, label="Trajectory")
                target_dot, = ax.plot([], [], "r*", markersize=15, label="Target")
                current_dot, = ax.plot([], [], "go", markersize=8, label="Current")

                lines.append(line)
                target_dots.append(target_dot)
                current_dots.append(current_dot)

            axes[0].legend(loc="upper left", fontsize=8)

            step_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

            def update(frame):
                for i, (ix, iy) in enumerate(idx_pairs):
                    lines[i].set_data(positions[:frame+1, ix], positions[:frame+1, iy])
                    target_dots[i].set_data([targets[frame, ix]], [targets[frame, iy]])
                    current_dots[i].set_data([positions[frame, ix]], [positions[frame, iy]])
                dist = np.linalg.norm(positions[frame] - targets[frame])
                step_text.set_text(f"Step: {frame}/{len(trajectory_data)} | Distance to target: {dist:.3f}m")
                return lines + target_dots + current_dots + [step_text]

            anim = FuncAnimation(fig, update, frames=len(trajectory_data), interval=33, blit=True)

            # Try saving as mp4, fallback to gif
            video_path = os.path.join(video_dir, f"trajectory_{timestamp}.gif")
            anim.save(video_path, writer=PillowWriter(fps=30))
            print(f"Trajectory animation saved: {video_path}")

            # Also save static plot
            static_path = os.path.join(video_dir, f"trajectory_{timestamp}.png")
            fig_static, axes_s = plt.subplots(1, 3, figsize=(18, 5))
            fig_static.suptitle("Quadrotor MPC - Full Trajectory", fontsize=14)

            for ax, (xlabel, ylabel, title), (ix, iy) in zip(axes_s, labels, idx_pairs):
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)
                ax.plot(positions[:, ix], positions[:, iy], "b-", linewidth=1, alpha=0.7, label="Trajectory")
                ax.plot(positions[0, ix], positions[0, iy], "gs", markersize=10, label="Start")
                ax.plot(positions[-1, ix], positions[-1, iy], "go", markersize=10, label="End")
                ax.plot(targets[-1, ix], targets[-1, iy], "r*", markersize=15, label="Target")
            axes_s[0].legend(fontsize=8)
            fig_static.savefig(static_path, dpi=150, bbox_inches="tight")
            print(f"Static trajectory plot saved: {static_path}")

            plt.close("all")

        except Exception as e:
            print(f"Visualization error: {e}")
            # Fallback: save raw data
            data_path = os.path.join(video_dir, f"trajectory_{timestamp}.npz")
            np.savez(data_path,
                     positions=np.array([d["obs"][:, 0:3] for d in trajectory_data]),
                     targets=np.array([d["obs"][:, 13:16] for d in trajectory_data]))
            print(f"Raw trajectory data saved: {data_path}")

    # Print final statistics
    print("-" * 40)
    print("Evaluation Results:")

    if episode_rewards:
        print(f"  Completed episodes: {len(episode_rewards)}")
        print(
            f"  Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}"
        )
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
