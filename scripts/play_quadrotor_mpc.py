#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation script for trained Quadrotor MPC agent.

This script loads a trained policy and visualizes its performance
in the quadrotor trajectory tracking environment.

Usage:
    # Run with trained checkpoint
    python scripts/play_quadrotor_mpc.py --checkpoint path/to/model.pt

    # Run with random policy (for testing environment)
    python scripts/play_quadrotor_mpc.py --random

    # Run with specific number of episodes
    python scripts/play_quadrotor_mpc.py --checkpoint path/to/model.pt --num_episodes 10

    # Record video
    python scripts/play_quadrotor_mpc.py --checkpoint path/to/model.pt --video
"""

import os
os.environ["OMNI_KIT_ALLOW_ROOT"] = "1"
# 限制 Vulkan 内存分配
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Evaluate Quadrotor MPC agent")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to trained model checkpoint"
)
parser.add_argument(
    "--num_envs", type=int, default=4,
    help="Number of parallel environments"
)
parser.add_argument(
    "--num_episodes", type=int, default=5,
    help="Number of episodes to run"
)
parser.add_argument("--random", action="store_true", help="Use random policy")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_file", type=str, default="quadrotor_mpc.mp4", help="Video filename")

# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras
args_cli.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports
import os
import time

import numpy as np
import torch

# Add source to path
SOURCE_DIR = Path(__file__).parent.parent / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(SOURCE_DIR))

from RL_Bezier_MPC.envs import QuadrotorMPCEnv, QuadrotorMPCEnvCfg


class RandomPolicy:
    """Random policy for testing."""

    def __init__(self, action_dim: int, device: str = "cpu"):
        self.action_dim = action_dim
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        # Random actions in [-1, 1]
        return torch.rand((batch_size, self.action_dim), device=self.device) * 2 - 1


class StraightLinePolicy:
    """Policy that generates straight-line trajectories towards target."""

    def __init__(self, action_dim: int, device: str = "cpu"):
        self.action_dim = action_dim
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate Bezier params for straight line to target.

        Observation structure:
            [0:3] - position
            [3:7] - quaternion
            [7:10] - linear velocity
            [10:13] - angular velocity
            [13:16] - target position
            [16] - trajectory phase
        """
        batch_size = obs.shape[0]
        actions = torch.zeros((batch_size, self.action_dim), device=self.device)

        # Extract current position and target
        current_pos = obs[:, 0:3]
        target_pos = obs[:, 13:16]

        # Direction to target
        direction = target_pos - current_pos
        distance = torch.norm(direction, dim=1, keepdim=True)

        # Normalize and scale to action range [-1, 1]
        max_displacement = 2.0  # Should match env config
        direction_normalized = direction / (distance + 1e-6)
        scale = torch.clamp(distance / max_displacement, 0, 1)

        # Control points along straight line
        # P0 offset = 0 (always starts at current position)
        # P1 offset = 1/3 of direction
        # P2 offset = 2/3 of direction
        # P3 offset = full direction (target)

        actions[:, 0:3] = 0.0  # P0 offset (must be zero)
        actions[:, 3:6] = direction_normalized * scale / 3  # P1
        actions[:, 6:9] = direction_normalized * scale * 2 / 3  # P2
        actions[:, 9:12] = direction_normalized * scale  # P3

        return actions


def load_policy(checkpoint_path: str, obs_dim: int, action_dim: int, device: str):
    """Load trained policy from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        device: Device to load model on.

    Returns:
        Policy network.
    """
    # Try loading RSL-RL policy
    try:
        from rsl_rl.modules import ActorCritic

        # Create actor-critic with matching architecture
        policy = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=[256, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()

        print(f"Loaded policy from: {checkpoint_path}")
        return lambda obs: policy.act_inference(obs)

    except Exception as e:
        print(f"Could not load RSL-RL policy: {e}")
        print("Falling back to straight-line policy")
        return StraightLinePolicy(action_dim, device)


def run_evaluation(env, policy, num_episodes: int, record_video: bool = False):
    """Run evaluation episodes.

    Args:
        env: Environment instance.
        policy: Policy function.
        num_episodes: Number of episodes to run.
        record_video: Whether to record video.

    Returns:
        Dictionary with evaluation statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Statistics
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []

    # Video frames
    frames = [] if record_video else None

    obs, _ = env.reset()
    episode_reward = np.zeros(env.num_envs)
    episode_length = np.zeros(env.num_envs)

    completed_episodes = 0

    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("-" * 40)

    while completed_episodes < num_episodes:
        # Get action from policy
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
            else:
                obs_tensor = obs.to(device)

            actions = policy(obs_tensor)

            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(actions)

        # Record video frame
        if record_video and hasattr(env, "render"):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Accumulate statistics
        episode_reward += reward if isinstance(reward, np.ndarray) else reward.cpu().numpy()
        episode_length += 1

        # Check for episode completions
        done = terminated | truncated
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()

        for i in range(env.num_envs):
            if done[i]:
                episode_rewards.append(episode_reward[i])
                episode_lengths.append(episode_length[i])

                # Compute average tracking error from info if available
                if "tracking_error" in info:
                    tracking_errors.append(info["tracking_error"][i])

                episode_reward[i] = 0
                episode_length[i] = 0
                completed_episodes += 1

                if completed_episodes <= num_episodes:
                    print(
                        f"Episode {completed_episodes}: "
                        f"Reward = {episode_rewards[-1]:.2f}, "
                        f"Length = {episode_lengths[-1]}"
                    )

        obs = next_obs

    # Compute statistics
    stats = {
        "num_episodes": len(episode_rewards),
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
    }

    if tracking_errors:
        stats["mean_tracking_error"] = np.mean(tracking_errors)

    print("-" * 40)
    print(f"Evaluation Results ({stats['num_episodes']} episodes):")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"  Mean Episode Length: {stats['mean_length']:.1f}")
    print(f"  Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")

    if "mean_tracking_error" in stats:
        print(f"  Mean Tracking Error: {stats['mean_tracking_error']:.4f} m")

    # Save video if recorded
    if record_video and frames:
        save_video(frames, args_cli.video_file)

    return stats


def save_video(frames: list, filename: str, fps: int = 30):
    """Save frames as video file.

    Args:
        frames: List of RGB frames.
        filename: Output filename.
        fps: Frames per second.
    """
    try:
        import imageio

        print(f"\nSaving video to: {filename}")
        imageio.mimsave(filename, frames, fps=fps)
        print(f"Video saved ({len(frames)} frames)")

    except ImportError:
        print("imageio not available. Install with: pip install imageio[ffmpeg]")

        # Fallback: save frames as images
        import os
        os.makedirs("frames", exist_ok=True)
        for i, frame in enumerate(frames):
            import matplotlib.pyplot as plt
            plt.imsave(f"frames/frame_{i:04d}.png", frame)
        print(f"Saved {len(frames)} frames to frames/ directory")


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Quadrotor MPC Bezier Trajectory Evaluation")
    print("=" * 60)

    # Create environment configuration
    env_cfg = QuadrotorMPCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    print(f"Number of environments: {env_cfg.scene.num_envs}")

    # Create environment
    env = QuadrotorMPCEnv(
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else "human",
    )

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")

    # Load or create policy
    if args_cli.random:
        print("Using random policy")
        policy = RandomPolicy(action_dim, device)
    elif args_cli.checkpoint:
        policy = load_policy(args_cli.checkpoint, obs_dim, action_dim, device)
    else:
        print("Using straight-line policy (no checkpoint specified)")
        policy = StraightLinePolicy(action_dim, device)

    # Run evaluation
    try:
        stats = run_evaluation(
            env,
            policy,
            num_episodes=args_cli.num_episodes,
            record_video=args_cli.video,
        )
    finally:
        env.close()

    return stats


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()