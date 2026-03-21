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

    # Record GIF animation
    python scripts/play_quadruped_mpc.py --checkpoint model.pt --num_envs 1 --headless --record_gif

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
parser.add_argument(
    "--record_gif", action="store_true",
    help="Record an animated GIF for each episode (saved to plot_dir)",
)
parser.add_argument(
    "--gif_fps", type=int, default=10,
    help="Frames per second for GIF output (default: 10). "
         "Simulation runs at 50 Hz; every Nth step is captured where N=50/gif_fps.",
)

# Evaluation / academic logging
parser.add_argument(
    "--eval_mode", action="store_true",
    help="Enable extended data logging (joints, torques, velocities, actions, MPC stats) "
         "for academic evaluation. Saves enriched NPZ files.",
)
parser.add_argument(
    "--push_time", type=float, default=5.0,
    help="Time (seconds) at which to apply lateral disturbance force (default: 5.0). "
         "Only active when --eval_mode is set and --push_force > 0.",
)
parser.add_argument(
    "--push_force", type=float, default=0.0,
    help="Magnitude of lateral (Y-axis) disturbance force in Newtons (default: 0 = disabled). "
         "Set to e.g. 60.0 for push-recovery experiment.",
)
parser.add_argument(
    "--push_duration", type=float, default=0.2,
    help="Duration of push disturbance in seconds (default: 0.2).",
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


def render_gif_frames_batch(positions_log, orientations_log, rewards_log,
                            target_pos, capture_indices, max_steps,
                            guard_count, fps):
    """Render all GIF frames in one batch, reusing a single Figure object.

    Reusing the figure is ~5-10x faster than creating/destroying per frame
    because matplotlib figure initialisation is expensive.

    Args:
        positions_log    : full list of (3,) positions for the episode.
        orientations_log : full list of (4,) quaternions.
        rewards_log      : full list of scalar rewards.
        target_pos       : (3,) target position.
        capture_indices  : list of step indices to render.
        max_steps        : episode max steps.
        guard_count      : total MPC Guard failures in episode.
        fps              : GIF fps (used for title only).

    Returns:
        list of (H, W, 3) uint8 numpy arrays.
    """
    import io
    from PIL import Image as _Image

    positions_arr = np.array(positions_log)

    # Pre-compute full pitch/roll arrays (cheap; avoids per-frame recomputation)
    rolls_full, pitches_full = [], []
    for q in orientations_log:
        w, x, y, z = q
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        rolls_full.append(np.degrees(np.arctan2(sinr, cosr)))
        sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitches_full.append(np.degrees(np.arcsin(sinp)))
    rolls_full = np.array(rolls_full)
    pitches_full = np.array(pitches_full)
    t_full = np.arange(len(positions_log)) * 0.02

    # Compute fixed XY axis limits from full trajectory (stable across frames)
    all_x, all_y = positions_arr[:, 0], positions_arr[:, 1]
    cx = (all_x.max() + all_x.min()) / 2
    cy = (all_y.max() + all_y.min()) / 2
    pad = 0.6
    x_half = max((all_x.max() - all_x.min()) / 2, 0.5) + pad
    y_half = max((all_y.max() - all_y.min()) / 2, 0.5) + pad
    xy_lim = (cx - x_half, cx + x_half, cy - y_half, cy + y_half)

    # ── Create figure ONCE, reuse for all frames ──────────────────────────
    fig = plt.figure(figsize=(8, 4), dpi=80)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.48)
    ax_xy = fig.add_subplot(gs[:, :2])
    ax_h  = fig.add_subplot(gs[0, 2])
    ax_rp = fig.add_subplot(gs[1, 2])

    # Static elements drawn once
    ax_xy.plot(target_pos[0], target_pos[1], "r*", markersize=12, label="target",
               zorder=5)
    ax_xy.plot(positions_arr[0, 0], positions_arr[0, 1], "go", markersize=8,
               label="start", zorder=5)
    ax_xy.set_xlim(xy_lim[0], xy_lim[1])
    ax_xy.set_ylim(xy_lim[2], xy_lim[3])
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("X (m)", fontsize=8)
    ax_xy.set_ylabel("Y (m)", fontsize=8)
    ax_xy.grid(True, alpha=0.25)

    ax_h.axhline(0.28, color="g", linestyle="--", linewidth=0.8)
    ax_h.axhline(0.12, color="r", linestyle="--", linewidth=0.8)
    ax_h.set_ylim(0.0, 0.45)
    ax_h.set_xlabel("t (s)", fontsize=7)
    ax_h.set_ylabel("Z (m)", fontsize=7)
    ax_h.set_title("Height", fontsize=8)
    ax_h.tick_params(labelsize=6)
    ax_h.grid(True, alpha=0.25)

    ax_rp.axhline(0, color="k", linewidth=0.5)
    ax_rp.set_ylim(-50, 50)
    ax_rp.set_xlabel("t (s)", fontsize=7)
    ax_rp.set_ylabel("deg", fontsize=7)
    ax_rp.set_title("Pitch / Roll", fontsize=8)
    ax_rp.tick_params(labelsize=6)
    ax_rp.grid(True, alpha=0.25)

    frames = []
    n = len(capture_indices)
    for fi, idx in enumerate(capture_indices):
        if fi % 10 == 0:
            print(f"    frame {fi + 1}/{n} ...", flush=True)

        T = idx + 1
        pos = positions_arr[:T]
        t_arr = t_full[:T]
        trail_len = min(T, 80)

        # Clear dynamic artists only
        ax_xy.lines[:] = [ax_xy.lines[0], ax_xy.lines[1]]  # keep target + start
        ax_h.lines.clear()
        ax_rp.lines.clear()

        # XY trail + robot marker
        ax_xy.plot(pos[-trail_len:, 0], pos[-trail_len:, 1],
                   "b-", linewidth=1.2, alpha=0.6)
        ax_xy.plot(pos[-1, 0], pos[-1, 1], "bs", markersize=9, zorder=6)

        time_s = T * 0.02
        ax_xy.set_title(
            f"t={time_s:.1f}s  step={T}/{max_steps}  "
            f"guard={guard_count}  Σr={np.sum(rewards_log[:T]):.0f}",
            fontsize=8,
        )

        # Height
        ax_h.plot(t_arr, pos[:, 2], "b-", linewidth=1)
        ax_h.axhline(0.28, color="g", linestyle="--", linewidth=0.8)
        ax_h.axhline(0.12, color="r", linestyle="--", linewidth=0.8)

        # Pitch / Roll
        ax_rp.plot(t_arr, pitches_full[:T], "g-", linewidth=1, label="pitch")
        ax_rp.plot(t_arr, rolls_full[:T],   "r-", linewidth=1, label="roll")
        if fi == 0:
            ax_rp.legend(fontsize=6, loc="upper right")

        # Snapshot to PNG bytes → PIL → numpy
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        frame = np.array(_Image.open(buf).convert("RGB"))
        buf.close()
        frames.append(frame)

    plt.close(fig)
    return frames


def save_gif(frames, save_path, fps=10):
    """Save a list of numpy RGB frames as an animated GIF.

    Args:
        frames   : list of (H, W, 3) uint8 numpy arrays.
        save_path: output .gif file path.
        fps      : frames per second.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [GIF] PIL not found. Install with: pip install Pillow")
        return

    if not frames:
        print("  [GIF] No frames to save.")
        return

    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,          # loop forever
    )
    size_kb = os.path.getsize(save_path) / 1024
    print(f"  GIF saved: {save_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


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
    policy_fn = None
    if args_cli.checkpoint and not args_cli.random:
        try:
            import torch.nn as nn

            # Load checkpoint and inspect contents
            checkpoint = torch.load(
                args_cli.checkpoint, map_location=device, weights_only=False,
            )
            state_dict = checkpoint["model_state_dict"]

            # Print checkpoint keys for debugging
            print(f"\nCheckpoint keys: {list(state_dict.keys())[:20]}")

            # ---- Build actor network manually (version-independent) ----
            # Architecture: obs_dim → 256 → 256 → 128 → action_dim (ELU activation)
            obs_dim = env_cfg.observation_space    # 45
            action_dim = env_cfg.action_space      # 12 (fixed gait) or 15 (full)
            hidden_dims = [256, 256, 128]

            # Build sequential actor layers
            actor_layers = []
            in_dim = obs_dim
            for h_dim in hidden_dims:
                actor_layers.append(nn.Linear(in_dim, h_dim))
                actor_layers.append(nn.ELU())
                in_dim = h_dim
            actor_layers.append(nn.Linear(in_dim, action_dim))
            actor_net = nn.Sequential(*actor_layers).to(device)

            # Load actor weights from checkpoint
            # RSL-RL stores as "actor.0.weight", "actor.0.bias", "actor.2.weight", ...
            # In nn.Sequential: layer 0=Linear, 1=ELU, 2=Linear, 3=ELU, 4=Linear, 5=ELU, 6=Linear
            actor_state = {}
            for key, val in state_dict.items():
                if key.startswith("actor."):
                    # Remove "actor." prefix to match nn.Sequential keys
                    actor_state[key[len("actor."):]] = val

            if actor_state:
                actor_net.load_state_dict(actor_state)
                print(f"Loaded actor network weights ({len(actor_state)} tensors)")
            else:
                raise RuntimeError("No 'actor.*' keys found in checkpoint!")

            actor_net.eval()

            # ---- Load observation normalizer if present ----
            obs_mean = None
            obs_var = None
            # Check various possible normalizer key patterns
            for mean_key in ["actor_obs_normalizer._mean", "obs_normalizer._mean",
                             "actor_obs_normalizer.running_mean"]:
                if mean_key in state_dict:
                    obs_mean = state_dict[mean_key].to(device)
                    break

            for var_key in ["actor_obs_normalizer._var", "obs_normalizer._var",
                            "actor_obs_normalizer.running_var"]:
                if var_key in state_dict:
                    obs_var = state_dict[var_key].to(device)
                    break

            if obs_mean is not None and obs_var is not None:
                obs_std = torch.sqrt(obs_var + 1e-8)
                print(f"Loaded observation normalizer (mean shape: {obs_mean.shape})")

                def policy_fn(obs):
                    normalized = (obs - obs_mean) / obs_std
                    return actor_net(normalized)
            else:
                print("No observation normalizer found, using raw observations")
                def policy_fn(obs):
                    return actor_net(obs)

            policy = "loaded"
            print(f"Successfully loaded policy from: {args_cli.checkpoint}")

        except Exception as e:
            import traceback
            print(f"Warning: Could not load policy: {e}")
            traceback.print_exc()
            print("Using random actions instead.")
            policy = None
            policy_fn = None

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

        # Extended eval logging buffers (only populated when --eval_mode)
        joint_pos_log  = []  # (T, 12)
        joint_vel_log  = []  # (T, 12)
        torque_log     = []  # (T, 12)
        actions_log    = []  # (T, action_dim)
        lin_vel_b_log  = []  # (T, 3)  body-frame linear velocity
        ang_vel_b_log  = []  # (T, 3)  body-frame angular velocity
        mpc_cost_log   = []  # (T,)
        mpc_conv_log   = []  # (T,)  bool

        episode_reward = torch.zeros(args_cli.num_envs, device=device)
        episode_length = 0
        done = False

        max_steps = args_cli.max_steps

        # Push disturbance timing (in step indices at 50 Hz)
        push_active = args_cli.eval_mode and args_cli.push_force > 0.0
        push_start_step = int(args_cli.push_time / 0.02)
        push_end_step   = int((args_cli.push_time + args_cli.push_duration) / 0.02)
        _zero_force  = torch.zeros(args_cli.num_envs, 1, 3, device=device)
        _zero_torque = torch.zeros(args_cli.num_envs, 1, 3, device=device)

        # GIF recording: only track which steps to render (rendering deferred to after episode)
        gif_capture_every = max(1, int(50 / args_cli.gif_fps))
        gif_guard_count = 0

        while not done and episode_length < max_steps:
            # Get actions
            if policy is not None and policy_fn is not None:
                with torch.no_grad():
                    actions = policy_fn(obs_tensor)
            else:
                # Random actions
                actions = torch.rand(
                    args_cli.num_envs, env_cfg.action_space, device=device,
                ) * 2 - 1

            # Push disturbance injection (before step so physics sees the force this tick)
            if push_active:
                if push_start_step <= episode_length < push_end_step:
                    push_f = _zero_force.clone()
                    push_f[:, 0, 1] = args_cli.push_force  # Y-axis lateral push
                    try:
                        env.robot.set_external_force_and_torque(push_f, _zero_torque, body_ids=[0])
                    except Exception:
                        pass  # API may differ across Isaac Lab versions
                else:
                    try:
                        env.robot.set_external_force_and_torque(_zero_force, _zero_torque, body_ids=[0])
                    except Exception:
                        pass

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

            # Extended eval logging
            if args_cli.eval_mode:
                joint_pos_log.append(robot_data.joint_pos[0, :12].cpu().numpy().copy())
                joint_vel_log.append(robot_data.joint_vel[0, :12].cpu().numpy().copy())
                # Feedforward torques buffered in env (written to robot just before physics)
                torque_log.append(env._pending_joint_efforts[0].cpu().numpy().copy())
                actions_log.append(actions[0].cpu().numpy().copy())
                # Body-frame velocities (IsaacLab provides these directly)
                try:
                    lin_vel_b_log.append(robot_data.root_lin_vel_b[0].cpu().numpy().copy())
                    ang_vel_b_log.append(robot_data.root_ang_vel_b[0].cpu().numpy().copy())
                except AttributeError:
                    # Fallback: use world-frame velocities
                    lin_vel_b_log.append(robot_data.root_lin_vel_w[0].cpu().numpy().copy())
                    ang_vel_b_log.append(robot_data.root_ang_vel_w[0].cpu().numpy().copy())
                mpc_cost_log.append(float(env._last_mpc_costs[0]))
                mpc_conv_log.append(bool(env._last_mpc_converged[0]))

            # GIF: count guard fires for later annotation
            if args_cli.record_gif:
                try:
                    gif_guard_count = int(getattr(env, "_guard_fires_total", gif_guard_count))
                except Exception:
                    pass

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

        # Save trajectory data for offline GIF generation (avoids matplotlib/
        # Omniverse rendering conflict that causes savefig to hang).
        if args_cli.record_gif:
            os.makedirs(args_cli.plot_dir, exist_ok=True)
            npz_path = os.path.join(args_cli.plot_dir, f"episode_{episode + 1}_data.npz")
            np.savez(
                npz_path,
                positions=np.array(positions_log),
                orientations=np.array(orientations_log),
                rewards=np.array(rewards_log),
                target_pos=target_pos,
                max_steps=max_steps,
                guard_count=gif_guard_count,
                gif_fps=args_cli.gif_fps,
            )
            print(f"  Data saved to: {npz_path}  (run make_gif.py to render GIF)")

        # Extended eval data save (academic analysis)
        if args_cli.eval_mode:
            os.makedirs(args_cli.plot_dir, exist_ok=True)
            eval_npz_path = os.path.join(args_cli.plot_dir, f"episode_{episode + 1}_eval.npz")
            save_kwargs = dict(
                positions=np.array(positions_log),
                orientations=np.array(orientations_log),
                rewards=np.array(rewards_log),
                target_pos=target_pos,
                max_steps=max_steps,
                dt=np.float64(0.02),
                # Extended channels
                joint_pos=np.array(joint_pos_log),
                joint_vel=np.array(joint_vel_log),
                torques=np.array(torque_log),
                actions=np.array(actions_log),
                lin_vel_b=np.array(lin_vel_b_log),
                ang_vel_b=np.array(ang_vel_b_log),
                mpc_cost=np.array(mpc_cost_log),
                mpc_converged=np.array(mpc_conv_log),
                # Disturbance metadata
                push_time=np.float64(args_cli.push_time),
                push_force=np.float64(args_cli.push_force),
                push_duration=np.float64(args_cli.push_duration),
            )
            np.savez(eval_npz_path, **save_kwargs)
            channels = list(save_kwargs.keys())
            print(f"  Eval data saved: {eval_npz_path}  ({len(channels)} channels: {channels})")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Episodes: {len(episode_rewards_all)}")
    print(f"Average reward: {np.mean(episode_rewards_all):.2f} +/- {np.std(episode_rewards_all):.2f}")
    print(f"Average length: {np.mean(episode_lengths_all):.1f} steps")
    print(f"\nTrajectory plots saved to: {args_cli.plot_dir}/")
    if args_cli.record_gif:
        print(f"Trajectory data saved to:  {args_cli.plot_dir}/*_data.npz")
        print(f"To render GIFs, run:")
        print(f"  python scripts/make_gif.py --data_dir {args_cli.plot_dir}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
