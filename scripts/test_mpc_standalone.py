#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone MPC diagnostic test: Does Crocoddyl MPC work inside Isaac Lab?

This script bypasses the RL policy entirely and feeds a FIXED straight-line
CoM reference trajectory to the MPC controller, running in the FULL Isaac Lab
physics simulation. This isolates whether the problem is:
  (A) MPC ↔ Isaac Lab integration (joint mapping, torque application, physics mismatch)
  (B) RL training (bad trajectories, reward shaping, etc.)

If the robot FALLS in this test → problem is (A), fix MPC integration.
If the robot WALKS stably → problem is (B), focus on RL training.

Usage:
    python scripts/test_mpc_standalone.py --num_envs 1 --headless
    python scripts/test_mpc_standalone.py --num_envs 1 --headless --max_steps 500
    python scripts/test_mpc_standalone.py --num_envs 1 --headless --mode stand
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Standalone MPC Diagnostic Test")

parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of parallel environments (default: 1)",
)
parser.add_argument(
    "--max_steps", type=int, default=500,
    help="Maximum MPC steps to run (default: 500, = 10 seconds at 50Hz)",
)
parser.add_argument(
    "--mode", type=str, default="walk",
    choices=["stand", "walk", "walk_fast"],
    help="Test mode: stand (stay in place), walk (slow forward), walk_fast (faster)",
)
parser.add_argument(
    "--plot_dir", type=str, default="plots/mpc_diagnostic",
    help="Directory to save diagnostic plots",
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add source to path
SOURCE_DIR = Path(__file__).parent.parent / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(SOURCE_DIR))

from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg


def generate_fixed_bezier_params(mode: str = "walk") -> np.ndarray:
    """Generate fixed Bezier control point parameters for testing.

    These are hand-crafted to produce a straight-line forward trajectory,
    similar to what a well-trained RL policy should output.

    Args:
        mode: "stand" = stay in place, "walk" = slow forward, "walk_fast" = faster

    Returns:
        Bezier parameter array (12D): 4 control points × 3D
        Values are in the DENORMALIZED action space (meters offset).
    """
    if mode == "stand":
        # P0 = [0,0,0], P1-P3 = very small offsets → robot stays in place
        params = np.array([
            0.0, 0.0, 0.0,    # P0: start offset (always zero)
            0.05, 0.0, 0.0,   # P1: tiny forward
            0.05, 0.0, 0.0,   # P2: tiny forward
            0.05, 0.0, 0.0,   # P3: end (barely moves)
        ])
    elif mode == "walk":
        # Gentle forward trajectory (~0.5m over 1.5s = 0.33 m/s)
        params = np.array([
            0.0, 0.0, 0.0,     # P0: start
            0.15, 0.0, 0.0,    # P1: control point
            0.35, 0.0, 0.0,    # P2: control point
            0.50, 0.0, 0.0,    # P3: end (~0.5m forward)
        ])
    elif mode == "walk_fast":
        # Moderate forward trajectory (~1.0m over 1.5s = 0.67 m/s)
        params = np.array([
            0.0, 0.0, 0.0,     # P0: start
            0.30, 0.0, 0.0,    # P1: control point
            0.70, 0.0, 0.0,    # P2: control point
            1.00, 0.0, 0.0,    # P3: end (~1.0m forward)
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return params


def normalize_params_to_actions(params: np.ndarray, action_low: np.ndarray,
                                 action_high: np.ndarray) -> np.ndarray:
    """Convert denormalized Bezier params to normalized [-1, 1] action space.

    The environment expects actions in [-1, 1] and denormalizes internally.
    So we need to reverse the denormalization:
        denorm = 0.5 * (action + 1) * (high - low) + low
        action = 2 * (denorm - low) / (high - low) - 1

    Args:
        params: Denormalized Bezier parameters (12D)
        action_low: Lower bounds of action space
        action_high: Upper bounds of action space

    Returns:
        Normalized actions in [-1, 1]
    """
    action_range = action_high - action_low
    # Avoid division by zero
    action_range = np.where(np.abs(action_range) < 1e-8, 1.0, action_range)
    normalized = 2.0 * (params - action_low) / action_range - 1.0
    return np.clip(normalized, -1.0, 1.0)


def plot_diagnostic(positions, orientations, target_pos, rewards,
                    joint_torques_log, joint_positions_log, mpc_converged_log,
                    mpc_costs_log, mode, save_dir):
    """Generate comprehensive 6-panel diagnostic plot.

    Args:
        positions: (T, 3) CoM positions
        orientations: (T, 4) quaternions (w, x, y, z)
        target_pos: (3,) target position
        rewards: (T,) per-step reward
        joint_torques_log: (T, 12) applied joint torques
        joint_positions_log: (T, 12) joint positions
        mpc_converged_log: (T,) bool convergence flags
        mpc_costs_log: (T,) MPC costs
        mode: test mode string
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    T = len(positions)
    time_steps = np.arange(T) * 0.02  # MPC dt = 0.02s

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"MPC Standalone Diagnostic | Mode: {mode} | {T} steps ({T*0.02:.1f}s)\n"
        f"Final height: {positions[-1, 2]:.3f}m | "
        f"Total reward: {np.sum(rewards):.2f} | "
        f"MPC convergence: {np.mean(mpc_converged_log)*100:.0f}%",
        fontsize=13
    )

    # --- Panel 1: XY Trajectory ---
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], "b-", linewidth=1.5, label="CoM path")
    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="Start")
    ax.plot(positions[-1, 0], positions[-1, 1], "rs", markersize=10, label="End")
    ax.plot(target_pos[0], target_pos[1], "r*", markersize=15, label="Target")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory (Bird's Eye)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Height (Z) over time ---
    ax = axes[0, 1]
    ax.plot(time_steps, positions[:, 2], "b-", linewidth=1.5, label="CoM height")
    ax.axhline(y=0.4, color="g", linestyle="--", alpha=0.5, label="Standing (0.40m)")
    ax.axhline(y=0.12, color="r", linestyle="--", alpha=0.5, label="Fall threshold (0.12m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("CoM Height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Body Orientation ---
    ax = axes[1, 0]
    rolls, pitches, yaws = [], [], []
    for q in orientations:
        w, x, y, z = q
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
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
    ax.set_title("Body Orientation (Roll/Pitch/Yaw)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Joint Torques ---
    ax = axes[1, 1]
    if len(joint_torques_log) > 0:
        torques = np.array(joint_torques_log)
        leg_names = ["FR_hip", "FR_thigh", "FR_calf",
                     "FL_hip", "FL_thigh", "FL_calf",
                     "RR_hip", "RR_thigh", "RR_calf",
                     "RL_hip", "RL_thigh", "RL_calf"]
        # Plot by leg (average of 3 joints per leg)
        t_torques = np.arange(len(torques)) * 0.02
        colors = ["red", "blue", "green", "orange"]
        for leg_idx, (leg_name, color) in enumerate(
            zip(["FR", "FL", "RR", "RL"], colors)):
            leg_torques = torques[:, leg_idx*3:(leg_idx+1)*3]
            ax.plot(t_torques, np.linalg.norm(leg_torques, axis=1),
                    color=color, linewidth=0.8, alpha=0.7, label=f"{leg_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque magnitude (N·m)")
        ax.set_title("Joint Torques by Leg")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Joint Positions ---
    ax = axes[2, 0]
    if len(joint_positions_log) > 0:
        jpos = np.array(joint_positions_log)
        t_jpos = np.arange(len(jpos)) * 0.02
        # Isaac Lab joint order: FR_hip, FL_hip, RR_hip, RL_hip, FR_thigh, ...
        # Plot hip, thigh, calf averages across legs
        for j in range(min(12, jpos.shape[1])):
            ax.plot(t_jpos, jpos[:, j], linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint position (rad)")
        ax.set_title("Joint Positions (all 12 joints)")
    ax.grid(True, alpha=0.3)

    # --- Panel 6: MPC Cost & Convergence ---
    ax = axes[2, 1]
    if len(mpc_costs_log) > 0:
        t_mpc = np.arange(len(mpc_costs_log)) * 0.02
        ax.semilogy(t_mpc, np.clip(mpc_costs_log, 1e-6, None),
                    "b-", linewidth=1, label="MPC cost")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MPC Cost (log scale)")
        ax.set_title(f"MPC Cost (convergence rate: {np.mean(mpc_converged_log)*100:.0f}%)")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"mpc_diagnostic_{mode}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDiagnostic plot saved to: {save_path}")


def main():
    """Main MPC diagnostic test."""
    print("=" * 70)
    print("MPC STANDALONE DIAGNOSTIC TEST")
    print("=" * 70)
    print(f"Mode: {args_cli.mode}")
    print(f"Max steps: {args_cli.max_steps} ({args_cli.max_steps * 0.02:.1f}s)")
    print(f"Purpose: Test if MPC can control Go2 in Isaac Lab WITHOUT RL")
    print("=" * 70)

    # Create environment configuration
    env_cfg = QuadrupedMPCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.fix_gait_params = False  # Fixed gait for diagnostic
    env_cfg.mpc_verbose = True  # Enable verbose MPC debugging (first 5 solves)

    # Force target to be straight ahead (simple test)
    env_cfg.target_pos_range = (2.0, 2.0, 0.0, 0.0, 0.0, 0.0)  # Fixed at x=2.0

    # Create environment
    print("\nCreating environment...")
    env = QuadrupedMPCEnv(cfg=env_cfg)
    device = env.device

    # Get action bounds from environment
    action_low = env.action_low.cpu().numpy()
    action_high = env.action_high.cpu().numpy()
    print(f"Action bounds low:  {action_low}")
    print(f"Action bounds high: {action_high}")

    # Generate fixed Bezier parameters
    bezier_params = generate_fixed_bezier_params(args_cli.mode)
    print(f"\nFixed Bezier params (denormalized): {bezier_params}")

    # Convert to normalized actions [-1, 1]
    normalized_actions = normalize_params_to_actions(
        bezier_params, action_low, action_high
    )
    print(f"Normalized actions: {normalized_actions}")

    # Verify denormalization round-trip
    denorm_check = 0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low
    print(f"Denorm round-trip:  {denorm_check}")
    print(f"Round-trip error:   {np.max(np.abs(denorm_check - bezier_params)):.6f}")

    # Create action tensor (same for all steps — fixed trajectory)
    action_tensor = torch.tensor(
        normalized_actions, dtype=torch.float32, device=device
    ).unsqueeze(0).expand(args_cli.num_envs, -1)

    # Reset environment
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION")
    print("=" * 70)
    obs, _ = env.reset()

    # Print initial state
    robot_data = env.robot.data
    init_pos = robot_data.root_pos_w[0].cpu().numpy()
    init_quat = robot_data.root_quat_w[0].cpu().numpy()
    print(f"Initial position: {init_pos}")
    print(f"Initial orientation (quat): {init_quat}")

    # Log buffers
    positions_log = []
    orientations_log = []
    rewards_log = []
    joint_torques_log = []
    joint_positions_log = []
    mpc_converged_log = []
    mpc_costs_log = []

    # Run simulation
    for step in range(args_cli.max_steps):
        # Apply FIXED actions (bypassing RL)
        obs, rewards, terminated, truncated, info = env.step(action_tensor)

        # Print detailed MPC info for first 3 steps
        if step < 3 and hasattr(env, 'last_mpc_solutions') and env.last_mpc_solutions[0] is not None:
            sol = env.last_mpc_solutions[0]
            print(f"\n  [Step {step}] MPC: converged={sol.converged}, "
                  f"iters={sol.iterations}, cost={sol.cost:.1f}, "
                  f"solve_time={sol.solve_time*1000:.0f}ms", flush=True)
            print(f"    u[0]: [{', '.join(f'{v:.2f}' for v in sol.control[:6])}...] "
                  f"|u|={np.linalg.norm(sol.control):.2f}", flush=True)

        # Log data (env 0 only)
        pos = robot_data.root_pos_w[0].cpu().numpy().copy()
        quat = robot_data.root_quat_w[0].cpu().numpy().copy()
        jpos = robot_data.joint_pos[0].cpu().numpy().copy()

        positions_log.append(pos)
        orientations_log.append(quat)
        rewards_log.append(rewards[0].cpu().item())
        joint_positions_log.append(jpos[:12])

        # Log MPC info
        if hasattr(env, '_pending_joint_efforts'):
            torques = env._pending_joint_efforts[0].cpu().numpy().copy()
            joint_torques_log.append(torques[:12])
        if hasattr(env, '_last_mpc_converged'):
            mpc_converged_log.append(env._last_mpc_converged[0])
        if hasattr(env, '_last_mpc_costs'):
            mpc_costs_log.append(env._last_mpc_costs[0])

        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            height = pos[2]
            # Compute pitch from quaternion
            w, x, y, z = quat
            sinp = 2.0 * (w * y - z * x)
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = np.degrees(np.arcsin(sinp))

            conv_rate = np.mean(mpc_converged_log[-50:]) * 100 if mpc_converged_log else 0
            print(f"  Step {step+1:4d}/{args_cli.max_steps}: "
                  f"pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}], "
                  f"pitch={pitch:+.1f}°, "
                  f"reward={rewards[0].item():+.3f}, "
                  f"MPC conv={conv_rate:.0f}%")

        # Check termination
        dones = terminated | truncated
        if dones.any():
            print(f"\n  *** TERMINATED at step {step + 1} "
                  f"(terminated={terminated[0].item()}, truncated={truncated[0].item()}) ***")
            break

    # Convert logs to arrays
    positions_arr = np.array(positions_log)
    orientations_arr = np.array(orientations_log)
    rewards_arr = np.array(rewards_log)
    mpc_converged_arr = np.array(mpc_converged_log) if mpc_converged_log else np.array([])
    mpc_costs_arr = np.array(mpc_costs_log) if mpc_costs_log else np.array([])

    # Get target position
    target_pos = env.target_positions[0].cpu().numpy()

    # ========================= DIAGNOSTIC SUMMARY =========================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    T = len(positions_arr)
    final_height = positions_arr[-1, 2]
    min_height = np.min(positions_arr[:, 2])
    max_height = np.max(positions_arr[:, 2])

    # Compute pitch over time
    pitches = []
    for q in orientations_arr:
        w, x, y, z = q
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitches.append(np.degrees(np.arcsin(sinp)))
    pitches = np.array(pitches)

    print(f"Test mode:         {args_cli.mode}")
    print(f"Steps completed:   {T} / {args_cli.max_steps} ({T * 0.02:.1f}s)")
    print(f"Final position:    [{positions_arr[-1, 0]:+.3f}, {positions_arr[-1, 1]:+.3f}, {positions_arr[-1, 2]:.3f}]")
    print(f"Target position:   [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    print(f"Height range:      [{min_height:.3f}, {max_height:.3f}]m (standing: 0.40m)")
    print(f"Pitch range:       [{np.min(pitches):.1f}°, {np.max(pitches):.1f}°]")
    print(f"Total reward:      {np.sum(rewards_arr):.2f}")

    if len(mpc_converged_arr) > 0:
        print(f"MPC convergence:   {np.mean(mpc_converged_arr)*100:.1f}%")

    # Distance traveled
    x_traveled = positions_arr[-1, 0] - positions_arr[0, 0]
    y_traveled = positions_arr[-1, 1] - positions_arr[0, 1]
    total_dist = np.sqrt(x_traveled**2 + y_traveled**2)
    print(f"X displacement:    {x_traveled:+.3f}m")
    print(f"Y displacement:    {y_traveled:+.3f}m")
    print(f"Total distance:    {total_dist:.3f}m")

    # Check joint torques
    if len(joint_torques_log) > 0:
        torques_arr = np.array(joint_torques_log)
        max_torque = np.max(np.abs(torques_arr))
        mean_torque = np.mean(np.abs(torques_arr))
        all_zero = np.allclose(torques_arr, 0.0, atol=1e-6)
        print(f"Max |torque|:      {max_torque:.3f} N·m")
        print(f"Mean |torque|:     {mean_torque:.3f} N·m")
        if all_zero:
            print("  *** WARNING: ALL TORQUES ARE ZERO — MPC not generating commands! ***")

    # ========================= DIAGNOSIS =========================
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    fell = final_height < 0.15
    pitch_collapsed = np.max(np.abs(pitches)) > 40

    if fell and pitch_collapsed:
        print("🔴 ROBOT FELL — Pitch collapsed and height dropped below 0.15m")
        print("   → MPC cannot stabilize the robot in Isaac Lab physics.")
        print("   → Problem is MPC ↔ Isaac Lab integration.")
        print("   Possible causes:")
        print("     1. Joint ordering still wrong (Isaac Lab vs Pinocchio)")
        print("     2. Torque magnitudes too small/large for Isaac Lab physics")
        print("     3. URDF/USD model mismatch between Pinocchio and Isaac Lab")
        print("     4. Friction/contact parameters mismatch")
        print("     5. Coordinate frame conventions differ")
    elif fell:
        print("🟡 ROBOT FELL (but pitch was ok) — Height dropped too low")
        print("   → Possible leg coordination issue or insufficient support forces.")
    elif pitch_collapsed:
        print("🟡 ROBOT TILTED — Pitch exceeded 40° but didn't fully fall")
        print("   → MPC is partially working but orientation control is weak.")
    elif T < args_cli.max_steps * 0.8:
        print("🟡 EARLY TERMINATION — Robot survived but terminated early")
        print(f"   → Lasted {T * 0.02:.1f}s out of {args_cli.max_steps * 0.02:.1f}s")
    else:
        if args_cli.mode == "stand":
            if abs(x_traveled) < 0.3 and abs(y_traveled) < 0.3:
                print("🟢 ROBOT IS STANDING STABLY — MPC works for balance!")
            else:
                print("🟡 ROBOT DRIFTED while standing — MPC works but has bias")
        else:
            if x_traveled > 0.2:
                print("🟢 ROBOT IS WALKING FORWARD — MPC works in Isaac Lab!")
                print(f"   → Traveled {x_traveled:.2f}m forward in {T*0.02:.1f}s")
                print("   → Problem is in RL training, not MPC integration.")
            elif x_traveled < -0.2:
                print("🟡 ROBOT WALKED BACKWARD — MPC has direction issue")
                print("   → Check coordinate frame sign conventions.")
            else:
                print("🟡 ROBOT IS STABLE BUT NOT MOVING — MPC stabilizes but doesn't track")
                print("   → Trajectory tracking may not be working correctly.")

    if len(joint_torques_log) > 0 and all_zero:
        print("\n⚠️  ALL TORQUES ARE ZERO!")
        print("   → MPC is not generating any control commands.")
        print("   → Check MPC solver initialization, Pinocchio model loading,")
        print("     and whether CROCODDYL_AVAILABLE is True.")

    # Generate plot
    print("\n" + "=" * 70)
    plot_diagnostic(
        positions_arr, orientations_arr, target_pos, rewards_arr,
        joint_torques_log, joint_positions_log,
        mpc_converged_arr if len(mpc_converged_arr) > 0 else [True] * T,
        mpc_costs_arr if len(mpc_costs_arr) > 0 else [0.0] * T,
        args_cli.mode, args_cli.plot_dir,
    )

    # Cleanup
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
