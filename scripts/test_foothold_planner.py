#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for FootholdPlanner visualization.

This script demonstrates how the FootholdPlanner computes foot landing
positions from a CoM Bezier trajectory. It visualizes:
1. The CoM trajectory (curved path)
2. Foot landing positions for each swing phase
3. Swing trajectories for each foot

This is a key test for curve walking: inner legs should take shorter
steps than outer legs during turns.

Usage:
    python scripts/test_foothold_planner.py

Requirements:
    - matplotlib for visualization
    - numpy
"""

import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Skipping visualization.")

import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source", "RL_Bezier_MPC"))

from RL_Bezier_MPC.trajectory import BezierTrajectoryGenerator
from RL_Bezier_MPC.gait import GaitScheduler, FootholdPlanner, ContactSequence


def create_curved_com_trajectory(
    start_pos: np.ndarray,
    turn_angle: float = np.pi / 4,  # 45 degree turn
    forward_distance: float = 1.0,
    num_waypoints: int = 75,
) -> tuple:
    """Create a curved CoM trajectory using Bezier curves.

    Args:
        start_pos: Starting position (3,).
        turn_angle: Angle of turn in radians.
        forward_distance: Forward distance to travel.
        num_waypoints: Number of waypoints.

    Returns:
        Tuple of (trajectory, bezier_params).
    """
    trajectory_gen = BezierTrajectoryGenerator(
        degree=3,
        state_dim=3,
        max_displacement=2.0,
    )

    # Create curved trajectory by placing control points off the straight line
    # P0 = start (offset = 0)
    # P1 = forward + slight turn
    # P2 = more forward + more turn
    # P3 = final position with turn

    # Direction vectors
    forward = np.array([forward_distance, 0.0, 0.0])
    turn_offset = np.array([0.0, forward_distance * np.sin(turn_angle), 0.0])

    params = np.zeros(12)
    params[0:3] = 0.0  # P0
    params[3:6] = forward / 3  # P1 - 1/3 forward
    params[6:9] = 2 * forward / 3 + turn_offset / 2  # P2 - 2/3 forward + half turn
    params[9:12] = forward + turn_offset  # P3 - full forward + turn

    trajectory = trajectory_gen.params_to_waypoints(
        params=params,
        dt=0.02,  # 50 Hz
        horizon=1.5,  # 1.5 seconds
        start_position=start_pos,
    )

    return trajectory, params


def test_foothold_planner():
    """Main test function for FootholdPlanner."""
    print("=" * 60)
    print("FootholdPlanner Test")
    print("=" * 60)

    # Initialize components
    gait_scheduler = GaitScheduler()
    foothold_planner = FootholdPlanner(
        hip_offsets={
            "LF": np.array([+0.2, +0.1, 0.0]),
            "RF": np.array([+0.2, -0.1, 0.0]),
            "LH": np.array([-0.2, +0.1, 0.0]),
            "RH": np.array([-0.2, -0.1, 0.0]),
        },
        step_height=0.05,
    )

    # Create curved CoM trajectory
    start_pos = np.array([0.0, 0.0, 0.35])  # Standing height
    com_trajectory, bezier_params = create_curved_com_trajectory(
        start_pos=start_pos,
        turn_angle=np.pi / 4,  # 45 degree turn
        forward_distance=1.0,
    )

    print(f"\nCoM trajectory shape: {com_trajectory.shape}")
    print(f"Start position: {start_pos}")
    print(f"End position: {com_trajectory[-1]}")

    # Generate gait sequence (trot)
    contact_sequence = gait_scheduler.generate(
        gait_type="trot",
        step_duration=0.15,
        support_duration=0.05,
        num_cycles=4,
    )

    print(f"\nContact sequence: {len(contact_sequence)} phases")
    print(f"Total duration: {contact_sequence.total_duration:.2f}s")

    # Initial foot positions (standing)
    initial_foot_positions = foothold_planner.get_footholds_at_time(
        com_position=start_pos,
        heading=0.0,  # Initially facing forward (+x)
    )

    print("\nInitial foot positions:")
    for name, pos in initial_foot_positions.items():
        print(f"  {name}: {pos}")

    # Plan footholds
    foothold_plans = foothold_planner.plan_footholds(
        com_trajectory=com_trajectory,
        contact_sequence=contact_sequence,
        current_foot_positions=initial_foot_positions,
        dt=0.02,
    )

    # Analyze step lengths
    step_lengths = foothold_planner.compute_step_lengths(foothold_plans)

    print("\nStep lengths by foot:")
    for foot_name, lengths in step_lengths.items():
        if lengths:
            print(f"  {foot_name}: {[f'{l:.3f}m' for l in lengths]}")

    # Check curve walking behavior
    # During left turn, left feet should take shorter steps
    if step_lengths["LF"] and step_lengths["RF"]:
        avg_left = np.mean([
            np.mean(step_lengths["LF"]),
            np.mean(step_lengths["LH"])
        ])
        avg_right = np.mean([
            np.mean(step_lengths["RF"]),
            np.mean(step_lengths["RH"])
        ])

        print(f"\nCurve walking analysis:")
        print(f"  Average left foot step:  {avg_left:.4f}m")
        print(f"  Average right foot step: {avg_right:.4f}m")

        if avg_left < avg_right:
            print("  ✓ Left feet take shorter steps (correct for left turn)")
        else:
            print("  ! Expected left feet to take shorter steps")

    # Visualization
    if HAS_MATPLOTLIB:
        visualize_footholds(com_trajectory, foothold_plans, initial_foot_positions)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    return foothold_plans


def visualize_footholds(
    com_trajectory: np.ndarray,
    foothold_plans: dict,
    initial_foot_positions: dict,
):
    """Visualize CoM trajectory and footholds.

    Args:
        com_trajectory: CoM waypoints (T, 3).
        foothold_plans: Output from FootholdPlanner.
        initial_foot_positions: Initial standing foot positions.
    """
    fig = plt.figure(figsize=(15, 5))

    # Bird's eye view (XY plane)
    ax1 = fig.add_subplot(131)
    ax1.set_title("Bird's Eye View (XY)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot CoM trajectory
    ax1.plot(com_trajectory[:, 0], com_trajectory[:, 1], "b-", linewidth=2, label="CoM")

    # Colors for each foot
    foot_colors = {
        "LF": "green",
        "RF": "red",
        "LH": "cyan",
        "RH": "orange",
    }

    # Plot initial foot positions
    for foot_name, pos in initial_foot_positions.items():
        ax1.scatter(pos[0], pos[1], c=foot_colors[foot_name], s=100, marker="o", alpha=0.5)

    # Plot footholds
    for foot_name, plans in foothold_plans.items():
        color = foot_colors[foot_name]
        for i, plan in enumerate(plans):
            # Landing position
            ax1.scatter(plan.end_pos[0], plan.end_pos[1], c=color, s=80, marker="x")

            # Swing trajectory
            if plan.trajectory is not None:
                ax1.plot(
                    plan.trajectory[:, 0],
                    plan.trajectory[:, 1],
                    c=color,
                    alpha=0.4,
                    linewidth=1,
                )

    ax1.legend()

    # Side view (XZ plane)
    ax2 = fig.add_subplot(132)
    ax2.set_title("Side View (XZ)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.grid(True, alpha=0.3)

    # Plot CoM trajectory
    ax2.plot(com_trajectory[:, 0], com_trajectory[:, 2], "b-", linewidth=2, label="CoM")

    # Plot swing trajectories
    for foot_name, plans in foothold_plans.items():
        color = foot_colors[foot_name]
        for plan in plans:
            if plan.trajectory is not None:
                ax2.plot(
                    plan.trajectory[:, 0],
                    plan.trajectory[:, 2],
                    c=color,
                    alpha=0.6,
                    linewidth=1,
                )

    ax2.legend()

    # 3D view
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_title("3D View")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")

    # Plot CoM trajectory
    ax3.plot3D(
        com_trajectory[:, 0],
        com_trajectory[:, 1],
        com_trajectory[:, 2],
        "b-",
        linewidth=2,
        label="CoM",
    )

    # Plot swing trajectories
    for foot_name, plans in foothold_plans.items():
        color = foot_colors[foot_name]
        for plan in plans:
            if plan.trajectory is not None:
                ax3.plot3D(
                    plan.trajectory[:, 0],
                    plan.trajectory[:, 1],
                    plan.trajectory[:, 2],
                    c=color,
                    alpha=0.6,
                    linewidth=1,
                )

            # Landing points
            ax3.scatter(
                plan.end_pos[0],
                plan.end_pos[1],
                plan.end_pos[2],
                c=color,
                s=50,
                marker="x",
            )

    ax3.legend()

    plt.tight_layout()
    plt.savefig("foothold_planner_visualization.png", dpi=150)
    print("\nVisualization saved to: foothold_planner_visualization.png")
    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test FootholdPlanner")
    parser.add_argument(
        "--gait",
        type=str,
        default="trot",
        choices=["trot", "walk", "pace", "bound"],
        help="Gait type to test",
    )
    parser.add_argument(
        "--turn-angle",
        type=float,
        default=45.0,
        help="Turn angle in degrees",
    )
    args = parser.parse_args()

    test_foothold_planner()


if __name__ == "__main__":
    main()
