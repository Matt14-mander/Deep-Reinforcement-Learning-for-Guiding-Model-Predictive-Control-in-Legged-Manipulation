#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone gait module test with Crocoddyl visualization.

This script tests the complete gait pipeline WITHOUT IsaacLab:
1. CoM Bezier trajectory generation
2. Gait scheduling (contact sequence)
3. Foothold planning
4. OCP construction with Crocoddyl
5. FDDP solving and visualization

Can run on MacBook or any machine with:
- Python 3.8+
- crocoddyl
- pinocchio
- example-robot-data (for robot models)
- meshcat (for visualization)
- matplotlib (for 2D plots)

Usage:
    # Basic test with matplotlib visualization
    python scripts/test_gait_standalone.py

    # With Meshcat 3D visualization
    python scripts/test_gait_standalone.py --meshcat

    # Test different gaits
    python scripts/test_gait_standalone.py --gait walk --meshcat

    # Test curved trajectory
    python scripts/test_gait_standalone.py --trajectory curve_left --meshcat

Installation on MacBook:
    pip install crocoddyl pinocchio example-robot-data meshcat matplotlib numpy
"""

import argparse
import time
import numpy as np
from typing import Optional, Tuple, Dict

# =============================================================================
# Import gait modules (no IsaacLab dependency)
# =============================================================================
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "..", "source", "RL_Bezier_MPC")
sys.path.insert(0, SOURCE_DIR)

from RL_Bezier_MPC.trajectory import BezierTrajectoryGenerator
from RL_Bezier_MPC.trajectory.bezier_foot_trajectory import BezierFootTrajectory
from RL_Bezier_MPC.gait import (
    GaitScheduler,
    FootholdPlanner,
    ContactSequence,
    ContactPhase,
)
from RL_Bezier_MPC.utils.math_utils import (
    bezier_curve,
    bezier_tangent,
    heading_from_tangent,
    rotation_matrix_z,
)

# =============================================================================
# Check optional dependencies
# =============================================================================
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import crocoddyl
    import pinocchio
    HAS_CROCODDYL = True
except ImportError:
    HAS_CROCODDYL = False
    print("Warning: crocoddyl/pinocchio not available.")
    print("Install with: pip install crocoddyl pinocchio")

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    HAS_MESHCAT = True
except ImportError:
    HAS_MESHCAT = False
    print("Warning: meshcat not available. Install with: pip install meshcat")

try:
    import example_robot_data
    HAS_ROBOT_DATA = True
except ImportError:
    HAS_ROBOT_DATA = False
    print("Warning: example-robot-data not available.")
    print("Install with: pip install example-robot-data")


# =============================================================================
# Trajectory Generation
# =============================================================================

def create_com_trajectory(
    start_pos: np.ndarray,
    trajectory_type: str = "straight",
    distance: float = 0.5,
    turn_angle: float = np.pi / 4,
    duration: float = 1.5,
    dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create CoM trajectory using Bezier curves.

    Args:
        start_pos: Starting CoM position (3,).
        trajectory_type: "straight", "curve_left", "curve_right", "s_curve".
        distance: Forward distance to travel in meters.
        turn_angle: Turn angle for curved trajectories.
        duration: Trajectory duration in seconds.
        dt: Sampling timestep.

    Returns:
        Tuple of (trajectory, bezier_params).
    """
    trajectory_gen = BezierTrajectoryGenerator(
        degree=3,
        state_dim=3,
        max_displacement=2.0,
    )

    params = np.zeros(12)

    if trajectory_type == "straight":
        # Straight line forward (+x direction)
        params[3:6] = [distance / 3, 0.0, 0.0]
        params[6:9] = [2 * distance / 3, 0.0, 0.0]
        params[9:12] = [distance, 0.0, 0.0]

    elif trajectory_type == "curve_left":
        # Curved path turning left (+y direction)
        lateral = distance * np.sin(turn_angle)
        forward = distance * np.cos(turn_angle)
        params[3:6] = [forward / 3, 0.0, 0.0]
        params[6:9] = [2 * forward / 3, lateral / 2, 0.0]
        params[9:12] = [forward, lateral, 0.0]

    elif trajectory_type == "curve_right":
        # Curved path turning right (-y direction)
        lateral = -distance * np.sin(turn_angle)
        forward = distance * np.cos(turn_angle)
        params[3:6] = [forward / 3, 0.0, 0.0]
        params[6:9] = [2 * forward / 3, lateral / 2, 0.0]
        params[9:12] = [forward, lateral, 0.0]

    elif trajectory_type == "s_curve":
        # S-curve (left then right)
        params[3:6] = [distance / 3, distance * 0.15, 0.0]
        params[6:9] = [2 * distance / 3, -distance * 0.15, 0.0]
        params[9:12] = [distance, 0.0, 0.0]

    elif trajectory_type == "circle":
        # Circular arc
        radius = distance / turn_angle
        params[3:6] = [radius * np.sin(turn_angle / 3),
                       radius * (1 - np.cos(turn_angle / 3)), 0.0]
        params[6:9] = [radius * np.sin(2 * turn_angle / 3),
                       radius * (1 - np.cos(2 * turn_angle / 3)), 0.0]
        params[9:12] = [radius * np.sin(turn_angle),
                        radius * (1 - np.cos(turn_angle)), 0.0]

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    trajectory = trajectory_gen.params_to_waypoints(
        params=params,
        dt=dt,
        horizon=duration,
        start_position=start_pos,
    )

    return trajectory, params


# =============================================================================
# Gait Pipeline Test
# =============================================================================

def test_gait_pipeline(
    gait_type: str = "trot",
    trajectory_type: str = "straight",
    distance: float = 0.5,
    duration: float = 1.5,
    standing_height: float = 0.24,
) -> Dict:
    """Test the complete gait pipeline.

    Args:
        gait_type: "trot", "walk", "pace", "bound".
        trajectory_type: "straight", "curve_left", etc.
        distance: Forward distance.
        duration: Trajectory duration.
        standing_height: Robot standing height (CoM height).

    Returns:
        Dict containing all computed data for visualization.
    """
    print("=" * 60)
    print(f"Gait Pipeline Test: {gait_type.upper()} - {trajectory_type}")
    print("=" * 60)

    # Initialize components
    gait_scheduler = GaitScheduler()
    foothold_planner = FootholdPlanner(
        hip_offsets={
            "LF": np.array([+0.15, +0.05, 0.0]),  # Solo12 dimensions
            "RF": np.array([+0.15, -0.05, 0.0]),
            "LH": np.array([-0.15, +0.05, 0.0]),
            "RH": np.array([-0.15, -0.05, 0.0]),
        },
        step_height=0.04,
    )
    foot_traj_gen = BezierFootTrajectory(step_height=0.04)

    # Create CoM trajectory
    start_pos = np.array([0.0, 0.0, standing_height])
    com_trajectory, bezier_params = create_com_trajectory(
        start_pos=start_pos,
        trajectory_type=trajectory_type,
        distance=distance,
        duration=duration,
    )

    print(f"\nCoM Trajectory:")
    print(f"  Shape: {com_trajectory.shape}")
    print(f"  Start: {com_trajectory[0]}")
    print(f"  End: {com_trajectory[-1]}")
    print(f"  Distance: {np.linalg.norm(com_trajectory[-1] - com_trajectory[0]):.3f}m")

    # Generate contact sequence
    contact_sequence = gait_scheduler.generate(
        gait_type=gait_type,
        step_duration=0.12,
        support_duration=0.04,
        num_cycles=4,
    )

    print(f"\nContact Sequence ({gait_type}):")
    print(f"  Phases: {len(contact_sequence)}")
    print(f"  Duration: {contact_sequence.total_duration:.2f}s")

    # Show gait pattern
    print(f"\n  Gait pattern:")
    for i, phase in enumerate(contact_sequence.phases[:6]):
        swing_str = ",".join(phase.swing_feet) if phase.swing_feet else "none"
        print(f"    Phase {i}: swing=[{swing_str}] dur={phase.duration:.3f}s")
    if len(contact_sequence) > 6:
        print(f"    ... ({len(contact_sequence) - 6} more phases)")

    # Get initial foot positions
    initial_foot_positions = foothold_planner.get_footholds_at_time(
        com_position=start_pos,
        heading=0.0,
    )

    print(f"\nInitial Foot Positions:")
    for name, pos in initial_foot_positions.items():
        print(f"  {name}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}]")

    # Plan footholds
    foothold_plans = foothold_planner.plan_footholds(
        com_trajectory=com_trajectory,
        contact_sequence=contact_sequence,
        current_foot_positions=initial_foot_positions,
        dt=0.02,
    )

    # Compute step lengths
    step_lengths = foothold_planner.compute_step_lengths(foothold_plans)

    print(f"\nStep Lengths:")
    for foot_name, lengths in step_lengths.items():
        if lengths:
            avg = np.mean(lengths)
            print(f"  {foot_name}: avg={avg:.4f}m, steps={len(lengths)}")

    # Analyze curve walking
    if trajectory_type in ["curve_left", "curve_right"]:
        left_avg = np.mean([
            np.mean(step_lengths["LF"]) if step_lengths["LF"] else 0,
            np.mean(step_lengths["LH"]) if step_lengths["LH"] else 0,
        ])
        right_avg = np.mean([
            np.mean(step_lengths["RF"]) if step_lengths["RF"] else 0,
            np.mean(step_lengths["RH"]) if step_lengths["RH"] else 0,
        ])

        print(f"\nCurve Walking Analysis:")
        print(f"  Left feet avg:  {left_avg:.4f}m")
        print(f"  Right feet avg: {right_avg:.4f}m")
        print(f"  Ratio: {left_avg/right_avg:.2f}" if right_avg > 0 else "")

        if trajectory_type == "curve_left" and left_avg < right_avg:
            print("  ✓ Correct: inner (left) feet take shorter steps")
        elif trajectory_type == "curve_right" and right_avg < left_avg:
            print("  ✓ Correct: inner (right) feet take shorter steps")

    # Collect results
    results = {
        "com_trajectory": com_trajectory,
        "bezier_params": bezier_params,
        "contact_sequence": contact_sequence,
        "foothold_plans": foothold_plans,
        "initial_foot_positions": initial_foot_positions,
        "step_lengths": step_lengths,
        "gait_type": gait_type,
        "trajectory_type": trajectory_type,
        "standing_height": standing_height,
    }

    return results


# =============================================================================
# Matplotlib Visualization
# =============================================================================

def visualize_matplotlib(results: Dict, save_path: Optional[str] = None):
    """Visualize gait results using matplotlib.

    Args:
        results: Output from test_gait_pipeline.
        save_path: Optional path to save figure.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualization")
        return

    com_traj = results["com_trajectory"]
    foothold_plans = results["foothold_plans"]
    initial_feet = results["initial_foot_positions"]

    fig = plt.figure(figsize=(16, 12))

    # Color scheme
    foot_colors = {
        "LF": "#2ecc71",  # Green
        "RF": "#e74c3c",  # Red
        "LH": "#3498db",  # Blue
        "RH": "#f39c12",  # Orange
    }

    # 1. Bird's eye view (XY)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Bird's Eye View (XY)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot CoM trajectory
    ax1.plot(com_traj[:, 0], com_traj[:, 1], "k-", linewidth=2.5, label="CoM", zorder=10)
    ax1.scatter(com_traj[0, 0], com_traj[0, 1], c="green", s=100, marker="o", zorder=11, label="Start")
    ax1.scatter(com_traj[-1, 0], com_traj[-1, 1], c="red", s=100, marker="*", zorder=11, label="End")

    # Plot initial foot positions
    for foot_name, pos in initial_feet.items():
        ax1.scatter(pos[0], pos[1], c=foot_colors[foot_name], s=80, marker="s",
                   alpha=0.5, edgecolors='black', linewidths=1)

    # Plot foothold trajectories and landing points
    for foot_name, plans in foothold_plans.items():
        color = foot_colors[foot_name]
        for i, plan in enumerate(plans):
            # Landing point
            ax1.scatter(plan.end_pos[0], plan.end_pos[1], c=color, s=60, marker="x", linewidths=2)

            # Swing trajectory
            if plan.trajectory is not None:
                ax1.plot(plan.trajectory[:, 0], plan.trajectory[:, 1],
                        c=color, alpha=0.4, linewidth=1.5)

            # Connect start to end
            ax1.plot([plan.start_pos[0], plan.end_pos[0]],
                    [plan.start_pos[1], plan.end_pos[1]],
                    c=color, alpha=0.2, linewidth=1, linestyle='--')

    # Legend
    for foot_name, color in foot_colors.items():
        ax1.scatter([], [], c=color, s=60, marker="x", label=foot_name)
    ax1.legend(loc="upper left", fontsize=9)

    # 2. Side view (XZ)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Side View (XZ)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.grid(True, alpha=0.3)

    # CoM trajectory
    ax2.plot(com_traj[:, 0], com_traj[:, 2], "k-", linewidth=2.5, label="CoM")

    # Swing trajectories
    for foot_name, plans in foothold_plans.items():
        color = foot_colors[foot_name]
        for plan in plans:
            if plan.trajectory is not None:
                ax2.plot(plan.trajectory[:, 0], plan.trajectory[:, 2],
                        c=color, alpha=0.6, linewidth=1.5)

    # Ground line
    x_range = [com_traj[:, 0].min() - 0.1, com_traj[:, 0].max() + 0.1]
    ax2.axhline(y=0, color='brown', linewidth=2, label='Ground')
    ax2.legend(loc="upper right", fontsize=9)

    # 3. 3D view
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title("3D View", fontsize=12, fontweight='bold')
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")

    # CoM trajectory
    ax3.plot3D(com_traj[:, 0], com_traj[:, 1], com_traj[:, 2],
              "k-", linewidth=2.5, label="CoM")

    # Swing trajectories
    for foot_name, plans in foothold_plans.items():
        color = foot_colors[foot_name]
        for plan in plans:
            if plan.trajectory is not None:
                ax3.plot3D(plan.trajectory[:, 0], plan.trajectory[:, 1], plan.trajectory[:, 2],
                          c=color, alpha=0.6, linewidth=1.5)
            # Landing point
            ax3.scatter(plan.end_pos[0], plan.end_pos[1], plan.end_pos[2],
                       c=color, s=30, marker="x")

    # Ground plane (simplified)
    ax3.set_zlim(0, results["standing_height"] * 1.5)

    # 4. Step length analysis
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Step Lengths by Foot", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Step Number")
    ax4.set_ylabel("Step Length (m)")
    ax4.grid(True, alpha=0.3)

    for foot_name, lengths in results["step_lengths"].items():
        if lengths:
            color = foot_colors[foot_name]
            steps = list(range(1, len(lengths) + 1))
            ax4.plot(steps, lengths, "o-", c=color, label=foot_name, linewidth=2, markersize=8)

    ax4.legend(loc="upper right", fontsize=9)

    plt.suptitle(f"Gait: {results['gait_type'].upper()} | Trajectory: {results['trajectory_type']}",
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


# =============================================================================
# Meshcat 3D Visualization
# =============================================================================

def visualize_meshcat(results: Dict, animate: bool = True):
    """Visualize gait results using Meshcat 3D viewer.

    Args:
        results: Output from test_gait_pipeline.
        animate: If True, animate the gait sequence.
    """
    if not HAS_MESHCAT:
        print("Meshcat not available, skipping 3D visualization")
        return

    print("\nStarting Meshcat visualization...")
    print("Open the URL in your browser to view the 3D scene")

    # Create visualizer
    vis = meshcat.Visualizer()
    vis.open()

    # Clear previous objects
    vis.delete()

    # Colors
    foot_colors = {
        "LF": 0x2ecc71,  # Green
        "RF": 0xe74c3c,  # Red
        "LH": 0x3498db,  # Blue
        "RH": 0xf39c12,  # Orange
    }

    com_traj = results["com_trajectory"]
    foothold_plans = results["foothold_plans"]
    initial_feet = results["initial_foot_positions"]
    standing_height = results["standing_height"]

    # Draw ground plane
    vis["ground"].set_object(
        g.Box([2.0, 2.0, 0.01]),
        g.MeshLambertMaterial(color=0x8B4513, opacity=0.5)
    )
    vis["ground"].set_transform(tf.translation_matrix([0.5, 0, -0.005]))

    # Draw CoM trajectory as line
    com_line_vertices = com_traj.T  # Shape: (3, N)
    vis["com_trajectory"].set_object(
        g.Line(g.PointsGeometry(com_line_vertices),
               g.LineBasicMaterial(color=0x000000, linewidth=3))
    )

    # Draw body (simplified box)
    body_size = [0.3, 0.1, 0.05]
    vis["body"].set_object(
        g.Box(body_size),
        g.MeshLambertMaterial(color=0x808080, opacity=0.8)
    )

    # Draw feet as spheres
    foot_radius = 0.015
    for foot_name, pos in initial_feet.items():
        color = foot_colors[foot_name]
        vis[f"foot_{foot_name}"].set_object(
            g.Sphere(foot_radius),
            g.MeshLambertMaterial(color=color)
        )
        vis[f"foot_{foot_name}"].set_transform(tf.translation_matrix(pos))

    # Draw foot trajectories
    for foot_name, plans in foothold_plans.items():
        color = foot_colors[foot_name]
        for i, plan in enumerate(plans):
            if plan.trajectory is not None and len(plan.trajectory) > 1:
                traj_vertices = plan.trajectory.T
                vis[f"swing_{foot_name}_{i}"].set_object(
                    g.Line(g.PointsGeometry(traj_vertices),
                           g.LineBasicMaterial(color=color, linewidth=2))
                )

            # Landing marker
            vis[f"landing_{foot_name}_{i}"].set_object(
                g.Sphere(0.01),
                g.MeshLambertMaterial(color=color)
            )
            vis[f"landing_{foot_name}_{i}"].set_transform(
                tf.translation_matrix(plan.end_pos)
            )

    # Animation
    if animate:
        print("\nAnimating gait sequence...")
        print("Press Ctrl+C to stop")

        dt = 0.02
        num_frames = len(com_traj)

        # Build foot position timeline
        foot_timelines = {foot: [] for foot in initial_feet.keys()}
        for foot_name in initial_feet.keys():
            current_pos = initial_feet[foot_name].copy()
            plan_idx = 0
            plans = foothold_plans[foot_name]

            for frame in range(num_frames):
                t = frame * dt

                # Check if in swing phase
                in_swing = False
                if plan_idx < len(plans):
                    plan = plans[plan_idx]
                    if plan.swing_start_time <= t < plan.swing_end_time:
                        in_swing = True
                        # Interpolate within swing trajectory
                        swing_progress = (t - plan.swing_start_time) / (plan.swing_end_time - plan.swing_start_time)
                        swing_idx = int(swing_progress * (len(plan.trajectory) - 1))
                        swing_idx = min(swing_idx, len(plan.trajectory) - 1)
                        current_pos = plan.trajectory[swing_idx].copy()
                    elif t >= plan.swing_end_time:
                        current_pos = plan.end_pos.copy()
                        plan_idx += 1

                foot_timelines[foot_name].append(current_pos.copy())

        try:
            for frame in range(num_frames):
                # Update body position
                body_pos = com_traj[frame].copy()
                body_pos[2] = standing_height  # Keep at standing height

                # Compute heading from trajectory tangent
                if frame < num_frames - 1:
                    tangent = com_traj[min(frame + 1, num_frames - 1)] - com_traj[max(frame - 1, 0)]
                    yaw = np.arctan2(tangent[1], tangent[0])
                else:
                    yaw = 0

                # Body transform
                body_transform = tf.translation_matrix(body_pos) @ tf.rotation_matrix(yaw, [0, 0, 1])
                vis["body"].set_transform(body_transform)

                # Update feet
                for foot_name in initial_feet.keys():
                    foot_pos = foot_timelines[foot_name][frame]
                    vis[f"foot_{foot_name}"].set_transform(tf.translation_matrix(foot_pos))

                time.sleep(dt * 2)  # Slow down for visualization

        except KeyboardInterrupt:
            print("\nAnimation stopped")

    print("\nMeshcat visualization ready. Keep the browser window open.")
    print("URL:", vis.url())

    # Keep the server running
    input("Press Enter to close...")


# =============================================================================
# Crocoddyl OCP Test
# =============================================================================

def test_crocoddyl_ocp(results: Dict):
    """Test OCP construction with Crocoddyl.

    Args:
        results: Output from test_gait_pipeline.
    """
    if not HAS_CROCODDYL or not HAS_ROBOT_DATA:
        print("\nSkipping Crocoddyl OCP test (missing dependencies)")
        return

    print("\n" + "=" * 60)
    print("Crocoddyl OCP Test")
    print("=" * 60)

    try:
        # Load Solo12 robot
        robot = example_robot_data.load_solo12()
        rmodel = robot.model
        rdata = rmodel.createData()

        print(f"\nLoaded robot: Solo12")
        print(f"  nq (config dim): {rmodel.nq}")
        print(f"  nv (velocity dim): {rmodel.nv}")
        print(f"  njoints: {rmodel.njoints}")

        # Create state and actuation models
        state = crocoddyl.StateMultibody(rmodel)
        actuation = crocoddyl.ActuationModelFloatingBase(state)

        print(f"  state.nx: {state.nx}")
        print(f"  actuation.nu: {actuation.nu}")

        # Get foot frame IDs
        foot_frame_names = {
            "LF": "FL_FOOT",
            "RF": "FR_FOOT",
            "LH": "HL_FOOT",
            "RH": "HR_FOOT",
        }

        foot_frame_ids = {}
        for foot_name, frame_name in foot_frame_names.items():
            try:
                frame_id = rmodel.getFrameId(frame_name)
                foot_frame_ids[foot_name] = frame_id
                print(f"  {foot_name} frame ID: {frame_id}")
            except Exception as e:
                print(f"  Warning: Could not find frame {frame_name}: {e}")

        # Build a simple OCP for one step
        print("\nBuilding OCP...")

        com_traj = results["com_trajectory"]
        contact_sequence = results["contact_sequence"]

        # Initial state (standing)
        q0 = pinocchio.neutral(rmodel)
        q0[2] = results["standing_height"]  # Set height
        v0 = np.zeros(rmodel.nv)
        x0 = np.concatenate([q0, v0])

        # Create running models
        running_models = []
        dt = 0.02
        num_knots = min(25, len(com_traj))  # Limit horizon

        for k in range(num_knots):
            # Contact model (all feet in contact for simplicity)
            contact_model = crocoddyl.ContactModelMultiple(state, actuation.nu)
            for foot_name, frame_id in foot_frame_ids.items():
                contact = crocoddyl.ContactModel3D(
                    state, frame_id,
                    np.array([0.0, 0.0, 0.0]),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    actuation.nu,
                    np.array([0.0, 0.0])
                )
                contact_model.addContact(f"contact_{foot_name}", contact)

            # Cost model
            cost_model = crocoddyl.CostModelSum(state, actuation.nu)

            # CoM tracking cost
            com_target = com_traj[min(k, len(com_traj) - 1)]
            com_residual = crocoddyl.ResidualModelCoMPosition(state, com_target, actuation.nu)
            com_cost = crocoddyl.CostModelResidual(state, com_residual)
            cost_model.addCost("comTrack", com_cost, 1e4)

            # State regularization
            state_residual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
            state_cost = crocoddyl.CostModelResidual(state, state_residual)
            cost_model.addCost("stateReg", state_cost, 1e1)

            # Control regularization
            ctrl_residual = crocoddyl.ResidualModelControl(state, actuation.nu)
            ctrl_cost = crocoddyl.CostModelResidual(state, ctrl_residual)
            cost_model.addCost("ctrlReg", ctrl_cost, 1e-1)

            # Differential model
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                state, actuation, contact_model, cost_model
            )

            # Integrated model
            model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)
            running_models.append(model)

        # Terminal model
        terminal_contact = crocoddyl.ContactModelMultiple(state, actuation.nu)
        for foot_name, frame_id in foot_frame_ids.items():
            contact = crocoddyl.ContactModel3D(
                state, frame_id,
                np.array([0.0, 0.0, 0.0]),
                pinocchio.LOCAL_WORLD_ALIGNED,
                actuation.nu,
                np.array([0.0, 0.0])
            )
            terminal_contact.addContact(f"contact_{foot_name}", contact)

        terminal_cost = crocoddyl.CostModelSum(state, actuation.nu)
        com_target = com_traj[-1]
        com_residual = crocoddyl.ResidualModelCoMPosition(state, com_target, actuation.nu)
        terminal_cost.addCost("comTrack", crocoddyl.CostModelResidual(state, com_residual), 1e5)

        terminal_dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, terminal_contact, terminal_cost
        )
        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_dmodel, 0.0)

        # Create shooting problem
        problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

        print(f"  Running models: {len(running_models)}")
        print(f"  Horizon: {len(running_models) * dt:.2f}s")

        # Create solver
        solver = crocoddyl.SolverFDDP(problem)
        solver.th_stop = 1e-4

        # Solve
        print("\nSolving OCP...")
        t_start = time.time()
        converged = solver.solve([], [], 100, False)
        t_solve = time.time() - t_start

        print(f"  Converged: {converged}")
        print(f"  Iterations: {solver.iter}")
        print(f"  Cost: {solver.cost:.2f}")
        print(f"  Solve time: {t_solve * 1000:.1f}ms")

        # Extract solution
        xs = np.array(solver.xs)
        us = np.array(solver.us)

        print(f"\nSolution:")
        print(f"  States shape: {xs.shape}")
        print(f"  Controls shape: {us.shape}")
        print(f"  First control: {us[0, :6]}...")

        return solver, xs, us

    except Exception as e:
        print(f"\nError in Crocoddyl OCP test: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone gait module test with visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_gait_standalone.py                           # Basic test
    python test_gait_standalone.py --gait walk               # Test walk gait
    python test_gait_standalone.py --trajectory curve_left   # Test curved path
    python test_gait_standalone.py --meshcat                 # 3D visualization
    python test_gait_standalone.py --meshcat --animate       # Animated 3D view
    python test_gait_standalone.py --crocoddyl               # Test with Crocoddyl OCP
        """
    )

    parser.add_argument(
        "--gait", type=str, default="trot",
        choices=["trot", "walk", "pace", "bound"],
        help="Gait type (default: trot)"
    )
    parser.add_argument(
        "--trajectory", type=str, default="straight",
        choices=["straight", "curve_left", "curve_right", "s_curve", "circle"],
        help="Trajectory type (default: straight)"
    )
    parser.add_argument(
        "--distance", type=float, default=0.5,
        help="Forward distance in meters (default: 0.5)"
    )
    parser.add_argument(
        "--meshcat", action="store_true",
        help="Use Meshcat 3D visualization"
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Animate the gait in Meshcat"
    )
    parser.add_argument(
        "--crocoddyl", action="store_true",
        help="Test Crocoddyl OCP construction"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save matplotlib figure to path"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib visualization"
    )

    args = parser.parse_args()

    # Run gait pipeline test
    results = test_gait_pipeline(
        gait_type=args.gait,
        trajectory_type=args.trajectory,
        distance=args.distance,
    )

    # Visualizations
    if not args.no_plot and HAS_MATPLOTLIB:
        save_path = args.save or f"gait_{args.gait}_{args.trajectory}.png"
        visualize_matplotlib(results, save_path)

    if args.meshcat:
        visualize_meshcat(results, animate=args.animate)

    # Crocoddyl OCP test
    if args.crocoddyl:
        test_crocoddyl_ocp(results)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
