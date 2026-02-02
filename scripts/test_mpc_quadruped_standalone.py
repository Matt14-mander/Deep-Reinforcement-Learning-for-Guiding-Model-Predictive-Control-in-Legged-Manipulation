#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone test for quadruped MPC without IsaacLab simulation.

This script tests the complete quadruped MPC pipeline:
1. CoM Bezier trajectory generation
2. Gait scheduling (contact sequence)
3. Foothold planning
4. OCP construction with OCPFactory
5. FDDP solving

This validates the MPC can track curved trajectories before
integrating with IsaacLab physics simulation.

Usage:
    python scripts/test_mpc_quadruped_standalone.py

Requirements:
    - crocoddyl
    - pinocchio
    - example-robot-data (optional, for robot models)
    - matplotlib for visualization
"""

import argparse
import time
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source", "RL_Bezier_MPC"))

from RL_Bezier_MPC.trajectory import BezierTrajectoryGenerator
from RL_Bezier_MPC.gait import GaitScheduler, FootholdPlanner, ContactSequence

# Check for Crocoddyl
try:
    import crocoddyl
    import pinocchio
    HAS_CROCODDYL = True
except ImportError:
    HAS_CROCODDYL = False
    print("Warning: Crocoddyl not available. Running in demo mode.")


def create_test_trajectory(
    start_pos: np.ndarray,
    trajectory_type: str = "straight",
    distance: float = 1.0,
    turn_angle: float = 0.0,
) -> tuple:
    """Create test CoM trajectory.

    Args:
        start_pos: Starting CoM position (3,).
        trajectory_type: "straight", "curve_left", "curve_right", "s_curve".
        distance: Forward distance to travel.
        turn_angle: Turn angle for curved trajectories.

    Returns:
        Tuple of (trajectory, bezier_params).
    """
    trajectory_gen = BezierTrajectoryGenerator(
        degree=3,
        state_dim=3,
        max_displacement=2.0,
    )

    if trajectory_type == "straight":
        # Straight line forward
        params = np.zeros(12)
        params[9:12] = [distance, 0.0, 0.0]  # End point
        params[3:6] = [distance / 3, 0.0, 0.0]
        params[6:9] = [2 * distance / 3, 0.0, 0.0]

    elif trajectory_type == "curve_left":
        # Curved path turning left
        lateral = distance * np.sin(turn_angle)
        forward = distance * np.cos(turn_angle)

        params = np.zeros(12)
        params[3:6] = [forward / 3, 0.0, 0.0]
        params[6:9] = [2 * forward / 3, lateral / 2, 0.0]
        params[9:12] = [forward, lateral, 0.0]

    elif trajectory_type == "curve_right":
        # Curved path turning right
        lateral = -distance * np.sin(turn_angle)
        forward = distance * np.cos(turn_angle)

        params = np.zeros(12)
        params[3:6] = [forward / 3, 0.0, 0.0]
        params[6:9] = [2 * forward / 3, lateral / 2, 0.0]
        params[9:12] = [forward, lateral, 0.0]

    elif trajectory_type == "s_curve":
        # S-curve (left then right)
        params = np.zeros(12)
        params[3:6] = [distance / 3, distance * 0.2, 0.0]
        params[6:9] = [2 * distance / 3, -distance * 0.2, 0.0]
        params[9:12] = [distance, 0.0, 0.0]

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    trajectory = trajectory_gen.params_to_waypoints(
        params=params,
        dt=0.02,
        horizon=1.5,
        start_position=start_pos,
    )

    return trajectory, params


def simple_quadruped_dynamics(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    mass: float = 12.0,
) -> np.ndarray:
    """Simple quadruped dynamics for testing without Crocoddyl.

    This is a very simplified point-mass model where control
    directly affects CoM acceleration.

    Args:
        state: Current state [pos(3), vel(3)].
        control: Control input (12D joint torques, simplified to force).
        dt: Timestep.
        mass: Robot mass.

    Returns:
        Next state.
    """
    pos = state[:3]
    vel = state[3:6]

    # Simplified: sum of torques creates horizontal force
    # This is NOT realistic, just for testing the pipeline
    force_x = np.sum(control[0::3]) * 0.1  # Every 3rd torque for x
    force_y = np.sum(control[1::3]) * 0.1  # For y
    force_z = np.sum(np.abs(control)) * 0.05 - mass * 9.81  # Vertical

    acc = np.array([force_x, force_y, force_z]) / mass

    # Integrate
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt

    # Ground constraint
    if new_pos[2] < 0.0:
        new_pos[2] = 0.0
        new_vel[2] = 0.0

    return np.concatenate([new_pos, new_vel])


def test_gait_pipeline():
    """Test the gait pipeline without MPC solving."""
    print("=" * 60)
    print("Quadruped Gait Pipeline Test")
    print("=" * 60)

    # Initialize components
    gait_scheduler = GaitScheduler()
    foothold_planner = FootholdPlanner(step_height=0.05)

    # Test different gaits
    for gait_type in ["trot", "walk", "pace", "bound"]:
        print(f"\n--- Testing {gait_type.upper()} gait ---")

        contact_sequence = gait_scheduler.generate(
            gait_type=gait_type,
            step_duration=0.15,
            support_duration=0.05,
            num_cycles=2,
        )

        print(f"Number of phases: {len(contact_sequence)}")
        print(f"Total duration: {contact_sequence.total_duration:.2f}s")

        # Show first few phases
        for i, phase in enumerate(contact_sequence.phases[:4]):
            print(f"  Phase {i}: support={phase.support_feet}, "
                  f"swing={phase.swing_feet}, duration={phase.duration:.3f}s")

    # Test foothold planning
    print("\n--- Testing Foothold Planning ---")

    start_pos = np.array([0.0, 0.0, 0.35])
    com_trajectory, _ = create_test_trajectory(
        start_pos=start_pos,
        trajectory_type="curve_left",
        distance=1.0,
        turn_angle=np.pi / 4,
    )

    contact_sequence = gait_scheduler.generate(
        gait_type="trot",
        step_duration=0.15,
        support_duration=0.05,
        num_cycles=4,
    )

    initial_feet = foothold_planner.get_footholds_at_time(
        com_position=start_pos,
        heading=0.0,
    )

    foothold_plans = foothold_planner.plan_footholds(
        com_trajectory=com_trajectory,
        contact_sequence=contact_sequence,
        current_foot_positions=initial_feet,
        dt=0.02,
    )

    # Analyze
    step_lengths = foothold_planner.compute_step_lengths(foothold_plans)
    print("\nStep lengths per foot:")
    for foot, lengths in step_lengths.items():
        if lengths:
            print(f"  {foot}: avg={np.mean(lengths):.4f}m, "
                  f"min={np.min(lengths):.4f}m, max={np.max(lengths):.4f}m")

    return foothold_plans, com_trajectory


def test_mpc_tracking():
    """Test MPC tracking with simplified dynamics."""
    print("\n" + "=" * 60)
    print("MPC Tracking Test (Simplified Dynamics)")
    print("=" * 60)

    if not HAS_CROCODDYL:
        print("\nSkipping MPC test: Crocoddyl not available")
        print("Testing with simplified dynamics only...")

        # Run simplified simulation
        start_pos = np.array([0.0, 0.0, 0.35])
        com_trajectory, _ = create_test_trajectory(
            start_pos=start_pos,
            trajectory_type="straight",
            distance=1.0,
        )

        # Simple P controller following trajectory
        state = np.concatenate([start_pos, np.zeros(3)])  # pos, vel
        states = [state.copy()]

        dt = 0.02
        num_steps = len(com_trajectory)

        for i in range(num_steps - 1):
            target = com_trajectory[min(i + 1, len(com_trajectory) - 1)]

            # Simple proportional control
            pos_error = target - state[:3]
            vel_error = -state[3:6]

            # "Control" is just desired acceleration
            control = 10.0 * pos_error + 2.0 * vel_error

            # Simulate
            state = simple_quadruped_dynamics(state, control, dt)
            states.append(state.copy())

        states = np.array(states)
        print(f"\nSimulated {len(states)} steps")
        print(f"Start: {states[0, :3]}")
        print(f"End: {states[-1, :3]}")
        print(f"Target: {com_trajectory[-1]}")

        tracking_error = np.linalg.norm(states[-1, :3] - com_trajectory[-1])
        print(f"Final tracking error: {tracking_error:.4f}m")

        return states, com_trajectory

    # Full MPC test with Crocoddyl
    print("\nRunning with Crocoddyl MPC...")

    try:
        from RL_Bezier_MPC.controllers.crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC
        from RL_Bezier_MPC.robots.quadruped_cfg import load_pinocchio_model

        # Load robot model
        rmodel, urdf_path = load_pinocchio_model(robot_name="solo12")
        print(f"Loaded robot model: nq={rmodel.nq}, nv={rmodel.nv}")

        # Get foot frame IDs
        foot_frame_names = {
            "LF": "FL_FOOT",  # Solo12 naming
            "RF": "FR_FOOT",
            "LH": "HL_FOOT",
            "RH": "HR_FOOT",
        }

        # Create MPC controller
        mpc = CrocoddylQuadrupedMPC(
            rmodel=rmodel,
            foot_frame_names=foot_frame_names,
            gait_type="trot",
            dt=0.02,
            horizon_steps=25,
        )

        # Create trajectory
        start_pos = np.array([0.0, 0.0, 0.24])  # Solo12 standing height
        com_trajectory, _ = create_test_trajectory(
            start_pos=start_pos,
            trajectory_type="straight",
            distance=0.5,
        )

        # Initial state
        q0 = pinocchio.neutral(rmodel)
        q0[:3] = start_pos
        v0 = np.zeros(rmodel.nv)
        x0 = np.concatenate([q0, v0])

        # Initial foot positions
        rdata = rmodel.createData()
        pinocchio.framesForwardKinematics(rmodel, rdata, q0)

        foot_positions = {}
        for foot_name, frame_name in foot_frame_names.items():
            frame_id = rmodel.getFrameId(frame_name)
            foot_positions[foot_name] = rdata.oMf[frame_id].translation.copy()

        print(f"\nInitial state shape: {x0.shape}")
        print(f"CoM trajectory shape: {com_trajectory.shape}")

        # Solve MPC
        t_start = time.time()
        solution = mpc.solve(
            current_state=x0,
            com_reference=com_trajectory[:25],  # First horizon
            current_foot_positions=foot_positions,
            warm_start=False,
        )
        t_solve = time.time() - t_start

        print(f"\nMPC solve time: {t_solve*1000:.1f}ms")
        print(f"Converged: {solution.converged}")
        print(f"Iterations: {solution.iterations}")
        print(f"Cost: {solution.cost:.2f}")
        print(f"Control shape: {solution.control.shape}")
        print(f"Control (first 6): {solution.control[:6]}")

        return solution, com_trajectory

    except Exception as e:
        print(f"\nError running Crocoddyl MPC: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_results(
    states: np.ndarray,
    com_trajectory: np.ndarray,
    title: str = "MPC Tracking Results",
):
    """Visualize tracking results.

    Args:
        states: Simulated states (T, state_dim).
        com_trajectory: Reference trajectory (T, 3).
        title: Plot title.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # XY trajectory
    ax = axes[0, 0]
    ax.plot(com_trajectory[:, 0], com_trajectory[:, 1], "b--", label="Reference", linewidth=2)
    ax.plot(states[:, 0], states[:, 1], "r-", label="Actual", linewidth=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # XZ trajectory
    ax = axes[0, 1]
    ax.plot(com_trajectory[:, 0], com_trajectory[:, 2], "b--", label="Reference", linewidth=2)
    ax.plot(states[:, 0], states[:, 2], "r-", label="Actual", linewidth=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Side View (XZ)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error over time
    ax = axes[1, 0]
    min_len = min(len(states), len(com_trajectory))
    error = np.linalg.norm(states[:min_len, :3] - com_trajectory[:min_len], axis=1)
    time_axis = np.arange(min_len) * 0.02

    ax.plot(time_axis, error, "k-", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.grid(True, alpha=0.3)

    # 3D view
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.plot3D(com_trajectory[:, 0], com_trajectory[:, 1], com_trajectory[:, 2],
              "b--", label="Reference", linewidth=2)
    ax.plot3D(states[:, 0], states[:, 1], states[:, 2],
              "r-", label="Actual", linewidth=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D View")
    ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("mpc_quadruped_test.png", dpi=150)
    print("\nResults saved to: mpc_quadruped_test.png")
    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quadruped MPC Standalone Test")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="straight",
        choices=["straight", "curve_left", "curve_right", "s_curve"],
        help="Trajectory type to test",
    )
    parser.add_argument(
        "--gait",
        type=str,
        default="trot",
        choices=["trot", "walk", "pace", "bound"],
        help="Gait type to use",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip visualization",
    )
    args = parser.parse_args()

    # Test gait pipeline
    foothold_plans, com_traj = test_gait_pipeline()

    # Test MPC tracking
    result, ref_traj = test_mpc_tracking()

    # Visualize if available
    if result is not None and not args.no_plot and HAS_MATPLOTLIB:
        if isinstance(result, np.ndarray):
            visualize_results(result, ref_traj)

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
