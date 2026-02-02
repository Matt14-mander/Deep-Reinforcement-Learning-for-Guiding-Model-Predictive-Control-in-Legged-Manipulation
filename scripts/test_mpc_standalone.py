#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone test for Bezier trajectory generation and Crocoddyl MPC.

This script tests the trajectory generator and MPC controller without
requiring IsaacLab or GPU resources. It:
1. Generates a Bezier trajectory from random parameters
2. Simulates quadrotor dynamics with MPC tracking
3. Plots the results for visualization

Run with:
    python scripts/test_mpc_standalone.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add the source directory to path
SOURCE_DIR = Path(__file__).parent.parent / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(SOURCE_DIR))

from RL_Bezier_MPC.trajectory import BezierTrajectoryGenerator


def simulate_quadrotor_simple(
    initial_state: np.ndarray,
    control: np.ndarray,
    dt: float,
    mass: float = 0.027,
    inertia: np.ndarray = None,
) -> np.ndarray:
    """Simple quadrotor dynamics simulation (Euler integration).

    Args:
        initial_state: State [pos(3), quat(4), vel(3), omega(3)]
        control: Control [thrust, tau_x, tau_y, tau_z]
        dt: Timestep
        mass: Quadrotor mass
        inertia: 3x3 inertia matrix

    Returns:
        Next state after one timestep.
    """
    if inertia is None:
        inertia = np.diag([1.4e-5, 1.4e-5, 2.17e-5])

    inertia_inv = np.linalg.inv(inertia)
    gravity = np.array([0.0, 0.0, -9.81])

    # Unpack state
    pos = initial_state[0:3].copy()
    quat = initial_state[3:7].copy()
    vel = initial_state[7:10].copy()
    omega = initial_state[10:13].copy()

    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)

    # Unpack control
    thrust = control[0]
    torque = control[1:4]

    # Quaternion to rotation matrix
    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

    # Translational dynamics
    thrust_body = np.array([0.0, 0.0, thrust])
    thrust_world = R @ thrust_body
    acc = thrust_world / mass + gravity

    # Rotational dynamics
    omega_cross_I_omega = np.cross(omega, inertia @ omega)
    angular_acc = inertia_inv @ (torque - omega_cross_I_omega)

    # Quaternion derivative
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quat_multiply(omega_quat, quat)

    # Euler integration
    pos_new = pos + vel * dt
    vel_new = vel + acc * dt
    quat_new = quat + quat_dot * dt
    quat_new = quat_new / np.linalg.norm(quat_new)  # Normalize
    omega_new = omega + angular_acc * dt

    return np.concatenate([pos_new, quat_new, vel_new, omega_new])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def test_bezier_trajectory():
    """Test Bezier trajectory generation."""
    print("=" * 60)
    print("Testing Bezier Trajectory Generator")
    print("=" * 60)

    # Create generator
    generator = BezierTrajectoryGenerator(
        degree=3,
        state_dim=3,
        max_displacement=2.0,
    )

    print(f"Parameter dimension: {generator.get_param_dim()}")
    print(f"State dimension: {generator.get_state_dim()}")

    low, high = generator.get_param_bounds()
    print(f"Parameter bounds: [{low[0]:.2f}, {high[3]:.2f}]")

    # Generate a test trajectory
    # P0 offset = [0, 0, 0] (start at current position)
    # P1 offset = [0.5, 0.5, 0.3] (first control point)
    # P2 offset = [1.0, 0.0, 0.5] (second control point)
    # P3 offset = [1.5, -0.3, 0.0] (end point)
    params = np.array([
        0.0, 0.0, 0.0,      # P0 offset (must be zero)
        0.5, 0.5, 0.3,      # P1 offset
        1.0, 0.0, 0.5,      # P2 offset
        1.5, -0.3, 0.0,     # P3 offset (target)
    ])

    start_position = np.array([0.0, 0.0, 1.0])  # Start at height 1m
    dt = 0.02  # 50 Hz
    horizon = 1.5  # 1.5 seconds

    waypoints = generator.params_to_waypoints(
        params=params,
        dt=dt,
        horizon=horizon,
        start_position=start_position,
    )

    print(f"Generated {len(waypoints)} waypoints over {horizon}s horizon")
    print(f"Start position: {waypoints[0]}")
    print(f"End position: {waypoints[-1]}")

    # Also compute velocity and acceleration
    velocity = generator.get_velocity(params, dt, horizon, start_position)
    acceleration = generator.get_acceleration(params, dt, horizon, start_position)

    print(f"Max velocity magnitude: {np.max(np.linalg.norm(velocity, axis=1)):.3f} m/s")
    print(f"Max acceleration magnitude: {np.max(np.linalg.norm(acceleration, axis=1)):.3f} m/s²")

    return waypoints, velocity, acceleration, start_position


def test_mpc_tracking(reference_trajectory: np.ndarray, start_position: np.ndarray):
    """Test MPC trajectory tracking with simple simulation.

    Args:
        reference_trajectory: Reference positions (T x 3)
        start_position: Initial position

    Returns:
        Tuple of (actual_trajectory, controls)
    """
    print("\n" + "=" * 60)
    print("Testing MPC Trajectory Tracking")
    print("=" * 60)

    try:
        from RL_Bezier_MPC.controllers import CrocoddylQuadrotorMPC
    except ImportError as e:
        print(f"Crocoddyl not available: {e}")
        print("Skipping MPC test. Install crocoddyl to run this test.")
        return None, None

    # Create MPC controller
    mpc = CrocoddylQuadrotorMPC(
        dt=0.02,
        horizon_steps=25,
        mass=0.027,
        position_weight=100.0,
        velocity_weight=10.0,
        control_weight=0.1,
        thrust_max=0.6,
        torque_max=0.01,
        max_iterations=10,
        verbose=False,
    )

    print(f"MPC control dimension: {mpc.get_control_dim()}")
    print(f"MPC state dimension: {mpc.get_state_dim()}")
    print(f"MPC horizon: {mpc.horizon_steps} steps")

    # Initial state: hovering at start position
    state = np.zeros(13)
    state[0:3] = start_position
    state[3] = 1.0  # Identity quaternion (w, x, y, z)

    # Simulation parameters
    sim_dt = 0.02  # Match MPC rate
    num_steps = len(reference_trajectory)

    # Storage
    actual_trajectory = [state[0:3].copy()]
    controls = []
    solve_times = []

    print(f"Running simulation for {num_steps} steps...")

    for i in range(num_steps - 1):
        # Get reference slice for MPC horizon
        ref_start = i
        ref_end = min(i + mpc.horizon_steps, len(reference_trajectory))
        reference_slice = reference_trajectory[ref_start:ref_end]

        # Solve MPC
        solution = mpc.solve(
            current_state=state,
            reference_trajectory=reference_slice,
            warm_start=True,
        )

        controls.append(solution.control)
        solve_times.append(solution.solve_time)

        # Apply control and simulate
        state = simulate_quadrotor_simple(state, solution.control, sim_dt)

        # Store position
        actual_trajectory.append(state[0:3].copy())

        # Progress update
        if (i + 1) % 20 == 0:
            pos_error = np.linalg.norm(state[0:3] - reference_trajectory[i])
            print(f"  Step {i+1}/{num_steps-1}, pos error: {pos_error:.4f} m, "
                  f"solve time: {solution.solve_time*1000:.1f} ms")

    actual_trajectory = np.array(actual_trajectory)
    controls = np.array(controls)

    # Statistics
    tracking_errors = np.linalg.norm(
        actual_trajectory[:-1] - reference_trajectory[:-1], axis=1
    )
    print(f"\nTracking Statistics:")
    print(f"  Mean position error: {np.mean(tracking_errors):.4f} m")
    print(f"  Max position error: {np.max(tracking_errors):.4f} m")
    print(f"  Mean solve time: {np.mean(solve_times)*1000:.2f} ms")

    return actual_trajectory, controls


def plot_results(
    reference: np.ndarray,
    actual: np.ndarray = None,
    velocity: np.ndarray = None,
    controls: np.ndarray = None,
    save_path: str = None,
):
    """Plot trajectory results.

    Args:
        reference: Reference trajectory (T x 3)
        actual: Actual trajectory (T x 3), optional
        velocity: Velocity profile (T x 3), optional
        controls: Control inputs (T x 4), optional
        save_path: Path to save figure, optional
    """
    # Defer matplotlib import to avoid Isaac Sim compatibility issues
    # Must set backend via env var BEFORE importing matplotlib
    import os
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(15, 10))

    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(reference[:, 0], reference[:, 1], reference[:, 2],
             'b-', label='Reference', linewidth=2)
    if actual is not None:
        ax1.plot(actual[:, 0], actual[:, 1], actual[:, 2],
                 'r--', label='Actual', linewidth=2)
    ax1.scatter(*reference[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(*reference[-1], c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # Position over time
    ax2 = fig.add_subplot(2, 2, 2)
    t = np.arange(len(reference)) * 0.02
    ax2.plot(t, reference[:, 0], 'b-', label='X ref')
    ax2.plot(t, reference[:, 1], 'g-', label='Y ref')
    ax2.plot(t, reference[:, 2], 'r-', label='Z ref')
    if actual is not None:
        t_actual = np.arange(len(actual)) * 0.02
        ax2.plot(t_actual, actual[:, 0], 'b--', label='X actual')
        ax2.plot(t_actual, actual[:, 1], 'g--', label='Y actual')
        ax2.plot(t_actual, actual[:, 2], 'r--', label='Z actual')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)

    # Velocity or tracking error
    ax3 = fig.add_subplot(2, 2, 3)
    if actual is not None and len(actual) == len(reference):
        errors = np.linalg.norm(actual - reference, axis=1)
        ax3.plot(t, errors, 'k-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Tracking Error (m)')
        ax3.set_title('Position Tracking Error')
        ax3.grid(True)
    elif velocity is not None:
        vel_mag = np.linalg.norm(velocity, axis=1)
        ax3.plot(t, vel_mag, 'b-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Velocity Magnitude')
        ax3.grid(True)

    # Control inputs
    ax4 = fig.add_subplot(2, 2, 4)
    if controls is not None:
        t_ctrl = np.arange(len(controls)) * 0.02
        ax4.plot(t_ctrl, controls[:, 0], 'k-', label='Thrust')
        ax4.plot(t_ctrl, controls[:, 1], 'r-', label='τx')
        ax4.plot(t_ctrl, controls[:, 2], 'g-', label='τy')
        ax4.plot(t_ctrl, controls[:, 3], 'b-', label='τz')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Control')
        ax4.set_title('Control Inputs')
        ax4.legend()
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'No control data', ha='center', va='center',
                 transform=ax4.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    else:
        # With Agg backend, show() does nothing - inform user
        print("Plot created but not saved. Use --save <path> to save the figure.")


def main():
    """Run standalone MPC test."""
    parser = argparse.ArgumentParser(description="Test Bezier + MPC without IsaacLab")
    parser.add_argument("--no-mpc", action="store_true",
                        help="Skip MPC test (only test trajectory generation)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting (useful when matplotlib is unavailable)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save plot (required when using Isaac Sim Python)")
    args = parser.parse_args()

    # Test trajectory generation
    waypoints, velocity, acceleration, start_pos = test_bezier_trajectory()

    # Test MPC tracking
    actual = None
    controls = None
    if not args.no_mpc:
        actual, controls = test_mpc_tracking(waypoints, start_pos)

    # Plot results
    if not args.no_plot:
        if args.save is None:
            print("\nNote: Using Agg backend (no display). Use --save <path> to save plot.")
        plot_results(
            reference=waypoints,
            actual=actual,
            velocity=velocity,
            controls=controls,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()