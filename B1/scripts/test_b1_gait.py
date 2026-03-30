#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test B1 quadruped gait module with Crocoddyl visualization.

This script tests our custom gait pipeline with the B1 robot model:
1. Load B1 model from example-robot-data
2. Generate CoM Bezier trajectory
3. Plan footholds using our FootholdPlanner
4. Build OCP using our pipeline (not Crocoddyl's demo classes)
5. Solve with FDDP
6. Visualize with display and plot

Usage:
    python test_b1_gait.py                          # Basic test
    python test_b1_gait.py --display                # With Meshcat/Gepetto visualization
    python test_b1_gait.py --plot                   # With matplotlib plots
    python test_b1_gait.py --display --plot         # Both
    python test_b1_gait.py --gait trot --display    # Test trotting gait
    python test_b1_gait.py --trajectory curve_left --display  # Test curve walking
"""

import argparse
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for quadruped_mpc module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
B1_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, B1_DIR)

# Import dependencies
try:
    import example_robot_data
    import pinocchio
    import crocoddyl
    HAS_CROCODDYL = True
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Install with: pip install crocoddyl pinocchio example-robot-data")
    sys.exit(1)

# Import our gait modules
from quadruped_mpc.trajectory import BezierTrajectoryGenerator, BezierFootTrajectory
from quadruped_mpc.gait import (
    GaitScheduler,
    FootholdPlanner,
    ContactSequence,
    ContactPhase,
    OCPFactory,
)
from quadruped_mpc.utils.math_utils import (
    bezier_curve,
    heading_from_tangent,
    rotation_matrix_z,
)

# Handle Ctrl+C gracefully
signal.signal(signal.SIGINT, signal.SIG_DFL)


# =============================================================================
# B1 Robot Configuration
# =============================================================================

class B1RobotConfig:
    """Configuration for B1 quadruped robot from example-robot-data.

    Note: B1 foot frame naming convention (from change_b1_model.py):
        - FR_foot (Front Right)
        - FL_foot (Front Left)
        - RR_foot (Rear Right)
        - RL_foot (Rear Left)

    Our naming convention:
        - LF (Left Front) = FL_foot
        - RF (Right Front) = FR_foot
        - LH (Left Hind) = RL_foot
        - RH (Right Hind) = RR_foot
    """

    # Mapping from our convention to B1 frame names
    FOOT_FRAME_NAMES = {
        "LF": "FL_foot",  # Left Front
        "RF": "FR_foot",  # Right Front
        "LH": "RL_foot",  # Left Hind
        "RH": "RR_foot",  # Right Hind
    }

    # Hip offsets in body frame (approximate for B1)
    # B1 is larger than Solo12, dimensions from URDF
    HIP_OFFSETS = {
        "LF": np.array([+0.3, +0.1, 0.0]),   # Front-left
        "RF": np.array([+0.3, -0.1, 0.0]),   # Front-right
        "LH": np.array([-0.3, +0.1, 0.0]),   # Hind-left
        "RH": np.array([-0.3, -0.1, 0.0]),   # Hind-right
    }

    # Default gait parameters for B1
    GAIT_PARAMS = {
        "step_duration": 0.15,
        "support_duration": 0.05,
        "step_height": 0.15,
    }

    # Standing height (CoM height)
    STANDING_HEIGHT = 0.45


def load_b1_robot():
    """Load B1 robot model from example-robot-data.

    Returns:
        Tuple of (robot, rmodel, rdata, q0, v0, x0)
    """
    print("Loading B1 robot model...")

    # Load B1 model
    b1 = example_robot_data.load("b1")
    rmodel = b1.model
    rdata = rmodel.createData()

    # Get initial configuration (standing pose)
    q0 = rmodel.referenceConfigurations["standing"].copy()
    v0 = pinocchio.utils.zero(rmodel.nv)
    x0 = np.concatenate([q0, v0])

    print(f"  Model loaded: B1")
    print(f"  nq (config dim): {rmodel.nq}")
    print(f"  nv (velocity dim): {rmodel.nv}")
    print(f"  nx (state dim): {rmodel.nq + rmodel.nv}")

    # Get foot frame IDs
    foot_frame_ids = {}
    print(f"  Foot frames:")
    for our_name, b1_name in B1RobotConfig.FOOT_FRAME_NAMES.items():
        try:
            frame_id = rmodel.getFrameId(b1_name)
            foot_frame_ids[our_name] = frame_id
            print(f"    {our_name} ({b1_name}): frame_id={frame_id}")
        except Exception as e:
            print(f"    Warning: Could not find frame {b1_name}: {e}")

    return b1, rmodel, rdata, q0, v0, x0, foot_frame_ids


# =============================================================================
# CoM Trajectory Generation
# =============================================================================

def create_com_trajectory(
    start_pos: np.ndarray,
    trajectory_type: str = "straight",
    distance: float = 0.5,
    turn_angle: float = np.pi / 3,
    duration: float = 2.0,
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
        params[3:6] = [distance / 3, 0.0, 0.0]
        params[6:9] = [2 * distance / 3, 0.0, 0.0]
        params[9:12] = [distance, 0.0, 0.0]

    elif trajectory_type == "curve_left":
        lateral = distance * np.sin(turn_angle)
        forward = distance * np.cos(turn_angle)
        params[3:6] = [forward / 3, 0.0, 0.0]
        params[6:9] = [2 * forward / 3, lateral / 2, 0.0]
        params[9:12] = [forward, lateral, 0.0]

    elif trajectory_type == "curve_right":
        lateral = -distance * np.sin(turn_angle)
        forward = distance * np.cos(turn_angle)
        params[3:6] = [forward / 3, 0.0, 0.0]
        params[6:9] = [2 * forward / 3, lateral / 2, 0.0]
        params[9:12] = [forward, lateral, 0.0]

    elif trajectory_type == "s_curve":
        params[3:6] = [distance / 3, distance * 0.3, 0.0]
        params[6:9] = [2 * distance / 3, -distance * 0.3, 0.0]
        params[9:12] = [distance, 0.0, 0.0]

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
# OCP Building with Our Pipeline
# =============================================================================

def build_ocp_with_our_pipeline(
    rmodel: "pinocchio.Model",
    x0: np.ndarray,
    com_trajectory: np.ndarray,
    contact_sequence: ContactSequence,
    foothold_plans: Dict[str, List],
    foot_frame_ids: Dict[str, int],
    dt: float = 0.02,
    heading_trajectory: Optional[np.ndarray] = None,
) -> Tuple["crocoddyl.ShootingProblem", List]:
    """Build OCP using OCPFactory with full constraints.

    Uses OCPFactory which includes all necessary costs/constraints:
    - Friction cone constraints (prevents downward GRF)
    - State bounds (joint limits)
    - Weighted state regularization (base orientation, joint, velocity weights)
    - Body orientation tracking (for curve walking)
    - Proper cost weights (1e6 for CoM/foot tracking)

    Args:
        rmodel: Pinocchio robot model.
        x0: Initial state.
        com_trajectory: Dense CoM waypoints.
        contact_sequence: Contact sequence from GaitScheduler.
        foothold_plans: Foothold plans from FootholdPlanner.
        foot_frame_ids: Mapping from foot name to frame ID.
        dt: OCP timestep.
        heading_trajectory: Target yaw angles per timestep. Optional.

    Returns:
        Tuple of (problem, running_models)
    """
    factory = OCPFactory(
        rmodel=rmodel,
        foot_frame_ids=foot_frame_ids,
        mu=0.7,
    )
    # Use B1 standing configuration for regularization (not pinocchio.neutral)
    factory.x0 = x0

    problem = factory.build_problem(
        x0=x0,
        contact_sequence=contact_sequence,
        com_trajectory=com_trajectory,
        foot_trajectories=foothold_plans,
        dt=dt,
        heading_trajectory=heading_trajectory,
    )

    return problem, list(problem.runningModels)


# =============================================================================
# Main Test Function
# =============================================================================

def test_b1_gait(
    gait_type: str = "trot",
    trajectory_type: str = "straight",
    distance: float = 2.0,
    with_display: bool = False,
    with_plot: bool = False,
):
    """Test our gait pipeline with B1 robot.

    Args:
        gait_type: Type of gait ("trot", "walk", "pace", "bound").
        trajectory_type: Type of trajectory ("straight", "curve_left", etc.).
        distance: Forward distance in meters.
        with_display: If True, show 3D visualization.
        with_plot: If True, show plots.
    """
    print("=" * 70)
    print(f"B1 Gait Test: {gait_type.upper()} - {trajectory_type}")
    print("=" * 70)

    # Load B1 robot
    b1, rmodel, rdata, q0, v0, x0, foot_frame_ids = load_b1_robot()

    # Get standing CoM position
    pinocchio.centerOfMass(rmodel, rdata, q0)
    com_start = rdata.com[0].copy()
    print(f"\nInitial CoM position: [{com_start[0]:.3f}, {com_start[1]:.3f}, {com_start[2]:.3f}]")

    # Initialize gait components with B1-specific parameters
    gait_scheduler = GaitScheduler()
    foothold_planner = FootholdPlanner(
        hip_offsets=B1RobotConfig.HIP_OFFSETS,
        step_height=B1RobotConfig.GAIT_PARAMS["step_height"],
    )

    # Create CoM trajectory
    # Clamp distance: negative sentinel (-5) or very large values reduce OCP tractability.
    # Original working version used distance=0.5m, duration=2.0s, ~80 OCP nodes.
    # Beyond ~1.0m the 150-node full-trajectory OCP stops converging and the display
    # shows a "floating" CoM that isn't driven by foot contacts.
    if distance <= 0:
        distance = 0.8  # default for display: short enough for FDDP to converge
    distance = min(distance, 1.0)  # cap at 1.0m for display quality

    print("\nGenerating CoM trajectory...")
    duration = 2.0  # shorter horizon → fewer OCP nodes → better convergence for display
    dt = 0.02
    com_trajectory, bezier_params = create_com_trajectory(
        start_pos=com_start,
        trajectory_type=trajectory_type,
        distance=distance,
        duration=duration,
        dt=dt,
    )

    print(f"  Trajectory shape: {com_trajectory.shape}")
    print(f"  Start: [{com_trajectory[0, 0]:.3f}, {com_trajectory[0, 1]:.3f}, {com_trajectory[0, 2]:.3f}]")
    print(f"  End: [{com_trajectory[-1, 0]:.3f}, {com_trajectory[-1, 1]:.3f}, {com_trajectory[-1, 2]:.3f}]")

    # Generate contact sequence — cap num_cycles to trajectory duration so the
    # contact sequence doesn't overflow the CoM trajectory.  When num_cycles > needed,
    # the extra OCP nodes all target com_trajectory[-1] and all footholds pile up at
    # the same endpoint → robot appears to have legs "bunched together" in the display.
    step_dur    = B1RobotConfig.GAIT_PARAMS["step_duration"]
    support_dur = B1RobotConfig.GAIT_PARAMS["support_duration"]
    n_groups    = len(GaitScheduler.GAIT_PATTERNS[gait_type]["swing_groups"])
    cycle_dur   = n_groups * (step_dur + support_dur)  # seconds per full gait cycle
    num_cycles  = max(2, int(duration / cycle_dur) + 1)  # +1: first half-cycle is shorter

    print(f"\nGenerating {gait_type} contact sequence...")
    print(f"  cycle_dur={cycle_dur:.2f}s  →  num_cycles={num_cycles}")
    contact_sequence = gait_scheduler.generate(
        gait_type=gait_type,
        step_duration=step_dur,
        support_duration=support_dur,
        num_cycles=num_cycles,
    )

    print(f"  Number of phases: {len(contact_sequence)}")
    print(f"  Total duration: {contact_sequence.total_duration:.2f}s")

    # Get initial foot positions
    print("\nComputing initial foot positions...")
    initial_foot_positions = foothold_planner.get_footholds_at_time(
        com_position=com_start,
        heading=0.0,
    )

    for name, pos in initial_foot_positions.items():
        print(f"  {name}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}]")

    # Plan footholds
    print("\nPlanning footholds...")
    foothold_plans = foothold_planner.plan_footholds(
        com_trajectory=com_trajectory,
        contact_sequence=contact_sequence,
        current_foot_positions=initial_foot_positions,
        dt=dt,
    )

    # Compute step lengths
    step_lengths = foothold_planner.compute_step_lengths(foothold_plans)
    print("\nStep lengths:")
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
        if right_avg > 0:
            print(f"  Ratio: {left_avg/right_avg:.2f}")

        if trajectory_type == "curve_left" and left_avg < right_avg:
            print("  [OK] Inner (left) feet take shorter steps")
        elif trajectory_type == "curve_right" and right_avg < left_avg:
            print("  [OK] Inner (right) feet take shorter steps")

    # Compute heading trajectory from CoM trajectory tangent
    heading_trajectory = np.zeros(len(com_trajectory))
    for i in range(len(com_trajectory)):
        if i == 0 and len(com_trajectory) > 1:
            tangent = (com_trajectory[1] - com_trajectory[0]) / dt
        elif i >= len(com_trajectory) - 1:
            tangent = (com_trajectory[-1] - com_trajectory[-2]) / dt
        else:
            tangent = (com_trajectory[i + 1] - com_trajectory[i - 1]) / (2 * dt)
        heading_trajectory[i] = heading_from_tangent(tangent[:2])

    # Build OCP
    print("\nBuilding OCP with OCPFactory (includes friction cone, state bounds, etc.)...")
    problem, running_models = build_ocp_with_our_pipeline(
        rmodel=rmodel,
        x0=x0,
        com_trajectory=com_trajectory,
        contact_sequence=contact_sequence,
        foothold_plans=foothold_plans,
        foot_frame_ids=foot_frame_ids,
        dt=dt,
        heading_trajectory=heading_trajectory,
    )

    print(f"  Running models: {len(running_models)}")
    print(f"  Horizon: {len(running_models) * dt:.2f}s")

    # Create solver
    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-4

    # Set callbacks
    if with_plot:
        solver.setCallbacks([
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ])
    else:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    # Initial guess (quasi-static)
    print("\nSolving OCP...")
    xs = [x0] * (problem.T + 1)
    us = problem.quasiStatic([x0] * problem.T)

    t_start = time.time()
    converged = solver.solve(xs, us, 300, False)  # 100→300: full-traj OCP has 150+ nodes
    t_solve = time.time() - t_start

    print(f"\n  Converged: {converged}")
    print(f"  Iterations: {solver.iter}")
    print(f"  Cost: {solver.cost:.2f}")
    print(f"  Solve time: {t_solve * 1000:.1f}ms")

    # Extract solution
    xs_sol = np.array(solver.xs)
    us_sol = np.array(solver.us)

    print(f"\nSolution:")
    print(f"  States shape: {xs_sol.shape}")
    print(f"  Controls shape: {us_sol.shape}")

    # Display
    if with_display:
        print("\nStarting visualization...")
        try:
            import gepetto
            gepetto.corbaserver.Client()
            cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
            display = crocoddyl.GepettoDisplay(b1, 4, 4, cameraTF)
            print("  Using Gepetto viewer")
        except Exception:
            display = crocoddyl.MeshcatDisplay(b1)
            print("  Using Meshcat viewer")

        display.rate = -1
        display.freq = 1

        # Set initial camera: side+above view looking at trajectory midpoint
        if hasattr(display, "viewer"):
            try:
                traj_mid_x = (com_start[0] + com_trajectory[-1, 0]) / 2.0
                _cp = np.array([traj_mid_x, 4.0, 2.0])   # side 4m, height 2m
                _ct = np.array([traj_mid_x, 0.0, 0.45])  # aim at CoM height
                _z  = _cp - _ct;  _z /= np.linalg.norm(_z)
                _up = np.array([0., 0., 1.])
                _x  = np.cross(_up, _z);  _x /= np.linalg.norm(_x)
                _y  = np.cross(_z, _x)
                _T  = np.eye(4)
                _T[:3, 0] = _x;  _T[:3, 1] = _y
                _T[:3, 2] = _z;  _T[:3, 3] = _cp
                display.viewer["/Cameras/default"].set_transform(_T)
                print(f"  Camera: side+above at [{_cp[0]:.1f}, {_cp[1]:.1f}, {_cp[2]:.1f}]"
                      f"  →  [{_ct[0]:.1f}, {_ct[1]:.1f}, {_ct[2]:.1f}]")
            except Exception as _e:
                print(f"  (Camera set skipped: {_e})")

        print("\n  Press Ctrl+C to stop playback...")
        try:
            while True:
                display.displayFromSolver(solver)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  Stopped.")

    # Plot
    if with_plot:
        print("\nGenerating plots...")
        try:
            from crocoddyl.utils.quadruped import plotSolution

            # Plot solution
            plotSolution([solver], figIndex=1, show=False)

            # Plot convergence
            log = solver.getCallbacks()[1]
            crocoddyl.plotConvergence(
                log.costs,
                log.pregs,
                log.dregs,
                log.grads,
                log.stops,
                log.steps,
                figTitle=f"B1 {gait_type} - {trajectory_type}",
                figIndex=2,
                show=True,
            )
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

            # Fallback to matplotlib
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # CoM trajectory
            ax1 = axes[0, 0]
            ax1.plot(com_trajectory[:, 0], com_trajectory[:, 1], 'b-', linewidth=2, label='Reference')
            ax1.scatter(com_trajectory[0, 0], com_trajectory[0, 1], c='g', s=100, marker='o', label='Start')
            ax1.scatter(com_trajectory[-1, 0], com_trajectory[-1, 1], c='r', s=100, marker='*', label='End')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('CoM Trajectory (XY)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')

            # Joint positions
            ax2 = axes[0, 1]
            num_joints = rmodel.nq - 7  # Exclude floating base
            for j in range(min(6, num_joints)):  # Plot first 6 joints
                ax2.plot(xs_sol[:, 7 + j], label=f'Joint {j}')
            ax2.set_xlabel('Time step')
            ax2.set_ylabel('Joint position (rad)')
            ax2.set_title('Joint Positions')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            # Joint torques
            ax3 = axes[1, 0]
            for j in range(min(6, us_sol.shape[1])):
                ax3.plot(us_sol[:, j], label=f'Torque {j}')
            ax3.set_xlabel('Time step')
            ax3.set_ylabel('Torque (Nm)')
            ax3.set_title('Joint Torques')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

            # Step lengths
            ax4 = axes[1, 1]
            foot_colors = {'LF': 'g', 'RF': 'r', 'LH': 'b', 'RH': 'orange'}
            for foot_name, lengths in step_lengths.items():
                if lengths:
                    ax4.plot(range(1, len(lengths) + 1), lengths, 'o-',
                            color=foot_colors[foot_name], label=foot_name, linewidth=2)
            ax4.set_xlabel('Step Number')
            ax4.set_ylabel('Step Length (m)')
            ax4.set_title('Step Lengths by Foot')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.suptitle(f'B1 {gait_type.upper()} - {trajectory_type}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'b1_{gait_type}_{trajectory_type}.png', dpi=150)
            print(f"  Saved figure to: b1_{gait_type}_{trajectory_type}.png")
            plt.show()

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

    return solver, xs_sol, us_sol


# =============================================================================
# Velocity Comparison Utilities
# =============================================================================

def _quat_to_rotation_matrix(qxyzw: np.ndarray) -> np.ndarray:
    """Convert Pinocchio quaternion (qx, qy, qz, qw) to 3×3 rotation matrix."""
    qx, qy, qz, qw = qxyzw
    return np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qw*qz),      2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),       1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),       1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def extract_velocities_world(xs_sol: np.ndarray, nq: int = 19) -> dict:
    """Extract body-frame velocities from Crocoddyl solution and rotate to world frame.

    Pinocchio state layout (nq=19, nv=18):
        q  = [x,y,z, qx,qy,qz,qw, joint×12]   indices 0:19
        v  = [vx_b,vy_b,vz_b, wx_b,wy_b,wz_b, dq×12]  indices 19:37

    All v[0:6] are expressed in the LOCAL (body) frame.
    We rotate to world frame for comparison with the reference trajectory.

    Returns dict with keys:
        vx, vy, vz  — linear velocity in world frame (m/s)
        wx, wy, wz  — angular velocity in world frame (rad/s)
        roll, pitch — orientation angles (rad)
    """
    T = len(xs_sol)
    vx = np.zeros(T); vy = np.zeros(T); vz = np.zeros(T)
    wx = np.zeros(T); wy = np.zeros(T); wz = np.zeros(T)
    roll = np.zeros(T); pitch = np.zeros(T)

    for i, x in enumerate(xs_sol):
        # Rotation matrix from body frame to world frame
        R = _quat_to_rotation_matrix(x[3:7])          # (qx, qy, qz, qw)

        # Linear velocity: body → world
        v_w = R @ x[nq:nq + 3]
        vx[i], vy[i], vz[i] = v_w

        # Angular velocity: body → world
        w_w = R @ x[nq + 3:nq + 6]
        wx[i], wy[i], wz[i] = w_w

        # Roll and pitch from quaternion
        qx_i, qy_i, qz_i, qw_i = x[3], x[4], x[5], x[6]
        roll[i]  = np.arctan2(2*(qw_i*qx_i + qy_i*qz_i),
                               1 - 2*(qx_i**2 + qy_i**2))
        sinp     = np.clip(2*(qw_i*qy_i - qz_i*qx_i), -1.0, 1.0)
        pitch[i] = np.arcsin(sinp)

    return dict(vx=vx, vy=vy, vz=vz, wx=wx, wy=wy, wz=wz,
                roll=roll, pitch=pitch)


def compute_desired_velocities(
    com_trajectory: np.ndarray,
    heading_trajectory: np.ndarray,
    dt: float = 0.02,
) -> dict:
    """Compute desired velocity signals by differentiating the reference trajectories.

    Returns dict with keys:
        vx, vy  — world-frame linear velocity derivatives of CoM trajectory (m/s)
        wz      — yaw rate derivative of heading trajectory (rad/s)
    """
    # np.gradient uses central differences; accurate at internal points
    des_vx = np.gradient(com_trajectory[:, 0], dt)
    des_vy = np.gradient(com_trajectory[:, 1], dt)
    des_wz = np.gradient(heading_trajectory, dt)
    return dict(vx=des_vx, vy=des_vy, wz=des_wz)


def run_gait_for_comparison(
    gait_type: str,
    trajectory_type: str = "curve_left",
    distance: float = 1.5,
    duration: float = 3.0,
    dt: float = 0.02,
) -> Optional[dict]:
    """Run full B1 MPC pipeline for one gait and return velocity data.

    Returns dict with 'actual', 'desired', 'time', 'com_trajectory',
    'heading_trajectory'; or None if solve failed.
    """
    print(f"\n{'─'*55}")
    print(f"  Running gait: {gait_type.upper()}  |  traj: {trajectory_type}")
    print(f"{'─'*55}")

    try:
        b1, rmodel, rdata, q0, v0, x0, foot_frame_ids = load_b1_robot()
        pinocchio.centerOfMass(rmodel, rdata, q0)
        com_start = rdata.com[0].copy()

        gait_scheduler  = GaitScheduler()
        foothold_planner = FootholdPlanner(
            hip_offsets=B1RobotConfig.HIP_OFFSETS,
            step_height=B1RobotConfig.GAIT_PARAMS["step_height"],
        )

        com_trajectory, _ = create_com_trajectory(
            start_pos=com_start,
            trajectory_type=trajectory_type,
            distance=distance,
            duration=duration,
            dt=dt,
        )

        _step_dur    = B1RobotConfig.GAIT_PARAMS["step_duration"]
        _support_dur = B1RobotConfig.GAIT_PARAMS["support_duration"]
        _n_groups    = len(GaitScheduler.GAIT_PATTERNS[gait_type]["swing_groups"])
        _cycle_dur   = _n_groups * (_step_dur + _support_dur)
        _num_cycles  = max(2, int(duration / _cycle_dur) + 1)

        contact_sequence = gait_scheduler.generate(
            gait_type=gait_type,
            step_duration=_step_dur,
            support_duration=_support_dur,
            num_cycles=_num_cycles,
        )

        initial_foot_positions = foothold_planner.get_footholds_at_time(
            com_position=com_start, heading=0.0,
        )
        foothold_plans = foothold_planner.plan_footholds(
            com_trajectory=com_trajectory,
            contact_sequence=contact_sequence,
            current_foot_positions=initial_foot_positions,
            dt=dt,
        )

        # Heading trajectory
        heading_trajectory = np.zeros(len(com_trajectory))
        for i in range(len(com_trajectory)):
            if i == 0:
                tangent = (com_trajectory[1] - com_trajectory[0]) / dt
            elif i >= len(com_trajectory) - 1:
                tangent = (com_trajectory[-1] - com_trajectory[-2]) / dt
            else:
                tangent = (com_trajectory[i + 1] - com_trajectory[i - 1]) / (2 * dt)
            heading_trajectory[i] = heading_from_tangent(tangent[:2])

        problem, _ = build_ocp_with_our_pipeline(
            rmodel=rmodel,
            x0=x0,
            com_trajectory=com_trajectory,
            contact_sequence=contact_sequence,
            foothold_plans=foothold_plans,
            foot_frame_ids=foot_frame_ids,
            dt=dt,
            heading_trajectory=heading_trajectory,
        )

        solver = crocoddyl.SolverFDDP(problem)
        solver.th_stop = 1e-4
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

        xs_init = [x0] * (problem.T + 1)
        us_init = problem.quasiStatic([x0] * problem.T)

        t0 = time.time()
        converged = solver.solve(xs_init, us_init, 300, False)
        solve_time = time.time() - t0

        print(f"  Converged={converged}  cost={solver.cost:.1f}  "
              f"time={solve_time*1000:.0f}ms  nodes={problem.T}")

        xs_sol = np.array(solver.xs)           # (T+1, 37)
        nq = rmodel.nq                          # 19 for B1

        # Time axis — use the shorter of OCP horizon and reference trajectory
        T_sol  = len(xs_sol)
        T_ref  = len(com_trajectory)
        T_use  = min(T_sol, T_ref)
        t_axis = np.arange(T_use) * dt

        actual  = extract_velocities_world(xs_sol[:T_use], nq=nq)
        desired = compute_desired_velocities(
            com_trajectory[:T_use], heading_trajectory[:T_use], dt=dt
        )

        return dict(
            gait=gait_type,
            actual=actual,
            desired=desired,
            time=t_axis,
            com_trajectory=com_trajectory,
            heading_trajectory=heading_trajectory,
            converged=converged,
            cost=solver.cost,
            solve_time=solve_time,
        )

    except Exception as exc:
        print(f"  ERROR running {gait_type}: {exc}")
        import traceback; traceback.print_exc()
        return None


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Causal moving-average smoothing (no look-ahead).

    Pads the beginning with the first value to avoid boundary shrinkage.
    Window size is automatically clamped to signal length.
    """
    window = min(window, len(signal))
    if window <= 1:
        return signal.copy()
    kernel = np.ones(window) / window
    # 'same' mode: pad start with edge value for causal feel
    padded = np.concatenate([np.full(window - 1, signal[0]), signal])
    return np.convolve(padded, kernel, mode="valid")[: len(signal)]


def plot_velocity_comparison(
    results: list,
    trajectory_type: str = "curve_left",
    save_path: Optional[str] = None,
    show: bool = True,
    smooth_window: int = 8,
) -> str:
    """Generate the 4-panel velocity/time comparison figure.

    Each panel shows:
      - Raw actual velocity: thin, semi-transparent line (gait oscillations visible)
      - Smoothed actual velocity: thick solid line (overall tracking trend)
      - Desired velocity: black dashed line (shared reference across all gaits)

    Panels:
      1. CoM VEL X  (world frame) vs Desired VEL X
      2. CoM VEL Y  (world frame) vs Desired VEL Y
      3. CoM ANG Z  (yaw rate, world frame) vs Desired ANG Z
      4. CoM ROLL & PITCH (body orientation angles)

    Args:
        results      : list of dicts from run_gait_for_comparison().
        trajectory_type: used in figure title.
        save_path    : if given, save PNG to this path.
        show         : if True, call plt.show().
        smooth_window: moving-average window in timesteps (dt=0.02s per step).
                       Default 8 → ~0.16s window, smooths out step-cycle noise
                       while preserving low-frequency tracking errors.

    Returns:
        Path where figure was saved (or empty string).
    """
    import matplotlib
    matplotlib.use("Agg" if not show else matplotlib.get_backend())
    import matplotlib.pyplot as plt

    # Colour/style per gait
    GAIT_STYLES = {
        "trot":  {"color": "#1f77b4", "ls": "-",  "lw": 2.0},
        "walk":  {"color": "#ff7f0e", "ls": "--", "lw": 2.0},
        "pace":  {"color": "#2ca02c", "ls": "-.", "lw": 2.0},
        "bound": {"color": "#d62728", "ls": ":",  "lw": 2.0},
    }
    DES_STYLE  = {"color": "black", "ls": "--", "lw": 1.8, "alpha": 0.85,
                  "zorder": 10}
    RAW_ALPHA  = 0.22   # raw oscillating line: barely visible background
    SMOOTH_LW  = 2.2    # smoothed trend line width

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"B1 Quadruped — Gait Comparison on Curved Trajectory  [{trajectory_type}]\n"
        f"(thin = raw MPC solution, thick = {smooth_window}-step moving average, "
        f"dashed = desired reference)",
        fontsize=12, fontweight="bold",
    )

    ax_vx  = axes[0, 0]   # Panel 1: VEL X
    ax_vy  = axes[0, 1]   # Panel 2: VEL Y
    ax_wz  = axes[1, 0]   # Panel 3: ANG Z (yaw rate)
    ax_rp  = axes[1, 1]   # Panel 4: ROLL & PITCH

    desired_plotted = False   # plot desired only once (shared reference)

    for res in results:
        if res is None:
            continue
        gait = res["gait"]
        t    = res["time"]
        act  = res["actual"]
        des  = res["desired"]
        sty  = GAIT_STYLES.get(gait, {"color": "gray", "ls": "-", "lw": 1.5})
        col  = sty["color"]
        lbl  = gait.capitalize()

        def _plot_channel(ax, raw, desired_sig, des_label="Desired"):
            smooth = _moving_average(raw, smooth_window)
            # Raw: very transparent background
            ax.plot(t, raw, color=col, lw=0.7, alpha=RAW_ALPHA)
            # Smoothed trend: solid, full opacity, labelled
            ax.plot(t, smooth, color=col, ls=sty["ls"], lw=SMOOTH_LW,
                    label=lbl)
            if not desired_plotted:
                ax.plot(t, desired_sig, label=des_label, **DES_STYLE)

        # --- Panel 1: VEL X ---
        _plot_channel(ax_vx, act["vx"], des["vx"])

        # --- Panel 2: VEL Y ---
        _plot_channel(ax_vy, act["vy"], des["vy"])

        # --- Panel 3: ANG Z ---
        _plot_channel(ax_wz, act["wz"], des["wz"])

        # --- Panel 4: ROLL & PITCH (no "desired" — target is 0 for both) ---
        roll_deg  = np.degrees(act["roll"])
        pitch_deg = np.degrees(act["pitch"])
        s_roll    = _moving_average(roll_deg,  smooth_window)
        s_pitch   = _moving_average(pitch_deg, smooth_window)
        ax_rp.plot(t, roll_deg,  color=col, lw=0.7, alpha=RAW_ALPHA)
        ax_rp.plot(t, pitch_deg, color=col, lw=0.7, alpha=RAW_ALPHA)
        ax_rp.plot(t, s_roll,  color=col, ls=sty["ls"],  lw=SMOOTH_LW,
                   label=f"{lbl} Roll")
        ax_rp.plot(t, s_pitch, color=col, ls=":", lw=1.4, alpha=0.8,
                   label=f"{lbl} Pitch")

        desired_plotted = True  # subsequent gaits skip desired line

    # Decorations — Panel 1
    ax_vx.set_xlabel("Time (s)")
    ax_vx.set_ylabel("Velocity (m/s)")
    ax_vx.set_title("CoM VEL X  vs  Desired VEL X")
    ax_vx.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax_vx.legend(fontsize=9, loc="upper right")
    ax_vx.grid(True, alpha=0.3)

    # Decorations — Panel 2
    ax_vy.set_xlabel("Time (s)")
    ax_vy.set_ylabel("Velocity (m/s)")
    ax_vy.set_title("CoM VEL Y  vs  Desired VEL Y")
    ax_vy.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax_vy.legend(fontsize=9, loc="upper right")
    ax_vy.grid(True, alpha=0.3)

    # Decorations — Panel 3
    ax_wz.set_xlabel("Time (s)")
    ax_wz.set_ylabel("Angular velocity (rad/s)")
    ax_wz.set_title("CoM ANG Z (Yaw Rate)  vs  Desired ANG Z")
    ax_wz.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax_wz.legend(fontsize=9, loc="upper right")
    ax_wz.grid(True, alpha=0.3)

    # Decorations — Panel 4
    ax_rp.set_xlabel("Time (s)")
    ax_rp.set_ylabel("Angle (deg)")
    ax_rp.set_title("CoM ROLL & PITCH  (solid=Roll, dotted=Pitch)")
    ax_rp.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax_rp.legend(fontsize=7, loc="upper right", ncol=2)
    ax_rp.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    out_path = ""
    if save_path is None:
        # Auto-name based on trajectory type
        save_path = f"b1_velocity_comparison_{trajectory_type}.png"
    out_path = save_path
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot_velocity_comparison] Saved → {os.path.abspath(out_path)}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def run_velocity_comparison(
    trajectory_type: str = "curve_left",
    gaits: Optional[list] = None,
    distance: float = 1.5,
    duration: float = 3.0,
    dt: float = 0.02,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Run all gaits on the same curved trajectory and plot velocity comparison.

    Args:
        trajectory_type: "curve_left", "curve_right", "s_curve", or "straight".
        gaits          : list of gait names to compare (default: trot, walk, pace).
        distance       : total distance for the trajectory (m).
        duration       : trajectory time horizon (s).
        dt             : timestep (s).
        save_path      : output PNG path (auto-named if None).
        show           : display interactive plot window.
    """
    if gaits is None:
        gaits = ["trot", "walk", "pace"]

    print("\n" + "=" * 60)
    print(f"  Velocity Comparison — {trajectory_type}")
    print(f"  Gaits: {gaits}")
    print("=" * 60)

    results = []
    for gait in gaits:
        res = run_gait_for_comparison(
            gait_type=gait,
            trajectory_type=trajectory_type,
            distance=distance,
            duration=duration,
            dt=dt,
        )
        results.append(res)

    if all(r is None for r in results):
        print("ERROR: all gait runs failed — no data to plot.")
        return

    out = plot_velocity_comparison(
        results=results,
        trajectory_type=trajectory_type,
        save_path=save_path,
        show=show,
    )

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  {'Gait':<8} {'Converged':<12} {'Cost':>12} {'Time(ms)':>10}")
    print("  " + "-" * 46)
    for res in results:
        if res is not None:
            print(f"  {res['gait']:<8} {str(res['converged']):<12} "
                  f"{res['cost']:>12.1f} {res['solve_time']*1000:>10.0f}")
    print("=" * 60)
    print(f"\nFigure saved to: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test B1 quadruped gait with our custom pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_b1_gait.py                              # Basic test
    python test_b1_gait.py --display                    # With 3D visualization
    python test_b1_gait.py --plot                       # With plots
    python test_b1_gait.py --display --plot             # Both
    python test_b1_gait.py --gait trot --display        # Trotting with display
    python test_b1_gait.py --trajectory curve_left      # Curve walking test

    # Velocity comparison across gaits (generates 4-panel figure):
    python test_b1_gait.py --vel_compare
    python test_b1_gait.py --vel_compare --trajectory curve_right
    python test_b1_gait.py --vel_compare --gaits trot walk pace bound
    python test_b1_gait.py --vel_compare --save_fig results/vel_compare.png
        """
    )

    parser.add_argument(
        "--gait", type=str, default="trot",
        choices=["trot", "walk", "pace", "bound"],
        help="Gait type (default: trot)"
    )
    parser.add_argument(
        "--trajectory", type=str, default="straight",
        choices=["straight", "curve_left", "curve_right", "s_curve"],
        help="Trajectory type (default: straight)"
    )
    parser.add_argument(
        "--distance", type=float, default=-5,
        help="Forward distance in meters (default: 0.5)"
    )
    parser.add_argument(
        "--display", "-d", action="store_true",
        help="Show 3D visualization (Meshcat/Gepetto)"
    )
    parser.add_argument(
        "--plot", "-p", action="store_true",
        help="Show plots"
    )
    # ---- Velocity comparison flags ----
    parser.add_argument(
        "--vel_compare", action="store_true",
        help=(
            "Run velocity comparison: solve all gaits on the same curved trajectory "
            "and generate 4-panel velocity/time figure. "
            "Uses --trajectory (default: curve_left) and --gaits."
        ),
    )
    parser.add_argument(
        "--gaits", type=str, nargs="+",
        default=["trot", "walk", "pace"],
        metavar="GAIT",
        choices=["trot", "walk", "pace", "bound"],
        help="Gaits to include in comparison (default: trot walk pace)",
    )
    parser.add_argument(
        "--vel_traj", type=str, default="curve_left",
        choices=["straight", "curve_left", "curve_right", "s_curve"],
        help="Trajectory type for velocity comparison (default: curve_left)",
    )
    parser.add_argument(
        "--save_fig", type=str, default=None,
        metavar="PATH",
        help="Save velocity comparison figure to this path (e.g. results/compare.png)",
    )

    args = parser.parse_args()

    # Also check environment variables (like change_b1_model.py)
    with_display = args.display or "CROCODDYL_DISPLAY" in os.environ
    with_plot = args.plot or "CROCODDYL_PLOT" in os.environ

    if args.vel_compare:
        # ---- Velocity comparison mode ----
        traj = args.vel_traj
        dist = args.distance if args.distance > 0 else 1.5
        run_velocity_comparison(
            trajectory_type=traj,
            gaits=args.gaits,
            distance=dist,
            save_path=args.save_fig,
            show=with_plot,
        )
    else:
        # ---- Standard single-gait test ----
        # Negative distance (-5) is a sentinel meaning "use function default"
        dist = args.distance if args.distance > 0 else 2.0
        test_b1_gait(
            gait_type=args.gait,
            trajectory_type=args.trajectory,
            distance=dist,
            with_display=with_display,
            with_plot=with_plot,
        )


if __name__ == "__main__":
    main()
