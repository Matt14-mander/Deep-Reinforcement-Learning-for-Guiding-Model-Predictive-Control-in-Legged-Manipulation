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
) -> Tuple["crocoddyl.ShootingProblem", List]:
    """Build OCP using our custom pipeline (not Crocoddyl's demo).

    Args:
        rmodel: Pinocchio robot model.
        x0: Initial state.
        com_trajectory: Dense CoM waypoints.
        contact_sequence: Contact sequence from GaitScheduler.
        foothold_plans: Foothold plans from FootholdPlanner.
        foot_frame_ids: Mapping from foot name to frame ID.
        dt: OCP timestep.

    Returns:
        Tuple of (problem, running_models)
    """
    # Create state and actuation models
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)

    nu = actuation.nu

    # Build running models
    running_models = []
    knot_index = 0

    # Track swing indices for each foot
    foot_swing_indices = {foot: 0 for foot in foot_frame_ids.keys()}

    for phase in contact_sequence.phases:
        num_knots = max(1, round(phase.duration / dt))

        # Get support foot frame IDs
        support_foot_ids = [
            foot_frame_ids[foot]
            for foot in phase.support_feet
            if foot in foot_frame_ids
        ]

        for knot in range(num_knots):
            # Get CoM target
            traj_idx = min(knot_index, len(com_trajectory) - 1)
            com_target = com_trajectory[traj_idx]

            # Create contact model
            contact_model = crocoddyl.ContactModelMultiple(state, nu)
            for foot_id in support_foot_ids:
                contact = crocoddyl.ContactModel3D(
                    state, foot_id,
                    np.array([0.0, 0.0, 0.0]),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                    np.array([0.0, 50.0])
                )
                contact_model.addContact(f"contact_{foot_id}", contact)

            # Create cost model
            cost_model = crocoddyl.CostModelSum(state, nu)

            # CoM tracking cost
            com_residual = crocoddyl.ResidualModelCoMPosition(state, com_target, nu)
            com_cost = crocoddyl.CostModelResidual(state, com_residual)
            cost_model.addCost("comTrack", com_cost, 1e4)

            # Swing foot tracking costs
            for foot_name in phase.swing_feet:
                if foot_name not in foot_frame_ids:
                    continue

                frame_id = foot_frame_ids[foot_name]
                swing_idx = foot_swing_indices[foot_name]

                if swing_idx < len(foothold_plans.get(foot_name, [])):
                    plan = foothold_plans[foot_name][swing_idx]

                    if plan.trajectory is not None and len(plan.trajectory) > 0:
                        traj_knot = min(knot, len(plan.trajectory) - 1)
                        target_pos = plan.trajectory[traj_knot]
                    else:
                        alpha = knot / max(1, num_knots - 1)
                        target_pos = (1 - alpha) * plan.start_pos + alpha * plan.end_pos

                    foot_residual = crocoddyl.ResidualModelFrameTranslation(
                        state, frame_id, target_pos, nu
                    )
                    foot_cost = crocoddyl.CostModelResidual(state, foot_residual)
                    cost_model.addCost(f"footTrack_{foot_name}", foot_cost, 1e5)

            # State regularization
            state_residual = crocoddyl.ResidualModelState(state, x0, nu)
            state_cost = crocoddyl.CostModelResidual(state, state_residual)
            cost_model.addCost("stateReg", state_cost, 1e1)

            # Control regularization
            ctrl_residual = crocoddyl.ResidualModelControl(state, nu)
            ctrl_cost = crocoddyl.CostModelResidual(state, ctrl_residual)
            cost_model.addCost("ctrlReg", ctrl_cost, 1e-1)

            # Differential model
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                state, actuation, contact_model, cost_model
            )

            # Integrated model
            model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)
            running_models.append(model)
            knot_index += 1

        # Update swing indices after swing phase
        if phase.phase_type == "swing":
            for foot_name in phase.swing_feet:
                if foot_name in foot_swing_indices:
                    foot_swing_indices[foot_name] += 1

    # Terminal model (all feet in contact)
    all_foot_ids = list(foot_frame_ids.values())

    terminal_contact = crocoddyl.ContactModelMultiple(state, nu)
    for foot_id in all_foot_ids:
        contact = crocoddyl.ContactModel3D(
            state, foot_id,
            np.array([0.0, 0.0, 0.0]),
            pinocchio.LOCAL_WORLD_ALIGNED,
            nu,
            np.array([0.0, 0.0])
        )
        terminal_contact.addContact(f"contact_{foot_id}", contact)

    terminal_cost = crocoddyl.CostModelSum(state, nu)

    # Terminal CoM tracking (heavier weight)
    com_target = com_trajectory[-1]
    com_residual = crocoddyl.ResidualModelCoMPosition(state, com_target, nu)
    terminal_cost.addCost("comTrack", crocoddyl.CostModelResidual(state, com_residual), 1e5)

    # Terminal state regularization
    state_residual = crocoddyl.ResidualModelState(state, x0, nu)
    terminal_cost.addCost("stateReg", crocoddyl.CostModelResidual(state, state_residual), 1e2)

    terminal_dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, terminal_contact, terminal_cost
    )
    terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_dmodel, 0.0)

    # Create shooting problem
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

    return problem, running_models


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
    print("\nGenerating CoM trajectory...")
    duration = 3.0
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

    # Generate contact sequence
    print(f"\nGenerating {gait_type} contact sequence...")
    contact_sequence = gait_scheduler.generate(
        gait_type=gait_type,
        step_duration=B1RobotConfig.GAIT_PARAMS["step_duration"],
        support_duration=B1RobotConfig.GAIT_PARAMS["support_duration"],
        num_cycles=12,
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

    # Build OCP
    print("\nBuilding OCP with our pipeline...")
    problem, running_models = build_ocp_with_our_pipeline(
        rmodel=rmodel,
        x0=x0,
        com_trajectory=com_trajectory,
        contact_sequence=contact_sequence,
        foothold_plans=foothold_plans,
        foot_frame_ids=foot_frame_ids,
        dt=dt,
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
    converged = solver.solve(xs, us, 100, False)
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
        "--distance", type=float, default=0.5,
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

    args = parser.parse_args()

    # Also check environment variables (like change_b1_model.py)
    with_display = args.display or "CROCODDYL_DISPLAY" in os.environ
    with_plot = args.plot or "CROCODDYL_PLOT" in os.environ

    test_b1_gait(
        gait_type=args.gait,
        trajectory_type=args.trajectory,
        distance=args.distance,
        with_display=with_display,
        with_plot=with_plot,
    )


if __name__ == "__main__":
    main()
