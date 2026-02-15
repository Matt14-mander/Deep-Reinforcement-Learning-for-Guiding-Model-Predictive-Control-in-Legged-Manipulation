#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRF extraction/visualization and MPC solve performance benchmark for B1 quadruped.

This script provides two analysis tools for the B1 gait pipeline:

1. **GRF Analysis** (--grf): Extract ground reaction forces from Crocoddyl solver
   data, verify physical constraints (Fz > 0, friction cone), and plot force
   time series per foot.

2. **Performance Benchmark** (--benchmark): Measure FDDP solve time vs horizon
   length, cold-start vs warm-start, and iteration counts.

Usage:
    python test_b1_analysis.py --grf                  # GRF analysis only
    python test_b1_analysis.py --benchmark            # Performance benchmark only
    python test_b1_analysis.py --grf --benchmark      # Both
    python test_b1_analysis.py --grf --gait walk      # GRF for walk gait
    python test_b1_analysis.py --grf --trajectory curve_left  # GRF for curve
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
# Shared: B1 Robot Config & Loading (reuse from test_b1_gait.py)
# =============================================================================

# B1 configuration
B1_FOOT_FRAME_NAMES = {
    "LF": "FL_foot",
    "RF": "FR_foot",
    "LH": "RL_foot",
    "RH": "RR_foot",
}

B1_HIP_OFFSETS = {
    "LF": np.array([+0.3, +0.1, 0.0]),
    "RF": np.array([+0.3, -0.1, 0.0]),
    "LH": np.array([-0.3, +0.1, 0.0]),
    "RH": np.array([-0.3, -0.1, 0.0]),
}

B1_GAIT_PARAMS = {
    "step_duration": 0.15,
    "support_duration": 0.05,
    "step_height": 0.15,
}


def load_b1_robot():
    """Load B1 robot model."""
    b1 = example_robot_data.load("b1")
    rmodel = b1.model
    rdata = rmodel.createData()
    q0 = rmodel.referenceConfigurations["standing"].copy()
    v0 = pinocchio.utils.zero(rmodel.nv)
    x0 = np.concatenate([q0, v0])

    foot_frame_ids = {}
    for our_name, b1_name in B1_FOOT_FRAME_NAMES.items():
        frame_id = rmodel.getFrameId(b1_name)
        foot_frame_ids[our_name] = frame_id

    return b1, rmodel, rdata, q0, v0, x0, foot_frame_ids


def create_com_trajectory(
    start_pos: np.ndarray,
    trajectory_type: str = "straight",
    distance: float = 0.5,
    duration: float = 3.0,
    dt: float = 0.02,
) -> np.ndarray:
    """Create CoM trajectory using Bezier curves."""
    trajectory_gen = BezierTrajectoryGenerator(degree=3, state_dim=3, max_displacement=2.0)
    params = np.zeros(12)

    if trajectory_type == "straight":
        params[3:6] = [distance / 3, 0.0, 0.0]
        params[6:9] = [2 * distance / 3, 0.0, 0.0]
        params[9:12] = [distance, 0.0, 0.0]
    elif trajectory_type == "curve_left":
        turn_angle = np.pi / 3
        lateral = distance * np.sin(turn_angle)
        forward = distance * np.cos(turn_angle)
        params[3:6] = [forward / 3, 0.0, 0.0]
        params[6:9] = [2 * forward / 3, lateral / 2, 0.0]
        params[9:12] = [forward, lateral, 0.0]
    elif trajectory_type == "curve_right":
        turn_angle = np.pi / 3
        lateral = -distance * np.sin(turn_angle)
        forward = distance * np.cos(turn_angle)
        params[3:6] = [forward / 3, 0.0, 0.0]
        params[6:9] = [2 * forward / 3, lateral / 2, 0.0]
        params[9:12] = [forward, lateral, 0.0]
    elif trajectory_type == "s_curve":
        params[3:6] = [distance / 3, distance * 0.3, 0.0]
        params[6:9] = [2 * distance / 3, -distance * 0.3, 0.0]
        params[9:12] = [distance, 0.0, 0.0]

    trajectory = trajectory_gen.params_to_waypoints(
        params=params, dt=dt, horizon=duration, start_position=start_pos,
    )
    return trajectory


def solve_gait_problem(
    gait_type: str = "trot",
    trajectory_type: str = "straight",
    distance: float = 0.5,
    duration: float = 3.0,
    dt: float = 0.02,
    max_iterations: int = 100,
    verbose: bool = True,
):
    """Build and solve a complete gait OCP. Returns solver and metadata.

    Returns:
        Tuple of (solver, rmodel, foot_frame_ids, contact_sequence, dt, com_trajectory)
    """
    b1, rmodel, rdata, q0, v0, x0, foot_frame_ids = load_b1_robot()
    pinocchio.centerOfMass(rmodel, rdata, q0)
    com_start = rdata.com[0].copy()

    gait_scheduler = GaitScheduler()
    foothold_planner = FootholdPlanner(
        hip_offsets=B1_HIP_OFFSETS,
        step_height=B1_GAIT_PARAMS["step_height"],
    )

    com_trajectory = create_com_trajectory(
        start_pos=com_start,
        trajectory_type=trajectory_type,
        distance=distance,
        duration=duration,
        dt=dt,
    )

    contact_sequence = gait_scheduler.generate(
        gait_type=gait_type,
        step_duration=B1_GAIT_PARAMS["step_duration"],
        support_duration=B1_GAIT_PARAMS["support_duration"],
        num_cycles=12,
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
        if i == 0 and len(com_trajectory) > 1:
            tangent = (com_trajectory[1] - com_trajectory[0]) / dt
        elif i >= len(com_trajectory) - 1:
            tangent = (com_trajectory[-1] - com_trajectory[-2]) / dt
        else:
            tangent = (com_trajectory[i + 1] - com_trajectory[i - 1]) / (2 * dt)
        heading_trajectory[i] = heading_from_tangent(tangent[:2])

    factory = OCPFactory(rmodel=rmodel, foot_frame_ids=foot_frame_ids, mu=0.7)
    factory.x0 = x0

    problem = factory.build_problem(
        x0=x0,
        contact_sequence=contact_sequence,
        com_trajectory=com_trajectory,
        foot_trajectories=foothold_plans,
        dt=dt,
        heading_trajectory=heading_trajectory,
    )

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-4

    if verbose:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    xs = [x0] * (problem.T + 1)
    us = problem.quasiStatic([x0] * problem.T)

    t_start = time.time()
    converged = solver.solve(xs, us, max_iterations, False)
    t_solve = time.time() - t_start

    if verbose:
        print(f"\n  Converged: {converged}")
        print(f"  Iterations: {solver.iter}")
        print(f"  Cost: {solver.cost:.2f}")
        print(f"  Solve time: {t_solve * 1000:.1f}ms")

    return solver, rmodel, foot_frame_ids, contact_sequence, dt, com_trajectory


# =============================================================================
# GRF Extraction
# =============================================================================

def extract_grf(
    solver: "crocoddyl.SolverFDDP",
    foot_frame_ids: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Extract ground reaction forces from Crocoddyl solver data.

    After solving, the contact forces are stored in:
        solver.problem.runningDatas[i].differential.multibody.contacts
            .contacts[contact_key].f

    The force `f` is a pinocchio.Force object with .linear and .angular components.
    For 3D contacts (ContactModel3D), we extract the linear part (Fx, Fy, Fz).

    Args:
        solver: Solved FDDP solver.
        foot_frame_ids: Dict mapping foot name to frame ID.

    Returns:
        Dict mapping foot name to force array of shape (T, 3).
        Forces are in world frame (LOCAL_WORLD_ALIGNED).
    """
    problem = solver.problem
    T = len(problem.runningDatas)

    # Initialize force arrays
    forces = {name: np.zeros((T, 3)) for name in foot_frame_ids}

    # Build reverse lookup: frame_id -> foot_name
    id_to_name = {fid: name for name, fid in foot_frame_ids.items()}

    for t in range(T):
        data = problem.runningDatas[t]

        # Skip impulse models (they don't have differential.multibody)
        if not hasattr(data, "differential"):
            continue

        diff_data = data.differential
        if not hasattr(diff_data, "multibody"):
            continue

        contact_data = diff_data.multibody.contacts

        # Iterate over contacts in this node
        for contact_key, contact_item in contact_data.contacts.todict().items():
            # Extract frame ID from contact key (format: "contact_{frame_id}")
            try:
                frame_id = int(contact_key.split("_")[1])
            except (IndexError, ValueError):
                continue

            if frame_id not in id_to_name:
                continue

            foot_name = id_to_name[frame_id]

            # Extract force (pinocchio.Force)
            f = contact_item.f
            # f.linear gives [Fx, Fy, Fz] in the contact frame
            # Since we use LOCAL_WORLD_ALIGNED, this is in world frame
            forces[foot_name][t] = f.linear.copy()

    return forces


def analyze_grf(
    forces: Dict[str, np.ndarray],
    dt: float,
    mu: float = 0.7,
) -> Dict[str, dict]:
    """Analyze GRF for physical validity.

    Checks:
    1. Fz > 0 when in contact (no pulling the ground)
    2. Friction cone: sqrt(Fx² + Fy²) <= mu * Fz
    3. Force magnitude statistics

    Args:
        forces: Dict from extract_grf.
        dt: Timestep for time axis.
        mu: Friction coefficient.

    Returns:
        Dict mapping foot name to analysis results.
    """
    results = {}

    for foot_name, force_array in forces.items():
        T = len(force_array)
        time_axis = np.arange(T) * dt

        # Identify contact timesteps (Fz != 0 means in contact)
        in_contact = np.abs(force_array[:, 2]) > 1e-6
        num_contact_steps = np.sum(in_contact)

        if num_contact_steps == 0:
            results[foot_name] = {
                "num_contact_steps": 0,
                "fz_violations": 0,
                "cone_violations": 0,
                "mean_fz": 0.0,
                "max_fz": 0.0,
                "mean_tangential": 0.0,
            }
            continue

        # Check Fz > 0 (upward force)
        fz = force_array[in_contact, 2]
        fz_violations = np.sum(fz < 0)
        fz_violation_ratio = fz_violations / num_contact_steps

        # Check friction cone
        ft = np.sqrt(force_array[in_contact, 0] ** 2 + force_array[in_contact, 1] ** 2)
        cone_limit = mu * np.abs(fz)
        cone_violations = np.sum(ft > cone_limit + 1e-6)
        cone_violation_ratio = cone_violations / num_contact_steps

        # Force statistics
        mean_fz = np.mean(fz[fz > 0]) if np.any(fz > 0) else 0.0
        max_fz = np.max(np.abs(fz))
        mean_tangential = np.mean(ft)

        results[foot_name] = {
            "num_contact_steps": int(num_contact_steps),
            "fz_violations": int(fz_violations),
            "fz_violation_ratio": float(fz_violation_ratio),
            "cone_violations": int(cone_violations),
            "cone_violation_ratio": float(cone_violation_ratio),
            "mean_fz": float(mean_fz),
            "max_fz": float(max_fz),
            "mean_tangential": float(mean_tangential),
            "max_tangential": float(np.max(ft)) if len(ft) > 0 else 0.0,
        }

    return results


def print_grf_report(
    analysis: Dict[str, dict],
    gait_type: str,
    trajectory_type: str,
):
    """Print a formatted GRF analysis report."""
    print("\n" + "=" * 70)
    print(f"GRF Analysis Report: {gait_type.upper()} - {trajectory_type}")
    print("=" * 70)

    all_ok = True

    for foot_name in ["LF", "RF", "LH", "RH"]:
        if foot_name not in analysis:
            continue

        a = analysis[foot_name]
        print(f"\n  {foot_name}:")
        print(f"    Contact steps: {a['num_contact_steps']}")

        if a["num_contact_steps"] == 0:
            print(f"    (no contact)")
            continue

        # Fz check
        if a["fz_violations"] > 0:
            print(f"    ❌ Fz violations: {a['fz_violations']} "
                  f"({a['fz_violation_ratio']:.1%}) - DOWNWARD FORCES DETECTED")
            all_ok = False
        else:
            print(f"    ✓ Fz: all upward ({a['mean_fz']:.1f}N avg, {a['max_fz']:.1f}N max)")

        # Friction cone check
        if a["cone_violations"] > 0:
            print(f"    ❌ Cone violations: {a['cone_violations']} "
                  f"({a['cone_violation_ratio']:.1%})")
            all_ok = False
        else:
            print(f"    ✓ Friction cone: satisfied "
                  f"(tangential {a['mean_tangential']:.1f}N avg)")

    print(f"\n  Overall: {'✓ ALL CONSTRAINTS SATISFIED' if all_ok else '❌ VIOLATIONS DETECTED'}")
    print("=" * 70)

    return all_ok


def plot_grf(
    forces: Dict[str, np.ndarray],
    dt: float,
    gait_type: str,
    trajectory_type: str,
    mu: float = 0.7,
    save_path: Optional[str] = None,
):
    """Plot GRF time series for all feet.

    Creates a 4x1 subplot (one per foot) showing:
    - Fx (red), Fy (green), Fz (blue) time series
    - Fz = 0 line (dashed)
    - Friction cone limit (gray shaded region)

    Args:
        forces: Dict from extract_grf.
        dt: Timestep.
        gait_type: Gait name for title.
        trajectory_type: Trajectory name for title.
        mu: Friction coefficient for cone visualization.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    foot_order = ["LF", "RF", "LH", "RH"]
    foot_colors = {"LF": "#2ecc71", "RF": "#e74c3c", "LH": "#3498db", "RH": "#e67e22"}

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for idx, foot_name in enumerate(foot_order):
        ax = axes[idx]
        force = forces.get(foot_name, np.zeros((1, 3)))
        T = len(force)
        t = np.arange(T) * dt

        # Plot force components
        ax.plot(t, force[:, 0], "r-", linewidth=1.2, alpha=0.8, label="Fx (forward)")
        ax.plot(t, force[:, 1], "g-", linewidth=1.2, alpha=0.8, label="Fy (lateral)")
        ax.plot(t, force[:, 2], "b-", linewidth=2.0, label="Fz (vertical)")

        # Fz = 0 line
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Mark Fz < 0 violations
        in_contact = np.abs(force[:, 2]) > 1e-6
        fz_violations = in_contact & (force[:, 2] < 0)
        if np.any(fz_violations):
            ax.scatter(
                t[fz_violations], force[fz_violations, 2],
                c="red", marker="x", s=50, zorder=5, label="Fz < 0 violation"
            )

        # Friction cone limit visualization
        # When Fz > 0, tangential force should be < mu * Fz
        ft = np.sqrt(force[:, 0] ** 2 + force[:, 1] ** 2)
        cone_limit = mu * np.maximum(force[:, 2], 0)
        cone_violations = in_contact & (ft > cone_limit + 1e-6)
        if np.any(cone_violations):
            ax.scatter(
                t[cone_violations],
                ft[cone_violations],
                c="orange", marker="^", s=30, zorder=5, label="Cone violation"
            )

        ax.set_ylabel(f"{foot_name}\nForce (N)")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)

        # Color the foot name
        ax.annotate(
            foot_name, xy=(0.02, 0.85), xycoords="axes fraction",
            fontsize=14, fontweight="bold", color=foot_colors[foot_name],
        )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Ground Reaction Forces: {gait_type.upper()} - {trajectory_type}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved GRF plot to: {save_path}")
    plt.show()


def plot_grf_summary(
    forces: Dict[str, np.ndarray],
    dt: float,
    gait_type: str,
    trajectory_type: str,
    mu: float = 0.7,
    save_path: Optional[str] = None,
):
    """Plot GRF summary: friction cone diagram and Fz histogram.

    Args:
        forces: Dict from extract_grf.
        dt: Timestep.
        gait_type: Gait name.
        trajectory_type: Trajectory name.
        mu: Friction coefficient.
        save_path: Optional save path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    foot_colors = {"LF": "#2ecc71", "RF": "#e74c3c", "LH": "#3498db", "RH": "#e67e22"}

    # Left: Friction cone diagram (Ft vs Fz)
    ax1 = axes[0]
    max_fz = 0
    for foot_name, force in forces.items():
        in_contact = np.abs(force[:, 2]) > 1e-6
        if not np.any(in_contact):
            continue

        fz = force[in_contact, 2]
        ft = np.sqrt(force[in_contact, 0] ** 2 + force[in_contact, 1] ** 2)
        ax1.scatter(fz, ft, c=foot_colors[foot_name], s=8, alpha=0.5, label=foot_name)
        max_fz = max(max_fz, np.max(np.abs(fz)))

    # Draw cone boundary
    fz_range = np.linspace(0, max_fz * 1.1, 100)
    ax1.plot(fz_range, mu * fz_range, "k--", linewidth=2, label=f"Cone (μ={mu})")
    ax1.fill_between(fz_range, 0, mu * fz_range, alpha=0.1, color="green", label="Feasible")

    ax1.axvline(x=0, color="red", linestyle=":", alpha=0.5, label="Fz=0")
    ax1.set_xlabel("Fz (vertical, N)")
    ax1.set_ylabel("Ft (tangential, N)")
    ax1.set_title("Friction Cone Diagram")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Fz histogram
    ax2 = axes[1]
    for foot_name, force in forces.items():
        in_contact = np.abs(force[:, 2]) > 1e-6
        if not np.any(in_contact):
            continue

        fz = force[in_contact, 2]
        ax2.hist(
            fz, bins=50, alpha=0.5, color=foot_colors[foot_name],
            label=f"{foot_name} (n={len(fz)})", edgecolor="none",
        )

    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Fz=0")
    ax2.set_xlabel("Fz (N)")
    ax2.set_ylabel("Count")
    ax2.set_title("Vertical Force Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"GRF Summary: {gait_type.upper()} - {trajectory_type}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved GRF summary to: {save_path}")
    plt.show()


# =============================================================================
# Performance Benchmark
# =============================================================================

def benchmark_solve_time(
    horizons: Optional[List[int]] = None,
    gait_type: str = "trot",
    trajectory_type: str = "straight",
    num_repeats: int = 3,
) -> Dict[str, list]:
    """Benchmark FDDP solve time vs horizon length.

    For each horizon length:
    1. Build OCP with that many running models (truncated from full problem)
    2. Cold-start solve (quasi-static initial guess)
    3. Warm-start solve (shift previous solution)

    Args:
        horizons: List of horizon lengths to test.
        gait_type: Gait type for the OCP.
        trajectory_type: Trajectory type.
        num_repeats: Number of times to repeat each measurement.

    Returns:
        Dict with benchmark results.
    """
    if horizons is None:
        horizons = [10, 25, 50, 75, 100, 150]

    print("=" * 70)
    print(f"MPC Solve Performance Benchmark: {gait_type} - {trajectory_type}")
    print("=" * 70)

    # Load robot once
    b1, rmodel, rdata, q0, v0, x0, foot_frame_ids = load_b1_robot()
    pinocchio.centerOfMass(rmodel, rdata, q0)
    com_start = rdata.com[0].copy()

    gait_scheduler = GaitScheduler()
    foothold_planner = FootholdPlanner(
        hip_offsets=B1_HIP_OFFSETS,
        step_height=B1_GAIT_PARAMS["step_height"],
    )

    results = {
        "horizons": [],
        "cold_start_mean": [],
        "cold_start_std": [],
        "warm_start_mean": [],
        "warm_start_std": [],
        "cold_iterations": [],
        "warm_iterations": [],
        "cold_converged": [],
        "warm_converged": [],
        "cold_cost": [],
        "warm_cost": [],
    }

    dt = 0.02
    duration = 5.0  # Enough for longest horizon

    com_trajectory = create_com_trajectory(
        start_pos=com_start,
        trajectory_type=trajectory_type,
        distance=1.5,
        duration=duration,
        dt=dt,
    )

    contact_sequence = gait_scheduler.generate(
        gait_type=gait_type,
        step_duration=B1_GAIT_PARAMS["step_duration"],
        support_duration=B1_GAIT_PARAMS["support_duration"],
        num_cycles=20,
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

    heading_trajectory = np.zeros(len(com_trajectory))
    for i in range(len(com_trajectory)):
        if i == 0 and len(com_trajectory) > 1:
            tangent = (com_trajectory[1] - com_trajectory[0]) / dt
        elif i >= len(com_trajectory) - 1:
            tangent = (com_trajectory[-1] - com_trajectory[-2]) / dt
        else:
            tangent = (com_trajectory[i + 1] - com_trajectory[i - 1]) / (2 * dt)
        heading_trajectory[i] = heading_from_tangent(tangent[:2])

    factory = OCPFactory(rmodel=rmodel, foot_frame_ids=foot_frame_ids, mu=0.7)
    factory.x0 = x0

    # Build full problem once
    full_problem = factory.build_problem(
        x0=x0,
        contact_sequence=contact_sequence,
        com_trajectory=com_trajectory,
        foot_trajectories=foothold_plans,
        dt=dt,
        heading_trajectory=heading_trajectory,
    )
    full_models = list(full_problem.runningModels)
    max_available = len(full_models)

    print(f"\n  Full problem has {max_available} running models")
    print(f"  Testing horizons: {horizons}")
    print(f"  Repeats per measurement: {num_repeats}")

    for horizon in horizons:
        if horizon > max_available:
            print(f"\n  Skipping horizon={horizon} (only {max_available} models available)")
            continue

        print(f"\n  Horizon = {horizon} ({horizon * dt:.2f}s):")

        # Truncate problem to this horizon
        models_subset = full_models[:horizon]
        terminal = full_problem.terminalModel
        sub_problem = crocoddyl.ShootingProblem(x0, models_subset, terminal)

        cold_times = []
        cold_iters = []
        cold_conv = []
        cold_costs = []
        warm_times = []
        warm_iters = []
        warm_conv = []
        warm_costs = []

        for rep in range(num_repeats):
            # Cold-start solve
            solver = crocoddyl.SolverFDDP(sub_problem)
            solver.th_stop = 1e-4

            xs_init = [x0] * (horizon + 1)
            us_init = sub_problem.quasiStatic([x0] * horizon)

            t0 = time.time()
            conv = solver.solve(xs_init, us_init, 100, False)
            t_cold = time.time() - t0

            cold_times.append(t_cold)
            cold_iters.append(solver.iter)
            cold_conv.append(bool(conv))
            cold_costs.append(float(solver.cost))

            # Warm-start solve (shift solution by 1 step)
            prev_xs = list(solver.xs)
            prev_us = list(solver.us)

            # Simulate "next step": shift by one
            xs_warm = [x0] + prev_xs[2:]
            while len(xs_warm) < horizon + 1:
                xs_warm.append(prev_xs[-1].copy())
            xs_warm = xs_warm[:horizon + 1]

            us_warm = prev_us[1:]
            while len(us_warm) < horizon:
                us_warm.append(prev_us[-1].copy())
            us_warm = us_warm[:horizon]

            solver2 = crocoddyl.SolverFDDP(sub_problem)
            solver2.th_stop = 1e-4
            solver2.setCandidate(xs_warm, us_warm, False)

            t0 = time.time()
            conv2 = solver2.solve([], [], 100, False, 0.0)
            t_warm = time.time() - t0

            warm_times.append(t_warm)
            warm_iters.append(solver2.iter)
            warm_conv.append(bool(conv2))
            warm_costs.append(float(solver2.cost))

        results["horizons"].append(horizon)
        results["cold_start_mean"].append(np.mean(cold_times) * 1000)
        results["cold_start_std"].append(np.std(cold_times) * 1000)
        results["warm_start_mean"].append(np.mean(warm_times) * 1000)
        results["warm_start_std"].append(np.std(warm_times) * 1000)
        results["cold_iterations"].append(np.mean(cold_iters))
        results["warm_iterations"].append(np.mean(warm_iters))
        results["cold_converged"].append(np.mean(cold_conv))
        results["warm_converged"].append(np.mean(warm_conv))
        results["cold_cost"].append(np.mean(cold_costs))
        results["warm_cost"].append(np.mean(warm_costs))

        print(f"    Cold: {np.mean(cold_times)*1000:.1f}±{np.std(cold_times)*1000:.1f}ms, "
              f"{np.mean(cold_iters):.0f} iters, "
              f"conv={np.mean(cold_conv):.0%}")
        print(f"    Warm: {np.mean(warm_times)*1000:.1f}±{np.std(warm_times)*1000:.1f}ms, "
              f"{np.mean(warm_iters):.0f} iters, "
              f"conv={np.mean(warm_conv):.0%}")
        speedup = np.mean(cold_times) / np.mean(warm_times) if np.mean(warm_times) > 0 else float("inf")
        print(f"    Speedup: {speedup:.1f}x")

    return results


def print_benchmark_report(results: Dict[str, list]):
    """Print formatted benchmark results table."""
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"{'Horizon':>8} {'T(s)':>6} | {'Cold(ms)':>10} {'Iter':>5} {'Conv':>5} | "
          f"{'Warm(ms)':>10} {'Iter':>5} {'Conv':>5} | {'Speedup':>8}")
    print("-" * 82)

    for i, h in enumerate(results["horizons"]):
        dt_h = h * 0.02
        cold = results["cold_start_mean"][i]
        cold_s = results["cold_start_std"][i]
        warm = results["warm_start_mean"][i]
        warm_s = results["warm_start_std"][i]
        c_it = results["cold_iterations"][i]
        w_it = results["warm_iterations"][i]
        c_cv = results["cold_converged"][i]
        w_cv = results["warm_converged"][i]
        speedup = cold / warm if warm > 0 else float("inf")

        print(f"{h:>8} {dt_h:>5.2f}s | "
              f"{cold:>7.1f}±{cold_s:<3.0f} {c_it:>5.0f} {c_cv:>4.0%} | "
              f"{warm:>7.1f}±{warm_s:<3.0f} {w_it:>5.0f} {w_cv:>4.0%} | "
              f"{speedup:>7.1f}x")

    # MPC feasibility analysis
    print("\n  MPC Feasibility (real-time requirement: solve < dt):")
    for i, h in enumerate(results["horizons"]):
        dt_ms = h * 0.02 * 1000  # not dt, but the actual control period
        # MPC period is typically 20ms (dt=0.02)
        mpc_period_ms = 20.0
        warm_ms = results["warm_start_mean"][i]
        feasible = warm_ms < mpc_period_ms
        ratio = warm_ms / mpc_period_ms
        status = "✓ REAL-TIME" if feasible else "❌ TOO SLOW"
        print(f"    H={h:>3}: {warm_ms:.1f}ms / {mpc_period_ms:.0f}ms = {ratio:.1%} [{status}]")


def plot_benchmark(
    results: Dict[str, list],
    save_path: Optional[str] = None,
):
    """Plot benchmark results.

    Creates a 2x2 subplot:
    - Top-left: Solve time vs horizon
    - Top-right: Iterations vs horizon
    - Bottom-left: Speedup ratio
    - Bottom-right: Convergence rate

    Args:
        results: Dict from benchmark_solve_time.
        save_path: Optional save path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    horizons = results["horizons"]

    # Top-left: Solve time vs horizon
    ax1 = axes[0, 0]
    ax1.errorbar(
        horizons, results["cold_start_mean"], yerr=results["cold_start_std"],
        fmt="o-", color="#e74c3c", linewidth=2, capsize=4, label="Cold start",
    )
    ax1.errorbar(
        horizons, results["warm_start_mean"], yerr=results["warm_start_std"],
        fmt="s-", color="#2ecc71", linewidth=2, capsize=4, label="Warm start",
    )
    ax1.axhline(y=20.0, color="gray", linestyle="--", alpha=0.7, label="dt=20ms (real-time)")
    ax1.set_xlabel("Horizon (steps)")
    ax1.set_ylabel("Solve time (ms)")
    ax1.set_title("Solve Time vs Horizon")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Top-right: Iterations vs horizon
    ax2 = axes[0, 1]
    ax2.plot(horizons, results["cold_iterations"], "o-", color="#e74c3c",
             linewidth=2, label="Cold start")
    ax2.plot(horizons, results["warm_iterations"], "s-", color="#2ecc71",
             linewidth=2, label="Warm start")
    ax2.set_xlabel("Horizon (steps)")
    ax2.set_ylabel("Iterations")
    ax2.set_title("Solver Iterations vs Horizon")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Speedup
    ax3 = axes[1, 0]
    speedups = [c / w if w > 0 else 0 for c, w in
                zip(results["cold_start_mean"], results["warm_start_mean"])]
    ax3.bar(range(len(horizons)), speedups, color="#3498db", alpha=0.7)
    ax3.set_xticks(range(len(horizons)))
    ax3.set_xticklabels([str(h) for h in horizons])
    ax3.set_xlabel("Horizon (steps)")
    ax3.set_ylabel("Speedup (cold/warm)")
    ax3.set_title("Warm-start Speedup")
    ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Convergence
    ax4 = axes[1, 1]
    ax4.plot(horizons, [c * 100 for c in results["cold_converged"]],
             "o-", color="#e74c3c", linewidth=2, label="Cold start")
    ax4.plot(horizons, [c * 100 for c in results["warm_converged"]],
             "s-", color="#2ecc71", linewidth=2, label="Warm start")
    ax4.set_xlabel("Horizon (steps)")
    ax4.set_ylabel("Convergence Rate (%)")
    ax4.set_title("Convergence vs Horizon")
    ax4.set_ylim(-5, 105)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle("MPC Solve Performance Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved benchmark plot to: {save_path}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="B1 GRF analysis and MPC performance benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_b1_analysis.py --grf                  # GRF analysis
    python test_b1_analysis.py --benchmark            # Performance benchmark
    python test_b1_analysis.py --grf --benchmark      # Both
    python test_b1_analysis.py --grf --gait walk      # Walk gait GRF
    python test_b1_analysis.py --grf --trajectory curve_left  # Curve GRF
        """
    )

    parser.add_argument("--grf", action="store_true", help="Run GRF extraction and analysis")
    parser.add_argument("--benchmark", action="store_true", help="Run solve performance benchmark")
    parser.add_argument(
        "--gait", type=str, default="trot",
        choices=["trot", "walk", "pace", "bound"],
        help="Gait type (default: trot)",
    )
    parser.add_argument(
        "--trajectory", type=str, default="straight",
        choices=["straight", "curve_left", "curve_right", "s_curve"],
        help="Trajectory type (default: straight)",
    )
    parser.add_argument(
        "--distance", type=float, default=0.5,
        help="Forward distance in meters (default: 0.5)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save plots to files",
    )
    parser.add_argument(
        "--all-gaits", action="store_true",
        help="Run GRF analysis for all gait types",
    )

    args = parser.parse_args()

    if not args.grf and not args.benchmark:
        print("No analysis specified. Use --grf and/or --benchmark.")
        print("Run with --help for usage.")
        return

    # GRF Analysis
    if args.grf:
        gaits = ["trot", "walk", "pace", "bound"] if args.all_gaits else [args.gait]

        for gait in gaits:
            print(f"\n{'='*70}")
            print(f"Running GRF analysis: {gait} - {args.trajectory}")
            print(f"{'='*70}")

            solver, rmodel, foot_frame_ids, contact_seq, dt, com_traj = solve_gait_problem(
                gait_type=gait,
                trajectory_type=args.trajectory,
                distance=args.distance,
                verbose=True,
            )

            # Extract GRF
            forces = extract_grf(solver, foot_frame_ids)

            # Analyze
            analysis = analyze_grf(forces, dt, mu=0.7)

            # Report
            all_ok = print_grf_report(analysis, gait, args.trajectory)

            # Plot
            save_prefix = f"b1_{gait}_{args.trajectory}" if args.save else None
            plot_grf(
                forces, dt, gait, args.trajectory,
                save_path=f"{save_prefix}_grf.png" if save_prefix else None,
            )
            plot_grf_summary(
                forces, dt, gait, args.trajectory,
                save_path=f"{save_prefix}_grf_summary.png" if save_prefix else None,
            )

    # Performance Benchmark
    if args.benchmark:
        results = benchmark_solve_time(
            horizons=[10, 25, 50, 75, 100, 150],
            gait_type=args.gait,
            trajectory_type=args.trajectory,
            num_repeats=3,
        )

        print_benchmark_report(results)

        save_path = f"b1_benchmark_{args.gait}.png" if args.save else None
        plot_benchmark(results, save_path=save_path)


if __name__ == "__main__":
    main()
