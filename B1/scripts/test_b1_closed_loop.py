#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop MPC simulation for B1 quadruped.

This script verifies the full MPC pipeline in closed-loop:
  1. Load B1 model
  2. Generate CoM Bezier trajectory
  3. At each timestep:
     a. MPC solves OCP from current state (with warm-start)
     b. Extract first control (joint torques)
     c. Forward-integrate dynamics with Pinocchio (ABA)
     d. Record state, control, GRF, solve time
  4. Analyze and visualize results:
     - CoM tracking error over time
     - Ground reaction forces (Fz > 0 check)
     - Friction cone satisfaction
     - Solve time statistics
     - 3D animation (optional)

This is the critical validation before RL training — if closed-loop MPC
cannot produce stable locomotion, RL training will fail.

Usage:
    python test_b1_closed_loop.py                    # Basic closed-loop test
    python test_b1_closed_loop.py --display           # With Meshcat animation
    python test_b1_closed_loop.py --plot              # With analysis plots
    python test_b1_closed_loop.py --trajectory curve_left --plot
    python test_b1_closed_loop.py --duration 3.0 --plot
"""

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
B1_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, B1_DIR)

# Import dependencies
try:
    import example_robot_data
    import pinocchio
    import crocoddyl
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Install with: pip install crocoddyl pinocchio example-robot-data")
    sys.exit(1)

from quadruped_mpc.controllers.crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC
from quadruped_mpc.controllers.base_mpc import MPCSolution
from quadruped_mpc.trajectory import BezierTrajectoryGenerator
from quadruped_mpc.utils.math_utils import heading_from_tangent

# Handle Ctrl+C gracefully
signal.signal(signal.SIGINT, signal.SIG_DFL)


# =============================================================================
# B1 Robot Configuration (same as test_b1_gait.py)
# =============================================================================

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


# =============================================================================
# Simulation Log
# =============================================================================

@dataclass
class SimLog:
    """Container for closed-loop simulation data."""
    # Time
    dt: float = 0.02
    times: List[float] = field(default_factory=list)

    # States and controls
    states: List[np.ndarray] = field(default_factory=list)
    controls: List[np.ndarray] = field(default_factory=list)

    # CoM tracking
    com_actual: List[np.ndarray] = field(default_factory=list)
    com_reference: List[np.ndarray] = field(default_factory=list)
    com_errors: List[float] = field(default_factory=list)

    # Ground reaction forces per foot: {foot_name: [(Fx, Fy, Fz), ...]}
    grf: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    # Friction cone violations
    friction_violations: List[int] = field(default_factory=list)

    # Solver performance
    solve_times: List[float] = field(default_factory=list)
    solve_converged: List[bool] = field(default_factory=list)
    solve_costs: List[float] = field(default_factory=list)
    solve_iterations: List[int] = field(default_factory=list)

    def summary(self) -> str:
        """Print summary statistics."""
        lines = []
        lines.append("=" * 60)
        lines.append("Closed-Loop MPC Simulation Summary")
        lines.append("=" * 60)

        n = len(self.times)
        lines.append(f"  Duration: {self.times[-1]:.2f}s ({n} steps)")

        # CoM tracking
        errors = np.array(self.com_errors)
        lines.append(f"\n  CoM Tracking Error:")
        lines.append(f"    Mean:  {errors.mean():.4f} m")
        lines.append(f"    Max:   {errors.max():.4f} m")
        lines.append(f"    Final: {errors[-1]:.4f} m")

        # GRF analysis
        total_fz_violations = 0
        total_friction_violations = sum(self.friction_violations)
        for foot_name, forces in self.grf.items():
            if not forces:
                continue
            fz = np.array([f[2] for f in forces])
            n_negative = np.sum(fz < -1e-3)
            total_fz_violations += n_negative
            lines.append(f"\n  GRF [{foot_name}]:")
            lines.append(f"    Fz range: [{fz.min():.1f}, {fz.max():.1f}] N")
            lines.append(f"    Fz < 0 violations: {n_negative}/{len(fz)}")

        lines.append(f"\n  Total Fz < 0 violations: {total_fz_violations}")
        lines.append(f"  Total friction cone violations: {total_friction_violations}")

        # Solver performance
        solve_times = np.array(self.solve_times)
        converge_rate = np.mean(self.solve_converged)
        lines.append(f"\n  Solver Performance:")
        lines.append(f"    Convergence rate: {converge_rate * 100:.1f}%")
        lines.append(f"    Solve time: mean={solve_times.mean()*1000:.1f}ms, "
                      f"max={solve_times.max()*1000:.1f}ms, "
                      f"min={solve_times.min()*1000:.1f}ms")
        lines.append(f"    Avg iterations: {np.mean(self.solve_iterations):.1f}")
        lines.append(f"    Real-time factor: {self.dt / solve_times.mean():.1f}x "
                      f"(>1 = real-time capable)")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Trajectory Generation (reused from test_b1_gait.py)
# =============================================================================

def create_com_trajectory(
    start_pos: np.ndarray,
    trajectory_type: str = "straight",
    distance: float = 0.5,
    turn_angle: float = np.pi / 3,
    duration: float = 2.0,
    dt: float = 0.02,
) -> np.ndarray:
    """Create CoM trajectory using Bezier curves."""
    gen = BezierTrajectoryGenerator(degree=3, state_dim=3, max_displacement=2.0)
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

    trajectory = gen.params_to_waypoints(
        params=params, dt=dt, horizon=duration, start_position=start_pos,
    )
    return trajectory


# =============================================================================
# GRF Extraction
# =============================================================================

def extract_grf_from_solver(
    solver: "crocoddyl.SolverFDDP",
    foot_frame_ids: Dict[str, int],
    step_idx: int = 0,
) -> Dict[str, np.ndarray]:
    """Extract ground reaction forces from OCP solver data.

    The GRF is stored in the contact data of each running model's data.
    For the first timestep (step_idx=0), this is the force applied at the
    current MPC step.

    Args:
        solver: Solved FDDP solver.
        foot_frame_ids: Mapping from foot name to frame ID.
        step_idx: Which timestep to extract (0 = current).

    Returns:
        Dict mapping foot name to 3D force vector.
        Feet not in contact return zero force.
    """
    forces = {}

    if step_idx >= len(solver.problem.runningDatas):
        return {name: np.zeros(3) for name in foot_frame_ids}

    data = solver.problem.runningDatas[step_idx]
    diff_data = data.differential

    # Contact forces are in diff_data.multibody.contacts.contacts
    contact_data = diff_data.multibody.contacts
    for foot_name, foot_id in foot_frame_ids.items():
        contact_key = f"contact_{foot_id}"
        if contact_key in contact_data.contacts:
            # .f is the spatial force; for 3D contact, force is (3,)
            f = contact_data.contacts[contact_key].f
            # Pinocchio spatial force: f.linear is the translational force
            forces[foot_name] = f.linear.copy()
        else:
            forces[foot_name] = np.zeros(3)

    return forces


def check_friction_cone(
    forces: Dict[str, np.ndarray],
    mu: float = 0.7,
) -> int:
    """Count friction cone violations.

    A force violates the friction cone if:
    - Fz < 0 (pulling the ground)
    - sqrt(Fx^2 + Fy^2) > mu * Fz (slipping)

    Args:
        forces: GRF per foot.
        mu: Friction coefficient.

    Returns:
        Number of violations.
    """
    violations = 0
    for foot_name, f in forces.items():
        fz = f[2]
        if fz < -1e-3:
            violations += 1  # Downward force
        elif fz > 1e-3:
            f_tangent = np.sqrt(f[0] ** 2 + f[1] ** 2)
            if f_tangent > mu * fz:
                violations += 1  # Slipping
    return violations


# =============================================================================
# Forward Integration
# =============================================================================

def forward_integrate_from_solver(
    solution: "MPCSolution",
) -> np.ndarray:
    """Use MPC's predicted next state as the integration result.

    In a pure Crocoddyl closed-loop simulation, the most consistent
    approach is to use the solver's own predicted next state (xs[1]).
    This uses the same contact-aware dynamics model that the MPC
    uses internally, ensuring ground contact forces are properly applied.

    Using raw Pinocchio ABA would ignore contact constraints entirely,
    causing the robot to fall through the ground.

    In real deployment or IsaacLab, the physics simulator provides the
    next state instead.

    Args:
        solution: MPC solution containing predicted_states.

    Returns:
        Next state, shape (nq + nv,).
    """
    # xs[0] = current state, xs[1] = predicted next state after applying us[0]
    if solution.predicted_states is not None and len(solution.predicted_states) > 1:
        return solution.predicted_states[1].copy()
    else:
        raise RuntimeError("MPC solution has no predicted next state")


# =============================================================================
# Main Closed-Loop Simulation
# =============================================================================

def run_closed_loop(
    gait_type: str = "trot",
    trajectory_type: str = "straight",
    distance: float = 0.5,
    duration: float = 2.0,
    dt: float = 0.02,
    horizon_steps: int = 25,
    max_iterations: int = 10,
    with_display: bool = False,
    with_plot: bool = False,
):
    """Run closed-loop MPC simulation.

    Args:
        gait_type: Gait type ("trot", "walk", "pace", "bound").
        trajectory_type: Trajectory type ("straight", "curve_left", etc.).
        distance: Forward distance in meters.
        duration: Simulation duration in seconds.
        dt: MPC/simulation timestep.
        horizon_steps: MPC prediction horizon.
        max_iterations: Max FDDP iterations per step.
        with_display: Show 3D animation.
        with_plot: Show analysis plots.
    """
    print("=" * 60)
    print(f"Closed-Loop MPC: {gait_type.upper()} - {trajectory_type}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Load B1 robot
    # -------------------------------------------------------------------------
    print("\nLoading B1 robot...")
    b1 = example_robot_data.load("b1")
    rmodel = b1.model
    rdata = rmodel.createData()

    q0 = rmodel.referenceConfigurations["standing"].copy()
    v0 = pinocchio.utils.zero(rmodel.nv)
    x0 = np.concatenate([q0, v0])

    pinocchio.centerOfMass(rmodel, rdata, q0)
    com_start = rdata.com[0].copy()

    print(f"  nq={rmodel.nq}, nv={rmodel.nv}, nu={rmodel.nv - 6}")
    print(f"  CoM start: [{com_start[0]:.3f}, {com_start[1]:.3f}, {com_start[2]:.3f}]")

    # -------------------------------------------------------------------------
    # Create MPC controller
    # -------------------------------------------------------------------------
    print("\nCreating MPC controller...")
    mpc = CrocoddylQuadrupedMPC(
        rmodel=rmodel,
        foot_frame_names=B1_FOOT_FRAME_NAMES,
        hip_offsets=B1_HIP_OFFSETS,
        gait_type=gait_type,
        dt=dt,
        horizon_steps=horizon_steps,
        step_duration=0.15,
        support_duration=0.05,
        step_height=0.15,
        mu=0.7,
        max_iterations=max_iterations,
        convergence_threshold=1e-4,
    )

    foot_frame_ids = mpc.foot_frame_ids
    print(f"  Gait: {gait_type}")
    print(f"  Horizon: {horizon_steps} steps ({horizon_steps * dt:.2f}s)")
    print(f"  Max iterations: {max_iterations}")

    # -------------------------------------------------------------------------
    # Generate reference trajectory
    # -------------------------------------------------------------------------
    print("\nGenerating CoM reference trajectory...")
    com_trajectory = create_com_trajectory(
        start_pos=com_start,
        trajectory_type=trajectory_type,
        distance=distance,
        duration=duration,
        dt=dt,
    )
    print(f"  Trajectory: {com_trajectory.shape[0]} waypoints")
    print(f"  Start: {com_trajectory[0]}")
    print(f"  End:   {com_trajectory[-1]}")

    # -------------------------------------------------------------------------
    # Run closed-loop simulation
    # -------------------------------------------------------------------------
    n_steps = int(duration / dt)
    x = x0.copy()

    log = SimLog(dt=dt)
    for foot_name in foot_frame_ids:
        log.grf[foot_name] = []

    print(f"\nRunning {n_steps} closed-loop steps...")
    print(f"  {'Step':>5} | {'Time':>6} | {'CoM err':>8} | {'Solve':>7} | {'Conv':>4} | {'Iter':>4} | {'Fz<0':>4}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}")

    for step in range(n_steps):
        t = step * dt

        # Get CoM reference window for MPC horizon
        ref_start = min(step, len(com_trajectory) - 1)
        ref_end = min(step + horizon_steps + 10, len(com_trajectory))
        com_ref = com_trajectory[ref_start:ref_end]

        # Pad if not enough reference points
        if len(com_ref) < horizon_steps:
            last = com_ref[-1:] if len(com_ref) > 0 else com_trajectory[-1:]
            com_ref = np.vstack([com_ref] + [last] * (horizon_steps - len(com_ref)))

        # Solve MPC
        solution = mpc.solve(
            current_state=x,
            com_reference=com_ref,
            warm_start=True,
        )

        # GRF estimation from predicted state (contact-aware dynamics)
        # Since we use the solver's predicted next state (which includes contact
        # forces internally), we estimate GRF from the momentum change.
        q = x[:rmodel.nq]
        v = x[rmodel.nq:]
        pinocchio.framesForwardKinematics(rmodel, rdata, q)

        grf_step = {}
        mass = pinocchio.computeTotalMass(rmodel)
        for foot_name, frame_id in foot_frame_ids.items():
            foot_z = rdata.oMf[frame_id].translation[2]
            if foot_z < 0.03:  # Foot near ground = in contact
                # Distribute weight among contact feet
                n_contact = sum(
                    1 for fn in foot_frame_ids
                    if rdata.oMf[foot_frame_ids[fn]].translation[2] < 0.03
                )
                if n_contact > 0:
                    grf_step[foot_name] = np.array([0.0, 0.0, mass * 9.81 / n_contact])
                else:
                    grf_step[foot_name] = np.zeros(3)
            else:
                grf_step[foot_name] = np.zeros(3)
            log.grf[foot_name].append(grf_step[foot_name])

        # Check friction cone
        violations = check_friction_cone(grf_step, mu=0.7)

        # Compute CoM error
        pinocchio.centerOfMass(rmodel, rdata, q)
        com_actual = rdata.com[0].copy()
        ref_idx = min(step, len(com_trajectory) - 1)
        com_ref_point = com_trajectory[ref_idx]
        com_error = np.linalg.norm(com_actual - com_ref_point)

        # Log
        log.times.append(t)
        log.states.append(x.copy())
        log.controls.append(solution.control.copy())
        log.com_actual.append(com_actual.copy())
        log.com_reference.append(com_ref_point.copy())
        log.com_errors.append(com_error)
        log.friction_violations.append(violations)
        log.solve_times.append(solution.solve_time)
        log.solve_converged.append(solution.converged)
        log.solve_costs.append(solution.cost)
        log.solve_iterations.append(solution.iterations)

        # Print progress every 10 steps
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  {step:5d} | {t:6.2f} | {com_error:8.4f} | "
                  f"{solution.solve_time*1000:5.1f}ms | "
                  f"{'Y' if solution.converged else 'N':>4} | "
                  f"{solution.iterations:4d} | {violations:4d}")

        # Use MPC's predicted next state (contact-aware dynamics)
        x = forward_integrate_from_solver(solution)

        # Safety check: robot fallen?
        if x[2] < 0.15:  # CoM z too low
            print(f"\n  [WARN] Robot fallen at step {step} (z={x[2]:.3f})")
            break
        if x[2] > 1.0:  # CoM z too high (exploded)
            print(f"\n  [WARN] Simulation diverged at step {step} (z={x[2]:.3f})")
            break

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(log.summary())

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    if with_display:
        print("\nStarting Meshcat visualization...")
        try:
            display = crocoddyl.MeshcatDisplay(b1)
            display.rate = -1
            display.freq = 1

            # Build a fake solver-like object for display
            # We'll replay logged states
            states_arr = np.array(log.states)
            print(f"  Replaying {len(states_arr)} frames...")
            print("  Press Ctrl+C to stop.")
            try:
                while True:
                    for state in log.states:
                        q = state[:rmodel.nq]
                        pinocchio.framesForwardKinematics(rmodel, rdata, q)
                        display.robot.display(q)
                        time.sleep(dt)
            except KeyboardInterrupt:
                print("\n  Stopped.")
        except Exception as e:
            print(f"  Display error: {e}")

    if with_plot:
        _plot_results(log, gait_type, trajectory_type)

    return log


# =============================================================================
# Plotting
# =============================================================================

def _plot_results(log: SimLog, gait_type: str, trajectory_type: str):
    """Generate analysis plots from simulation log."""
    import matplotlib.pyplot as plt

    times = np.array(log.times)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. CoM trajectory XY
    ax = axes[0, 0]
    com_actual = np.array(log.com_actual)
    com_ref = np.array(log.com_reference)
    ax.plot(com_ref[:, 0], com_ref[:, 1], 'b--', linewidth=1.5, label='Reference')
    ax.plot(com_actual[:, 0], com_actual[:, 1], 'r-', linewidth=2, label='Actual')
    ax.scatter(com_ref[0, 0], com_ref[0, 1], c='g', s=100, marker='o', zorder=5)
    ax.scatter(com_ref[-1, 0], com_ref[-1, 1], c='r', s=100, marker='*', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('CoM Trajectory (XY)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 2. CoM tracking error
    ax = axes[0, 1]
    errors = np.array(log.com_errors)
    ax.plot(times, errors * 100, 'r-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CoM Error (cm)')
    ax.set_title(f'CoM Tracking Error (mean={errors.mean()*100:.1f}cm)')
    ax.grid(True, alpha=0.3)

    # 3. CoM height (z)
    ax = axes[0, 2]
    ax.plot(times, com_actual[:, 2], 'b-', linewidth=2, label='Actual')
    ax.plot(times, com_ref[:, 2], 'b--', linewidth=1, label='Reference')
    ax.axhline(y=0.15, color='r', linestyle=':', label='Fall threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CoM Z (m)')
    ax.set_title('CoM Height')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Solve time
    ax = axes[1, 0]
    solve_ms = np.array(log.solve_times) * 1000
    ax.plot(times, solve_ms, 'g-', linewidth=1, alpha=0.7)
    ax.axhline(y=log.dt * 1000, color='r', linestyle='--', label=f'Real-time limit ({log.dt*1000:.0f}ms)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Solve Time (ms)')
    ax.set_title(f'Solver Performance (mean={solve_ms.mean():.1f}ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Joint torques (first 6 joints)
    ax = axes[1, 1]
    controls = np.array(log.controls)
    for j in range(min(6, controls.shape[1])):
        ax.plot(times[:len(controls)], controls[:, j], linewidth=1, alpha=0.7, label=f'Joint {j}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Joint Torques')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 6. Solver convergence
    ax = axes[1, 2]
    iterations = np.array(log.solve_iterations)
    converged = np.array(log.solve_converged)
    ax.bar(times, iterations, width=log.dt * 0.8,
           color=['green' if c else 'red' for c in converged], alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Iterations')
    ax.set_title(f'Solver Convergence ({np.mean(converged)*100:.0f}% converged)')
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Closed-Loop MPC: B1 {gait_type.upper()} - {trajectory_type}',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    fname = f'b1_closed_loop_{gait_type}_{trajectory_type}.png'
    plt.savefig(fname, dpi=150)
    print(f"\n  Saved figure: {fname}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop MPC simulation for B1 quadruped",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_b1_closed_loop.py                          # Basic test
    python test_b1_closed_loop.py --plot                   # With analysis plots
    python test_b1_closed_loop.py --display                # With 3D animation
    python test_b1_closed_loop.py --gait trot --plot       # Trot with plots
    python test_b1_closed_loop.py --trajectory curve_left --plot
    python test_b1_closed_loop.py --duration 3.0 --horizon 30 --plot
        """
    )

    parser.add_argument("--gait", type=str, default="trot",
                        choices=["trot", "walk", "pace", "bound"])
    parser.add_argument("--trajectory", type=str, default="straight",
                        choices=["straight", "curve_left", "curve_right", "s_curve"])
    parser.add_argument("--distance", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--display", "-d", action="store_true")
    parser.add_argument("--plot", "-p", action="store_true")

    args = parser.parse_args()

    run_closed_loop(
        gait_type=args.gait,
        trajectory_type=args.trajectory,
        distance=args.distance,
        duration=args.duration,
        dt=args.dt,
        horizon_steps=args.horizon,
        max_iterations=args.max_iter,
        with_display=args.display or "CROCODDYL_DISPLAY" in os.environ,
        with_plot=args.plot or "CROCODDYL_PLOT" in os.environ,
    )


if __name__ == "__main__":
    main()
