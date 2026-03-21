#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""B1 quadruped MPC demo — merged replacement for poster_demo.py and test_b1_gait.py.

Combines every useful feature from both scripts:

  3-D visualisation (Meshcat / Gepetto)
    • Full B1 URDF mesh via crocoddyl.MeshcatDisplay (or GepettoDisplay)
    • Friction cones + GRF arrows rendered automatically by displayFromSolver()
    • Custom static scene objects added to display.robot.viewer:
        – CoM reference trajectory (coloured tube)
        – Bézier control polygon (dashed line + sphere handles)
        – Foot landing-target spheres (one colour per foot)
        – Foot swing arc traces (small dots along each swing trajectory)
        – Start / end markers

  Analysis (printed to console)
    • Step lengths per foot (average + count)
    • Curve-walking inner/outer leg ratio verification
    • OCP solve time, convergence, cost

  Matplotlib plots (--plot)
    • crocoddyl.plotSolution  — base CoM, joint positions, torques, GRFs
    • crocoddyl.plotConvergence — cost / step-size convergence log

  Export
    • --save-gif FILE   — animated side panel (top-down path + GRF + torques)
    • --save-mp4 FILE   — same as GIF but MP4 (requires ffmpeg)
    • --save-panel FILE — single static PNG of the side panel

Usage
-----
    cd B1/scripts

    # Live Meshcat in browser + full analysis
    python b1_demo.py --display

    # Slow playback for poster audience, 3 loops
    python b1_demo.py --display --fps 8 --loops 4

    # Curved left trajectory + step analysis
    python b1_demo.py --display --trajectory curve_left --distance 1.2

    # Walk gait with convergence plots
    python b1_demo.py --display --plot --gait walk

    # Export GIF (no browser window needed)
    python b1_demo.py --save-gif demo.gif --no-meshcat

    # Save static panel PNG
    python b1_demo.py --save-panel demo.png --no-meshcat

    # Full: display + plot + save GIF
    python b1_demo.py --display --plot --save-gif trot_curve.gif --trajectory curve_left

    # Use Gepetto if available, fall back to Meshcat
    python b1_demo.py --display --gepetto
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
B1_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, B1_DIR)

# ── robot deps ────────────────────────────────────────────────────────────────
try:
    import example_robot_data
    import pinocchio
    import crocoddyl
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("pip install crocoddyl pinocchio example-robot-data meshcat")
    sys.exit(1)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.animation as mpl_animation

from quadruped_mpc.trajectory import BezierTrajectoryGenerator
from quadruped_mpc.gait import GaitScheduler, FootholdPlanner, OCPFactory
from quadruped_mpc.utils.math_utils import heading_from_tangent
from quadruped_mpc.utils.meshcat_viz import mc_sphere, mc_line

# ── B1 constants ──────────────────────────────────────────────────────────────
FOOT_FRAME_NAMES = {
    "LF": "FL_foot", "RF": "FR_foot",
    "LH": "RL_foot", "RH": "RR_foot",
}
HIP_OFFSETS = {
    "LF": np.array([+0.3, +0.1, 0.0]),
    "RF": np.array([+0.3, -0.1, 0.0]),
    "LH": np.array([-0.3, +0.1, 0.0]),
    "RH": np.array([-0.3, -0.1, 0.0]),
}
GAIT_PARAMS = {"step_duration": 0.15, "support_duration": 0.05, "step_height": 0.15}
FOOT_COLORS = {"LF": "#27ae60", "RF": "#e74c3c", "LH": "#2980b9", "RH": "#e67e22"}
FOOT_COLORS_MPL = FOOT_COLORS   # alias

# Gepetto camera transform (front-left-above view)
GEPETTO_CAM = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

# ── matplotlib style ──────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#30363d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COL,
    "grid.color": GRID_COL, "grid.alpha": 0.35,
    "text.color": TEXT_COL, "axes.labelcolor": TEXT_COL,
    "xtick.color": TEXT_COL, "ytick.color": TEXT_COL,
    "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 1.8,
})


# ═════════════════════════════════════════════════════════════════════════════
# Robot loading
# ═════════════════════════════════════════════════════════════════════════════

def load_b1():
    b1 = example_robot_data.load("b1")
    rmodel = b1.model
    rdata  = rmodel.createData()
    q0     = rmodel.referenceConfigurations["standing"].copy()
    v0     = pinocchio.utils.zero(rmodel.nv)
    x0     = np.concatenate([q0, v0])
    pinocchio.centerOfMass(rmodel, rdata, q0)
    com0   = rdata.com[0].copy()
    foot_ids = {k: rmodel.getFrameId(v) for k, v in FOOT_FRAME_NAMES.items()}

    print(f"  Model: B1  nq={rmodel.nq}  nv={rmodel.nv}  nx={rmodel.nq+rmodel.nv}")
    print(f"  CoM standing: [{com0[0]:.3f}, {com0[1]:.3f}, {com0[2]:.3f}]")
    for name, fid in foot_ids.items():
        print(f"  {name}: frame_id={fid}")

    return b1, rmodel, rdata, q0, v0, x0, com0, foot_ids


# ═════════════════════════════════════════════════════════════════════════════
# Trajectory generation
# ═════════════════════════════════════════════════════════════════════════════

def make_trajectory(com0, traj_type="straight", distance=1.0,
                    duration=3.0, dt=0.02):
    """Generate CoM Bézier trajectory and return (traj, ctrl_pts, params)."""
    gen    = BezierTrajectoryGenerator(degree=3, state_dim=3, max_displacement=5.0)
    params = np.zeros(12)

    if traj_type == "straight":
        params[3:6]  = [distance / 3,       0.0,             0.0]
        params[6:9]  = [2 * distance / 3,   0.0,             0.0]
        params[9:12] = [distance,            0.0,             0.0]
    elif traj_type == "curve_left":
        a   = np.pi / 3
        lat = distance * np.sin(a);  fwd = distance * np.cos(a)
        params[3:6]  = [fwd / 3,       0.0,      0.0]
        params[6:9]  = [2 * fwd / 3,   lat / 2,  0.0]
        params[9:12] = [fwd,            lat,      0.0]
    elif traj_type == "curve_right":
        a   = np.pi / 3
        lat = -distance * np.sin(a);  fwd = distance * np.cos(a)
        params[3:6]  = [fwd / 3,       0.0,      0.0]
        params[6:9]  = [2 * fwd / 3,   lat / 2,  0.0]
        params[9:12] = [fwd,            lat,      0.0]
    elif traj_type == "s_curve":
        params[3:6]  = [distance / 3,       distance * 0.30,  0.0]
        params[6:9]  = [2 * distance / 3,  -distance * 0.30,  0.0]
        params[9:12] = [distance,            0.0,              0.0]
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type!r}")

    traj     = gen.params_to_waypoints(params=params, dt=dt, horizon=duration,
                                        start_position=com0)
    ctrl_pts = np.vstack([
        com0,
        com0 + params[3:6],
        com0 + params[6:9],
        com0 + params[9:12],
    ])
    return traj, ctrl_pts, params


def compute_heading(traj, dt):
    N = len(traj)
    h = np.zeros(N)
    for i in range(N):
        if i == 0:       tang = (traj[1]  - traj[0])  / dt
        elif i == N - 1: tang = (traj[-1] - traj[-2]) / dt
        else:            tang = (traj[i+1]- traj[i-1])/ (2*dt)
        h[i] = heading_from_tangent(tang[:2])
    return h


# ═════════════════════════════════════════════════════════════════════════════
# OCP build & solve
# ═════════════════════════════════════════════════════════════════════════════

def build_and_solve(b1, rmodel, rdata, q0, v0, x0, com0, foot_ids,
                    gait_type, traj_type, distance, duration=3.0, dt=0.02,
                    verbose=True):
    """Run the full pipeline and return all data."""
    traj, ctrl_pts, params = make_trajectory(com0, traj_type, distance, duration, dt)
    heading = compute_heading(traj, dt)

    sched   = GaitScheduler()
    planner = FootholdPlanner(hip_offsets=HIP_OFFSETS,
                              step_height=GAIT_PARAMS["step_height"])

    seq = sched.generate(
        gait_type=gait_type,
        step_duration=GAIT_PARAMS["step_duration"],
        support_duration=GAIT_PARAMS["support_duration"],
        num_cycles=12,
    )
    init_feet  = planner.get_footholds_at_time(com_position=com0, heading=0.0)
    foot_plans = planner.plan_footholds(
        com_trajectory=traj, contact_sequence=seq,
        current_foot_positions=init_feet, dt=dt,
    )

    factory   = OCPFactory(rmodel=rmodel, foot_frame_ids=foot_ids, mu=0.7)
    factory.x0 = x0
    problem   = factory.build_problem(
        x0=x0, contact_sequence=seq, com_trajectory=traj,
        foot_trajectories=foot_plans, dt=dt, heading_trajectory=heading,
    )

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-4
    callbacks = [crocoddyl.CallbackLogger()]
    if verbose:
        callbacks.insert(0, crocoddyl.CallbackVerbose())
    solver.setCallbacks(callbacks)

    xs0 = [x0] * (problem.T + 1)
    us0 = problem.quasiStatic([x0] * problem.T)
    t0  = time.time()
    converged = solver.solve(xs0, us0, 100, False)
    t_solve   = time.time() - t0

    return dict(
        solver=solver, traj=traj, ctrl_pts=ctrl_pts, params=params,
        foot_plans=foot_plans, seq=seq, heading=heading,
        dt=dt, converged=converged, t_solve=t_solve,
    )


# ═════════════════════════════════════════════════════════════════════════════
# GRF extractor
# ═════════════════════════════════════════════════════════════════════════════

def extract_grf(solver, foot_ids):
    problem = solver.problem
    T       = len(problem.runningDatas)
    forces  = {n: np.zeros((T, 3)) for n in foot_ids}
    id2name = {fid: n for n, fid in foot_ids.items()}
    for t in range(T):
        data = problem.runningDatas[t]
        if not hasattr(data, "differential"):
            continue
        diff = data.differential
        if not hasattr(diff, "multibody"):
            continue
        for key, item in diff.multibody.contacts.contacts.todict().items():
            try:
                fid = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            if fid in id2name:
                forces[id2name[fid]][t] = item.f.linear.copy()
    return forces


# ═════════════════════════════════════════════════════════════════════════════
# Analysis: step lengths + curve-walking ratio
# ═════════════════════════════════════════════════════════════════════════════

def print_analysis(foot_plans, traj_type):
    """Replicate test_b1_gait's step-length analysis."""
    print("\nStep lengths:")
    lengths = {}
    for foot_name, plans in foot_plans.items():
        plan_list = plans if isinstance(plans, list) else []
        ls = []
        for entry in plan_list:
            start = end = None
            if hasattr(entry, "start_pos") and hasattr(entry, "end_pos"):
                start, end = entry.start_pos, entry.end_pos
            elif isinstance(entry, dict):
                start = entry.get("start_pos", entry.get("start"))
                end   = entry.get("end_pos",   entry.get("end"))
            if start is not None and end is not None:
                ls.append(np.linalg.norm(np.array(end[:2]) - np.array(start[:2])))
        lengths[foot_name] = ls
        if ls:
            print(f"  {foot_name}: avg={np.mean(ls):.4f}m  count={len(ls)}")
        else:
            print(f"  {foot_name}: (no swing phases detected)")

    if traj_type in ("curve_left", "curve_right"):
        left_avg  = np.mean(lengths.get("LF", [0.0]) + lengths.get("LH", [0.0]) or [0.0])
        right_avg = np.mean(lengths.get("RF", [0.0]) + lengths.get("RH", [0.0]) or [0.0])
        print(f"\nCurve-walking analysis:")
        print(f"  Left  feet avg: {left_avg:.4f} m")
        print(f"  Right feet avg: {right_avg:.4f} m")
        if right_avg > 0:
            ratio = left_avg / right_avg
            print(f"  L/R ratio: {ratio:.3f}")
            if traj_type == "curve_left" and left_avg < right_avg:
                print("  [OK] Inner (left) feet take shorter steps ✓")
            elif traj_type == "curve_right" and right_avg < left_avg:
                print("  [OK] Inner (right) feet take shorter steps ✓")
            else:
                print("  [WARN] Unexpected inner/outer step ratio")


# ═════════════════════════════════════════════════════════════════════════════
# Meshcat / Gepetto display
# ═════════════════════════════════════════════════════════════════════════════

def init_display(b1, prefer_gepetto=False):
    """Create a crocoddyl display.  Returns (display, display_type_str)."""
    if prefer_gepetto:
        try:
            import gepetto
            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(b1, 4, 4, GEPETTO_CAM)
            return display, "gepetto"
        except Exception:
            print("  Gepetto not available — falling back to Meshcat.")

    display = crocoddyl.MeshcatDisplay(b1)
    print(f"  Meshcat URL: {display.robot.viewer.url()}")
    return display, "meshcat"


def add_custom_scene(display, traj, ctrl_pts, foot_plans):
    """Draw Bézier trajectory, control polygon and foothold targets in the viewer.

    Uses display.robot.viewer (the Meshcat viewer inside the Crocoddyl display)
    so custom objects coexist with the automatic friction-cone / GRF rendering.
    """
    viewer = display.robot.viewer

    # ── CoM reference trajectory ─────────────────────────────────────────────
    mc_line(viewer, "custom/com_traj",
            traj.tolist(), "#ffffff", lw=4)

    # ── Bézier control polygon ────────────────────────────────────────────────
    mc_line(viewer, "custom/bezier_poly",
            ctrl_pts.tolist(), "#f39c12", lw=3)

    for pi, cp in enumerate(ctrl_pts):
        # P0, P3 = endpoints (larger); P1, P2 = handles (smaller, transparent)
        is_handle = pi in (1, 2)
        mc_sphere(viewer, f"custom/ctrl_pt/{pi}",
                  cp, radius=0.025 if is_handle else 0.04,
                  color_hex="#f39c12",
                  opacity=0.55 if is_handle else 0.95)

    # ── Start / end markers ───────────────────────────────────────────────────
    mc_sphere(viewer, "custom/start", traj[0],  0.06, "#2ecc71", opacity=0.95)
    mc_sphere(viewer, "custom/end",   traj[-1], 0.08, "#e74c3c", opacity=0.95)

    # ── Foot landing-target spheres ───────────────────────────────────────────
    for foot_name, color_hex in FOOT_COLORS.items():
        plans = foot_plans.get(foot_name, [])
        if not isinstance(plans, list):
            continue
        for pi, entry in enumerate(plans):
            # Extract landing position from FootholdPlan or dict
            pos = None
            if hasattr(entry, "end_pos"):
                pos = entry.end_pos
            elif hasattr(entry, "position"):
                pos = np.array(entry.position)
            elif isinstance(entry, dict):
                pos = entry.get("end_pos", entry.get("position", entry.get("pos")))
            elif isinstance(entry, np.ndarray):
                pos = entry
            if pos is None or len(pos) < 3:
                continue
            mc_sphere(viewer, f"custom/footholds/{foot_name}/{pi}",
                      np.array(pos), 0.022, color_hex, opacity=0.75)

        # Foot swing arc traces (dots along parabolic swing path)
        for pi, entry in enumerate(plans):
            traj_pts = None
            if hasattr(entry, "trajectory") and entry.trajectory is not None:
                traj_pts = entry.trajectory
            elif isinstance(entry, dict) and "trajectory" in entry:
                traj_pts = entry["trajectory"]
            if traj_pts is None or len(traj_pts) < 2:
                continue
            for ti in range(0, len(traj_pts), 4):
                mc_sphere(viewer, f"custom/swing/{foot_name}/{pi}/{ti}",
                          traj_pts[ti], 0.009, color_hex, opacity=0.35)

    print("  Custom scene objects added (Bézier curve, footholds, swing arcs).")


def play_display(display, solver, fps=20, loops=3, display_type="meshcat"):
    """Playback loop using displayFromSolver (friction cones rendered automatically)."""
    display.rate = -1   # manual timing
    display.freq = 1    # show every frame

    print(f"\n  Playing animation — {fps} fps × {loops} loops.")
    print("  Keep the browser / viewer tab open.")
    dt_frame = 1.0 / fps

    for loop in range(loops):
        print(f"  Loop {loop + 1}/{loops}…", end="  ", flush=True)
        t0 = time.time()
        display.displayFromSolver(solver)
        elapsed = time.time() - t0
        # displayFromSolver is synchronous; add a brief pause between loops
        time.sleep(max(0, dt_frame * 2))
        print(f"done ({elapsed:.1f}s)")


# ═════════════════════════════════════════════════════════════════════════════
# Matplotlib plots (plotSolution + plotConvergence)
# ═════════════════════════════════════════════════════════════════════════════

def show_crocoddyl_plots(solver, gait_type, traj_type):
    """Try crocoddyl.plotSolution + plotConvergence; fall back to custom matplotlib."""
    try:
        from crocoddyl.utils.quadruped import plotSolution
        print("  crocoddyl.plotSolution…")
        plotSolution([solver], figIndex=1, show=False)
    except Exception as e:
        print(f"  plotSolution unavailable ({e}) — using custom plots.")
        _fallback_plots(solver)

    # Convergence log
    try:
        log = next(cb for cb in solver.getCallbacks()
                   if isinstance(cb, crocoddyl.CallbackLogger))
        print("  crocoddyl.plotConvergence…")
        crocoddyl.plotConvergence(
            log.costs, log.pregs, log.dregs,
            log.grads, log.stops, log.steps,
            figTitle=f"B1 {gait_type} — {traj_type}",
            figIndex=10, show=False,
        )
    except Exception as e:
        print(f"  plotConvergence unavailable ({e})")

    plt.show()


def _fallback_plots(solver):
    """Custom matplotlib fallback when crocoddyl.utils.quadruped is unavailable."""
    xs = np.array(solver.xs)
    us = np.array(solver.us)
    T  = len(xs)
    dt = 0.02
    t  = np.arange(T) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK_BG)
    fig.suptitle("MPC Solution  —  B1 Quadruped", fontsize=12, fontweight="bold")

    # CoM trajectory
    ax = axes[0, 0]; ax.set_facecolor(PANEL_BG)
    ax.plot(xs[:, 0], xs[:, 1], color="#3498db", lw=2)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("CoM Trajectory (XY)"); ax.set_aspect("equal")

    # Joint positions (first 6)
    ax = axes[0, 1]; ax.set_facecolor(PANEL_BG)
    jnames = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
    for j in range(min(6, xs.shape[1] - 7)):
        ax.plot(t, xs[:, 7 + j], label=jnames[j], lw=1.4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Joint pos (rad)")
    ax.set_title("Front-leg Joint Positions"); ax.legend(fontsize=7, ncol=2)

    # Joint torques (first 6)
    ax = axes[1, 0]; ax.set_facecolor(PANEL_BG)
    for j in range(min(6, us.shape[1])):
        ax.plot(np.arange(len(us)) * dt, us[:, j], label=jnames[j], lw=1.4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Torque (Nm)")
    ax.set_title("Front-leg Joint Torques"); ax.legend(fontsize=7, ncol=2)

    # CoM height
    ax = axes[1, 1]; ax.set_facecolor(PANEL_BG)
    ax.plot(t, xs[:, 2], color="#e74c3c", lw=2, label="CoM Z")
    ax.axhline(xs[0, 2], color=MUTED, lw=1, ls="--", label="Initial Z")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Z height (m)")
    ax.set_title("CoM Height"); ax.legend(fontsize=8)

    plt.tight_layout()


# ═════════════════════════════════════════════════════════════════════════════
# Matplotlib side-panel animation (for GIF / MP4 / static PNG)
# ═════════════════════════════════════════════════════════════════════════════

def build_side_panel_anim(solver_data, forces, dt, fps,
                           gait_type, traj_type):
    """Build the animated 3-panel matplotlib figure."""
    solver    = solver_data["solver"]
    traj      = solver_data["traj"]
    ctrl_pts  = solver_data["ctrl_pts"]

    xs = np.array(solver.xs)
    us = np.array(solver.us)
    T  = len(xs)
    foot_order = ["LF", "RF", "LH", "RH"]
    t_arr      = np.arange(T) * dt

    fig = plt.figure(figsize=(13, 7), facecolor=DARK_BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    ax_traj = fig.add_subplot(gs[:, 0])   # Left: CoM top-down
    ax_grf  = fig.add_subplot(gs[0, 1])   # Top-right: GRF Fz
    ax_torq = fig.add_subplot(gs[1, 1])   # Bot-right: torques

    for ax in [ax_traj, ax_grf, ax_torq]:
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)

    # ── Map ──────────────────────────────────────────────────────────────────
    ax_traj.plot(traj[:, 0], traj[:, 1], "--",
                 color=MUTED, lw=1.5, label="Reference")
    ax_traj.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], "o--",
                 color="#f39c12", ms=7, lw=1, alpha=0.7, label="Control pts")
    for i, (x, y) in enumerate(ctrl_pts[:, :2]):
        ax_traj.annotate(f"P{i}", (x, y), xytext=(5, 5),
                         textcoords="offset points", fontsize=8, color="#f39c12")

    com_line, = ax_traj.plot([], [], color="#3498db", lw=2.5, label="CoM")
    com_dot,  = ax_traj.plot([], [], "o", color="white", ms=10, zorder=10)
    ax_traj.set_xlabel("X (m)"); ax_traj.set_ylabel("Y (m)")
    ax_traj.set_title(f"CoM Trajectory  ({gait_type} / {traj_type})",
                      fontweight="bold")
    ax_traj.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL)
    ax_traj.set_xlim(xs[:, 0].min() - 0.1, xs[:, 0].max() + 0.1)
    ax_traj.set_ylim(xs[:, 1].min() - 0.15, xs[:, 1].max() + 0.15)
    ax_traj.set_aspect("equal")
    time_text = ax_traj.text(0.02, 0.96, "", transform=ax_traj.transAxes,
                              fontsize=9, va="top")

    # ── GRF ──────────────────────────────────────────────────────────────────
    for foot in foot_order:
        ax_grf.plot(t_arr[:len(forces[foot])], forces[foot][:, 2],
                    color=FOOT_COLORS[foot], lw=0.7, alpha=0.22)
    grf_lines = {f: ax_grf.plot([], [], color=FOOT_COLORS[f],
                                 lw=2.0, label=f)[0]
                 for f in foot_order}
    grf_vline = ax_grf.axvline(0, color="white", lw=1.5, ls=":")
    ax_grf.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax_grf.set_xlim(0, t_arr[-1])
    fz_max = max(np.abs(forces[f][:, 2]).max() for f in foot_order)
    ax_grf.set_ylim(-fz_max * 0.1, fz_max * 1.1)
    ax_grf.set_ylabel("Fz (N)")
    ax_grf.set_title("Vertical GRF (Stance = Fz > 0)", fontweight="bold")
    ax_grf.tick_params(labelbottom=False)
    ax_grf.legend(fontsize=7.5, ncol=4, facecolor=PANEL_BG, edgecolor=GRID_COL)

    # ── Torques ───────────────────────────────────────────────────────────────
    joint_names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
    colors_j    = plt.cm.Set2(np.linspace(0, 1, 6))
    for j in range(min(6, us.shape[1])):
        ax_torq.plot(t_arr[:len(us)], us[:, j],
                     color=colors_j[j], lw=0.7, alpha=0.22)
    torq_lines = [ax_torq.plot([], [], color=colors_j[j],
                                lw=2.0, label=joint_names[j])[0]
                  for j in range(min(6, us.shape[1]))]
    torq_vline = ax_torq.axvline(0, color="white", lw=1.5, ls=":")
    ax_torq.set_xlim(0, t_arr[-1])
    u_max = np.abs(us).max() * 1.1 if len(us) else 20
    ax_torq.set_ylim(-u_max, u_max)
    ax_torq.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax_torq.set_xlabel("Time (s)"); ax_torq.set_ylabel("Torque (Nm)")
    ax_torq.set_title("Front-Leg Joint Torques", fontweight="bold")
    ax_torq.legend(fontsize=7, ncol=3, facecolor=PANEL_BG, edgecolor=GRID_COL)

    fig.suptitle(
        f"Crocoddyl MPC — B1 {gait_type} / {traj_type}  "
        f"({'converged' if solver_data['converged'] else 'partial'}, "
        f"{solver_data['t_solve']*1000:.0f} ms)",
        fontsize=11, fontweight="bold",
    )

    def init():
        com_line.set_data([], [])
        com_dot.set_data([], [])
        for l in grf_lines.values():  l.set_data([], [])
        for l in torq_lines:          l.set_data([], [])
        return []

    def update(frame):
        t_now = frame * dt
        com_line.set_data(xs[:frame+1, 0], xs[:frame+1, 1])
        com_dot.set_data([xs[frame, 0]], [xs[frame, 1]])
        for foot in foot_order:
            n = min(frame+1, len(forces[foot]))
            grf_lines[foot].set_data(t_arr[:n], forces[foot][:n, 2])
        grf_vline.set_xdata([t_now, t_now])
        fu = min(frame, len(us) - 1)
        for j, line in enumerate(torq_lines):
            line.set_data(t_arr[:fu+1], us[:fu+1, j])
        torq_vline.set_xdata([t_now, t_now])
        time_text.set_text(f"t = {t_now:.2f} s  |  step {frame}/{T}")
        return (com_line, com_dot, grf_vline, torq_vline, time_text,
                *grf_lines.values(), *torq_lines)

    interval  = max(16, int(1000 / fps))
    anim_obj  = mpl_animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=interval, blit=True,
    )
    return fig, anim_obj


def save_static_panel(solver_data, forces, dt, path, gait_type, traj_type):
    """Save a single snapshot of the side panel (no animation)."""
    fig, anim_obj = build_side_panel_anim(
        solver_data, forces, dt, fps=20,
        gait_type=gait_type, traj_type=traj_type,
    )
    # Advance to the last frame for a fully-drawn static image
    anim_obj._init_drawn = True
    anim_obj._draw_frame(len(np.array(solver_data["solver"].xs)) - 1)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Static panel saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="B1 MPC demo — Meshcat/Gepetto + matplotlib export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python b1_demo.py --display                              # live Meshcat
    python b1_demo.py --display --fps 8 --loops 4           # slow for audience
    python b1_demo.py --display --trajectory curve_left     # curve trajectory
    python b1_demo.py --display --plot --gait walk          # walk + convergence plots
    python b1_demo.py --display --gepetto                   # prefer Gepetto
    python b1_demo.py --save-gif demo.gif --no-meshcat      # GIF only
    python b1_demo.py --save-mp4 demo.mp4 --no-meshcat      # MP4 only
    python b1_demo.py --save-panel demo.png --no-meshcat    # static PNG
    python b1_demo.py --display --plot --save-gif out.gif   # everything
        """,
    )
    parser.add_argument("--gait", default="trot",
                        choices=["trot", "walk", "pace", "bound"])
    parser.add_argument("--trajectory", default="straight",
                        choices=["straight", "curve_left", "curve_right", "s_curve"])
    parser.add_argument("--distance", type=float, default=1.5,
                        help="Forward/arc distance in metres (default 1.5)")
    parser.add_argument("--display", action="store_true",
                        help="Open Meshcat/Gepetto 3-D viewer")
    parser.add_argument("--gepetto", action="store_true",
                        help="Prefer Gepetto viewer (falls back to Meshcat)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Playback FPS (default 20)")
    parser.add_argument("--loops", type=int, default=3,
                        help="Number of playback loops (default 3)")
    parser.add_argument("--plot", action="store_true",
                        help="Show crocoddyl.plotSolution + plotConvergence")
    parser.add_argument("--no-meshcat", action="store_true",
                        help="Skip 3-D viewer (useful with --save-*)")
    parser.add_argument("--save-gif", metavar="FILE",
                        help="Save matplotlib animation as GIF")
    parser.add_argument("--save-mp4", metavar="FILE",
                        help="Save matplotlib animation as MP4 (needs ffmpeg)")
    parser.add_argument("--save-panel", metavar="FILE",
                        help="Save static side-panel PNG")
    args = parser.parse_args()

    # Backend: Agg when not plotting interactively
    if not args.plot and not (args.save_gif or args.save_mp4):
        matplotlib.use("Agg")

    print("=" * 60)
    print(f"B1 Demo  |  gait={args.gait}  traj={args.trajectory}  "
          f"dist={args.distance} m")
    print("=" * 60)

    # ── Load robot ────────────────────────────────────────────────────────────
    print("\nLoading B1 robot model…")
    b1, rmodel, rdata, q0, v0, x0, com0, foot_ids = load_b1()

    # ── Solve OCP ─────────────────────────────────────────────────────────────
    print(f"\nSolving OCP ({args.gait} / {args.trajectory})…")
    solver_data = build_and_solve(
        b1, rmodel, rdata, q0, v0, x0, com0, foot_ids,
        gait_type=args.gait, traj_type=args.trajectory,
        distance=args.distance, duration=3.0, dt=0.02, verbose=True,
    )

    solver    = solver_data["solver"]
    converged = solver_data["converged"]
    t_solve   = solver_data["t_solve"]
    xs        = np.array(solver.xs)
    us        = np.array(solver.us)
    forces    = extract_grf(solver, foot_ids)

    print(f"\n  converged={converged}  iter={solver.iter}  "
          f"cost={solver.cost:.0f}  time={t_solve*1000:.0f} ms")
    print(f"  States: {xs.shape}   Controls: {us.shape}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print_analysis(solver_data["foot_plans"], args.trajectory)

    # ── Meshcat / Gepetto 3-D display ─────────────────────────────────────────
    if args.display and not args.no_meshcat:
        print("\nInitialising display…")
        display, dtype = init_display(b1, prefer_gepetto=args.gepetto)

        # Add Bézier curve + footholds on top of the automatic friction cones
        add_custom_scene(display, solver_data["traj"],
                         solver_data["ctrl_pts"], solver_data["foot_plans"])

        input("\n  [Press Enter to start playback]")
        try:
            play_display(display, solver, fps=args.fps,
                         loops=args.loops, display_type=dtype)
        except KeyboardInterrupt:
            print("\n  Playback interrupted.")

    # ── Matplotlib convergence / solution plots ────────────────────────────────
    if args.plot:
        print("\nGenerating Crocoddyl plots…")
        matplotlib.use("TkAgg")
        show_crocoddyl_plots(solver, args.gait, args.trajectory)

    # ── Export ────────────────────────────────────────────────────────────────
    if args.save_panel:
        print(f"\nSaving static panel → {args.save_panel}")
        save_static_panel(solver_data, forces, dt=0.02,
                          path=args.save_panel,
                          gait_type=args.gait, traj_type=args.trajectory)

    if args.save_gif or args.save_mp4:
        print("\nBuilding matplotlib animation…")
        matplotlib.use("Agg")
        fig_anim, anim_obj = build_side_panel_anim(
            solver_data, forces, dt=0.02, fps=args.fps,
            gait_type=args.gait, traj_type=args.trajectory,
        )

        if args.save_gif:
            print(f"  Saving GIF → {args.save_gif}  (may take 30–90 s)…")
            writer = mpl_animation.PillowWriter(fps=args.fps)
            anim_obj.save(args.save_gif, writer=writer, dpi=120,
                          progress_callback=lambda i, n:
                              print(f"    frame {i}/{n}", end="\r"))
            print(f"\n    Done → {args.save_gif}")

        if args.save_mp4:
            print(f"  Saving MP4 → {args.save_mp4}…")
            writer = mpl_animation.FFMpegWriter(
                fps=args.fps, bitrate=2500,
                metadata={"title": f"B1 MPC {args.gait}/{args.trajectory}"})
            anim_obj.save(args.save_mp4, writer=writer, dpi=140,
                          progress_callback=lambda i, n:
                              print(f"    frame {i}/{n}", end="\r"))
            print(f"\n    Done → {args.save_mp4}")

        plt.close(fig_anim)

    print("\nDone!")


if __name__ == "__main__":
    main()
