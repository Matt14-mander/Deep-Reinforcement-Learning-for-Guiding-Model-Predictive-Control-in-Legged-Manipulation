#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate publication-quality figures for poster and paper.

Produces the following figures:
  Fig 1 — System Pipeline: RL policy → Bezier → OCP → Robot (block diagram)
  Fig 2 — Bezier Trajectory: 3D CoM path + Bezier control points + foothold plan
  Fig 3 — Gait Schedule: contact phase Gantt chart for all 4 feet
  Fig 4 — Pinocchio Skeleton Snapshots: robot FK at N timesteps overlaid on XZ plane
  Fig 5 — Ground Reaction Forces: Fx/Fy/Fz time series + friction cone diagram
  Fig 6 — MPC Performance: solve-time vs horizon (cold vs warm) + convergence rate

Usage:
    cd B1/scripts
    python poster_figures.py                     # all figures, display
    python poster_figures.py --save              # save PNG to figures/ folder
    python poster_figures.py --gait walk --save  # walk gait
    python poster_figures.py --fig 2 4 5 --save  # specific figures only
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # safe headless default; overridden below if --show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
B1_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, B1_DIR)

# ── robot deps ───────────────────────────────────────────────────────────────
try:
    import example_robot_data
    import pinocchio
    import crocoddyl
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("pip install crocoddyl pinocchio example-robot-data")
    sys.exit(1)

from quadruped_mpc.trajectory import BezierTrajectoryGenerator
from quadruped_mpc.gait import (
    GaitScheduler, FootholdPlanner,
    ContactSequence, ContactPhase, OCPFactory,
)
from quadruped_mpc.utils.math_utils import bezier_curve, heading_from_tangent

# ── B1 constants ─────────────────────────────────────────────────────────────
FOOT_FRAME_NAMES = {"LF": "FL_foot", "RF": "FR_foot", "LH": "RL_foot", "RH": "RR_foot"}
HIP_OFFSETS = {
    "LF": np.array([+0.3, +0.1, 0.0]),
    "RF": np.array([+0.3, -0.1, 0.0]),
    "LH": np.array([-0.3, +0.1, 0.0]),
    "RH": np.array([-0.3, -0.1, 0.0]),
}
GAIT_PARAMS = {"step_duration": 0.15, "support_duration": 0.05, "step_height": 0.15}
FOOT_COLORS = {"LF": "#27ae60", "RF": "#e74c3c", "LH": "#2980b9", "RH": "#e67e22"}

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_b1():
    b1 = example_robot_data.load("b1")
    rmodel = b1.model
    rdata = rmodel.createData()
    q0 = rmodel.referenceConfigurations["standing"].copy()
    v0 = pinocchio.utils.zero(rmodel.nv)
    x0 = np.concatenate([q0, v0])
    pinocchio.centerOfMass(rmodel, rdata, q0)
    com0 = rdata.com[0].copy()
    foot_ids = {k: rmodel.getFrameId(v) for k, v in FOOT_FRAME_NAMES.items()}
    return b1, rmodel, rdata, q0, v0, x0, com0, foot_ids


def make_trajectory(com0, traj_type="straight", distance=1.0, duration=3.0, dt=0.02):
    gen = BezierTrajectoryGenerator(degree=3, state_dim=3, max_displacement=5.0)
    params = np.zeros(12)
    if traj_type == "straight":
        params[3:6] = [distance / 3, 0.0, 0.0]
        params[6:9] = [2 * distance / 3, 0.0, 0.0]
        params[9:12] = [distance, 0.0, 0.0]
    elif traj_type == "curve_left":
        a = np.pi / 3
        lat = distance * np.sin(a); fwd = distance * np.cos(a)
        params[3:6] = [fwd / 3, 0.0, 0.0]
        params[6:9] = [2 * fwd / 3, lat / 2, 0.0]
        params[9:12] = [fwd, lat, 0.0]
    elif traj_type == "s_curve":
        params[3:6] = [distance / 3, distance * 0.3, 0.0]
        params[6:9] = [2 * distance / 3, -distance * 0.3, 0.0]
        params[9:12] = [distance, 0.0, 0.0]
    traj = gen.params_to_waypoints(params=params, dt=dt, horizon=duration, start_position=com0)
    ctrl_pts = np.vstack([com0, com0 + params[3:6], com0 + params[6:9], com0 + params[9:12]])
    return traj, ctrl_pts, params


def compute_heading(traj, dt):
    N = len(traj)
    heading = np.zeros(N)
    for i in range(N):
        if i == 0:
            tang = (traj[1] - traj[0]) / dt
        elif i == N - 1:
            tang = (traj[-1] - traj[-2]) / dt
        else:
            tang = (traj[i + 1] - traj[i - 1]) / (2 * dt)
        heading[i] = heading_from_tangent(tang[:2])
    return heading


def solve_ocp(gait_type, traj_type, distance=1.0, duration=3.0, dt=0.02, verbose=False):
    b1, rmodel, rdata, q0, v0, x0, com0, foot_ids = load_b1()
    traj, ctrl_pts, params = make_trajectory(com0, traj_type, distance, duration, dt)
    heading = compute_heading(traj, dt)

    sched = GaitScheduler()
    planner = FootholdPlanner(hip_offsets=HIP_OFFSETS, step_height=GAIT_PARAMS["step_height"])
    seq = sched.generate(
        gait_type=gait_type,
        step_duration=GAIT_PARAMS["step_duration"],
        support_duration=GAIT_PARAMS["support_duration"],
        num_cycles=12,
    )
    init_feet = planner.get_footholds_at_time(com_position=com0, heading=0.0)
    foot_plans = planner.plan_footholds(
        com_trajectory=traj, contact_sequence=seq,
        current_foot_positions=init_feet, dt=dt,
    )

    factory = OCPFactory(rmodel=rmodel, foot_frame_ids=foot_ids, mu=0.7)
    factory.x0 = x0
    problem = factory.build_problem(
        x0=x0, contact_sequence=seq, com_trajectory=traj,
        foot_trajectories=foot_plans, dt=dt, heading_trajectory=heading,
    )

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-4
    if verbose:
        solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
    else:
        solver.setCallbacks([crocoddyl.CallbackLogger()])

    xs0 = [x0] * (problem.T + 1)
    us0 = problem.quasiStatic([x0] * problem.T)
    t0 = time.time()
    converged = solver.solve(xs0, us0, 100, False)
    t_solve = time.time() - t0

    return (solver, rmodel, rdata, foot_ids, seq, traj, ctrl_pts,
            foot_plans, heading, dt, converged, t_solve, x0, q0)


def extract_grf(solver, foot_ids):
    problem = solver.problem
    T = len(problem.runningDatas)
    forces = {n: np.zeros((T, 3)) for n in foot_ids}
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
# Fig 1 — System Pipeline Block Diagram
# ═════════════════════════════════════════════════════════════════════════════

def fig_pipeline(save_dir=None):
    """Draw the RL→Bezier→MPC→Robot pipeline."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    boxes = [
        (1.0,  2.0, 2.2, 1.6, "#3498db", "white", "RL Policy\n(RSL-RL PPO)",
         "obs: body vel,\nfoot contact,\ngait phase"),
        (4.2,  2.0, 2.2, 1.6, "#9b59b6", "white", "Bézier\nTrajectory Gen",
         "degree-3 curve\nover 3 s horizon\n(151 waypoints)"),
        (7.4,  2.0, 2.4, 1.6, "#e74c3c", "white", "Crocoddyl MPC\n(FDDP Solver)",
         "horizon: 25 nodes\n12 actuators\nfriction cone"),
        (10.8, 2.0, 2.2, 1.6, "#27ae60", "white", "B1 Quadruped\n(Pinocchio)",
         "nq=19, nv=18\n12 joints\n4 feet"),
    ]

    for x, y, w, h, fc, tc, title, detail in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=fc, edgecolor="white", linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=tc, zorder=3)
        ax.text(x + w / 2, y + h * 0.25, detail, ha="center", va="center",
                fontsize=7.5, color=tc, alpha=0.9, zorder=3)

    # Arrows between boxes
    arrow_kw = dict(arrowstyle="-|>", color="#2c3e50", lw=2,
                    mutation_scale=18, zorder=1)
    arrow_labels = ["action\n(9 params)", "CoM waypoints\n+ heading", "joint torques\n(12 × Nm)"]
    arrow_xs = [(3.2, 4.2), (6.4, 7.4), (9.8, 10.8)]
    for (x0a, x1a), lbl in zip(arrow_xs, arrow_labels):
        ax.annotate("", xy=(x1a, 2.8), xytext=(x0a, 2.8),
                    arrowprops=dict(**arrow_kw))
        ax.text((x0a + x1a) / 2, 2.45, lbl, ha="center", va="top",
                fontsize=8, color="#555", style="italic")

    # Feedback arrow (robot state → RL)
    ax.annotate("", xy=(2.1, 0.9), xytext=(11.9, 0.9),
                arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=1.5,
                                connectionstyle="arc3,rad=0.0", mutation_scale=14))
    ax.text(7.0, 0.55, "robot state feedback  (proprioception @ 50 Hz)",
            ha="center", va="center", fontsize=9, color="#7f8c8d", style="italic")

    # MPC clock label
    ax.text(8.6, 3.85, "20 ms / step", ha="center", fontsize=8.5,
            color="#e74c3c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fdecea", ec="#e74c3c", lw=1))

    ax.set_title("System Architecture: Deep RL + Bézier MPC for Quadruped Locomotion",
                 fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    _save(fig, save_dir, "fig1_pipeline.png")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 2 — Bézier Trajectory + Foothold Plan (3D)
# ═════════════════════════════════════════════════════════════════════════════

def fig_bezier_trajectory(solver_data, save_dir=None):
    (solver, rmodel, rdata, foot_ids, seq, traj, ctrl_pts,
     foot_plans, heading, dt, converged, t_solve, x0, q0) = solver_data

    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1])

    # ── Left: 3D trajectory + footholds ──────────────────────────────────────
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
              "k-", lw=2, label="CoM trajectory", zorder=5)

    # Control polygon
    ax3d.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2],
              "k--", lw=1, alpha=0.5, label="Bézier control polygon")
    ax3d.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2],
                 c="black", s=60, zorder=6)

    # Footholds as colored dots
    for foot, color in FOOT_COLORS.items():
        plans = foot_plans.get(foot, [])
        if not plans:
            continue
        pts = []
        for entry in plans:
            if isinstance(entry, dict):
                pts.append(entry.get("position", entry.get("pos", None)))
            elif hasattr(entry, "position"):
                pts.append(np.array(entry.position))
            elif isinstance(entry, np.ndarray):
                pts.append(entry)
        pts = [p for p in pts if p is not None]
        if pts:
            pts = np.array(pts)
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=color, s=30, alpha=0.7, label=f"{foot} footholds")

    ax3d.scatter(*traj[0], c="green", s=120, marker="o", zorder=10, label="Start")
    ax3d.scatter(*traj[-1], c="red", s=120, marker="*", zorder=10, label="End")

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("CoM Bézier Trajectory & Foothold Plan", fontweight="bold")
    ax3d.legend(fontsize=7, loc="upper left")

    # ── Right: Top-down XY view ───────────────────────────────────────────────
    ax2d = fig.add_subplot(gs[1])
    ax2d.plot(traj[:, 0], traj[:, 1], "k-", lw=2, label="CoM")
    ax2d.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], "k--", lw=1, alpha=0.5)
    ax2d.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], c="black", s=60, zorder=6)

    for i, (x, y) in enumerate(ctrl_pts[:, :2]):
        ax2d.annotate(f"P{i}", (x, y), textcoords="offset points",
                      xytext=(5, 5), fontsize=8)

    for foot, color in FOOT_COLORS.items():
        plans = foot_plans.get(foot, [])
        pts = []
        for entry in plans:
            if isinstance(entry, dict):
                pts.append(entry.get("position", entry.get("pos", None)))
            elif hasattr(entry, "position"):
                pts.append(np.array(entry.position))
            elif isinstance(entry, np.ndarray):
                pts.append(entry)
        pts = [p for p in pts if p is not None]
        if pts:
            pts = np.array(pts)
            ax2d.scatter(pts[:, 0], pts[:, 1], c=color, s=25, alpha=0.7, label=foot)

    ax2d.scatter(*traj[0, :2], c="green", s=120, marker="o", zorder=10)
    ax2d.scatter(*traj[-1, :2], c="red", s=120, marker="*", zorder=10)

    ax2d.set_xlabel("X (m)")
    ax2d.set_ylabel("Y (m)")
    ax2d.set_title("Top-View: Footholds & Control Points", fontweight="bold")
    ax2d.set_aspect("equal")
    ax2d.legend(fontsize=8)

    plt.suptitle("Bézier Trajectory Generation and Foothold Planning",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir, "fig2_bezier_trajectory.png")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 3 — Gait Contact Schedule (Gantt Chart)
# ═════════════════════════════════════════════════════════════════════════════

def fig_gait_schedule(solver_data, save_dir=None):
    (solver, rmodel, rdata, foot_ids, seq, traj, ctrl_pts,
     foot_plans, heading, dt, converged, t_solve, x0, q0) = solver_data

    foot_order = ["LF", "RF", "LH", "RH"]
    foot_display = {"LF": "Left Front (LF)", "RF": "Right Front (RF)",
                    "LH": "Left Hind (LH)", "RH": "Right Hind (RH)"}

    fig, ax = plt.subplots(figsize=(14, 4))

    # Build contact array from contact_sequence
    phases = list(seq)  # ContactPhase objects
    t_end = min(seq.total_duration if hasattr(seq, "total_duration") else 3.0, 3.0)

    for yi, foot in enumerate(foot_order):
        t_cur = 0.0
        for phase in phases:
            if t_cur >= t_end:
                break
            dur = phase.duration
            in_contact = phase.is_foot_in_contact(foot)

            if in_contact:
                ax.barh(yi, min(dur, t_end - t_cur), left=t_cur, height=0.6,
                        color=FOOT_COLORS[foot], alpha=0.85)
            else:
                ax.barh(yi, min(dur, t_end - t_cur), left=t_cur, height=0.6,
                        color="#ecf0f1", alpha=0.4, edgecolor="#bdc3c7", lw=0.5)
            t_cur += dur

    ax.set_yticks(range(len(foot_order)))
    ax.set_yticklabels([foot_display[f] for f in foot_order])
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, t_end)
    ax.set_title("Gait Contact Schedule (Trot: Diagonal Pair Alternation)", fontweight="bold")

    stance_patch = mpatches.Patch(facecolor="#27ae60", alpha=0.85, label="Stance (contact)")
    swing_patch = mpatches.Patch(facecolor="#ecf0f1", edgecolor="#bdc3c7",
                                  alpha=0.6, label="Swing (airborne)")
    ax.legend(handles=[stance_patch, swing_patch], loc="upper right")

    # Mark gait cycle period
    step = GAIT_PARAMS["step_duration"]
    sup = GAIT_PARAMS["support_duration"]
    cycle = 2 * (step + sup)
    for t in np.arange(0, t_end, cycle):
        ax.axvline(x=t, color="#7f8c8d", lw=0.8, linestyle=":")
    ax.text(cycle / 2, -0.8, f"1 cycle = {cycle*1000:.0f} ms", ha="center",
            fontsize=8, color="#7f8c8d", style="italic")
    ax.annotate("", xy=(cycle, -0.65), xytext=(0, -0.65),
                arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.2))

    plt.tight_layout()
    _save(fig, save_dir, "fig3_gait_schedule.png")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 4 — Pinocchio Skeleton Snapshots (FK overlaid)
# ═════════════════════════════════════════════════════════════════════════════

def fig_skeleton_snapshots(solver_data, save_dir=None, n_frames=8):
    (solver, rmodel, rdata, foot_ids, seq, traj, ctrl_pts,
     foot_plans, heading, dt, converged, t_solve, x0, q0) = solver_data

    xs = np.array(solver.xs)
    T = len(xs)
    step = max(1, T // n_frames)
    frame_indices = list(range(0, T, step))[:n_frames]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_aspect("equal")

    # Color gradient: blue→red across time
    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(len(frame_indices) - 1, 1)) for i in range(len(frame_indices))]

    # Body links: pairs of frame names to draw as lines
    # B1 frame structure: base → hip → thigh → calf → foot
    link_pairs = [
        ("universe",  "FL_hip_joint"),
        ("FL_hip_joint", "FL_thigh_joint"),
        ("FL_thigh_joint", "FL_calf_joint"),
        ("FL_calf_joint", "FL_foot"),
        ("universe",  "FR_hip_joint"),
        ("FR_hip_joint", "FR_thigh_joint"),
        ("FR_thigh_joint", "FR_calf_joint"),
        ("FR_calf_joint", "FR_foot"),
        ("universe",  "RL_hip_joint"),
        ("RL_hip_joint", "RL_thigh_joint"),
        ("RL_thigh_joint", "RL_calf_joint"),
        ("RL_calf_joint", "RL_foot"),
        ("universe",  "RR_hip_joint"),
        ("RR_hip_joint", "RR_thigh_joint"),
        ("RR_thigh_joint", "RR_calf_joint"),
        ("RR_calf_joint", "RR_foot"),
    ]

    # Collect valid frame IDs available in model
    frame_name_to_id = {rmodel.frames[i].name: i for i in range(len(rmodel.frames))}

    for fi, (frame_idx, col) in enumerate(zip(frame_indices, colors)):
        q = xs[frame_idx, :rmodel.nq]
        pinocchio.forwardKinematics(rmodel, rdata, q)
        pinocchio.updateFramePlacements(rmodel, rdata)

        # Draw body rectangle (base CoM position)
        com_pos = xs[frame_idx, :3]  # approximate from state
        base_x = com_pos[0]
        base_z = com_pos[2]

        # Draw skeleton links (XZ plane view — side view)
        alpha = 0.3 + 0.7 * (fi / max(len(frame_indices) - 1, 1))
        lw = 1.0 + 1.5 * (fi / max(len(frame_indices) - 1, 1))

        for a_name, b_name in link_pairs:
            if a_name not in frame_name_to_id or b_name not in frame_name_to_id:
                continue
            aid = frame_name_to_id[a_name]
            bid = frame_name_to_id[b_name]
            pa = rdata.oMf[aid].translation
            pb = rdata.oMf[bid].translation
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]],
                    color=col, lw=lw, alpha=alpha, solid_capstyle="round")

        # Mark feet
        for fname in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            if fname in frame_name_to_id:
                fid = frame_name_to_id[fname]
                fp = rdata.oMf[fid].translation
                ax.plot(fp[0], fp[2], "o", color=col, ms=4, alpha=alpha)

        # Label first and last frame
        if fi == 0:
            ax.text(base_x, base_z + 0.08, "t=0", fontsize=7.5, ha="center",
                    color=col, fontweight="bold")
        elif fi == len(frame_indices) - 1:
            t_label = frame_idx * dt
            ax.text(base_x, base_z + 0.08, f"t={t_label:.1f}s", fontsize=7.5,
                    ha="center", color=col, fontweight="bold")

    # Ground line
    ax.axhline(y=0, color="#7f8c8d", lw=1.5, linestyle="-", label="Ground")
    ax.set_xlabel("X — forward (m)")
    ax.set_ylabel("Z — height (m)")
    ax.set_title(f"Robot Skeleton Snapshots from Pinocchio FK  ({n_frames} frames, earliest→latest: blue→red)",
                 fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T * dt))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("Time (s)")

    plt.tight_layout()
    _save(fig, save_dir, "fig4_skeleton_snapshots.png")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 5 — Ground Reaction Forces + Friction Cone
# ═════════════════════════════════════════════════════════════════════════════

def fig_grf(solver_data, save_dir=None, mu=0.7):
    (solver, rmodel, rdata, foot_ids, seq, traj, ctrl_pts,
     foot_plans, heading, dt, converged, t_solve, x0, q0) = solver_data

    forces = extract_grf(solver, foot_ids)
    foot_order = ["LF", "RF", "LH", "RH"]

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[2, 1], hspace=0.55, wspace=0.35)

    max_fz_all = 0
    for foot in foot_order:
        f = forces[foot]
        if np.any(np.abs(f[:, 2]) > 1):
            max_fz_all = max(max_fz_all, np.max(np.abs(f[:, 2])))

    for row, foot in enumerate(foot_order):
        f = forces[foot]
        T = len(f)
        t = np.arange(T) * dt

        ax = fig.add_subplot(gs[row, 0])
        ax.plot(t, f[:, 0], color="#e74c3c", lw=1.2, alpha=0.8, label="Fx")
        ax.plot(t, f[:, 1], color="#27ae60", lw=1.2, alpha=0.8, label="Fy")
        ax.plot(t, f[:, 2], color="#2980b9", lw=2.0, label="Fz")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_ylabel(f"{foot}\n(N)")
        if row == 0:
            ax.legend(ncol=3, fontsize=8, loc="upper right")
        if row < 3:
            ax.tick_params(labelbottom=False)
        in_contact = np.abs(f[:, 2]) > 1
        ax.fill_between(t, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -5,
                         ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5,
                         where=in_contact, alpha=0.08, color=FOOT_COLORS[foot])

    axes_row = fig.axes
    axes_row[-1].set_xlabel("Time (s)")

    # Friction cone scatter (right column, spans all rows)
    ax_cone = fig.add_subplot(gs[:, 1])
    for foot in foot_order:
        f = forces[foot]
        in_c = np.abs(f[:, 2]) > 1
        if not np.any(in_c):
            continue
        fz = f[in_c, 2]
        ft = np.sqrt(f[in_c, 0] ** 2 + f[in_c, 1] ** 2)
        ax_cone.scatter(fz, ft, c=FOOT_COLORS[foot], s=6, alpha=0.5, label=foot)

    fz_range = np.linspace(0, max_fz_all * 1.05, 200)
    ax_cone.plot(fz_range, mu * fz_range, "k--", lw=2, label=f"Cone (μ={mu})")
    ax_cone.fill_between(fz_range, 0, mu * fz_range, alpha=0.1, color="green")
    ax_cone.axvline(0, color="red", ls=":", lw=1, alpha=0.6)
    ax_cone.set_xlabel("Fz — vertical (N)")
    ax_cone.set_ylabel("Ft — tangential (N)")
    ax_cone.set_title("Friction Cone\nSatisfaction", fontweight="bold")
    ax_cone.legend(fontsize=8)
    ax_cone.set_xlim(-10, max_fz_all * 1.1)

    fig.suptitle("Ground Reaction Forces: Physical Constraint Verification",
                 fontsize=13, fontweight="bold")
    _save(fig, save_dir, "fig5_grf.png")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 6 — MPC Performance Benchmark
# ═════════════════════════════════════════════════════════════════════════════

def fig_mpc_performance(b1, rmodel, rdata, q0, v0, x0, com0, foot_ids,
                        gait_type, save_dir=None):
    """Benchmark cold vs warm start solve time over multiple horizons."""
    print("  Running MPC benchmark (this takes ~60 s)…")

    horizons = [10, 20, 30, 40, 50, 60, 75]
    dt = 0.02
    duration = 5.0
    n_rep = 3

    sched = GaitScheduler()
    planner = FootholdPlanner(hip_offsets=HIP_OFFSETS, step_height=GAIT_PARAMS["step_height"])
    traj, _, _ = make_trajectory(com0, "straight", 1.5, duration, dt)
    heading = compute_heading(traj, dt)
    seq = sched.generate(
        gait_type=gait_type,
        step_duration=GAIT_PARAMS["step_duration"],
        support_duration=GAIT_PARAMS["support_duration"],
        num_cycles=20,
    )
    init_feet = planner.get_footholds_at_time(com_position=com0, heading=0.0)
    foot_plans = planner.plan_footholds(
        com_trajectory=traj, contact_sequence=seq,
        current_foot_positions=init_feet, dt=dt,
    )

    factory = OCPFactory(rmodel=rmodel, foot_frame_ids=foot_ids, mu=0.7)
    factory.x0 = x0
    full_prob = factory.build_problem(
        x0=x0, contact_sequence=seq, com_trajectory=traj,
        foot_trajectories=foot_plans, dt=dt, heading_trajectory=heading,
    )
    full_models = list(full_prob.runningModels)

    cold_ms, cold_std, warm_ms, warm_std = [], [], [], []
    cold_iter, warm_iter = [], []
    cold_conv, warm_conv = [], []

    for H in horizons:
        H = min(H, len(full_models))
        sub = crocoddyl.ShootingProblem(x0, full_models[:H], full_prob.terminalModel)
        c_t, w_t, c_i, w_i, c_c, w_c = [], [], [], [], [], []
        for _ in range(n_rep):
            slv = crocoddyl.SolverFDDP(sub)
            slv.th_stop = 1e-4
            xs0 = [x0] * (H + 1)
            us0 = sub.quasiStatic([x0] * H)
            t0 = time.time()
            cv = slv.solve(xs0, us0, 100, False)
            c_t.append(time.time() - t0)
            c_i.append(slv.iter)
            c_c.append(float(cv))

            xs_w = [x0] + list(slv.xs[2:]) + [slv.xs[-1]] * 2
            xs_w = xs_w[:H + 1]
            us_w = list(slv.us[1:]) + [slv.us[-1]]
            us_w = us_w[:H]
            slv2 = crocoddyl.SolverFDDP(sub)
            slv2.th_stop = 1e-4
            slv2.setCandidate(xs_w, us_w, False)
            t0 = time.time()
            cv2 = slv2.solve([], [], 100, False, 0.0)
            w_t.append(time.time() - t0)
            w_i.append(slv2.iter)
            w_c.append(float(cv2))

        cold_ms.append(np.mean(c_t) * 1000)
        cold_std.append(np.std(c_t) * 1000)
        warm_ms.append(np.mean(w_t) * 1000)
        warm_std.append(np.std(w_t) * 1000)
        cold_iter.append(np.mean(c_i))
        warm_iter.append(np.mean(w_i))
        cold_conv.append(np.mean(c_c) * 100)
        warm_conv.append(np.mean(w_c) * 100)
        print(f"    H={H}: cold {cold_ms[-1]:.1f} ms | warm {warm_ms[-1]:.1f} ms")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Solve time
    ax = axes[0]
    ax.errorbar(horizons, cold_ms, yerr=cold_std, fmt="o-",
                color="#e74c3c", lw=2, capsize=4, label="Cold start")
    ax.errorbar(horizons, warm_ms, yerr=warm_std, fmt="s-",
                color="#27ae60", lw=2, capsize=4, label="Warm start")
    ax.axhline(20, color="#7f8c8d", ls="--", lw=1.5, label="dt = 20 ms")
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Solve time (ms)")
    ax.set_title("Solve Time vs Horizon")
    ax.legend()
    ax.set_yscale("log")

    # Iterations
    ax = axes[1]
    ax.plot(horizons, cold_iter, "o-", color="#e74c3c", lw=2, label="Cold start")
    ax.plot(horizons, warm_iter, "s-", color="#27ae60", lw=2, label="Warm start")
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("FDDP Iterations")
    ax.set_title("Solver Iterations vs Horizon")
    ax.legend()

    # Convergence rate
    ax = axes[2]
    ax.plot(horizons, cold_conv, "o-", color="#e74c3c", lw=2, label="Cold start")
    ax.plot(horizons, warm_conv, "s-", color="#27ae60", lw=2, label="Warm start")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Convergence Rate vs Horizon")
    ax.legend()

    plt.suptitle(f"MPC Solve Performance Benchmark  ({gait_type} gait)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir, "fig6_mpc_performance.png")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, save_dir, filename):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate poster/paper figures for RL-Bézier-MPC quadruped locomotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python poster_figures.py --save                  # all 6 figures → figures/
    python poster_figures.py --fig 1 2 3 --save      # only pipeline, trajectory, gait
    python poster_figures.py --fig 5 --gait walk     # GRF for walk gait
    python poster_figures.py --show                  # display instead of save
        """,
    )
    parser.add_argument("--gait", default="trot",
                        choices=["trot", "walk", "pace", "bound"])
    parser.add_argument("--trajectory", default="straight",
                        choices=["straight", "curve_left", "s_curve"])
    parser.add_argument("--distance", type=float, default=1.5)
    parser.add_argument("--save", action="store_true",
                        help="Save figures to figures/ folder")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively")
    parser.add_argument("--fig", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Which figures to generate (default: all 1–6)")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    save_dir = "figures" if args.save else None
    figs_to_make = set(args.fig)

    print("=" * 60)
    print(f"Generating poster figures  |  gait={args.gait}  traj={args.trajectory}")
    print("=" * 60)

    # Fig 1 — pipeline (no solver needed)
    if 1 in figs_to_make:
        print("\n[Fig 1] System pipeline diagram…")
        fig_pipeline(save_dir)

    # Load robot + solve OCP once for figs 2–5
    solver_data = None
    if figs_to_make & {2, 3, 4, 5}:
        print("\nSolving OCP (for figs 2–5)…")
        solver_data = solve_ocp(
            args.gait, args.trajectory, args.distance,
            duration=3.0, dt=0.02, verbose=True,
        )
        _, _, _, _, _, _, _, _, _, _, converged, t_solve, _, _ = solver_data
        print(f"  → converged={converged}, solve time={t_solve*1000:.0f} ms")

    if 2 in figs_to_make and solver_data:
        print("\n[Fig 2] Bézier trajectory + footholds…")
        fig_bezier_trajectory(solver_data, save_dir)

    if 3 in figs_to_make and solver_data:
        print("\n[Fig 3] Gait contact schedule…")
        fig_gait_schedule(solver_data, save_dir)

    if 4 in figs_to_make and solver_data:
        print("\n[Fig 4] Pinocchio skeleton snapshots…")
        fig_skeleton_snapshots(solver_data, save_dir)

    if 5 in figs_to_make and solver_data:
        print("\n[Fig 5] Ground reaction forces…")
        fig_grf(solver_data, save_dir)

    if 6 in figs_to_make:
        print("\n[Fig 6] MPC performance benchmark…")
        b1, rmodel, rdata, q0, v0, x0, com0, foot_ids = load_b1()
        fig_mpc_performance(b1, rmodel, rdata, q0, v0, x0, com0, foot_ids,
                            args.gait, save_dir)

    if args.show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
