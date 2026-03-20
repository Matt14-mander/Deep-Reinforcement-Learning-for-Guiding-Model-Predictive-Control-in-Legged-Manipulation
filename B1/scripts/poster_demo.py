#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive Pinocchio + Meshcat demo for poster presentation.

Features:
  • Full B1 robot mesh displayed via pinocchio.visualize.MeshcatVisualizer
  • CoM trajectory tube rendered as a colored path
  • Foothold targets shown as colored spheres (LF=green, RF=red, LH=blue, RH=orange)
  • GRF force arrows rendered as cylinders at each foot (scaled by magnitude)
  • Bezier control polygon shown as a wireframe path
  • Side matplotlib panel: live joint torques + GRF Fz + MPC cost (for saving frames)
  • Animated frame-by-frame playback with configurable speed
  • Save animation as MP4 or GIF (--save-mp4 / --save-gif)

Usage:
    cd B1/scripts

    # Live Meshcat demo in browser (open URL printed in terminal)
    python poster_demo.py

    # Slow playback for poster audience
    python poster_demo.py --fps 5

    # Save matplotlib side-panel animation as GIF
    python poster_demo.py --save-gif demo.gif

    # Curved trajectory demo
    python poster_demo.py --trajectory curve_left

    # Walk gait (more stable, good for poster)
    python poster_demo.py --gait walk
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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
    from pinocchio.visualize import MeshcatVisualizer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("pip install crocoddyl pinocchio example-robot-data meshcat")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

from quadruped_mpc.trajectory import BezierTrajectoryGenerator
from quadruped_mpc.gait import GaitScheduler, FootholdPlanner, OCPFactory
from quadruped_mpc.utils.math_utils import heading_from_tangent

# ── constants ─────────────────────────────────────────────────────────────────
FOOT_FRAME_NAMES = {"LF": "FL_foot", "RF": "FR_foot", "LH": "RL_foot", "RH": "RR_foot"}
HIP_OFFSETS = {
    "LF": np.array([+0.3, +0.1, 0.0]),
    "RF": np.array([+0.3, -0.1, 0.0]),
    "LH": np.array([-0.3, +0.1, 0.0]),
    "RH": np.array([-0.3, -0.1, 0.0]),
}
GAIT_PARAMS = {"step_duration": 0.15, "support_duration": 0.05, "step_height": 0.15}

# RGBA colors for meshcat (0–1 range)
FOOT_COLORS_RGB = {
    "LF": [0.15, 0.68, 0.38],   # green
    "RF": [0.91, 0.30, 0.24],   # red
    "LH": [0.16, 0.50, 0.73],   # blue
    "RH": [0.91, 0.50, 0.14],   # orange
}
FOOT_COLORS_MPL = {
    "LF": "#27ae60", "RF": "#e74c3c", "LH": "#2980b9", "RH": "#e67e22",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ═════════════════════════════════════════════════════════════════════════════
# Data preparation
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


def make_trajectory(com0, traj_type, distance, duration=3.0, dt=0.02):
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
    ctrl_pts = np.vstack([com0, com0 + params[3:6], com0 + params[6:9], com0 + params[9:12]])
    traj = gen.params_to_waypoints(params=params, dt=dt, horizon=duration, start_position=com0)
    return traj, ctrl_pts


def compute_heading(traj, dt):
    N = len(traj)
    h = np.zeros(N)
    for i in range(N):
        if i == 0:
            tang = (traj[1] - traj[0]) / dt
        elif i == N - 1:
            tang = (traj[-1] - traj[-2]) / dt
        else:
            tang = (traj[i + 1] - traj[i - 1]) / (2 * dt)
        h[i] = heading_from_tangent(tang[:2])
    return h


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


def solve_full(gait_type, traj_type, distance=1.0, dt=0.02, verbose=True):
    b1, rmodel, rdata, q0, v0, x0, com0, foot_ids = load_b1()
    traj, ctrl_pts = make_trajectory(com0, traj_type, distance, duration=3.0, dt=dt)
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
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    xs0 = [x0] * (problem.T + 1)
    us0 = problem.quasiStatic([x0] * problem.T)
    t0 = time.time()
    converged = solver.solve(xs0, us0, 100, False)
    t_solve = time.time() - t0
    print(f"  converged={converged}  iter={solver.iter}  cost={solver.cost:.0f}  "
          f"time={t_solve*1000:.0f} ms")

    return (b1, rmodel, rdata, q0, v0, x0, com0, foot_ids,
            solver, traj, ctrl_pts, foot_plans, seq, heading, dt, converged)


# ═════════════════════════════════════════════════════════════════════════════
# Meshcat visualization helpers
# ═════════════════════════════════════════════════════════════════════════════

def _meshcat_sphere(viz, name, pos, radius, color_rgba):
    """Add a sphere to meshcat scene."""
    try:
        import meshcat.geometry as mg
        import meshcat.transformations as mt
        sphere = mg.Sphere(radius)
        mat = mg.MeshLambertMaterial(color=_rgb_to_hex(*color_rgba[:3]),
                                      opacity=color_rgba[3] if len(color_rgba) > 3 else 1.0)
        viz.viewer[name].set_object(sphere, mat)
        T = np.eye(4)
        T[:3, 3] = pos
        viz.viewer[name].set_transform(T)
    except Exception:
        pass


def _meshcat_line(viz, name, pts, color_hex="#ffffff", lw=3):
    """Draw a polyline in meshcat."""
    try:
        import meshcat.geometry as mg
        vertices = np.array(pts).T.astype(np.float32)   # (3, N)
        line = mg.Line(mg.PointsGeometry(vertices),
                       mg.LineBasicMaterial(color=color_hex, linewidth=lw))
        viz.viewer[name].set_object(line)
    except Exception:
        pass


def _meshcat_arrow(viz, name, origin, direction, scale=1.0, color_hex="#ff0000"):
    """Draw a force arrow (cylinder) in meshcat."""
    try:
        import meshcat.geometry as mg
        import meshcat.transformations as mt

        length = np.linalg.norm(direction) * scale
        if length < 1e-4:
            return
        d = direction / np.linalg.norm(direction)

        # Rotation: align cylinder (Z-axis) with d
        z = np.array([0, 0, 1.0])
        axis = np.cross(z, d)
        angle = np.arccos(np.clip(np.dot(z, d), -1, 1))
        T = np.eye(4)
        T[:3, 3] = origin + direction * scale / 2
        if np.linalg.norm(axis) > 1e-6:
            T[:3, :3] = mt.rotation_matrix(angle, axis / np.linalg.norm(axis))[:3, :3]

        cyl = mg.Cylinder(length, 0.012)
        mat = mg.MeshLambertMaterial(color=color_hex, opacity=0.85)
        viz.viewer[name].set_object(cyl, mat)
        viz.viewer[name].set_transform(T)
    except Exception:
        pass


def _rgb_to_hex(r, g, b):
    return int(r * 255) << 16 | int(g * 255) << 8 | int(b * 255)


# ═════════════════════════════════════════════════════════════════════════════
# Setup Meshcat scene
# ═════════════════════════════════════════════════════════════════════════════

def setup_meshcat(b1, rmodel, rdata, traj, ctrl_pts, foot_plans):
    """Initialize Pinocchio MeshcatVisualizer and add static scene objects."""
    print("\nInitializing Pinocchio MeshcatVisualizer…")
    try:
        viz = MeshcatVisualizer(rmodel, b1.collision_model, b1.visual_model)
        viz.initViewer(open=True, loadModel=True)
        print(f"  Meshcat URL: {viz.viewer.url()}")
    except Exception as e:
        print(f"  Warning: MeshcatVisualizer failed ({e}), using fallback")
        viz = None

    if viz is None:
        return None

    # Draw CoM reference trajectory
    try:
        _meshcat_line(viz, "scene/com_trajectory", traj.tolist(), "#ffffff", lw=4)
    except Exception:
        pass

    # Draw Bezier control polygon
    try:
        _meshcat_line(viz, "scene/bezier_polygon", ctrl_pts.tolist(), "#f39c12", lw=3)
    except Exception:
        pass

    # Draw foothold targets
    for foot, color in FOOT_COLORS_RGB.items():
        plans = foot_plans.get(foot, [])
        for pi, entry in enumerate(plans):
            if isinstance(entry, dict):
                pos = entry.get("position", entry.get("pos", None))
            elif hasattr(entry, "position"):
                pos = np.array(entry.position)
            elif isinstance(entry, np.ndarray):
                pos = entry
            else:
                pos = None
            if pos is None:
                continue
            _meshcat_sphere(viz, f"scene/footholds/{foot}/{pi}",
                            pos, 0.025, color + [0.7])

    print("  Scene objects added.")
    return viz


# ═════════════════════════════════════════════════════════════════════════════
# Matplotlib side panel animation
# ═════════════════════════════════════════════════════════════════════════════

def build_side_panel(xs, us, forces, foot_ids, dt, title=""):
    """Build a 3-panel matplotlib figure for side-panel display / saving."""
    T = len(xs)
    time_arr = np.arange(T) * dt

    fig = plt.figure(figsize=(10, 8), facecolor="#1a1a2e")
    gs = GridSpec(3, 1, figure=fig, hspace=0.45)

    panel_bg = "#16213e"
    text_col = "#e8e8e8"
    plt.rcParams["text.color"] = text_col

    # Panel 1: Joint torques (first 6 joints)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(panel_bg)
    joint_names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
    colors_j = plt.cm.tab10(np.linspace(0, 1, 6))
    for j in range(min(6, us.shape[1])):
        ax1.plot(time_arr[:len(us)], us[:, j],
                 color=colors_j[j], lw=1.3, label=joint_names[j], alpha=0.85)
    ax1.set_ylabel("Torque (Nm)", color=text_col)
    ax1.set_title("Joint Torques (front legs)", color=text_col, fontsize=10)
    ax1.legend(fontsize=7, ncol=3, loc="upper right",
               facecolor="#2c3e50", labelcolor=text_col)
    ax1.tick_params(colors=text_col)
    ax1.spines[:].set_color("#4a4a6a")
    ax1.grid(alpha=0.2)

    # Panel 2: GRF Fz per foot
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(panel_bg)
    for foot, color in FOOT_COLORS_MPL.items():
        fz = forces[foot][:, 2]
        ax2.plot(time_arr[:len(fz)], fz, color=color, lw=1.5, label=foot)
    ax2.axhline(0, color="#7f8c8d", lw=0.8, ls="--")
    ax2.set_ylabel("Fz (N)", color=text_col)
    ax2.set_title("Vertical Ground Reaction Forces", color=text_col, fontsize=10)
    ax2.legend(fontsize=8, ncol=4, loc="upper right",
               facecolor="#2c3e50", labelcolor=text_col)
    ax2.tick_params(colors=text_col)
    ax2.spines[:].set_color("#4a4a6a")
    ax2.grid(alpha=0.2)

    # Panel 3: CoM XY trajectory (top-down)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(panel_bg)
    com_xy = xs[:, :2]  # first two states are x, y
    ax3.plot(com_xy[:, 0], com_xy[:, 1], color="#3498db", lw=2, label="CoM path")
    sc = ax3.scatter(com_xy[::5, 0], com_xy[::5, 1],
                     c=np.arange(0, len(com_xy), 5), cmap="coolwarm", s=8, zorder=5)
    plt.colorbar(sc, ax=ax3, label="Time step", shrink=0.7)
    ax3.set_xlabel("X (m)", color=text_col)
    ax3.set_ylabel("Y (m)", color=text_col)
    ax3.set_title("CoM Trajectory (Top View)", color=text_col, fontsize=10)
    ax3.set_aspect("equal")
    ax3.tick_params(colors=text_col)
    ax3.spines[:].set_color("#4a4a6a")
    ax3.grid(alpha=0.2)

    if title:
        fig.suptitle(title, fontsize=12, color=text_col, fontweight="bold")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Animated side panel (for GIF / MP4 export)
# ═════════════════════════════════════════════════════════════════════════════

def make_animation(xs, us, forces, foot_ids, dt, traj, ctrl_pts,
                   fps=15, gait_type="trot", traj_type="straight"):
    """Create a matplotlib animation showing real-time data panels."""
    T = min(len(xs), len(us) + 1)
    time_arr = np.arange(T) * dt
    foot_order = ["LF", "RF", "LH", "RH"]

    fig = plt.figure(figsize=(13, 7), facecolor="#1a1a2e")
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    bg = "#16213e"
    tc = "#e8e8e8"

    # ── axes setup ────────────────────────────────────────────────────────────
    ax_traj = fig.add_subplot(gs[:, 0])  # Left: CoM XY top-down
    ax_traj.set_facecolor(bg)

    ax_grf = fig.add_subplot(gs[0, 1])   # Top-right: GRF Fz
    ax_grf.set_facecolor(bg)

    ax_torq = fig.add_subplot(gs[1, 1])  # Bottom-right: joint torques
    ax_torq.set_facecolor(bg)

    for ax in [ax_traj, ax_grf, ax_torq]:
        ax.tick_params(colors=tc)
        ax.spines[:].set_color("#4a4a6a")
        ax.grid(alpha=0.2)

    # Static: reference CoM trajectory
    ax_traj.plot(traj[:, 0], traj[:, 1], "#555", lw=1.5, ls="--", label="Reference")
    ax_traj.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], "o--",
                 color="#f39c12", ms=8, lw=1, label="Control pts", alpha=0.7)
    com_line, = ax_traj.plot([], [], color="#3498db", lw=2.5, label="CoM actual")
    com_dot,  = ax_traj.plot([], [], "o", color="white", ms=10, zorder=10)
    ax_traj.set_xlabel("X (m)", color=tc)
    ax_traj.set_ylabel("Y (m)", color=tc)
    ax_traj.set_title(f"CoM Trajectory  ({gait_type}/{traj_type})", color=tc, fontsize=10)
    ax_traj.legend(fontsize=8, facecolor="#2c3e50", labelcolor=tc)
    ax_traj.set_xlim(xs[:, 0].min() - 0.1, xs[:, 0].max() + 0.1)
    ax_traj.set_ylim(xs[:, 1].min() - 0.1, xs[:, 1].max() + 0.1)
    ax_traj.set_aspect("equal")

    # GRF background reference
    for foot in foot_order:
        ax_grf.plot(time_arr[:len(forces[foot])], forces[foot][:, 2],
                    color=FOOT_COLORS_MPL[foot], lw=0.8, alpha=0.25)
    grf_lines = {foot: ax_grf.plot([], [], color=FOOT_COLORS_MPL[foot],
                                   lw=1.8, label=foot)[0]
                 for foot in foot_order}
    grf_vline = ax_grf.axvline(0, color="white", lw=1.5, ls=":")
    ax_grf.axhline(0, color="#7f8c8d", lw=0.7, ls="--")
    ax_grf.set_xlim(0, time_arr[-1])
    ax_grf.set_ylabel("Fz (N)", color=tc)
    ax_grf.set_title("Vertical GRF (Stance = Fz > 0)", color=tc, fontsize=10)
    ax_grf.legend(fontsize=7.5, ncol=4, facecolor="#2c3e50", labelcolor=tc)

    # Joint torques background
    colors_j = plt.cm.Set2(np.linspace(0, 1, min(6, us.shape[1])))
    joint_names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
    for j in range(min(6, us.shape[1])):
        ax_torq.plot(time_arr[:len(us)], us[:, j],
                     color=colors_j[j], lw=0.8, alpha=0.25)
    torq_lines = [ax_torq.plot([], [], color=colors_j[j],
                               lw=1.8, label=joint_names[j])[0]
                  for j in range(min(6, us.shape[1]))]
    torq_vline = ax_torq.axvline(0, color="white", lw=1.5, ls=":")
    ax_torq.set_xlim(0, time_arr[-1])
    ax_torq.set_xlabel("Time (s)", color=tc)
    ax_torq.set_ylabel("Torque (Nm)", color=tc)
    ax_torq.set_title("Joint Torques (Front Legs)", color=tc, fontsize=10)
    ax_torq.legend(fontsize=7, ncol=3, facecolor="#2c3e50", labelcolor=tc)

    time_text = ax_traj.text(0.02, 0.96, "", transform=ax_traj.transAxes,
                              color="white", fontsize=9, va="top")

    fig.suptitle("Crocoddyl MPC for Quadruped Locomotion  |  B1 Robot",
                 fontsize=12, color=tc, fontweight="bold")

    def init():
        com_line.set_data([], [])
        com_dot.set_data([], [])
        for line in grf_lines.values():
            line.set_data([], [])
        for line in torq_lines:
            line.set_data([], [])
        return [com_line, com_dot, grf_vline, torq_vline] + list(grf_lines.values()) + torq_lines

    def update(frame):
        t_now = frame * dt
        # CoM trajectory
        com_line.set_data(xs[:frame + 1, 0], xs[:frame + 1, 1])
        com_dot.set_data([xs[frame, 0]], [xs[frame, 1]])

        # GRF
        for foot in foot_order:
            grf_lines[foot].set_data(
                time_arr[:frame + 1],
                forces[foot][:frame + 1, 2],
            )
        grf_vline.set_xdata([t_now, t_now])

        # Torques
        f_u = min(frame, len(us) - 1)
        for j, line in enumerate(torq_lines):
            line.set_data(time_arr[:f_u + 1], us[:f_u + 1, j])
        torq_vline.set_xdata([t_now, t_now])

        time_text.set_text(f"t = {t_now:.2f} s  |  step {frame}/{T}")
        return [com_line, com_dot, grf_vline, torq_vline] + list(grf_lines.values()) + torq_lines

    interval = max(20, int(1000 / fps))
    n_frames = T
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=interval, blit=True,
    )
    return fig, anim


# ═════════════════════════════════════════════════════════════════════════════
# Meshcat playback
# ═════════════════════════════════════════════════════════════════════════════

def play_meshcat(viz, rmodel, rdata, foot_ids, xs, forces, dt, fps=25, loops=3):
    """Play back solver trajectory in Meshcat with GRF arrows."""
    if viz is None:
        print("  Meshcat not available — skipping 3D playback.")
        return

    print(f"\nPlaying Meshcat animation ({fps} fps, {loops} loops)…")
    print("  Keep the Meshcat browser tab open.")
    T = len(xs)
    dt_frame = 1.0 / fps
    frame_step = max(1, int((1.0 / fps) / dt))

    id2name = {fid: n for n, fid in foot_ids.items()}
    frame_name_to_id = {rmodel.frames[i].name: i for i in range(len(rmodel.frames))}

    for loop in range(loops):
        print(f"  Loop {loop + 1}/{loops}…")
        for t in range(0, T, frame_step):
            t0 = time.time()
            q = xs[t, :rmodel.nq]
            try:
                viz.display(q)
            except Exception:
                pass

            # GRF arrows
            pinocchio.forwardKinematics(rmodel, rdata, q)
            pinocchio.updateFramePlacements(rmodel, rdata)

            for foot, color in FOOT_COLORS_RGB.items():
                fid = foot_ids[foot]
                foot_pos = rdata.oMf[fid].translation.copy()
                grf = forces[foot][t] if t < len(forces[foot]) else np.zeros(3)
                scale = 0.002   # N → m visual scale
                _meshcat_arrow(
                    viz, f"grf/{foot}",
                    foot_pos, grf * scale,
                    scale=1.0,
                    color_hex=f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}",
                )

            elapsed = time.time() - t0
            remaining = dt_frame - elapsed
            if remaining > 0:
                time.sleep(remaining)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pinocchio Meshcat demo + matplotlib animation for poster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python poster_demo.py                           # live Meshcat demo
    python poster_demo.py --fps 10                  # slow playback
    python poster_demo.py --save-gif demo.gif       # save animation as GIF
    python poster_demo.py --save-mp4 demo.mp4       # save animation as MP4
    python poster_demo.py --gait walk               # walk gait
    python poster_demo.py --trajectory curve_left   # curved trajectory
    python poster_demo.py --no-meshcat --save-gif   # GIF only (no browser)
        """,
    )
    parser.add_argument("--gait", default="trot",
                        choices=["trot", "walk", "pace", "bound"])
    parser.add_argument("--trajectory", default="straight",
                        choices=["straight", "curve_left", "s_curve"])
    parser.add_argument("--distance", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=25,
                        help="Playback FPS (default 25)")
    parser.add_argument("--loops", type=int, default=3,
                        help="Meshcat playback loops (default 3)")
    parser.add_argument("--save-gif", metavar="FILE",
                        help="Save matplotlib animation as GIF")
    parser.add_argument("--save-mp4", metavar="FILE",
                        help="Save matplotlib animation as MP4")
    parser.add_argument("--save-panel", metavar="FILE",
                        help="Save static side panel as PNG")
    parser.add_argument("--no-meshcat", action="store_true",
                        help="Skip Meshcat browser window")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Poster Demo  |  gait={args.gait}  traj={args.trajectory}")
    print("=" * 60)

    # Solve
    print("\nSolving OCP…")
    (b1, rmodel, rdata, q0, v0, x0, com0, foot_ids,
     solver, traj, ctrl_pts, foot_plans, seq, heading, dt, converged) = solve_full(
        args.gait, args.trajectory, args.distance, dt=0.02, verbose=True,
    )

    xs = np.array(solver.xs)
    us = np.array(solver.us)
    forces = extract_grf(solver, foot_ids)

    print(f"\nTrajectory: {len(xs)} states × {rmodel.nq + rmodel.nv} dims")
    print(f"Controls:   {len(us)} × {rmodel.nv - 6} torques")

    # ── Meshcat 3D demo ───────────────────────────────────────────────────────
    if not args.no_meshcat:
        viz = setup_meshcat(b1, rmodel, rdata, traj, ctrl_pts, foot_plans)
        input("\n  [Press Enter to start Meshcat playback]")
        play_meshcat(viz, rmodel, rdata, foot_ids, xs, forces,
                     dt, fps=args.fps, loops=args.loops)
    else:
        viz = None

    # ── Static side panel ─────────────────────────────────────────────────────
    if args.save_panel:
        print(f"\nGenerating static side panel → {args.save_panel}")
        fig_panel = build_side_panel(
            xs, us, forces, foot_ids, dt,
            title=f"Crocoddyl MPC  |  B1 {args.gait} / {args.trajectory}",
        )
        fig_panel.savefig(args.save_panel, dpi=200, bbox_inches="tight",
                          facecolor=fig_panel.get_facecolor())
        print(f"  Saved → {args.save_panel}")

    # ── Animation export ──────────────────────────────────────────────────────
    if args.save_gif or args.save_mp4:
        print("\nBuilding matplotlib animation…")
        fig_anim, anim = make_animation(
            xs, us, forces, foot_ids, dt, traj, ctrl_pts,
            fps=args.fps, gait_type=args.gait, traj_type=args.trajectory,
        )

        if args.save_gif:
            print(f"  Saving GIF → {args.save_gif}  (may take 30–90 s)…")
            writer = animation.PillowWriter(fps=args.fps)
            anim.save(args.save_gif, writer=writer, dpi=120)
            print(f"  Saved → {args.save_gif}")

        if args.save_mp4:
            print(f"  Saving MP4 → {args.save_mp4}…")
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=2000)
            anim.save(args.save_mp4, writer=writer, dpi=150)
            print(f"  Saved → {args.save_mp4}")

    print("\nDone!")


if __name__ == "__main__":
    main()
