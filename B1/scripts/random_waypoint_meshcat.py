#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random waypoint navigation demo — Bézier MPC with full Pinocchio + Meshcat 3D view.

This is the *live browser* version of random_waypoint_demo.py.
Meshcat renders the full B1 robot mesh, the 3D Bézier curves, control polygons,
waypoint spheres, G1-continuity tangent arrows, GRF force arrows, and an
accumulating CoM trail — all inside a standard web browser (no RViz/Gepetto).

What the 3D scene shows
-----------------------
  Static (drawn once, persist for all time):
    • Ground plane (grey grid)
    • Waypoint spheres  (★ gold, numbered W1…Wn)
    • Per-segment CoM reference curves (coloured tubes, one hue per segment)
    • Per-segment Bézier control polygons (dashed lines + sphere handles)
    • G1-continuity tangent arrows at every segment junction
    • Foot swing arc traces (small spheres along swing trajectory)

  Dynamic (updated frame-by-frame):
    • Full B1 robot mesh (pinocchio viz.display)
    • 4 × GRF force arrows (scaled cylinders at each foot)
    • Accumulating CoM trail (line grows as robot walks)
    • "Active waypoint" indicator ring pulsing around current target

Optional matplotlib side panel (saved as GIF/MP4):
    • Top-down 2D map with all Bézier curves + robot position
    • Live GRF Fz time-series
    • Live front-leg joint torques

Usage
-----
    cd B1/scripts

    # Open Meshcat browser then start interactive animation
    python random_waypoint_meshcat.py

    # Reproducible demo (seed), slower playback, walk gait
    python random_waypoint_meshcat.py --seed 42 --fps 10 --gait walk

    # 8 waypoints, tighter turns
    python random_waypoint_meshcat.py --n-waypoints 8 --max-turn 40 --seed 7

    # Also save a matplotlib GIF (for poster)
    python random_waypoint_meshcat.py --seed 42 --save-gif rw_nav.gif

    # Also save a MP4
    python random_waypoint_meshcat.py --seed 42 --save-mp4 rw_nav.mp4

    # Pre-bake smooth animation into Meshcat (browser plays it automatically)
    python random_waypoint_meshcat.py --seed 42 --bake-animation

    # Skip Meshcat, only produce GIF (CI / headless)
    python random_waypoint_meshcat.py --seed 42 --no-meshcat --save-gif rw.gif
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
B1_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, B1_DIR)

# ── robot / MPC deps ──────────────────────────────────────────────────────────
try:
    import example_robot_data
    import pinocchio
    import crocoddyl
    from pinocchio.visualize import MeshcatVisualizer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("pip install crocoddyl pinocchio example-robot-data meshcat")
    sys.exit(1)

try:
    import meshcat
    import meshcat.geometry as mg
    import meshcat.transformations as mt
    HAS_MESHCAT = True
except ImportError:
    HAS_MESHCAT = False
    print("Warning: meshcat-python not found — 3D view disabled.")

from quadruped_mpc.utils.meshcat_viz import (
    hex_to_int, mc_sphere, mc_line, mc_cylinder, mc_cone, mc_delete,
    draw_friction_cone, draw_grf_arrow, draw_contact_viz,
)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from matplotlib.gridspec import GridSpec

from quadruped_mpc.trajectory import BezierTrajectoryGenerator
from quadruped_mpc.gait import GaitScheduler, FootholdPlanner, OCPFactory
from quadruped_mpc.utils.math_utils import heading_from_tangent

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

# Per-foot colours (rgb 0-1 and hex)
FOOT_RGB = {
    "LF": [0.15, 0.68, 0.38],
    "RF": [0.91, 0.30, 0.24],
    "LH": [0.16, 0.50, 0.73],
    "RH": [0.91, 0.50, 0.14],
}
FOOT_HEX = {k: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for k, (r, g, b) in FOOT_RGB.items()}

# Segment colour palette
SEGMENT_PALETTE = [
    "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
    "#f39c12", "#1abc9c", "#e67e22", "#e91e63",
    "#00bcd4", "#8bc34a",
]

# Visual-scale: Newton → metres for GRF arrows
GRF_SCALE = 0.0018

# B1 body dimensions
BODY_L, BODY_W = 0.55, 0.25

# ── matplotlib dark style ─────────────────────────────────────────────────────
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
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Waypoint:
    pos: np.ndarray      # (3,)
    heading: float       # yaw at arrival


@dataclass
class Segment:
    idx: int
    start_pos: np.ndarray   # (3,)
    target_pos: np.ndarray  # (3,)
    ctrl_pts: np.ndarray    # (4, 3)  absolute Bézier control points
    com_traj: np.ndarray    # (N, 3)  dense CoM reference
    xs: np.ndarray          # (T+1, nx)
    us: np.ndarray          # (T,   nu)
    forces: Dict            # foot → (T, 3)
    foot_plans: Dict        # raw foothold plan from FootholdPlanner
    converged: bool
    cost: float
    solve_ms: float
    color: str             # hex
    entry_tangent: np.ndarray  # (3,) for G1 viz


# ═════════════════════════════════════════════════════════════════════════════
# Robot loader
# ═════════════════════════════════════════════════════════════════════════════

def load_b1():
    b1 = example_robot_data.load("b1")
    rmodel = b1.model
    rdata  = rmodel.createData()
    q0 = rmodel.referenceConfigurations["standing"].copy()
    v0 = pinocchio.utils.zero(rmodel.nv)
    x0 = np.concatenate([q0, v0])
    pinocchio.centerOfMass(rmodel, rdata, q0)
    com0 = rdata.com[0].copy()
    foot_ids = {k: rmodel.getFrameId(v) for k, v in FOOT_FRAME_NAMES.items()}
    return b1, rmodel, rdata, q0, v0, x0, com0, foot_ids


# ═════════════════════════════════════════════════════════════════════════════
# Random waypoint generator
# ═════════════════════════════════════════════════════════════════════════════

class RandomWaypointGenerator:
    def __init__(self, dist_min=0.6, dist_max=1.3, max_turn_deg=55.0,
                 com_height=0.45, seed=None):
        self.dist_min   = dist_min
        self.dist_max   = dist_max
        self.max_turn   = np.deg2rad(max_turn_deg)
        self.com_height = com_height
        self.rng = np.random.default_rng(seed)

    def next(self, pos: np.ndarray, heading: float) -> Waypoint:
        d   = self.rng.uniform(self.dist_min, self.dist_max)
        dh  = self.rng.uniform(-self.max_turn, self.max_turn)
        yaw = heading + dh
        new_pos = np.array([pos[0] + d * np.cos(yaw),
                            pos[1] + d * np.sin(yaw),
                            self.com_height])
        return Waypoint(pos=new_pos, heading=yaw)


# ═════════════════════════════════════════════════════════════════════════════
# G1-continuous Bézier builder
# ═════════════════════════════════════════════════════════════════════════════

def build_g1_bezier(start, target, entry_tangent, dt=0.02, duration=3.0):
    """Cubic Bézier with G1 continuity at the start."""
    gen  = BezierTrajectoryGenerator(degree=3, state_dim=3, max_displacement=20.0)
    diff = target - start
    dist = np.linalg.norm(diff[:2])

    # Exit tangent: point roughly toward target
    exit_t = np.array([diff[0], diff[1], 0.0])
    if np.linalg.norm(exit_t) > 1e-6:
        exit_t /= np.linalg.norm(exit_t)

    blend = max(dist / 3.0, 0.15)
    P0 = start.copy()
    P1 = start  + entry_tangent * blend;  P1[2] = start[2]
    P2 = target - exit_t        * blend;  P2[2] = target[2]
    P3 = target.copy()
    ctrl_pts = np.array([P0, P1, P2, P3])

    offsets   = (ctrl_pts - P0).flatten()
    com_traj  = gen.params_to_waypoints(offsets, dt, duration, start_position=P0)

    # Heading per waypoint
    N = len(com_traj)
    heading_traj = np.zeros(N)
    for i in range(N):
        if i == 0:       tang = (com_traj[1]  - com_traj[0])  / dt
        elif i == N - 1: tang = (com_traj[-1] - com_traj[-2]) / dt
        else:            tang = (com_traj[i+1]- com_traj[i-1])/ (2*dt)
        heading_traj[i] = heading_from_tangent(tang[:2])

    return ctrl_pts, com_traj, heading_traj


def exit_tangent(com_traj, dt):
    if len(com_traj) < 2:
        return np.array([1.0, 0.0, 0.0])
    t = (com_traj[-1] - com_traj[-2]) / dt;  t[2] = 0.0
    n = np.linalg.norm(t)
    return t / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


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
# Solve one segment
# ═════════════════════════════════════════════════════════════════════════════

def solve_segment(rmodel, x0, foot_ids, com_traj, heading_traj, gait_type, dt=0.02):
    sched   = GaitScheduler()
    planner = FootholdPlanner(hip_offsets=HIP_OFFSETS, step_height=GAIT_PARAMS["step_height"])
    seq     = sched.generate(gait_type=gait_type,
                             step_duration=GAIT_PARAMS["step_duration"],
                             support_duration=GAIT_PARAMS["support_duration"],
                             num_cycles=12)
    com0       = com_traj[0].copy()
    init_feet  = planner.get_footholds_at_time(com_position=com0, heading=0.0)
    foot_plans = planner.plan_footholds(com_trajectory=com_traj,
                                        contact_sequence=seq,
                                        current_foot_positions=init_feet, dt=dt)

    factory = OCPFactory(rmodel=rmodel, foot_frame_ids=foot_ids, mu=0.7)
    factory.x0 = x0
    problem = factory.build_problem(x0=x0, contact_sequence=seq,
                                    com_trajectory=com_traj,
                                    foot_trajectories=foot_plans,
                                    dt=dt, heading_trajectory=heading_traj)

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-4
    xs0 = [x0] * (problem.T + 1)
    us0 = problem.quasiStatic([x0] * problem.T)
    t0  = time.time()
    converged = solver.solve(xs0, us0, 100, False)
    solve_ms  = (time.time() - t0) * 1000
    return solver, foot_plans, converged, solve_ms


# ═════════════════════════════════════════════════════════════════════════════
# Pre-compute all segments
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_segments(rmodel, x0_init, com0, foot_ids,
                         waypoints, gait_type="trot", dt=0.02, duration=3.0):
    segments = []
    cur_x   = x0_init.copy()
    cur_com = com0.copy()
    cur_tan = np.array([1.0, 0.0, 0.0])

    print(f"\nPre-computing {len(waypoints)} segments…")
    print("─" * 56)

    for i, wp in enumerate(waypoints):
        color = SEGMENT_PALETTE[i % len(SEGMENT_PALETTE)]
        print(f"  [{i+1}] ({cur_com[0]:.2f},{cur_com[1]:.2f}) → "
              f"({wp.pos[0]:.2f},{wp.pos[1]:.2f})", end="  ")

        ctrl_pts, com_traj, heading_traj = build_g1_bezier(
            cur_com, wp.pos, cur_tan, dt=dt, duration=duration)

        solver, foot_plans, converged, solve_ms = solve_segment(
            rmodel, cur_x, foot_ids, com_traj, heading_traj, gait_type, dt)

        xs = np.array(solver.xs)
        us = np.array(solver.us)
        forces = extract_grf(solver, foot_ids)

        print(f"{'✓' if converged else '⚠'}  cost={solver.cost:.0f}  "
              f"iter={solver.iter}  {solve_ms:.0f}ms")

        seg = Segment(
            idx=i, start_pos=cur_com.copy(), target_pos=wp.pos.copy(),
            ctrl_pts=ctrl_pts, com_traj=com_traj,
            xs=xs, us=us, forces=forces, foot_plans=foot_plans,
            converged=converged, cost=solver.cost, solve_ms=solve_ms,
            color=color, entry_tangent=cur_tan.copy(),
        )
        segments.append(seg)

        cur_x   = xs[-1].copy()
        cur_com = com_traj[-1].copy()
        cur_tan = exit_tangent(com_traj, dt)

    print("─" * 56)
    return segments


# ═════════════════════════════════════════════════════════════════════════════
# Concatenate all segment data into one long timeline
# ═════════════════════════════════════════════════════════════════════════════

def concat_segments(segments, dt):
    all_xs     = []
    all_us     = []
    all_forces = {f: [] for f in FOOT_RGB}
    all_seg_i  = []
    all_com_ref= []

    for si, seg in enumerate(segments):
        skip = 1 if si > 0 else 0
        all_xs.append(seg.xs[skip:])
        all_us.append(seg.us[skip:] if skip < len(seg.us) else seg.us)
        for f in FOOT_RGB:
            all_forces[f].append(seg.forces[f][skip:])
        all_seg_i.extend([si] * (len(seg.xs) - skip))
        all_com_ref.append(seg.com_traj[skip:])

    xs      = np.concatenate(all_xs,      axis=0)
    us      = np.concatenate(all_us,      axis=0)
    for f in FOOT_RGB:
        all_forces[f] = np.concatenate(all_forces[f], axis=0)
    seg_idx = np.array(all_seg_i, dtype=int)
    com_ref = np.concatenate(all_com_ref, axis=0)
    t_arr   = np.arange(len(xs)) * dt
    return xs, us, all_forces, seg_idx, com_ref, t_arr


# ═════════════════════════════════════════════════════════════════════════════
# Meshcat ground plane (local helper — not in shared module)
# ═════════════════════════════════════════════════════════════════════════════

def mc_box(viewer, path, center, size, color_hex, opacity=0.5):
    """Thin flat box — used for the ground plane."""
    if not HAS_MESHCAT:
        return
    mat = mg.MeshLambertMaterial(color=hex_to_int(color_hex), opacity=opacity)
    viewer[path].set_object(mg.Box(size), mat)
    T = np.eye(4); T[:3, 3] = np.asarray(center, dtype=float)
    viewer[path].set_transform(T)


# ═════════════════════════════════════════════════════════════════════════════
# Build Meshcat static scene
# ═════════════════════════════════════════════════════════════════════════════

def build_static_scene(viewer, segments, waypoints, com0):
    """Add all permanent scene objects once — no need to update per frame."""
    print("  Building Meshcat static scene…")

    # ── Ground plane ─────────────────────────────────────────────────────────
    mc_box(viewer, "scene/ground",
           center=[0, 0, -0.01],
           size=[30.0, 30.0, 0.004],
           color_hex="#1a2a1a", opacity=0.85)

    # ── Start marker ─────────────────────────────────────────────────────────
    mc_sphere(viewer, "scene/start", com0, 0.06, "#2ecc71", opacity=0.95)

    # ── Per-segment: CoM reference curve + Bézier control polygon ────────────
    for seg in segments:
        si   = seg.idx
        col  = seg.color

        # Dense CoM reference tube (as polyline)
        mc_line(viewer, f"scene/com_ref/{si}",
                seg.com_traj.tolist(), col, lw=3)

        # Control polygon  (P0–P1–P2–P3)
        mc_line(viewer, f"scene/ctrl_poly/{si}",
                seg.ctrl_pts.tolist(), col, lw=2)

        # Control point handles (small spheres)
        for pi, cp in enumerate(seg.ctrl_pts):
            alpha = 0.5 if pi in (1, 2) else 0.9   # inner pts more transparent
            mc_sphere(viewer, f"scene/ctrl_pts/{si}/{pi}",
                      cp, 0.025 if pi in (1, 2) else 0.04,
                      col, opacity=alpha)

        # G1-continuity tangent arrow at segment start
        if si > 0:
            tang_end = seg.start_pos + seg.entry_tangent * 0.25
            mc_line(viewer, f"scene/g1_arrow/{si}",
                    [seg.start_pos.tolist(), tang_end.tolist()], "#f1c40f", lw=3)
            mc_sphere(viewer, f"scene/g1_tip/{si}", tang_end, 0.018, "#f1c40f")

        # Foot swing arc traces (small spheres along each swing trajectory)
        for foot_name, plans in seg.foot_plans.items():
            f_col = FOOT_HEX[foot_name]
            for pi, entry in enumerate(plans if isinstance(plans, list) else []):
                traj_pts = None
                if hasattr(entry, "trajectory") and entry.trajectory is not None:
                    traj_pts = entry.trajectory
                elif isinstance(entry, dict) and "trajectory" in entry:
                    traj_pts = entry["trajectory"]
                if traj_pts is None or len(traj_pts) < 2:
                    continue
                # Draw every-3rd point to avoid clutter
                for ti in range(0, len(traj_pts), 3):
                    mc_sphere(viewer,
                              f"scene/swing/{si}/{foot_name}/{pi}/{ti}",
                              traj_pts[ti], 0.010, f_col, opacity=0.45)

    # ── Waypoint spheres (numbered) ───────────────────────────────────────────
    for i, wp in enumerate(waypoints):
        mc_sphere(viewer, f"scene/waypoints/{i}",
                  wp.pos, 0.07, "#f1c40f", opacity=0.9)
        # Small label sphere on top
        label_pos = wp.pos + np.array([0, 0, 0.14])
        mc_sphere(viewer, f"scene/wp_label/{i}",
                  label_pos, 0.025, "#ffffff", opacity=0.7)

    print("  Static scene ready.")


# ═════════════════════════════════════════════════════════════════════════════
# Live Meshcat playback (sleep-loop, real-time)
# ═════════════════════════════════════════════════════════════════════════════

def play_meshcat_live(viz, rmodel, rdata, foot_ids, segments,
                      xs, us, seg_idx, forces, dt, fps=20, loops=2):
    """Animate robot + dynamic elements at target FPS in the browser."""
    if viz is None:
        return

    viewer = viz.viewer
    T = len(xs)
    frame_step = max(1, int(round((1.0 / fps) / dt)))
    dt_frame   = 1.0 / fps

    print(f"\nMeshcat live playback — {fps} fps, {loops} loop(s).  "
          f"Keep the browser tab open.")

    # CoM trail: starts empty, extends each frame
    trail_pts = []

    for loop in range(loops):
        print(f"  Loop {loop + 1}/{loops}…")
        trail_pts.clear()

        for t in range(0, T, frame_step):
            t_wall0 = time.time()

            q  = xs[t, :rmodel.nq]
            si = int(seg_idx[t]) if t < len(seg_idx) else len(segments) - 1
            seg_col = segments[si].color

            # ── Robot pose ──────────────────────────────────────────────────
            try:
                viz.display(q)
            except Exception:
                pass

            # ── Foot positions (FK) ─────────────────────────────────────────
            pinocchio.forwardKinematics(rmodel, rdata, q)
            pinocchio.updateFramePlacements(rmodel, rdata)

            # ── Friction cone + GRF arrow (per foot) ────────────────────────
            for foot, fid in foot_ids.items():
                foot_pos = rdata.oMf[fid].translation.copy()
                grf = forces[foot][t] if t < len(forces[foot]) else np.zeros(3)
                # draw_contact_viz handles stance/swing detection internally
                draw_contact_viz(
                    viewer, f"dynamic/contact/{foot}",
                    foot_pos=foot_pos,
                    grf_world=grf,
                    mu=0.7,
                    grf_scale=GRF_SCALE,
                    cone_height=0.18,
                    n_spokes=8,
                    cone_color=FOOT_HEX[foot],
                    cone_opacity=0.50,
                )

            # ── CoM trail ───────────────────────────────────────────────────
            com_now = xs[t, :3].copy()
            trail_pts.append(com_now.tolist())
            if len(trail_pts) >= 2:
                mc_line(viewer, "dynamic/com_trail",
                        trail_pts, seg_col, lw=4)

            # ── Active target indicator (pulsing ring = sphere opacity) ──────
            if si < len(segments):
                tgt = segments[si].target_pos
                # Scale the sphere size gently to simulate pulse
                pulse = 0.07 + 0.02 * np.sin(t * dt * 4.0)
                mc_sphere(viewer, "dynamic/active_target",
                          tgt, pulse, segments[si].color, opacity=0.55)

            # ── Timing ──────────────────────────────────────────────────────
            elapsed = time.time() - t_wall0
            sleep   = dt_frame - elapsed
            if sleep > 0:
                time.sleep(sleep)

        # Clean up dynamic elements between loops
        mc_delete(viewer, "dynamic/com_trail")
        time.sleep(0.4)


# ═════════════════════════════════════════════════════════════════════════════
# Pre-baked keyframe animation (browser plays autonomously via Meshcat API)
# ═════════════════════════════════════════════════════════════════════════════

def bake_meshcat_animation(viz, rmodel, rdata, foot_ids, segments,
                           xs, us, seg_idx, forces, dt, fps=20):
    """Build a meshcat.animation.Animation object and push to browser.

    The browser plays this smoothly at the specified FPS without any
    Python timing loop — great for demos where you step away from the laptop.
    """
    if viz is None or not HAS_MESHCAT:
        print("  bake_meshcat_animation: Meshcat not available, skipping.")
        return

    try:
        from meshcat.animation import Animation
    except ImportError:
        print("  bake_meshcat_animation: meshcat.animation not available.")
        return

    print(f"\nBaking Meshcat animation ({fps} fps, {len(xs)} frames)…")
    anim = Animation(default_framerate=fps)

    T          = len(xs)
    frame_step = max(1, int(round(1.0 / (fps * dt))))
    trail_pts  = []
    baked_frames = 0

    for t in range(0, T, frame_step):
        q  = xs[t, :rmodel.nq]
        si = int(seg_idx[t]) if t < len(seg_idx) else len(segments) - 1

        # FK for foot positions
        pinocchio.forwardKinematics(rmodel, rdata, q)
        pinocchio.updateFramePlacements(rmodel, rdata)

        # Robot joints
        pinocchio.normalize(rmodel, q)
        Tmat = pinocchio.SE3ToXYZQUAT(rmodel.frames[0].placement)
        # Use Pinocchio's built-in animation helper if available
        try:
            with anim.at_frame(viz.viewer, baked_frames) as frame:
                # Set each joint transform
                for j_idx in range(rmodel.njoints):
                    joint_name = rmodel.names[j_idx]
                    pinocchio.forwardKinematics(rmodel, rdata, q)
                    placement = rdata.oMi[j_idx]
                    T4 = np.eye(4)
                    T4[:3, :3] = placement.rotation
                    T4[:3,  3] = placement.translation
                    frame[f"robot/{joint_name}"].set_transform(T4)
        except Exception:
            pass  # Older meshcat API — silently skip baked joints

        trail_pts.append(xs[t, :3].tolist())
        baked_frames += 1

    print(f"  Baked {baked_frames} keyframes.")

    try:
        viz.viewer.set_animation(anim, play=True, repetitions=0)
        print("  Animation pushed to browser (playing in loop).")
    except Exception as e:
        print(f"  set_animation failed: {e} — use live playback instead.")


# ═════════════════════════════════════════════════════════════════════════════
# Matplotlib side-panel animation (for GIF / MP4 export)
# ═════════════════════════════════════════════════════════════════════════════

def build_mpl_animation(segments, waypoints, com0, xs, us, forces, seg_idx,
                        com_ref, t_arr, dt, fps, gait_type):
    """2D top-down + GRF + torque panel — identical layout to random_waypoint_demo.py."""
    T          = len(xs)
    foot_order = ["LF", "RF", "LH", "RH"]

    fig = plt.figure(figsize=(15, 8), facecolor=DARK_BG)
    gs  = GridSpec(3, 3, figure=fig,
                   left=0.06, right=0.97, top=0.92, bottom=0.07,
                   hspace=0.55, wspace=0.40,
                   width_ratios=[1.8, 1, 1])

    ax_map  = fig.add_subplot(gs[:, 0])
    ax_grf  = fig.add_subplot(gs[0, 1:])
    ax_torq = fig.add_subplot(gs[1, 1:])
    ax_cost = fig.add_subplot(gs[2, 1:])

    for ax in [ax_map, ax_grf, ax_torq, ax_cost]:
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)

    # Map limits
    all_x = np.concatenate([s.xs[:, 0] for s in segments])
    all_y = np.concatenate([s.xs[:, 1] for s in segments])
    pad   = 0.4
    ax_map.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_map.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X — forward (m)")
    ax_map.set_ylabel("Y — lateral (m)")
    ax_map.set_title(f"Random Waypoint Navigation  |  B1 {gait_type} gait  "
                     f"(Bézier MPC + Meshcat, no RL)",
                     fontsize=11, fontweight="bold")

    # Static: all reference curves + control polygons (faint)
    for seg in segments:
        ax_map.plot(seg.com_traj[:, 0], seg.com_traj[:, 1],
                    color=seg.color, lw=1.0, alpha=0.20)
        cp = seg.ctrl_pts
        ax_map.plot(cp[:, 0], cp[:, 1], "--",
                    color=seg.color, lw=0.7, alpha=0.20)

    # Static: waypoint stars
    wp_pos = np.array([w.pos for w in waypoints])
    ax_map.scatter(wp_pos[:, 0], wp_pos[:, 1], s=120, marker="*",
                   c="#f1c40f", zorder=8, edgecolors=DARK_BG, linewidths=0.5)
    for i, wp in enumerate(waypoints):
        ax_map.annotate(f"W{i+1}", (wp.pos[0], wp.pos[1]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color="#f1c40f", alpha=0.85)
    ax_map.scatter(*com0[:2], s=200, marker="o", c="#2ecc71", zorder=10,
                   edgecolors="white", linewidths=1.5, label="Start")
    ax_map.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, loc="upper left")

    # Dynamic map elements
    past_line,      = ax_map.plot([], [], color="#cccccc", lw=1.0, alpha=0.5, zorder=3)
    active_bezier,  = ax_map.plot([], [], lw=2.5, alpha=0.9, zorder=6)
    ctrl_line,      = ax_map.plot([], [], "--o", lw=1.0, ms=6, alpha=0.75, zorder=5)
    robot_dot,      = ax_map.plot([], [], "o", ms=12, zorder=10,
                                   markeredgecolor="white", markeredgewidth=1.5)
    body_rect = plt.Polygon(np.zeros((4, 2)), closed=True,
                            facecolor="#3498db", edgecolor="white",
                            linewidth=1.2, alpha=0.8, zorder=9)
    ax_map.add_patch(body_rect)
    target_ring, = ax_map.plot([], [], "o", ms=22, alpha=0.35, zorder=7,
                                markeredgecolor="yellow", markerfacecolor="none",
                                markeredgewidth=2.5)
    seg_text = ax_map.text(0.02, 0.97, "", transform=ax_map.transAxes,
                            fontsize=9, va="top",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc=DARK_BG, ec=GRID_COL, alpha=0.8))

    # GRF axis
    ax_grf.set_xlim(0, t_arr[-1])
    fz_max = max(np.abs(forces[f][:, 2]).max() for f in foot_order)
    ax_grf.set_ylim(-fz_max * 0.1, fz_max * 1.1)
    ax_grf.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax_grf.set_ylabel("Fz (N)"); ax_grf.set_title("Ground Reaction Forces")
    ax_grf.tick_params(labelbottom=False)
    for foot in foot_order:
        ax_grf.plot(t_arr[:len(forces[foot])], forces[foot][:, 2],
                    color=FOOT_HEX[foot], lw=0.6, alpha=0.18)
    grf_lines = {f: ax_grf.plot([], [], color=FOOT_HEX[f], lw=1.8, label=f)[0]
                 for f in foot_order}
    grf_vl = ax_grf.axvline(0, color="white", lw=1.2, ls=":", alpha=0.6)
    ax_grf.legend(fontsize=7.5, ncol=4, facecolor=PANEL_BG, edgecolor=GRID_COL)

    # Torque axis
    ax_torq.set_xlim(0, t_arr[-1])
    u_max = np.abs(us).max() * 1.1 if len(us) else 20
    ax_torq.set_ylim(-u_max, u_max)
    ax_torq.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax_torq.set_ylabel("Torque (Nm)"); ax_torq.set_title("Front-Leg Joint Torques")
    ax_torq.tick_params(labelbottom=False)
    joint_names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
    colors_j    = plt.cm.Set2(np.linspace(0, 1, 6))
    for j in range(min(6, us.shape[1])):
        ax_torq.plot(t_arr[:len(us)], us[:, j], color=colors_j[j], lw=0.6, alpha=0.18)
    torq_lines = [ax_torq.plot([], [], color=colors_j[j], lw=1.8,
                               label=joint_names[j])[0]
                  for j in range(min(6, us.shape[1]))]
    torq_vl = ax_torq.axvline(0, color="white", lw=1.2, ls=":", alpha=0.6)
    ax_torq.legend(fontsize=7, ncol=3, facecolor=PANEL_BG, edgecolor=GRID_COL)

    # Cost axis
    ax_cost.set_xlim(0, t_arr[-1])
    ax_cost.set_ylim(0, max(s.cost for s in segments) * 1.25)
    ax_cost.set_xlabel("Time (s)"); ax_cost.set_ylabel("OCP Cost")
    ax_cost.set_title("MPC Solve Cost per Segment")
    seg_t0 = 0.0
    for seg in segments:
        seg_t1 = seg_t0 + len(seg.xs) * dt
        ax_cost.fill_between([seg_t0, seg_t1], [seg.cost, seg.cost],
                             alpha=0.42, color=seg.color)
        ax_cost.plot([seg_t0, seg_t1], [seg.cost, seg.cost], color=seg.color, lw=1.5)
        ax_cost.text((seg_t0 + seg_t1) / 2, seg.cost * 1.03,
                     "✓" if seg.converged else "⚠",
                     ha="center", fontsize=8, color=seg.color)
        if 0 < seg_t0 < t_arr[-1]:
            ax_cost.axvline(seg_t0, color=GRID_COL, lw=1.0, ls="--", alpha=0.7)
        seg_t0 = seg_t1
    cost_vl = ax_cost.axvline(0, color="white", lw=1.2, ls=":", alpha=0.6)

    time_txt = fig.text(0.5, 0.955, "", ha="center", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_BG,
                                  ec=GRID_COL, alpha=0.85))

    # Robot body rectangle helper
    def _footprint(cx, cy, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners = np.array([[+BODY_L/2, +BODY_W/2],
                             [-BODY_L/2, +BODY_W/2],
                             [-BODY_L/2, -BODY_W/2],
                             [+BODY_L/2, -BODY_W/2]])
        return corners @ R.T + np.array([cx, cy])

    def init():
        past_line.set_data([], [])
        active_bezier.set_data([], [])
        ctrl_line.set_data([], [])
        robot_dot.set_data([], [])
        body_rect.set_xy(np.zeros((4, 2)))
        target_ring.set_data([], [])
        return []

    def update(frame):
        t_now = t_arr[frame]
        si    = int(seg_idx[frame]) if frame < len(seg_idx) else len(segments) - 1
        seg   = segments[si]

        # CoM history
        past_line.set_data(xs[:frame+1, 0], xs[:frame+1, 1])

        # Active Bézier
        active_bezier.set_data(seg.com_traj[:, 0], seg.com_traj[:, 1])
        active_bezier.set_color(seg.color); active_bezier.set_linewidth(2.5)

        # Control polygon
        cp = seg.ctrl_pts
        ctrl_line.set_data(cp[:, 0], cp[:, 1])
        ctrl_line.set_color(seg.color)

        # Robot body
        rx, ry = xs[frame, 0], xs[frame, 1]
        yaw = (np.arctan2(xs[frame, 1] - xs[max(frame-2, 0), 1],
                          xs[frame, 0] - xs[max(frame-2, 0), 0])
               if frame > 2 else 0.0)
        robot_dot.set_data([rx], [ry]); robot_dot.set_color(seg.color)
        body_rect.set_xy(_footprint(rx, ry, yaw))
        body_rect.set_facecolor(seg.color)

        # Target ring
        target_ring.set_data([seg.target_pos[0]], [seg.target_pos[1]])
        target_ring.set_color(seg.color)

        # Segment label
        seg_text.set_text(f"Seg {si+1}/{len(segments)}  |  "
                          f"cost={seg.cost:.0f}  "
                          f"{'✓' if seg.converged else '⚠'}  "
                          f"{seg.solve_ms:.0f} ms")
        seg_text.get_bbox_patch().set_edgecolor(seg.color)

        # GRF
        for foot in foot_order:
            n = min(frame+1, len(forces[foot]))
            grf_lines[foot].set_data(t_arr[:n], forces[foot][:n, 2])
        grf_vl.set_xdata([t_now, t_now])

        # Torques
        fu = min(frame, len(us)-1)
        for j, line in enumerate(torq_lines):
            line.set_data(t_arr[:fu+1], us[:fu+1, j])
        torq_vl.set_xdata([t_now, t_now])

        cost_vl.set_xdata([t_now, t_now])
        time_txt.set_text(
            f"t = {t_now:.2f} s  |  frame {frame}/{T}  |  "
            f"waypoint {si+1}/{len(segments)}  "
            f"target: ({seg.target_pos[0]:.2f}, {seg.target_pos[1]:.2f}) m")
        return (past_line, active_bezier, ctrl_line, robot_dot,
                body_rect, target_ring, seg_text, grf_vl, torq_vl,
                cost_vl, time_txt, *grf_lines.values(), *torq_lines)

    interval = max(16, int(1000 / fps))
    anim_obj = mpl_animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=interval, blit=False,
    )
    return fig, anim_obj


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(segments):
    print("\n" + "═" * 56)
    print("Navigation Summary")
    print("═" * 56)
    total_d = sum(np.linalg.norm(s.target_pos[:2] - s.start_pos[:2])
                  for s in segments)
    n_ok    = sum(s.converged for s in segments)
    print(f"  Segments:       {len(segments)}")
    print(f"  Total distance: {total_d:.2f} m")
    print(f"  Converged:      {n_ok}/{len(segments)}")
    print(f"  Mean cost:      {np.mean([s.cost for s in segments]):.0f}")
    print(f"  Mean solve:     {np.mean([s.solve_ms for s in segments]):.0f} ms")
    print("─" * 56)
    for seg in segments:
        d  = np.linalg.norm(seg.target_pos[:2] - seg.start_pos[:2])
        ok = "✓" if seg.converged else "⚠"
        print(f"  [{seg.idx+1}] {ok} d={d:.2f}m  cost={seg.cost:.0f}"
              f"  {seg.solve_ms:.0f}ms  {seg.color}")
    print("═" * 56)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Random waypoint Bézier MPC — Pinocchio + Meshcat 3D demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-waypoints", type=int, default=6,
                        help="Number of random waypoints (default 6)")
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--gait",        default="trot",
                        choices=["trot", "walk", "pace"])
    parser.add_argument("--dist-min",    type=float, default=0.6)
    parser.add_argument("--dist-max",    type=float, default=1.3)
    parser.add_argument("--max-turn",    type=float, default=55.0,
                        help="Max heading change per waypoint (deg)")
    parser.add_argument("--duration",    type=float, default=3.0,
                        help="Bézier / OCP horizon per segment (s)")
    parser.add_argument("--fps",         type=float, default=20.0,
                        help="Meshcat / animation playback FPS")
    parser.add_argument("--loops",       type=int, default=2,
                        help="Meshcat live playback loops (default 2)")
    parser.add_argument("--bake-animation", action="store_true",
                        help="Push pre-baked keyframe animation to Meshcat browser")
    parser.add_argument("--no-meshcat",  action="store_true",
                        help="Skip Meshcat 3D view entirely")
    parser.add_argument("--save-gif",    metavar="FILE",
                        help="Save matplotlib side panel as GIF")
    parser.add_argument("--save-mp4",    metavar="FILE",
                        help="Save matplotlib side panel as MP4")
    parser.add_argument("--no-show",     action="store_true",
                        help="Don't open matplotlib window")
    args = parser.parse_args()

    # Matplotlib backend
    if args.no_show or args.save_gif or args.save_mp4:
        matplotlib.use("Agg")

    print("=" * 56)
    print("Random Waypoint Bézier MPC + Meshcat  |  B1 Quadruped")
    print("=" * 56)
    if args.seed is not None:
        print(f"  Seed:      {args.seed}")
    print(f"  Waypoints: {args.n_waypoints}  |  Gait: {args.gait}")
    print(f"  Distance:  [{args.dist_min}, {args.dist_max}] m")
    print(f"  Max turn:  ±{args.max_turn}°")

    # ── Load robot ────────────────────────────────────────────────────────────
    print("\nLoading B1 robot model…")
    b1, rmodel, rdata, q0, v0, x0, com0, foot_ids = load_b1()
    print(f"  nq={rmodel.nq}  nv={rmodel.nv}  com0={np.round(com0, 3)}")

    # ── Generate waypoints ────────────────────────────────────────────────────
    print("\nGenerating random waypoints…")
    gen = RandomWaypointGenerator(
        dist_min=args.dist_min, dist_max=args.dist_max,
        max_turn_deg=args.max_turn, com_height=com0[2], seed=args.seed)
    waypoints, cur_pos, cur_heading = [], com0.copy(), 0.0
    for i in range(args.n_waypoints):
        wp = gen.next(cur_pos, cur_heading)
        waypoints.append(wp)
        print(f"  W{i+1}: ({wp.pos[0]:+.2f}, {wp.pos[1]:+.2f}, {wp.pos[2]:.2f})")
        cur_pos, cur_heading = wp.pos, wp.heading

    # ── Solve all segments ────────────────────────────────────────────────────
    segments = compute_all_segments(
        rmodel, x0, com0, foot_ids, waypoints,
        gait_type=args.gait, dt=0.02, duration=args.duration)
    print_summary(segments)

    # ── Concatenate timeline ──────────────────────────────────────────────────
    xs, us, forces, seg_idx, com_ref, t_arr = concat_segments(segments, dt=0.02)

    # ── Meshcat 3D ────────────────────────────────────────────────────────────
    viz = None
    if not args.no_meshcat and HAS_MESHCAT:
        print("\nInitializing Pinocchio MeshcatVisualizer…")
        try:
            viz = MeshcatVisualizer(rmodel, b1.collision_model, b1.visual_model)
            viz.initViewer(open=True, loadModel=True)
            print(f"  Browser URL: {viz.viewer.url()}")
            print("  (Open the URL above in Chrome/Firefox)")

            # Draw static scene
            build_static_scene(viz.viewer, segments, waypoints, com0)

            # Let user see the static scene for 2 s
            time.sleep(2.0)

            if args.bake_animation:
                bake_meshcat_animation(
                    viz, rmodel, rdata, foot_ids, segments,
                    xs, us, seg_idx, forces, dt=0.02, fps=args.fps)
                print("  Browser is now playing the pre-baked animation.")
                print("  (Ctrl-C to continue to matplotlib / exit)")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            else:
                play_meshcat_live(
                    viz, rmodel, rdata, foot_ids, segments,
                    xs, us, seg_idx, forces, dt=0.02,
                    fps=args.fps, loops=args.loops)

        except Exception as e:
            print(f"  Meshcat initialisation failed: {e}")
            print("  Continuing without 3D view.")
            viz = None
    elif args.no_meshcat:
        print("\n[Meshcat disabled by --no-meshcat]")
    else:
        print("\n[meshcat-python not installed — skipping 3D view]")

    # ── Matplotlib side panel ─────────────────────────────────────────────────
    if args.save_gif or args.save_mp4 or not args.no_show:
        print("\nBuilding matplotlib side panel animation…")
        fig, anim_obj = build_mpl_animation(
            segments, waypoints, com0,
            xs, us, forces, seg_idx, com_ref, t_arr,
            dt=0.02, fps=args.fps, gait_type=args.gait)

        if args.save_gif:
            print(f"Saving GIF → {args.save_gif}  (may take a few minutes)…")
            writer = mpl_animation.PillowWriter(fps=args.fps)
            anim_obj.save(args.save_gif, writer=writer, dpi=110,
                          progress_callback=lambda i, n:
                              print(f"  frame {i}/{n}", end="\r"))
            print(f"\n  Saved → {args.save_gif}")

        if args.save_mp4:
            print(f"Saving MP4 → {args.save_mp4}…")
            writer = mpl_animation.FFMpegWriter(
                fps=args.fps, bitrate=2500,
                metadata={"title": "Random Waypoint Bézier MPC"})
            anim_obj.save(args.save_mp4, writer=writer, dpi=140,
                          progress_callback=lambda i, n:
                              print(f"  frame {i}/{n}", end="\r"))
            print(f"\n  Saved → {args.save_mp4}")

        if not args.no_show:
            plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
