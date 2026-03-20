#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random waypoint navigation demo — pure Bézier + Crocoddyl MPC (no RL).

Concept
-------
At each segment the navigator:
  1. Picks a *random* target point (bounded distance + heading change)
  2. Builds a **G1-continuous** cubic Bézier curve: entry tangent = previous
     exit tangent, exit tangent aimed at the next target
  3. Solves a Crocoddyl FDDP OCP with the Bézier as CoM reference
  4. Chains the solved trajectories (xs, us) end-to-end

All segments are pre-computed offline, then played back as a smooth animation:
  • Top-down 2D view: past CoM path, current Bézier segment with control polygon,
    all waypoints (visited / current / upcoming)
  • Robot "footprint" rectangle at current pose
  • Right panels: GRF Fz per foot, front-leg joint torques, live cost meter

Usage
-----
    cd B1/scripts

    # Interactive matplotlib window (default 6 waypoints)
    python random_waypoint_demo.py

    # More waypoints, slower playback
    python random_waypoint_demo.py --n-waypoints 10 --fps 8

    # Save as GIF for poster
    python random_waypoint_demo.py --save-gif random_nav.gif --n-waypoints 6

    # Save as MP4 for video
    python random_waypoint_demo.py --save-mp4 random_nav.mp4

    # Fixed random seed for reproducibility
    python random_waypoint_demo.py --seed 42

    # Walk gait (more conservative steps, more stable)
    python random_waypoint_demo.py --gait walk
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
B1_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, B1_DIR)

# ── dependencies ──────────────────────────────────────────────────────────────
try:
    import example_robot_data
    import pinocchio
    import crocoddyl
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("pip install crocoddyl pinocchio example-robot-data")
    sys.exit(1)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

from quadruped_mpc.trajectory import BezierTrajectoryGenerator
from quadruped_mpc.gait import GaitScheduler, FootholdPlanner, OCPFactory
from quadruped_mpc.utils.math_utils import heading_from_tangent

# ── B1 constants ──────────────────────────────────────────────────────────────
FOOT_FRAME_NAMES = {"LF": "FL_foot", "RF": "FR_foot", "LH": "RL_foot", "RH": "RR_foot"}
HIP_OFFSETS = {
    "LF": np.array([+0.3, +0.1, 0.0]),
    "RF": np.array([+0.3, -0.1, 0.0]),
    "LH": np.array([-0.3, +0.1, 0.0]),
    "RH": np.array([-0.3, -0.1, 0.0]),
}
GAIT_PARAMS = {"step_duration": 0.15, "support_duration": 0.05, "step_height": 0.15}
FOOT_COLORS = {"LF": "#27ae60", "RF": "#e74c3c", "LH": "#2980b9", "RH": "#e67e22"}

# Bézier curve palette (one color per segment, cycles)
SEGMENT_PALETTE = [
    "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
    "#f39c12", "#1abc9c", "#e67e22", "#e91e63",
    "#00bcd4", "#8bc34a",
]

# ── visual style ─────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#30363d"
TEXT_COL  = "#e6edf3"
MUTED     = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COL,
    "grid.color": GRID_COL,
    "grid.alpha": 0.4,
    "text.color": TEXT_COL,
    "axes.labelcolor": TEXT_COL,
    "xtick.color": TEXT_COL,
    "ytick.color": TEXT_COL,
    "font.size": 9,
    "axes.titlesize": 10,
    "lines.linewidth": 1.8,
})


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Waypoint:
    """A target point in world XY."""
    pos: np.ndarray        # (3,)  world position
    heading: float         # yaw angle at arrival


@dataclass
class Segment:
    """One solved navigation segment: start → target."""
    waypoint_idx: int
    start_pos: np.ndarray        # (3,)
    target_pos: np.ndarray       # (3,)
    ctrl_pts: np.ndarray         # (4, 3)  Bézier control points (absolute)
    com_traj: np.ndarray         # (N, 3)  dense CoM reference
    xs: np.ndarray               # (T+1, nx)
    us: np.ndarray               # (T, nu)
    forces: dict                 # foot → (T, 3)
    converged: bool
    cost: float
    solve_ms: float
    color: str


# ═════════════════════════════════════════════════════════════════════════════
# Robot loading
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


# ═════════════════════════════════════════════════════════════════════════════
# Random waypoint generator
# ═════════════════════════════════════════════════════════════════════════════

class RandomWaypointGenerator:
    """Generates random reachable waypoints from current position.

    Constraints:
      - Step distance: uniform in [dist_min, dist_max]
      - Heading change: uniform in [-max_turn, +max_turn]
      - Height: fixed (standing CoM height)
    """

    def __init__(
        self,
        dist_min: float = 0.6,
        dist_max: float = 1.4,
        max_turn_deg: float = 55.0,
        com_height: float = 0.45,
        seed: Optional[int] = None,
    ):
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.max_turn = np.deg2rad(max_turn_deg)
        self.com_height = com_height
        self.rng = np.random.default_rng(seed)

    def next(self, current_pos: np.ndarray, current_heading: float) -> Waypoint:
        """Sample a new waypoint from current position."""
        dist = self.rng.uniform(self.dist_min, self.dist_max)
        dheading = self.rng.uniform(-self.max_turn, self.max_turn)
        new_heading = current_heading + dheading

        dx = dist * np.cos(new_heading)
        dy = dist * np.sin(new_heading)
        pos = np.array([
            current_pos[0] + dx,
            current_pos[1] + dy,
            self.com_height,
        ])
        return Waypoint(pos=pos, heading=new_heading)


# ═════════════════════════════════════════════════════════════════════════════
# G1-continuous Bézier curve builder
# ═════════════════════════════════════════════════════════════════════════════

def build_g1_bezier(
    start: np.ndarray,
    target: np.ndarray,
    entry_tangent: np.ndarray,     # unit vector (3,) — direction entering start
    dt: float = 0.02,
    duration: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a G1-continuous cubic Bézier from start to target.

    G1 continuity means: the tangent direction at the START of this segment
    matches the tangent direction at the END of the previous segment.

    Control points:
        P0 = start
        P1 = start  +  entry_tangent * ||start→target|| / 3
             (preserves incoming velocity direction)
        P2 = target - exit_tangent  * ||start→target|| / 3
             (exit tangent aims at target from slightly behind)
        P3 = target

    Returns:
        (ctrl_pts, com_traj, heading_traj)
        ctrl_pts shape: (4, 3)
        com_traj shape: (N, 3)
        heading_traj shape: (N,)
    """
    gen = BezierTrajectoryGenerator(degree=3, state_dim=3, max_displacement=10.0)

    diff = target - start
    dist = np.linalg.norm(diff[:2])   # 2D distance
    exit_tangent = np.array([diff[0], diff[1], 0.0])
    if np.linalg.norm(exit_tangent) > 1e-6:
        exit_tangent /= np.linalg.norm(exit_tangent)

    # Blend length: 1/3 of chord
    blend = max(dist / 3.0, 0.15)

    P0 = start.copy()
    P1 = start + entry_tangent * blend
    P2 = target - exit_tangent * blend
    P3 = target.copy()

    # Keep height constant throughout (Z = start.z)
    # Allows a slight Z blend if target.z != start.z
    P1[2] = start[2]
    P2[2] = target[2]

    ctrl_pts = np.array([P0, P1, P2, P3])

    # Build dense trajectory via BezierTrajectoryGenerator
    # Use zero-offset params (ctrl_pts are absolute)
    # params_to_waypoints adds start_position, so we pass offsets relative to P0
    offsets = (ctrl_pts - P0).flatten()  # (12,)
    com_traj = gen.params_to_waypoints(
        params=offsets, dt=dt, horizon=duration, start_position=P0,
    )

    # Compute heading from trajectory tangents
    N = len(com_traj)
    heading_traj = np.zeros(N)
    for i in range(N):
        if i == 0:
            tang = (com_traj[1] - com_traj[0]) / dt
        elif i == N - 1:
            tang = (com_traj[-1] - com_traj[-2]) / dt
        else:
            tang = (com_traj[i + 1] - com_traj[i - 1]) / (2 * dt)
        heading_traj[i] = heading_from_tangent(tang[:2])

    return ctrl_pts, com_traj, heading_traj


def exit_tangent_from_traj(com_traj: np.ndarray, dt: float) -> np.ndarray:
    """Extract the exit tangent direction of a trajectory (last two points)."""
    if len(com_traj) < 2:
        return np.array([1.0, 0.0, 0.0])
    tang = (com_traj[-1] - com_traj[-2]) / dt
    tang[2] = 0.0
    norm = np.linalg.norm(tang)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return tang / norm


# ═════════════════════════════════════════════════════════════════════════════
# GRF extractor
# ═════════════════════════════════════════════════════════════════════════════

def extract_grf(solver, foot_ids: dict) -> dict:
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
# OCP solver for one segment
# ═════════════════════════════════════════════════════════════════════════════

def solve_segment(
    rmodel, x0: np.ndarray, foot_ids: dict,
    com_traj: np.ndarray, heading_traj: np.ndarray,
    gait_type: str, dt: float = 0.02,
) -> Tuple[crocoddyl.SolverFDDP, bool, float]:
    """Solve one OCP segment. Returns (solver, converged, solve_ms)."""
    sched = GaitScheduler()
    planner = FootholdPlanner(hip_offsets=HIP_OFFSETS, step_height=GAIT_PARAMS["step_height"])
    seq = sched.generate(
        gait_type=gait_type,
        step_duration=GAIT_PARAMS["step_duration"],
        support_duration=GAIT_PARAMS["support_duration"],
        num_cycles=12,
    )
    com0 = com_traj[0].copy()
    init_feet = planner.get_footholds_at_time(com_position=com0, heading=0.0)
    foot_plans = planner.plan_footholds(
        com_trajectory=com_traj, contact_sequence=seq,
        current_foot_positions=init_feet, dt=dt,
    )

    factory = OCPFactory(rmodel=rmodel, foot_frame_ids=foot_ids, mu=0.7)
    factory.x0 = x0
    problem = factory.build_problem(
        x0=x0, contact_sequence=seq, com_trajectory=com_traj,
        foot_trajectories=foot_plans, dt=dt, heading_trajectory=heading_traj,
    )

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-4

    xs0 = [x0] * (problem.T + 1)
    us0 = problem.quasiStatic([x0] * problem.T)

    t0 = time.time()
    converged = solver.solve(xs0, us0, 100, False)
    solve_ms = (time.time() - t0) * 1000

    return solver, converged, solve_ms


# ═════════════════════════════════════════════════════════════════════════════
# Pre-compute all segments
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_segments(
    rmodel, x0_init: np.ndarray, com0: np.ndarray, foot_ids: dict,
    waypoints: List[Waypoint],
    gait_type: str = "trot",
    dt: float = 0.02,
    duration: float = 3.0,
) -> List[Segment]:
    """Solve OCP for each waypoint and chain results."""
    segments = []

    current_x = x0_init.copy()
    current_com = com0.copy()
    # Initial entry tangent: face forward (+X)
    entry_tangent = np.array([1.0, 0.0, 0.0])

    print(f"\nPre-computing {len(waypoints)} segments…")
    print("─" * 55)

    for i, wp in enumerate(waypoints):
        color = SEGMENT_PALETTE[i % len(SEGMENT_PALETTE)]
        print(f"  Segment {i+1}/{len(waypoints)}: "
              f"({current_com[0]:.2f},{current_com[1]:.2f}) → "
              f"({wp.pos[0]:.2f},{wp.pos[1]:.2f})  [{gait_type}]")

        # Build G1-continuous Bézier
        ctrl_pts, com_traj, heading_traj = build_g1_bezier(
            start=current_com,
            target=wp.pos,
            entry_tangent=entry_tangent,
            dt=dt,
            duration=duration,
        )

        # Solve OCP
        solver, converged, solve_ms = solve_segment(
            rmodel, current_x, foot_ids,
            com_traj, heading_traj, gait_type, dt,
        )

        status = "✓" if converged else "⚠"
        print(f"    {status} cost={solver.cost:.0f}  iter={solver.iter}  "
              f"solve={solve_ms:.0f} ms")

        xs = np.array(solver.xs)
        us = np.array(solver.us)
        forces = extract_grf(solver, foot_ids)

        seg = Segment(
            waypoint_idx=i,
            start_pos=current_com.copy(),
            target_pos=wp.pos.copy(),
            ctrl_pts=ctrl_pts,
            com_traj=com_traj,
            xs=xs,
            us=us,
            forces=forces,
            converged=converged,
            cost=solver.cost,
            solve_ms=solve_ms,
            color=color,
        )
        segments.append(seg)

        # Chain: next segment starts from end of this one
        current_x = xs[-1].copy()
        current_com = xs[-1, :3].copy()   # approximate: first 3 states = x, y, z
        # Actually use com_traj endpoint for more accuracy
        current_com = com_traj[-1].copy()
        entry_tangent = exit_tangent_from_traj(com_traj, dt)

    print("─" * 55)
    print(f"All segments solved. Total states: "
          f"{sum(len(s.xs) for s in segments)}")
    return segments


# ═════════════════════════════════════════════════════════════════════════════
# Build concatenated timeline for animation
# ═════════════════════════════════════════════════════════════════════════════

def concat_segments(segments: List[Segment], dt: float):
    """Concatenate all segment data into one long timeline."""
    all_xs = []
    all_us = []
    all_forces = {foot: [] for foot in FOOT_COLORS}
    all_seg_idx = []   # which segment each frame belongs to
    all_com_ref = []   # CoM reference trajectory

    for si, seg in enumerate(segments):
        T = len(seg.xs)
        # Skip first state of subsequent segments (it's the last of previous)
        start = 1 if si > 0 else 0
        all_xs.append(seg.xs[start:])
        all_us.append(seg.us[start:] if start < len(seg.us) else seg.us)
        for foot in FOOT_COLORS:
            all_forces[foot].append(seg.forces[foot][start:])
        all_seg_idx.extend([si] * (T - start))
        all_com_ref.append(seg.com_traj[start:])

    xs = np.concatenate(all_xs, axis=0)
    us = np.concatenate(all_us, axis=0)
    for foot in FOOT_COLORS:
        all_forces[foot] = np.concatenate(all_forces[foot], axis=0)
    seg_idx = np.array(all_seg_idx, dtype=int)
    com_ref = np.concatenate(all_com_ref, axis=0)
    time_arr = np.arange(len(xs)) * dt

    return xs, us, all_forces, seg_idx, com_ref, time_arr


# ═════════════════════════════════════════════════════════════════════════════
# Animation builder
# ═════════════════════════════════════════════════════════════════════════════

def build_animation(
    segments: List[Segment],
    waypoints: List[Waypoint],
    com0: np.ndarray,
    dt: float,
    fps: float = 20.0,
    gait_type: str = "trot",
):
    """Build the full matplotlib animation."""

    xs, us, forces, seg_idx, com_ref, time_arr = concat_segments(segments, dt)
    T = len(xs)
    foot_order = ["LF", "RF", "LH", "RH"]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8), facecolor=DARK_BG)
    gs = GridSpec(
        3, 3, figure=fig,
        left=0.06, right=0.97, top=0.92, bottom=0.07,
        hspace=0.55, wspace=0.4,
        width_ratios=[1.8, 1, 1],
    )

    ax_map  = fig.add_subplot(gs[:, 0])      # Main: top-down map
    ax_grf  = fig.add_subplot(gs[0, 1:])     # Top-right: GRF Fz
    ax_torq = fig.add_subplot(gs[1, 1:])     # Mid-right: torques
    ax_cost = fig.add_subplot(gs[2, 1:])     # Bot-right: cost/info bar

    for ax in [ax_map, ax_grf, ax_torq, ax_cost]:
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)

    # ── compute axis limits from all trajectories ─────────────────────────────
    all_x = np.concatenate([s.xs[:, 0] for s in segments])
    all_y = np.concatenate([s.xs[:, 1] for s in segments])
    pad = 0.4
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.min() - pad, all_y.max() + pad)

    # ── MAP axis setup ────────────────────────────────────────────────────────
    ax_map.set_xlim(*xlim)
    ax_map.set_ylim(*ylim)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X — forward (m)", color=TEXT_COL)
    ax_map.set_ylabel("Y — lateral (m)", color=TEXT_COL)
    ax_map.set_title(f"Random Waypoint Navigation  |  B1 {gait_type} gait  (Bézier MPC, no RL)",
                     color=TEXT_COL, fontsize=11, fontweight="bold")

    # Draw all Bézier control polygons (faint, static background)
    for seg in segments:
        cp = seg.ctrl_pts
        ax_map.plot(cp[:, 0], cp[:, 1], "--",
                    color=seg.color, lw=0.7, alpha=0.25)

    # Draw all CoM reference curves (faint background)
    for seg in segments:
        ax_map.plot(seg.com_traj[:, 0], seg.com_traj[:, 1],
                    color=seg.color, lw=1.0, alpha=0.2)

    # Waypoint markers (static)
    wp_positions = np.array([w.pos for w in waypoints])
    ax_map.scatter(wp_positions[:, 0], wp_positions[:, 1],
                   s=120, marker="*", c="#f1c40f", zorder=8,
                   edgecolors=DARK_BG, linewidths=0.5, label="Waypoints")
    for i, wp in enumerate(waypoints):
        ax_map.annotate(f"W{i+1}", (wp.pos[0], wp.pos[1]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color="#f1c40f", alpha=0.8)

    # Start marker
    ax_map.scatter(*com0[:2], s=180, marker="o", c="#2ecc71", zorder=10,
                   edgecolors="white", linewidths=1.2, label="Start")

    ax_map.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
                  loc="upper left")

    # ── Dynamic map elements ──────────────────────────────────────────────────
    # Past CoM path (multi-colored by segment)
    past_line, = ax_map.plot([], [], color="#cccccc", lw=1.0, alpha=0.45,
                              zorder=3, label="_nolegend_")

    # Active segment: Bézier curve highlight
    active_bezier, = ax_map.plot([], [], lw=2.5, alpha=0.9, zorder=6)

    # Active segment: control polygon
    ctrl_line, = ax_map.plot([], [], "--o", lw=1.0, ms=6, alpha=0.7, zorder=5)

    # Robot position dot
    robot_dot, = ax_map.plot([], [], "o", ms=12, zorder=10,
                              markeredgecolor="white", markeredgewidth=1.5)

    # Robot "body rectangle" — shows orientation
    body_rect = plt.Polygon(np.zeros((4, 2)), closed=True,
                             facecolor="#3498db", edgecolor="white",
                             linewidth=1.2, alpha=0.85, zorder=9)
    ax_map.add_patch(body_rect)

    # Current target highlight ring
    target_ring, = ax_map.plot([], [], "o", ms=22, alpha=0.35, zorder=7,
                                markeredgecolor="yellow", markerfacecolor="none",
                                markeredgewidth=2.5)
    target_dot,  = ax_map.plot([], [], "*", ms=14, zorder=11, color="yellow")

    # Heading arrow on robot
    heading_arrow = ax_map.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=14),
        zorder=12,
    )

    # Segment label
    seg_text = ax_map.text(0.02, 0.97, "", transform=ax_map.transAxes,
                            color=TEXT_COL, fontsize=9, va="top",
                            bbox=dict(boxstyle="round,pad=0.3", fc=DARK_BG,
                                      ec=GRID_COL, alpha=0.8))

    # ── GRF axis ─────────────────────────────────────────────────────────────
    ax_grf.set_xlim(0, time_arr[-1])
    fz_all = np.concatenate([forces[f][:, 2] for f in foot_order])
    fz_range = max(np.abs(fz_all).max() * 1.1, 10)
    ax_grf.set_ylim(-fz_range * 0.15, fz_range * 1.05)
    ax_grf.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax_grf.set_ylabel("Fz (N)", color=TEXT_COL)
    ax_grf.set_title("Vertical Ground Reaction Forces", color=TEXT_COL)
    ax_grf.tick_params(labelbottom=False)

    # Faint background traces
    for foot in foot_order:
        ax_grf.plot(time_arr[:len(forces[foot])], forces[foot][:, 2],
                    color=FOOT_COLORS[foot], lw=0.6, alpha=0.18)
    grf_lines = {foot: ax_grf.plot([], [], color=FOOT_COLORS[foot],
                                    lw=1.8, label=foot)[0]
                 for foot in foot_order}
    grf_vline = ax_grf.axvline(0, color="white", lw=1.2, ls=":", alpha=0.6)
    ax_grf.legend(fontsize=7.5, ncol=4, facecolor=PANEL_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL)

    # ── Torque axis ──────────────────────────────────────────────────────────
    ax_torq.set_xlim(0, time_arr[-1])
    if len(us) > 0:
        u_range = max(np.abs(us).max() * 1.1, 5)
        ax_torq.set_ylim(-u_range, u_range)
    ax_torq.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax_torq.set_ylabel("Torque (Nm)", color=TEXT_COL)
    ax_torq.set_title("Front-Leg Joint Torques", color=TEXT_COL)
    ax_torq.tick_params(labelbottom=False)

    joint_names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
    colors_j = plt.cm.Set2(np.linspace(0, 1, min(6, us.shape[1])))
    for j in range(min(6, us.shape[1])):
        ax_torq.plot(time_arr[:len(us)], us[:, j],
                     color=colors_j[j], lw=0.6, alpha=0.18)
    torq_lines = [ax_torq.plot([], [], color=colors_j[j],
                                lw=1.8, label=joint_names[j])[0]
                  for j in range(min(6, us.shape[1]))]
    torq_vline = ax_torq.axvline(0, color="white", lw=1.2, ls=":", alpha=0.6)
    ax_torq.legend(fontsize=7, ncol=3, facecolor=PANEL_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL)

    # ── Cost / info axis ─────────────────────────────────────────────────────
    ax_cost.set_xlim(0, time_arr[-1])
    all_costs = [s.cost for s in segments]
    cost_range = max(all_costs) * 1.2 if all_costs else 1000
    ax_cost.set_ylim(0, cost_range)
    ax_cost.set_xlabel("Time (s)", color=TEXT_COL)
    ax_cost.set_ylabel("OCP Cost", color=TEXT_COL)
    ax_cost.set_title("MPC Solve Cost per Segment", color=TEXT_COL)

    # Draw a horizontal bar for each segment's cost
    seg_boundaries = [0.0]
    t_cursor = 0.0
    for seg in segments:
        t_cursor += len(seg.xs) * dt
        seg_boundaries.append(t_cursor)

    for i, seg in enumerate(segments):
        t0 = seg_boundaries[i]
        t1 = seg_boundaries[i + 1]
        ax_cost.fill_between([t0, t1], [seg.cost, seg.cost],
                             alpha=0.45, color=seg.color, step="pre")
        ax_cost.plot([t0, t1], [seg.cost, seg.cost],
                     color=seg.color, lw=1.5)
        # Convergence label
        label = "✓" if seg.converged else "⚠"
        ax_cost.text((t0 + t1) / 2, seg.cost * 1.02, label,
                     ha="center", fontsize=8, color=seg.color, alpha=0.8)
    cost_vline = ax_cost.axvline(0, color="white", lw=1.2, ls=":", alpha=0.6)

    # ── Segment boundary lines across all panels ──────────────────────────────
    for ax in [ax_grf, ax_torq, ax_cost]:
        for tb in seg_boundaries[1:-1]:
            ax.axvline(tb, color=GRID_COL, lw=1.0, ls="--", alpha=0.7)

    # ── Time label ────────────────────────────────────────────────────────────
    time_text = fig.text(0.5, 0.955, "", ha="center", va="top",
                         fontsize=10, color=TEXT_COL,
                         bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_BG,
                                   ec=GRID_COL, alpha=0.85))

    # ── B1 body dimensions for footprint ─────────────────────────────────────
    BODY_L = 0.55   # body length
    BODY_W = 0.25   # body width

    def robot_footprint(cx, cy, yaw):
        """4-corner rectangle of robot body."""
        corners = np.array([
            [+BODY_L / 2, +BODY_W / 2],
            [-BODY_L / 2, +BODY_W / 2],
            [-BODY_L / 2, -BODY_W / 2],
            [+BODY_L / 2, -BODY_W / 2],
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        rotated = corners @ R.T
        return rotated + np.array([cx, cy])

    # ── Smooth color for past path using segment membership ──────────────────
    def get_seg_color(frame):
        si = seg_idx[frame] if frame < len(seg_idx) else seg_idx[-1]
        return segments[si].color

    # ── Init and update functions ─────────────────────────────────────────────
    def init():
        past_line.set_data([], [])
        active_bezier.set_data([], [])
        ctrl_line.set_data([], [])
        robot_dot.set_data([], [])
        body_rect.set_xy(np.zeros((4, 2)))
        target_ring.set_data([], [])
        target_dot.set_data([], [])
        for line in grf_lines.values():
            line.set_data([], [])
        for line in torq_lines:
            line.set_data([], [])
        return []

    def update(frame):
        t_now = time_arr[frame]
        si = int(seg_idx[frame]) if frame < len(seg_idx) else len(segments) - 1
        seg = segments[si]

        # ── MAP ──────────────────────────────────────────────────────────────
        # Past CoM path (all frames up to now)
        cx = xs[:frame + 1, 0]
        cy = xs[:frame + 1, 1]
        past_line.set_data(cx, cy)

        # Current segment Bézier curve
        active_bezier.set_data(seg.com_traj[:, 0], seg.com_traj[:, 1])
        active_bezier.set_color(seg.color)
        active_bezier.set_linewidth(2.5)

        # Control polygon
        cp = seg.ctrl_pts
        ctrl_line.set_data(cp[:, 0], cp[:, 1])
        ctrl_line.set_color(seg.color)

        # Robot position + heading
        rx, ry = xs[frame, 0], xs[frame, 1]
        # Approximate heading from recent CoM velocity
        if frame > 2:
            dx = xs[frame, 0] - xs[frame - 2, 0]
            dy = xs[frame, 1] - xs[frame - 2, 1]
            yaw = np.arctan2(dy, dx) if abs(dx) + abs(dy) > 1e-4 else 0.0
        else:
            yaw = 0.0

        robot_dot.set_data([rx], [ry])
        robot_dot.set_color(seg.color)

        corners = robot_footprint(rx, ry, yaw)
        body_rect.set_xy(corners)
        body_rect.set_facecolor(seg.color)
        body_rect.set_alpha(0.8)

        # Heading arrow
        arrow_len = 0.3
        heading_arrow.set_position((rx, ry))
        heading_arrow.xy = (rx + arrow_len * np.cos(yaw),
                            ry + arrow_len * np.sin(yaw))
        heading_arrow.xytext = (rx, ry)

        # Current target
        tx, ty = seg.target_pos[0], seg.target_pos[1]
        target_ring.set_data([tx], [ty])
        target_ring.set_color(seg.color)
        target_dot.set_data([tx], [ty])
        target_dot.set_color(seg.color)

        # Segment info text
        seg_text.set_text(
            f"Seg {si+1}/{len(segments)}  |  "
            f"cost={seg.cost:.0f}  "
            f"{'✓ converged' if seg.converged else '⚠ partial'}  |  "
            f"{seg.solve_ms:.0f} ms"
        )
        seg_text.get_bbox_patch().set_edgecolor(seg.color)

        # ── GRF ──────────────────────────────────────────────────────────────
        for foot in foot_order:
            fz = forces[foot]
            n = min(frame + 1, len(fz))
            grf_lines[foot].set_data(time_arr[:n], fz[:n, 2])
        grf_vline.set_xdata([t_now, t_now])

        # ── Torques ──────────────────────────────────────────────────────────
        fu = min(frame, len(us) - 1)
        for j, line in enumerate(torq_lines):
            line.set_data(time_arr[:fu + 1], us[:fu + 1, j])
        torq_vline.set_xdata([t_now, t_now])

        # ── Cost cursor ──────────────────────────────────────────────────────
        cost_vline.set_xdata([t_now, t_now])

        # ── Title time ───────────────────────────────────────────────────────
        time_text.set_text(
            f"t = {t_now:.2f} s  |  frame {frame}/{T}  |  "
            f"waypoint {si+1} / {len(segments)}  target: "
            f"({tx:.2f}, {ty:.2f}) m"
        )

        return (past_line, active_bezier, ctrl_line, robot_dot,
                body_rect, target_ring, target_dot, grf_vline, torq_vline,
                cost_vline, seg_text, time_text,
                *grf_lines.values(), *torq_lines)

    interval = max(16, int(1000 / fps))
    anim_obj = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=interval, blit=False,  # blit=False for body_rect Polygon
    )

    return fig, anim_obj


# ═════════════════════════════════════════════════════════════════════════════
# Print summary
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(segments: List[Segment], waypoints: List[Waypoint]):
    print("\n" + "═" * 55)
    print("Navigation Summary")
    print("═" * 55)
    total_dist = sum(
        np.linalg.norm(s.target_pos[:2] - s.start_pos[:2]) for s in segments
    )
    converged_n = sum(s.converged for s in segments)
    print(f"  Segments:       {len(segments)}")
    print(f"  Total distance: {total_dist:.2f} m")
    print(f"  Converged:      {converged_n}/{len(segments)}")
    print(f"  Mean cost:      {np.mean([s.cost for s in segments]):.0f}")
    print(f"  Mean solve:     {np.mean([s.solve_ms for s in segments]):.0f} ms")
    print("─" * 55)
    for i, seg in enumerate(segments):
        d = np.linalg.norm(seg.target_pos[:2] - seg.start_pos[:2])
        status = "✓" if seg.converged else "⚠"
        print(f"  [{i+1}] {status} d={d:.2f}m  cost={seg.cost:.0f}"
              f"  {seg.solve_ms:.0f}ms")
    print("═" * 55)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Random waypoint Bézier MPC navigation demo (no RL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python random_waypoint_demo.py                      # 6 waypoints, show window
    python random_waypoint_demo.py --n-waypoints 10    # more waypoints
    python random_waypoint_demo.py --seed 42            # reproducible
    python random_waypoint_demo.py --gait walk         # walk gait
    python random_waypoint_demo.py --save-gif demo.gif  # save as GIF
    python random_waypoint_demo.py --save-mp4 demo.mp4  # save as MP4
    python random_waypoint_demo.py --fps 6             # slower animation
    python random_waypoint_demo.py --max-turn 30       # gentle turns only
        """,
    )
    parser.add_argument("--n-waypoints", type=int, default=6,
                        help="Number of random waypoints (default: 6)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--gait", default="trot",
                        choices=["trot", "walk", "pace"],
                        help="Gait type (default: trot)")
    parser.add_argument("--dist-min", type=float, default=0.6,
                        help="Min waypoint distance (default: 0.6 m)")
    parser.add_argument("--dist-max", type=float, default=1.3,
                        help="Max waypoint distance (default: 1.3 m)")
    parser.add_argument("--max-turn", type=float, default=55.0,
                        help="Max heading change per waypoint (deg, default: 55)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="OCP horizon per segment (s, default: 3.0)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Animation playback FPS (default: 20)")
    parser.add_argument("--save-gif", metavar="FILE",
                        help="Save animation as GIF")
    parser.add_argument("--save-mp4", metavar="FILE",
                        help="Save animation as MP4")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip displaying window (useful with --save-*)")

    args = parser.parse_args()

    if not args.no_show and not args.save_gif and not args.save_mp4:
        matplotlib.use("TkAgg")

    print("=" * 55)
    print("Random Waypoint Bézier MPC Demo  |  B1 Quadruped")
    print("=" * 55)
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    print(f"  Waypoints: {args.n_waypoints}  |  Gait: {args.gait}")
    print(f"  Distance: [{args.dist_min}, {args.dist_max}] m")
    print(f"  Max heading change: ±{args.max_turn}°")

    # Load robot
    print("\nLoading B1 robot model…")
    b1, rmodel, rdata, q0, v0, x0, com0, foot_ids = load_b1()
    print(f"  nq={rmodel.nq}  nv={rmodel.nv}  com0={com0}")

    # Generate waypoints
    print("\nGenerating random waypoints…")
    gen = RandomWaypointGenerator(
        dist_min=args.dist_min,
        dist_max=args.dist_max,
        max_turn_deg=args.max_turn,
        com_height=com0[2],
        seed=args.seed,
    )

    waypoints = []
    cur_pos = com0.copy()
    cur_heading = 0.0
    for i in range(args.n_waypoints):
        wp = gen.next(cur_pos, cur_heading)
        waypoints.append(wp)
        print(f"  W{i+1}: ({wp.pos[0]:+.2f}, {wp.pos[1]:+.2f}, {wp.pos[2]:.2f})")
        cur_pos = wp.pos
        cur_heading = wp.heading

    # Pre-compute all OCP segments
    segments = compute_all_segments(
        rmodel, x0, com0, foot_ids,
        waypoints,
        gait_type=args.gait,
        dt=0.02,
        duration=args.duration,
    )

    print_summary(segments, waypoints)

    # Build animation
    print("\nBuilding animation…")
    fig, anim_obj = build_animation(
        segments, waypoints, com0, dt=0.02,
        fps=args.fps, gait_type=args.gait,
    )

    if args.save_gif:
        print(f"Saving GIF → {args.save_gif}  (may take 1–3 min)…")
        writer = animation.PillowWriter(fps=args.fps)
        anim_obj.save(args.save_gif, writer=writer, dpi=110,
                      progress_callback=lambda i, n: print(f"  frame {i}/{n}", end="\r"))
        print(f"\n  Saved → {args.save_gif}")

    if args.save_mp4:
        print(f"Saving MP4 → {args.save_mp4}…")
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2500,
                                         metadata={"title": "Random Waypoint MPC Demo"})
        anim_obj.save(args.save_mp4, writer=writer, dpi=140,
                      progress_callback=lambda i, n: print(f"  frame {i}/{n}", end="\r"))
        print(f"\n  Saved → {args.save_mp4}")

    if not args.no_show:
        plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
