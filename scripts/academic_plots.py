#!/usr/bin/env python3
"""Academic evaluation plots for RL_Bezier_MPC.

Generates publication-quality figures from NPZ data collected by play_quadruped_mpc.py
with --eval_mode. No IsaacLab dependency — pure numpy + matplotlib.

Usage:
    # Normal eval figures (no push)
    python scripts/academic_plots.py --data eval_data/episode_1_eval.npz --out eval_figs/

    # Push-recovery figures (requires NPZ with push_force > 0)
    python scripts/academic_plots.py --data eval_data_push/episode_1_eval.npz --out eval_figs/ --push

Outputs (all PNG, 300 DPI):
    1. stability_roll_pitch_yaw.png    — body orientation angles over time
    2. velocity_tracking.png           — desired vs. actual forward/lateral velocity
    3. bezier_params_evolution.png     — RL actor Bezier control point outputs
    4. joint_torques.png               — 12-motor torque profiles
    5. phase_portrait_FL_calf.png      — FL calf limit-cycle phase portrait
    6. disturbance_rejection.png       — push recovery: lateral vel + action response
    7. obs_action_correlation.png      — pitch error vs. forward stride correlation
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────
# Colour/style constants
# ─────────────────────────────────────────────
LEG_COLORS = {
    "FL": "#1f77b4",   # blue
    "FR": "#ff7f0e",   # orange
    "RL": "#2ca02c",   # green
    "RR": "#d62728",   # red
}
# Isaac Lab joint order: 0-3 hips, 4-7 thighs, 8-11 calves
# Format: (leg_name, joint_type)
JOINT_LABELS = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]
# Map each Isaac joint index → leg name
JOINT_LEG = [lbl.split("_")[0] for lbl in JOINT_LABELS]
JOINT_TYPE_STYLE = {"hip": "solid", "thigh": "dashed", "calf": "dotted"}

TORQUE_LIMIT = 23.5  # N·m hardware limit for Unitree Go2


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _quat_to_euler(quats):
    """Convert (T, 4) quaternions (w, x, y, z) to (T, 3) roll/pitch/yaw in radians."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    # Roll
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    # Pitch
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.stack([roll, pitch, yaw], axis=1)


def _time_axis(data):
    T = len(data["positions"])
    dt = float(data.get("dt", 0.02))
    return np.arange(T) * dt


def _rms(arr):
    return float(np.sqrt(np.mean(arr ** 2)))


def _save(fig, path, dpi=300):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Figure 1 — Stability: Roll / Pitch / Yaw
# ─────────────────────────────────────────────

def fig1_stability(data, out_dir):
    """Body orientation angles over time with ±0.05 rad stability bands."""
    t = _time_axis(data)
    euler = _quat_to_euler(data["orientations"])  # (T, 3)
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Body Orientation Stability", fontsize=13, fontweight="bold")

    specs = [
        (roll,  "Roll",  "#e74c3c"),
        (pitch, "Pitch", "#2ecc71"),
        (yaw,   "Yaw",   "#3498db"),
    ]
    for ax, (angle, label, color) in zip(axes, specs):
        ax.plot(t, angle, color=color, linewidth=1.0, label=f"{label}  (RMS={_rms(angle):.4f} rad)")
        if label in ("Roll", "Pitch"):
            ax.axhspan(-0.05, 0.05, alpha=0.12, color="green", label="±0.05 rad band")
            ax.axhline(0.05, color="green", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.axhline(-0.05, color="green", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_ylabel(f"{label} (rad)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.4, 0.4)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "stability_roll_pitch_yaw.png"))


# ─────────────────────────────────────────────
# Figure 2 — Velocity Tracking
# ─────────────────────────────────────────────

def fig2_velocity_tracking(data, out_dir):
    """Desired vs. actual forward and lateral body-frame velocities."""
    t = _time_axis(data)
    lin_vel_b = data["lin_vel_b"]  # (T, 3)  [vx, vy, vz] in body frame

    # Infer desired velocity direction from start → target
    start_pos = data["positions"][0, :2]
    target_pos = data["target_pos"][:2]
    direction = target_pos - start_pos
    dist = np.linalg.norm(direction)
    desired_speed = 0.5  # m/s nominal
    if dist > 0.01:
        desired_vx = desired_speed * direction[0] / dist
    else:
        desired_vx = desired_speed

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Velocity Tracking", fontsize=13, fontweight="bold")

    # Forward velocity
    ax = axes[0]
    ax.plot(t, lin_vel_b[:, 0], color="#2980b9", linewidth=1.2, label="Actual $v_x$")
    ax.axhline(desired_vx, color="#e74c3c", linewidth=1.2, linestyle="--",
               label=f"Desired $v_x$ = {desired_vx:.2f} m/s")
    ax.set_ylabel("$v_x$ (m/s)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Lateral velocity
    ax = axes[1]
    ax.plot(t, lin_vel_b[:, 1], color="#27ae60", linewidth=1.2, label="Actual $v_y$ (lateral)")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5,
               label="Desired $v_y$ = 0.0 m/s")
    ax.set_ylabel("$v_y$ (m/s)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "velocity_tracking.png"))


# ─────────────────────────────────────────────
# Figure 3 — Bezier Parameter Evolution
# ─────────────────────────────────────────────

def fig3_bezier_params(data, out_dir):
    """RL actor output: Bezier control point X-components over time.

    Actions (12D when fix_gait_params=True):
        [0:3]  P0 offset (≈ always 0, start of curve)
        [3:6]  P1 offset (first intermediate)
        [6:9]  P2 offset (second intermediate)
        [9:12] P3 offset (end point / stride target)
    We plot P1_x, P2_x, P3_x as they encode stride shaping.
    """
    t = _time_axis(data)
    actions = data["actions"]  # (T, action_dim)

    if actions.shape[1] < 12:
        print("  [fig3] actions array has fewer than 12 columns, skipping Bezier plot.")
        return

    p1x = actions[:, 3]
    p2x = actions[:, 6]
    p3x = actions[:, 9]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Bezier Control Point Evolution (X-axis / Forward Stride)",
                 fontsize=13, fontweight="bold")

    pairs = [
        (p1x, "P1_x  (early pull)", "#8e44ad"),
        (p2x, "P2_x  (mid shaping)", "#2980b9"),
        (p3x, "P3_x  (stride end)",  "#e67e22"),
    ]
    for ax, (vals, label, color) in zip(axes, pairs):
        ax.plot(t, vals, color=color, linewidth=1.0, label=label)
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_ylabel("Offset (m)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "bezier_params_evolution.png"))


# ─────────────────────────────────────────────
# Figure 4 — Joint Torques
# ─────────────────────────────────────────────

def fig4_joint_torques(data, out_dir):
    """12-motor torque profiles with hardware limit annotation."""
    t = _time_axis(data)
    torques = data["torques"]  # (T, 12)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Joint Torque Profiles  (dashed = ±23.5 N·m hardware limit)",
                 fontsize=13, fontweight="bold")

    joint_groups = [
        ("Hip joints",   list(range(0, 4))),
        ("Thigh joints", list(range(4, 8))),
        ("Calf joints",  list(range(8, 12))),
    ]
    leg_order = ["FL", "FR", "RL", "RR"]

    for ax, (group_name, indices) in zip(axes, joint_groups):
        for i, joint_idx in enumerate(indices):
            leg = leg_order[i]
            ax.plot(t, torques[:, joint_idx],
                    color=LEG_COLORS[leg], linewidth=0.9,
                    label=JOINT_LABELS[joint_idx], alpha=0.9)

        ax.axhline(TORQUE_LIMIT, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.axhline(-TORQUE_LIMIT, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_ylabel("τ (N·m)", fontsize=9)
        ax.set_title(group_name, fontsize=9, pad=2)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.2)

    # Leg colour legend
    legend_patches = [Patch(color=LEG_COLORS[leg], label=leg) for leg in leg_order]
    axes[0].legend(handles=legend_patches + axes[0].lines[:4],
                   labels=[l.get_label() for l in legend_patches + axes[0].lines[:4]],
                   fontsize=7, ncol=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "joint_torques.png"))


# ─────────────────────────────────────────────
# Figure 5 — Phase Portrait (FL calf)
# ─────────────────────────────────────────────

def fig5_phase_portrait(data, out_dir):
    """FL calf phase portrait: joint angle vs. joint velocity, colored by time."""
    # Isaac Lab order: index 8 = FL_calf
    FL_CALF_IDX = 8

    joint_pos = data["joint_pos"]   # (T, 12)
    joint_vel = data["joint_vel"]   # (T, 12)
    T = len(joint_pos)

    q   = joint_pos[:, FL_CALF_IDX]
    dq  = joint_vel[:, FL_CALF_IDX]
    tcolor = np.linspace(0, 1, T)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(q, dq, c=tcolor, cmap="viridis", s=3, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Normalised time", fontsize=9)
    # Draw arrow at start and end for direction
    if T > 10:
        ax.annotate("", xy=(q[5], dq[5]), xytext=(q[0], dq[0]),
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
        ax.annotate("", xy=(q[-1], dq[-1]), xytext=(q[-6], dq[-6]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_xlabel("FL Calf angle $q$ (rad)", fontsize=10)
    ax.set_ylabel("FL Calf velocity $\\dot{q}$ (rad/s)", fontsize=10)
    ax.set_title("Phase Portrait — FL Calf Joint\n"
                 "(closed loop → periodic limit cycle)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    legend_elems = [
        Line2D([0], [0], color="green", lw=1.5, label="Trajectory start"),
        Line2D([0], [0], color="red",   lw=1.5, label="Trajectory end"),
    ]
    ax.legend(handles=legend_elems, fontsize=8)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "phase_portrait_FL_calf.png"))


# ─────────────────────────────────────────────
# Figure 6 — Disturbance Rejection (Push Recovery)
# ─────────────────────────────────────────────

def fig6_disturbance_rejection(data, out_dir):
    """Push recovery: lateral velocity spike + RL actor response (P3_y).

    Requires NPZ collected with --push_force > 0.
    """
    push_force = float(data.get("push_force", 0.0))
    push_time  = float(data.get("push_time",  5.0))
    push_dur   = float(data.get("push_duration", 0.2))

    if push_force <= 0.0:
        print("  [fig6] push_force == 0, disturbance rejection plot skipped.")
        print("         Re-run play_quadruped_mpc.py with --eval_mode --push_force 60")
        return

    t = _time_axis(data)
    lin_vel_b = data["lin_vel_b"]    # (T, 3)
    actions   = data["actions"]      # (T, action_dim)
    vy = lin_vel_b[:, 1]
    # P3_y: lateral endpoint of Bezier stride = actions[:,10]
    p3y = actions[:, 10] if actions.shape[1] > 10 else np.zeros_like(vy)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Push Recovery  ({push_force:.0f} N lateral at t={push_time:.1f}s, "
                 f"{push_dur:.2f}s)",
                 fontsize=13, fontweight="bold")

    # Top: lateral velocity
    ax = axes[0]
    ax.plot(t, vy, color="#2980b9", linewidth=1.2, label="Lateral velocity $v_y$")
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvspan(push_time, push_time + push_dur, alpha=0.15, color="red", label="Push window")
    ax.axvline(push_time, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_ylabel("$v_y$ (m/s)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Bottom: RL actor's lateral stride action
    ax = axes[1]
    ax.plot(t, p3y, color="#e67e22", linewidth=1.2, label="$P_{3,y}$ (lateral stride endpoint)")
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvspan(push_time, push_time + push_dur, alpha=0.15, color="red")
    ax.axvline(push_time, color="red", linewidth=1.2, linestyle="--", alpha=0.7,
               label="Push onset")
    ax.set_ylabel("$P_{3,y}$ offset (m)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "disturbance_rejection.png"))


# ─────────────────────────────────────────────
# Figure 7 — Observation–Action Correlation
# ─────────────────────────────────────────────

def fig7_obs_action_correlation(data, out_dir):
    """Pitch error vs. forward stride parameter (P3_x) on dual Y-axis.

    Demonstrates policy interpretability: when body pitches forward,
    the actor extends front stride (P3_x increases) to catch the fall.
    """
    t = _time_axis(data)
    euler = _quat_to_euler(data["orientations"])  # (T, 3)
    pitch = euler[:, 1]
    actions = data["actions"]  # (T, action_dim)
    p3x = actions[:, 9] if actions.shape[1] > 9 else np.zeros_like(pitch)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(t, pitch, color="#e74c3c", linewidth=1.1, label="Pitch angle (rad)")
    ax1.axhline(0.0, color="#e74c3c", linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.set_ylabel("Pitch (rad)", color="#e74c3c", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#e74c3c")

    l2, = ax2.plot(t, p3x, color="#2980b9", linewidth=1.1,
                   label="$P_{3,x}$ — forward stride endpoint (m)")
    ax2.set_ylabel("$P_{3,x}$ (m)", color="#2980b9", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#2980b9")

    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_title("Observation–Action Correlation\n"
                  "Pitch error ↔ Forward Bezier stride (policy interpretability)",
                  fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.2)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=8, loc="upper left")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "obs_action_correlation.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Academic evaluation plots for RL_Bezier_MPC")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to *_eval.npz file from play_quadruped_mpc.py --eval_mode")
    parser.add_argument("--out", type=str, default="eval_figs",
                        help="Output directory for PNG figures (default: eval_figs/)")
    parser.add_argument("--push", action="store_true",
                        help="Also generate disturbance rejection figure (requires push NPZ)")
    parser.add_argument("--figs", type=str, default="all",
                        help="Comma-separated list of figures to generate: "
                             "1=stability, 2=velocity, 3=bezier, 4=torques, "
                             "5=phase, 6=push, 7=correlation  (default: all)")
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)

    raw = np.load(str(data_path), allow_pickle=True)
    data = dict(raw)
    print(f"Loaded: {data_path}")
    print(f"  Channels: {list(data.keys())}")
    T = len(data["positions"])
    dt = float(data.get("dt", 0.02))
    print(f"  Steps: {T}  ({T * dt:.1f}s at {1/dt:.0f} Hz)")

    # Check required channels
    required = ["positions", "orientations", "rewards"]
    eval_required = ["joint_pos", "joint_vel", "torques", "actions", "lin_vel_b"]
    missing = [k for k in eval_required if k not in data]
    if missing:
        print(f"  WARNING: eval channels missing: {missing}")
        print("  Run play_quadruped_mpc.py with --eval_mode to capture full data.")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Decide which figures to run
    if args.figs == "all":
        fig_ids = {1, 2, 3, 4, 5, 6, 7}
    else:
        fig_ids = {int(x.strip()) for x in args.figs.split(",")}

    if args.push:
        fig_ids.add(6)

    print(f"\nGenerating figures: {sorted(fig_ids)}  →  {out_dir}/")

    if 1 in fig_ids:
        print("\n[1/7] Stability (roll/pitch/yaw)...")
        fig1_stability(data, out_dir)

    if 2 in fig_ids and "lin_vel_b" in data:
        print("[2/7] Velocity tracking...")
        fig2_velocity_tracking(data, out_dir)

    if 3 in fig_ids and "actions" in data:
        print("[3/7] Bezier parameter evolution...")
        fig3_bezier_params(data, out_dir)

    if 4 in fig_ids and "torques" in data:
        print("[4/7] Joint torques...")
        fig4_joint_torques(data, out_dir)

    if 5 in fig_ids and "joint_pos" in data and "joint_vel" in data:
        print("[5/7] Phase portrait (FL calf)...")
        fig5_phase_portrait(data, out_dir)

    if 6 in fig_ids:
        print("[6/7] Disturbance rejection...")
        fig6_disturbance_rejection(data, out_dir)

    if 7 in fig_ids and "actions" in data:
        print("[7/7] Observation-action correlation...")
        fig7_obs_action_correlation(data, out_dir)

    print(f"\nDone. Figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
