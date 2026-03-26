#!/usr/bin/env python3
"""
Offline benchmark comparison tool — Baseline MPC vs MPC+RL.

Records trajectories from both modes and produces a side-by-side comparison
plot (XY path, CoM height, distance to target, cumulative reward).

This script does NOT require Gazebo or ROS2 to be running. It replays
the data logged from two separate Gazebo runs (one baseline, one RL).

Data is collected by running the controller node with:
    --ros-args -p log_data:=true -p log_path:=/tmp/go2_baseline.npz

Then calling this script:
    python benchmark_compare.py \
        --baseline /tmp/go2_baseline.npz \
        --rl       /tmp/go2_rl.npz \
        --out      results/comparison.png

Alternatively, this script provides a Python API for recording data
from within the controller loop (see DataLogger class).
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data logger (to be used inside go2_mpc_node.py control loop)
# ---------------------------------------------------------------------------

@dataclass
class DataLogger:
    """Lightweight recorder for one Gazebo episode.

    Usage inside control loop:
        logger = DataLogger(mode="baseline")
        logger.record(pos, height, dist, torques, mpc_cost, converged)
        logger.save("/tmp/go2_baseline.npz")
    """
    mode:   str
    dt:     float = 0.02  # s per step

    positions:     list = field(default_factory=list)  # (3,) each
    heights:       list = field(default_factory=list)  # float
    distances:     list = field(default_factory=list)  # float
    torques_rms:   list = field(default_factory=list)  # float
    mpc_costs:     list = field(default_factory=list)  # float
    convergences:  list = field(default_factory=list)  # bool
    guard_fires:   list = field(default_factory=list)  # bool

    def record(
        self,
        pos:         np.ndarray,
        dist_target: float,
        torques:     Optional[np.ndarray] = None,
        mpc_cost:    float = 0.0,
        converged:   bool = True,
        guard_fired: bool = False,
    ):
        self.positions.append(pos.copy())
        self.heights.append(float(pos[2]))
        self.distances.append(float(dist_target))
        self.torques_rms.append(
            float(np.sqrt(np.mean(torques ** 2))) if torques is not None else 0.0
        )
        self.mpc_costs.append(float(mpc_cost))
        self.convergences.append(bool(converged))
        self.guard_fires.append(bool(guard_fired))

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mode        = self.mode,
            dt          = self.dt,
            positions   = np.array(self.positions),
            heights     = np.array(self.heights),
            distances   = np.array(self.distances),
            torques_rms = np.array(self.torques_rms),
            mpc_costs   = np.array(self.mpc_costs),
            convergences= np.array(self.convergences),
            guard_fires = np.array(self.guard_fires),
        )
        print(f"[DataLogger] Saved {len(self.positions)} steps to {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"baseline": "#E57373", "rl": "#42A5F5"}
LABELS = {"baseline": "Pure MPC (Baseline)", "rl": "MPC + RL (Ours)"}


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def plot_comparison(
    baseline_path: str,
    rl_path: str,
    out_path: str,
    target_pos: Optional[np.ndarray] = None,
):
    """Generate 2×3 comparison figure and save to out_path."""
    datasets = {}
    for key, path in [("baseline", baseline_path), ("rl", rl_path)]:
        if path and os.path.exists(path):
            datasets[key] = load_npz(path)
        else:
            print(f"[benchmark_compare] WARNING: {path} not found, skipping {key}")

    if not datasets:
        print("[benchmark_compare] No data to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Go2 Locomotion: Baseline MPC vs MPC + RL",
        fontsize=15, fontweight="bold",
    )

    for key, data in datasets.items():
        color = COLORS[key]
        label = LABELS[key]
        dt    = float(data["dt"])
        T     = len(data["positions"])
        time  = np.arange(T) * dt
        pos   = data["positions"]

        # ---- [0,0] XY trajectory ----
        ax = axes[0, 0]
        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.8, label=label)
        ax.plot(pos[0, 0], pos[0, 1], "o", color=color, markersize=8)
        ax.plot(pos[-1, 0], pos[-1, 1], "s", color=color, markersize=8)

    if target_pos is not None:
        axes[0, 0].plot(
            target_pos[0], target_pos[1],
            "r*", markersize=14, label="Target", zorder=5,
        )
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")
    axes[0, 0].set_title("XY Trajectory (Bird's Eye View)")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    for key, data in datasets.items():
        color = COLORS[key]
        label = LABELS[key]
        dt    = float(data["dt"])
        T     = len(data["distances"])
        time  = np.arange(T) * dt

        # ---- [0,1] Distance to target ----
        axes[0, 1].plot(time, data["distances"], color=color, linewidth=1.5, label=label)

        # ---- [0,2] CoM height ----
        axes[0, 2].plot(time, data["heights"], color=color, linewidth=1.5, label=label)

        # ---- [1,0] Torque RMS ----
        axes[1, 0].plot(time, data["torques_rms"], color=color, linewidth=1.2,
                        alpha=0.8, label=label)

        # ---- [1,1] MPC cost ----
        cost = np.clip(data["mpc_costs"], 0, 1e5)
        axes[1, 1].semilogy(time, cost + 1.0, color=color, linewidth=1.2,
                             alpha=0.8, label=label)

        # ---- [1,2] MPC convergence rate ----
        window = max(1, T // 50)
        conv_rate = np.convolve(
            data["convergences"].astype(float),
            np.ones(window) / window,
            mode="same",
        )
        axes[1, 2].plot(time, conv_rate * 100, color=color, linewidth=1.5, label=label)

    # Decorations
    axes[0, 1].set(xlabel="Time (s)", ylabel="Distance (m)",
                   title="Distance to Target")
    axes[0, 1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Success (0.5m)")
    axes[0, 1].legend(fontsize=9)

    axes[0, 2].set(xlabel="Time (s)", ylabel="Height (m)",
                   title="CoM Height")
    axes[0, 2].axhline(0.35, color="green", linestyle="--", alpha=0.5, label="Nominal (0.35m)")
    axes[0, 2].axhline(0.22, color="red",   linestyle="--", alpha=0.5, label="Min (0.22m)")
    axes[0, 2].legend(fontsize=9)

    axes[1, 0].set(xlabel="Time (s)", ylabel="Torque RMS (N·m)",
                   title="Joint Torque RMS")
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].set(xlabel="Time (s)", ylabel="MPC Cost (log scale)",
                   title="MPC Objective Cost")
    axes[1, 1].legend(fontsize=9)

    axes[1, 2].set(xlabel="Time (s)", ylabel="Convergence Rate (%)",
                   title="MPC Convergence Rate (rolling)")
    axes[1, 2].set_ylim(0, 105)
    axes[1, 2].legend(fontsize=9)

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark_compare] Saved comparison plot: {out_path}")

    # ---- Summary statistics ----
    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Baseline':>12} {'MPC+RL':>12}")
    print("=" * 60)
    metrics = {
        "baseline": datasets.get("baseline", {}),
        "rl":       datasets.get("rl", {}),
    }
    rows = [
        ("Min dist to target (m)",   "distances",   np.min),
        ("Mean CoM height (m)",       "heights",     np.mean),
        ("Std CoM height (m)",        "heights",     np.std),
        ("Mean torque RMS (N·m)",     "torques_rms", np.mean),
        ("MPC convergence rate (%)",  "convergences",lambda x: np.mean(x) * 100),
        ("Guard fire rate (%)",       "guard_fires", lambda x: np.mean(x) * 100),
    ]
    for name, key, fn in rows:
        vals = []
        for mode in ["baseline", "rl"]:
            d = metrics[mode]
            vals.append(f"{fn(d[key]):.3f}" if key in d else "  N/A")
        print(f"{name:<30} {vals[0]:>12} {vals[1]:>12}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline MPC vs MPC+RL from logged .npz files"
    )
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline .npz log")
    parser.add_argument("--rl",       type=str, required=True,
                        help="Path to RL+MPC .npz log")
    parser.add_argument("--out",      type=str, default="results/comparison.png",
                        help="Output plot path (default: results/comparison.png)")
    parser.add_argument("--target_x", type=float, default=2.0)
    parser.add_argument("--target_y", type=float, default=0.0)
    args = parser.parse_args()

    target = np.array([args.target_x, args.target_y, 0.0])
    plot_comparison(args.baseline, args.rl, args.out, target_pos=target)
