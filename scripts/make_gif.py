#!/usr/bin/env python3
"""Offline GIF renderer for Quadruped MPC play episodes.

Reads *_data.npz files saved by play_quadruped_mpc.py and renders
animated GIFs WITHOUT importing Isaac Sim / Omniverse (which causes
matplotlib savefig to hang inside the Isaac Sim process).

Usage:
    # Render all episodes in a directory
    python scripts/make_gif.py --data_dir plots/gif_test

    # Render a single file
    python scripts/make_gif.py --data_dir plots/gif_test --episode 1

    # Custom fps / output dir
    python scripts/make_gif.py --data_dir plots/gif_test --fps 15 --out_dir plots/gifs
"""

import argparse
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def render_frames_batch(positions, orientations, rewards, target_pos,
                        max_steps, guard_count, fps):
    """Render all GIF frames for one episode.

    Args:
        positions    : (T, 3) CoM positions.
        orientations : (T, 4) quaternions (w, x, y, z).
        rewards      : (T,)   per-step rewards.
        target_pos   : (3,)   target position.
        max_steps    : int    episode max steps.
        guard_count  : int    total MPC Guard failures.
        fps          : int    GIF fps (used for title only).

    Returns:
        list of PIL Image objects.
    """
    T_full = len(positions)
    capture_every = max(1, int(50 / fps))
    capture_indices = list(range(0, T_full, capture_every))
    n_frames = len(capture_indices)
    print(f"  Rendering {n_frames} frames (every {capture_every} steps) ...",
          flush=True)

    # Pre-compute roll / pitch for all steps
    rolls, pitches = [], []
    for q in orientations:
        w, x, y, z = q
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        rolls.append(np.degrees(np.arctan2(sinr, cosr)))
        sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitches.append(np.degrees(np.arcsin(sinp)))
    rolls = np.array(rolls)
    pitches = np.array(pitches)
    t_full = np.arange(T_full) * 0.02

    # Fixed XY axis limits (stable across all frames)
    ax_x, ax_y = positions[:, 0], positions[:, 1]
    cx = (ax_x.max() + ax_x.min()) / 2
    cy = (ax_y.max() + ax_y.min()) / 2
    pad = 0.6
    x_half = max((ax_x.max() - ax_x.min()) / 2, 0.5) + pad
    y_half = max((ax_y.max() - ax_y.min()) / 2, 0.5) + pad

    # Create figure ONCE and reuse
    fig = plt.figure(figsize=(8, 4), dpi=80)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.48)
    ax_xy = fig.add_subplot(gs[:, :2])
    ax_h  = fig.add_subplot(gs[0, 2])
    ax_rp = fig.add_subplot(gs[1, 2])

    # Static one-time elements
    ax_xy.plot(target_pos[0], target_pos[1], "r*", markersize=12,
               label="target", zorder=5)
    ax_xy.plot(positions[0, 0], positions[0, 1], "go", markersize=8,
               label="start", zorder=5)
    ax_xy.set_xlim(cx - x_half, cx + x_half)
    ax_xy.set_ylim(cy - y_half, cy + y_half)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("X (m)", fontsize=8)
    ax_xy.set_ylabel("Y (m)", fontsize=8)
    ax_xy.grid(True, alpha=0.25)

    ax_h.set_ylim(0.0, 0.45)
    ax_h.set_xlabel("t (s)", fontsize=7)
    ax_h.set_ylabel("Z (m)", fontsize=7)
    ax_h.set_title("Height", fontsize=8)
    ax_h.tick_params(labelsize=6)
    ax_h.grid(True, alpha=0.25)

    ax_rp.set_ylim(-50, 50)
    ax_rp.set_xlabel("t (s)", fontsize=7)
    ax_rp.set_ylabel("deg", fontsize=7)
    ax_rp.set_title("Pitch / Roll", fontsize=8)
    ax_rp.tick_params(labelsize=6)
    ax_rp.grid(True, alpha=0.25)

    pil_frames = []
    for fi, idx in enumerate(capture_indices):
        if fi % 20 == 0:
            print(f"    {fi + 1}/{n_frames}", flush=True)

        T = idx + 1
        pos = positions[:T]
        trail = pos[-min(T, 80):]

        # Clear dynamic lines only (keep static markers at index 0,1)
        ax_xy.lines[2:] = []
        ax_h.lines.clear()
        ax_rp.lines.clear()

        # XY trail + robot
        ax_xy.plot(trail[:, 0], trail[:, 1], "b-", linewidth=1.2, alpha=0.6)
        ax_xy.plot(pos[-1, 0], pos[-1, 1], "bs", markersize=9, zorder=6)
        ax_xy.set_title(
            f"t={T * 0.02:.1f}s  step={T}/{max_steps}  "
            f"guard={guard_count}  Σr={np.sum(rewards[:T]):.0f}",
            fontsize=8,
        )

        # Height
        ax_h.plot(t_full[:T], pos[:, 2], "b-", linewidth=1)
        ax_h.axhline(0.28, color="g", linestyle="--", linewidth=0.8)
        ax_h.axhline(0.12, color="r", linestyle="--", linewidth=0.8)

        # Pitch / Roll
        ax_rp.plot(t_full[:T], pitches[:T], "g-", linewidth=1)
        ax_rp.plot(t_full[:T], rolls[:T],   "r-", linewidth=1)
        ax_rp.axhline(0, color="k", linewidth=0.5)
        if fi == 0:
            ax_rp.plot([], [], "g-", label="pitch")
            ax_rp.plot([], [], "r-", label="roll")
            ax_rp.legend(fontsize=6, loc="upper right")

        # Save frame
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()

    plt.close(fig)
    return pil_frames


def save_gif(pil_frames, out_path, fps):
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  GIF saved → {out_path}  ({len(pil_frames)} frames, {size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Offline GIF renderer")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing *_data.npz files")
    parser.add_argument("--episode", type=int, default=None,
                        help="Render only this episode number (default: all)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override GIF fps (default: use value stored in npz)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for GIFs (default: same as data_dir)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find .npz files
    if args.episode is not None:
        npz_files = [data_dir / f"episode_{args.episode}_data.npz"]
    else:
        npz_files = sorted(data_dir.glob("*_data.npz"))

    if not npz_files:
        print(f"No *_data.npz files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(npz_files)} episode(s) to render.")

    for npz_path in npz_files:
        print(f"\nLoading {npz_path.name} ...")
        data = np.load(npz_path, allow_pickle=True)

        positions    = data["positions"]
        orientations = data["orientations"]
        rewards      = data["rewards"]
        target_pos   = data["target_pos"]
        max_steps    = int(data["max_steps"])
        guard_count  = int(data["guard_count"])
        fps          = args.fps if args.fps else int(data["gif_fps"])

        ep_name = npz_path.stem.replace("_data", "")
        gif_path = out_dir / f"{ep_name}.gif"

        frames = render_frames_batch(
            positions, orientations, rewards, target_pos,
            max_steps, guard_count, fps,
        )
        save_gif(frames, gif_path, fps)

    print("\nDone!")


if __name__ == "__main__":
    main()
