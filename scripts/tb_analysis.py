#!/usr/bin/env python3
"""TensorBoard log analysis for RL_Bezier_MPC training.

Reads TensorBoard event files and generates academic training-dynamics plots.
No simulation dependency — only requires tensorboard + matplotlib.

Usage:
    # Auto-detect latest run
    python scripts/tb_analysis.py --logdir logs/quadruped_mpc/ --out eval_figs/

    # Specific run directory
    python scripts/tb_analysis.py \
        --logdir logs/quadruped_mpc/2026-02-28_05-10-43 --out eval_figs/

    # List available scalar tags (no plots)
    python scripts/tb_analysis.py --logdir logs/quadruped_mpc/ --list_tags

Outputs:
    reward_components.png   — reward sub-components over training iterations
    training_dynamics.png   — combined overview: reward, episode length, policy loss
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# TensorBoard event loading
# ─────────────────────────────────────────────

def load_tb_scalars(logdir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all scalar time-series from TensorBoard event files.

    Returns:
        dict mapping tag → (steps_array, values_array)
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed. Run: pip install tensorboard")
        sys.exit(1)

    logdir = Path(logdir)
    # If given a parent dir, find the most-recently-modified run sub-directory
    event_files = list(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"ERROR: No TensorBoard event files found in {logdir}")
        sys.exit(1)

    # Use the most recent event file
    event_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    event_dir = str(event_files[0].parent)
    print(f"Loading TensorBoard events from: {event_dir}")

    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    ea.Reload()

    scalars = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events])
        values = np.array([e.value for e in events])
        scalars[tag] = (steps, values)

    return scalars


def _smooth(values, window=10):
    """Running-mean smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def _save(fig, path, dpi=300):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Tag classification helpers
# ─────────────────────────────────────────────

# Known reward-related substrings (RSL-RL convention)
REWARD_KEYWORDS = [
    "reward", "Reward",
    "mean_reward", "Episode/mean_reward",
    "train/mean_reward",
]

# Known training metric substrings
TRAINING_KEYWORDS = [
    "mean_episode_length", "episode_length",
    "loss", "Loss",
    "value_function", "clip_fraction",
    "kl", "entropy",
    "learning_rate",
]


def _find_tags(scalars: dict, keywords: list[str]) -> list[str]:
    """Return tags that contain any of the given keywords (case-sensitive substring)."""
    matched = []
    for tag in scalars:
        if any(kw in tag for kw in keywords):
            matched.append(tag)
    return sorted(matched)


# ─────────────────────────────────────────────
# Reward components figure
# ─────────────────────────────────────────────

def fig_reward_components(scalars: dict, out_dir: str):
    """Plot reward-related scalar tags.

    If the training code only logged total reward, this plots a single curve.
    If sub-components were logged (e.g. Train/reward_tracking, Train/reward_alive, ...),
    all are drawn on the same axes.
    """
    reward_tags = _find_tags(scalars, REWARD_KEYWORDS)
    if not reward_tags:
        print("  [reward] No reward tags found. Available tags:")
        for t in sorted(scalars.keys())[:30]:
            print(f"    {t}")
        return

    print(f"  [reward] Found {len(reward_tags)} reward tags:")
    for t in reward_tags:
        print(f"    {t}")

    # Prefer tags that look like sub-components (contain '/' and a specific component name)
    component_tags = [t for t in reward_tags
                      if "/" in t and "mean_reward" not in t.lower()]
    total_tags     = [t for t in reward_tags if t not in component_tags]

    # Colour cycle
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Training Reward Curves", fontsize=13, fontweight="bold")

    all_plot_tags = total_tags + component_tags
    for i, tag in enumerate(all_plot_tags):
        steps, values = scalars[tag]
        label = tag.split("/")[-1]  # Use last part of tag path as label
        is_total = tag in total_tags
        color = cmap(i % 10)
        lw = 1.8 if is_total else 1.0
        alpha_raw = 0.25 if not is_total else 0.15
        ax.plot(steps, values, alpha=alpha_raw, color=color, linewidth=0.5)
        ax.plot(steps, _smooth(values, window=20), color=color, linewidth=lw,
                label=label, linestyle=("solid" if is_total else "dashed"))

    ax.set_xlabel("Training Iteration", fontsize=9)
    ax.set_ylabel("Reward", fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "reward_components.png"))


# ─────────────────────────────────────────────
# Training dynamics overview
# ─────────────────────────────────────────────

def fig_training_dynamics(scalars: dict, out_dir: str):
    """Multi-panel overview: total reward, episode length, policy loss, KL divergence."""

    # Collect panels: (title, [list of (tag, label)], y_label)
    panels = []

    # 1. Mean reward
    r_tags = [t for t in _find_tags(scalars, ["mean_reward", "Episode/mean_reward",
                                               "Train/mean_reward"])
              if t in scalars]
    if r_tags:
        panels.append(("Mean Reward", [(r_tags[0], "mean reward")], "Reward"))

    # 2. Episode length
    ep_tags = [t for t in _find_tags(scalars, ["episode_length", "mean_episode"])
               if t in scalars]
    if ep_tags:
        panels.append(("Episode Length", [(ep_tags[0], "mean ep. length")], "Steps"))

    # 3. Policy/value loss
    loss_tags = [(t, t.split("/")[-1])
                 for t in _find_tags(scalars, ["loss", "Loss"]) if t in scalars]
    if loss_tags:
        panels.append(("Losses", loss_tags[:4], "Loss"))

    # 4. KL / entropy
    kl_tags = [(t, t.split("/")[-1])
               for t in _find_tags(scalars, ["kl", "entropy", "clip_fraction"])
               if t in scalars]
    if kl_tags:
        panels.append(("KL / Entropy", kl_tags[:3], "Value"))

    if not panels:
        print("  [dynamics] No recognisable training tags found. Use --list_tags "
              "to see what is available.")
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    fig.suptitle("Training Dynamics", fontsize=13, fontweight="bold")
    if n == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    for ax, (title, tag_list, ylabel) in zip(axes, panels):
        for i, (tag, lbl) in enumerate(tag_list):
            steps, values = scalars[tag]
            color = cmap(i % 10)
            ax.plot(steps, values, alpha=0.2, color=color, linewidth=0.5)
            ax.plot(steps, _smooth(values, window=20), color=color,
                    linewidth=1.4, label=lbl)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Training Iteration", fontsize=9)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "training_dynamics.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TensorBoard analysis for RL_Bezier_MPC training runs")
    parser.add_argument("--logdir", type=str,
                        default="logs/quadruped_mpc",
                        help="Path to TensorBoard log directory or run sub-directory")
    parser.add_argument("--out", type=str, default="eval_figs",
                        help="Output directory for PNG figures (default: eval_figs/)")
    parser.add_argument("--list_tags", action="store_true",
                        help="Print all available scalar tags and exit (no plots)")
    args = parser.parse_args()

    scalars = load_tb_scalars(args.logdir)
    print(f"Found {len(scalars)} scalar tags total.")

    if args.list_tags:
        print("\nAll available tags:")
        for tag in sorted(scalars.keys()):
            steps, values = scalars[tag]
            print(f"  {tag:<60}  steps={len(steps)}  "
                  f"last={values[-1]:.4f}  min={values.min():.4f}  max={values.max():.4f}")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"\nGenerating training figures → {args.out}/")

    print("[1/2] Reward components...")
    fig_reward_components(scalars, args.out)

    print("[2/2] Training dynamics overview...")
    fig_training_dynamics(scalars, args.out)

    print(f"\nDone. Figures saved to: {args.out}/")


if __name__ == "__main__":
    main()
