"""
Policy loader for ROS2 deployment.

Loads a trained RSL-RL checkpoint (.pt) and returns the actor network
and observation normalizer — with zero IsaacLab dependencies.

Checkpoint structure produced by RSL-RL:
    {
        'model_state_dict': {
            'actor.0.weight': ...,  'actor.0.bias': ...,
            'actor.2.weight': ...,  'actor.2.bias': ...,
            'actor.4.weight': ...,  'actor.4.bias': ...,
            'actor.6.weight': ...,  'actor.6.bias': ...,   (optional 4th layer)
            'obs_normalizer.mean': ...,
            'obs_normalizer.var': ...,
        }
    }

Usage:
    actor, obs_mean, obs_std = load_policy("logs/.../model_1098.pt")
    obs_norm = (obs - obs_mean) / obs_std
    with torch.no_grad():
        action = actor(torch.FloatTensor(obs_norm)).numpy()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Actor network (mirrors RSL-RL ActorCritic's actor head)
# ---------------------------------------------------------------------------

class ActorMLP(nn.Module):
    """Simple MLP actor — reconstructed from checkpoint weight shapes."""

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_actor_from_state_dict(state_dict: dict) -> ActorMLP:
    """Reconstruct actor MLP from checkpoint weight tensors.

    Scans for keys matching 'actor.<even_idx>.weight' and infers the
    network architecture automatically from tensor shapes.
    """
    # Collect (layer_idx, weight, bias) triples
    layer_weights = {}
    for k, v in state_dict.items():
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "actor" and parts[2] == "weight":
            idx = int(parts[1])
            layer_weights[idx] = {"weight": v}
        if len(parts) >= 3 and parts[0] == "actor" and parts[2] == "bias":
            idx = int(parts[1])
            if idx not in layer_weights:
                layer_weights[idx] = {}
            layer_weights[idx]["bias"] = v

    if not layer_weights:
        raise ValueError(
            "No 'actor.*' keys found in checkpoint. "
            "Keys present: " + str(list(state_dict.keys())[:10])
        )

    sorted_indices = sorted(layer_weights.keys())
    layers = []
    for idx in sorted_indices:
        w = layer_weights[idx]["weight"]
        b = layer_weights[idx].get("bias", None)
        out_features, in_features = w.shape
        linear = nn.Linear(in_features, out_features, bias=(b is not None))
        linear.weight.data.copy_(w)
        if b is not None:
            linear.bias.data.copy_(b)

        layers.append(linear)

        # Add ELU activation between layers (not after the last one)
        if idx != sorted_indices[-1]:
            layers.append(nn.ELU())

    return ActorMLP(layers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_policy(
    checkpoint_path: str | Path,
    obs_dim: int = 45,
    action_dim: int = 12,
    device: str = "cpu",
) -> tuple[ActorMLP, np.ndarray, np.ndarray]:
    """Load trained policy from RSL-RL checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        obs_dim: Expected observation dimension (default 45).
        action_dim: Expected action dimension (default 12 Bezier params).
        device: Torch device for inference ('cpu' or 'cuda').

    Returns:
        actor     : nn.Module ready for inference. Call with normalised obs.
        obs_mean  : (obs_dim,) numpy array — normaliser mean.
        obs_std   : (obs_dim,) numpy array — normaliser std (clipped ≥ 1e-5).

    Example:
        actor, mean, std = load_policy("model_1098.pt")
        obs_norm = (raw_obs - mean) / std
        with torch.no_grad():
            action = actor(torch.FloatTensor(obs_norm)).numpy()
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[PolicyLoader] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both direct state_dict and wrapped {'model_state_dict': ...}
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any("actor" in k for k in ckpt):
        state_dict = ckpt
    else:
        raise ValueError(
            "Unrecognised checkpoint format. "
            f"Top-level keys: {list(ckpt.keys())}"
        )

    # --- Actor network ---
    actor = _build_actor_from_state_dict(state_dict)
    actor.eval()
    actor.to(device)

    # --- Observation normaliser ---
    # RSL-RL stores running mean/variance under several possible key prefixes
    mean_key = next(
        (k for k in state_dict if "obs_normalizer" in k and "mean" in k), None
    )
    var_key = next(
        (k for k in state_dict if "obs_normalizer" in k and ("var" in k or "std" in k)), None
    )

    if mean_key is not None and var_key is not None:
        obs_mean = state_dict[mean_key].cpu().numpy().astype(np.float32)
        obs_var  = state_dict[var_key].cpu().numpy().astype(np.float32)
        # RSL-RL stores variance; convert to std
        obs_std  = np.sqrt(np.maximum(obs_var, 1e-10)).astype(np.float32)
    else:
        print(
            "[PolicyLoader] WARNING: obs_normalizer not found in checkpoint. "
            "Using identity normalisation (mean=0, std=1)."
        )
        obs_mean = np.zeros(obs_dim, dtype=np.float32)
        obs_std  = np.ones(obs_dim, dtype=np.float32)

    obs_std = np.maximum(obs_std, 1e-5)  # numerical safety

    print(
        f"[PolicyLoader] Actor architecture: "
        + " -> ".join(
            str(m.out_features)
            for m in actor.net
            if isinstance(m, nn.Linear)
        )
    )
    print(
        f"[PolicyLoader] obs_mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]"
    )

    return actor, obs_mean, obs_std
