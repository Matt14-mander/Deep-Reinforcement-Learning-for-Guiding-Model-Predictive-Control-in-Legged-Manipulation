"""Load RSL-RL actor checkpoints without importing Isaac Lab."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


_ACTOR_KEY = re.compile(r"(?:^|\.)actor\.(\d+)\.(weight|bias)$")


class ActorMLP(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class NormalizedActor(nn.Module):
    """Exportable normalizer + deterministic actor module."""

    def __init__(self, actor: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.actor = actor
        self.register_buffer("observation_mean", mean.reshape(-1).float())
        self.register_buffer("observation_std", torch.clamp(std.reshape(-1).float(), min=1e-6))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor((observation - self.observation_mean) / self.observation_std)


@dataclass(frozen=True)
class PolicyInfo:
    observation_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...]
    normalizer_mean_key: str | None
    normalizer_scale_key: str | None
    normalizer_scale_kind: str


@dataclass
class PolicyBundle:
    module: NormalizedActor
    info: PolicyInfo
    checkpoint_path: Path
    device: torch.device

    def act(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        single = value.ndim == 1
        if single:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != self.info.observation_dim:
            raise ValueError(
                f"Expected observation shape ({self.info.observation_dim},) or "
                f"(N, {self.info.observation_dim}), got {value.shape}"
            )
        with torch.inference_mode():
            output = self.module(torch.from_numpy(value).to(self.device)).cpu().numpy()
        return output[0] if single else output


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate
    if any(_ACTOR_KEY.search(str(key)) for key in checkpoint):
        return checkpoint
    raise ValueError(f"No actor state dict found; top-level keys: {list(checkpoint)[:12]}")


def _actor_layers(state_dict: dict[str, torch.Tensor]) -> tuple[ActorMLP, tuple[int, ...]]:
    tensors: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        match = _ACTOR_KEY.search(key)
        if match:
            tensors.setdefault(int(match.group(1)), {})[match.group(2)] = value
    if not tensors:
        raise ValueError("No keys ending in actor.<index>.weight were found")

    layers: list[nn.Module] = []
    widths: list[int] = []
    indices = sorted(tensors)
    for position, index in enumerate(indices):
        values = tensors[index]
        if "weight" not in values:
            raise ValueError(f"Actor layer {index} has no weight")
        weight = values["weight"].detach().cpu()
        bias = values.get("bias")
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        linear.weight.data.copy_(weight)
        if bias is not None:
            linear.bias.data.copy_(bias.detach().cpu())
        layers.append(linear)
        widths.append(int(weight.shape[0]))
        if position < len(indices) - 1:
            layers.append(nn.ELU())
    return ActorMLP(layers), tuple(widths[:-1])


def _normalizer(
    state_dict: dict[str, torch.Tensor], observation_dim: int
) -> tuple[torch.Tensor, torch.Tensor, str | None, str | None, str]:
    def candidates(kind: str) -> list[tuple[str, torch.Tensor]]:
        return [
            (key, value)
            for key, value in state_dict.items()
            if "obs_normalizer" in key.lower() and key.lower().endswith(kind)
        ]

    def choose(items: list[tuple[str, torch.Tensor]]) -> tuple[str, torch.Tensor] | None:
        valid = [(key, value) for key, value in items if value.numel() == observation_dim]
        valid.sort(key=lambda item: ("actor_obs_normalizer" not in item[0].lower(), len(item[0])))
        return valid[0] if valid else None

    mean_entry = choose(candidates("_mean") + candidates("running_mean") + candidates(".mean"))
    std_entry = choose(candidates("_std") + candidates("running_std") + candidates(".std"))
    var_entry = choose(candidates("_var") + candidates("running_var") + candidates(".var"))

    if mean_entry is None:
        mean = torch.zeros(observation_dim)
        mean_key = None
    else:
        mean_key, mean = mean_entry
        mean = mean.detach().cpu().float().reshape(-1)

    if std_entry is not None:
        scale_key, std = std_entry
        scale_kind = "std"
        std = std.detach().cpu().float().reshape(-1)
    elif var_entry is not None:
        scale_key, variance = var_entry
        scale_kind = "variance"
        std = torch.sqrt(torch.clamp(variance.detach().cpu().float().reshape(-1), min=1e-10))
    else:
        scale_key = None
        scale_kind = "identity"
        std = torch.ones(observation_dim)
    return mean, torch.clamp(std, min=1e-6), mean_key, scale_key, scale_kind


def load_policy(
    checkpoint_path: str | Path,
    *,
    expected_observation_dim: int | None = None,
    expected_action_dim: int | None = None,
    device: str = "cpu",
) -> PolicyBundle:
    """Load a deterministic actor and include observation normalization."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    target_device = torch.device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(checkpoint)
    actor, hidden_dims = _actor_layers(state_dict)
    linear_layers = [module for module in actor.net if isinstance(module, nn.Linear)]
    observation_dim = linear_layers[0].in_features
    action_dim = linear_layers[-1].out_features
    if expected_observation_dim is not None and observation_dim != expected_observation_dim:
        raise ValueError(f"Checkpoint observation dim is {observation_dim}, expected {expected_observation_dim}")
    if expected_action_dim is not None and action_dim != expected_action_dim:
        raise ValueError(f"Checkpoint action dim is {action_dim}, expected {expected_action_dim}")

    mean, std, mean_key, scale_key, scale_kind = _normalizer(state_dict, observation_dim)
    module = NormalizedActor(actor, mean, std).eval().to(target_device)
    return PolicyBundle(
        module=module,
        info=PolicyInfo(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            normalizer_mean_key=mean_key,
            normalizer_scale_key=scale_key,
            normalizer_scale_kind=scale_kind,
        ),
        checkpoint_path=path,
        device=target_device,
    )
