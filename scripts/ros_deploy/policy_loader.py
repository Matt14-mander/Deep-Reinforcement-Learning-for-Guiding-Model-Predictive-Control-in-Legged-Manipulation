"""
policy_loader.py — 加载 RSL-RL 训练的策略检查点，无需 IsaacLab 依赖

从 scripts/play_quadruped_mpc.py 提取，适配独立 ROS 节点使用。
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


def load_policy(
    checkpoint_path: str,
    obs_dim: int = 45,
    action_dim: int = 12,
    hidden_dims: Tuple[int, ...] = (256, 256, 128),
    device: str = "cpu",
) -> Tuple[Callable, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """加载训练好的 RL 策略检查点。

    Args:
        checkpoint_path: .pt 检查点文件路径
        obs_dim:         观测空间维度 (默认 45)
        action_dim:      动作空间维度 (默认 12，Stage 1 fixed gait)
        hidden_dims:     隐藏层尺寸 (默认 256→256→128)
        device:          PyTorch 设备 ("cpu" 或 "cuda")

    Returns:
        policy_fn:  obs (torch.Tensor) → action (torch.Tensor)
        obs_mean:   观测均值 (45,)，无则 None
        obs_std:    观测标准差 (45,)，无则 None

    Raises:
        RuntimeError: 检查点中找不到 actor 权重
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # ── 构建 actor 网络 ──────────────────────────────────────────────────────
    # 架构：obs_dim → hidden_dims[0] → ... → action_dim (ELU 激活)
    actor_layers = []
    in_dim = obs_dim
    for h_dim in hidden_dims:
        actor_layers.append(nn.Linear(in_dim, h_dim))
        actor_layers.append(nn.ELU())
        in_dim = h_dim
    actor_layers.append(nn.Linear(in_dim, action_dim))
    actor_net = nn.Sequential(*actor_layers).to(device)

    # RSL-RL 保存格式："actor.0.weight", "actor.2.weight", ...
    # nn.Sequential 索引：0=Linear, 1=ELU, 2=Linear, 3=ELU, ...
    actor_state = {
        key[len("actor."):]: val
        for key, val in state_dict.items()
        if key.startswith("actor.")
    }
    if not actor_state:
        raise RuntimeError(
            f"检查点 {checkpoint_path} 中未找到 'actor.*' 权重。\n"
            f"可用 keys: {list(state_dict.keys())[:20]}"
        )

    actor_net.load_state_dict(actor_state)
    actor_net.eval()
    print(f"[PolicyLoader] 加载 actor ({len(actor_state)} tensors) from {checkpoint_path}")

    # ── 加载观测归一化参数 ────────────────────────────────────────────────────
    obs_mean: Optional[torch.Tensor] = None
    obs_std:  Optional[torch.Tensor] = None

    for mean_key in [
        "actor_obs_normalizer._mean",
        "obs_normalizer._mean",
        "actor_obs_normalizer.running_mean",
    ]:
        if mean_key in state_dict:
            obs_mean = state_dict[mean_key].to(device)
            break

    # 优先使用预计算的 _std；回退到 sqrt(_var)
    for std_key in ["actor_obs_normalizer._std", "obs_normalizer._std"]:
        if std_key in state_dict:
            obs_std = state_dict[std_key].to(device)
            break
    if obs_std is None:
        for var_key in [
            "actor_obs_normalizer._var",
            "obs_normalizer._var",
            "actor_obs_normalizer.running_var",
        ]:
            if var_key in state_dict:
                obs_std = torch.sqrt(state_dict[var_key].to(device) + 1e-8)
                break

    if obs_mean is not None and obs_std is not None:
        obs_std_safe = torch.clamp(obs_std, min=1e-6)
        print(f"[PolicyLoader] 加载观测归一化器 (mean shape: {obs_mean.shape})")

        def policy_fn(obs: torch.Tensor) -> torch.Tensor:
            normalized = (obs - obs_mean) / obs_std_safe
            return actor_net(normalized)
    else:
        print("[PolicyLoader] WARNING: 未找到观测归一化器，使用原始观测（可能影响性能）")

        def policy_fn(obs: torch.Tensor) -> torch.Tensor:
            return actor_net(obs)

    return policy_fn, obs_mean, obs_std
