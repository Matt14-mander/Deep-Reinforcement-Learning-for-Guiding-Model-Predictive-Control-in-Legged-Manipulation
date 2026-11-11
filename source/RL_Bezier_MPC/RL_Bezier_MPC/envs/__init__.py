# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment module for RL+MPC Bezier control system."""

from .quadrotor_mpc_env import QuadrotorMPCEnv
from .quadrotor_mpc_env_cfg import QuadrotorMPCEnvCfg
from .quadruped_mpc_env import QuadrupedMPCEnv
from .quadruped_mpc_env_cfg import QuadrupedMPCEnvCfg

__all__ = [
    "QuadrotorMPCEnv",
    "QuadrotorMPCEnvCfg",
    "QuadrupedMPCEnv",
    "QuadrupedMPCEnvCfg",
]


from .go2_augmpc_env import Go2AugMPCEnv
from .go2_augmpc_env_cfg import Go2AugMPCEnvCfg
__all__ += ["Go2AugMPCEnv", "Go2AugMPCEnvCfg"]

import gymnasium as gym
gym.register(
    id="Isaac-Go2-AugMPC-Direct-v0",
    entry_point="RL_Bezier_MPC.envs:Go2AugMPCEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "RL_Bezier_MPC.envs:Go2AugMPCEnvCfg"},
)
