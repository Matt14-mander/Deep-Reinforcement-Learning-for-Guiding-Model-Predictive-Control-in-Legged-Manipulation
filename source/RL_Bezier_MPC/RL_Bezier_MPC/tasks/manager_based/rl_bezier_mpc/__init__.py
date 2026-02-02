# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations for RL+MPC Bezier control environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Legacy manager-based environment (cartpole placeholder)
gym.register(
    id="Template-Rl-Bezier-Mpc-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_bezier_mpc_env_cfg:RlBezierMpcEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Main Quadrotor MPC environment (DirectRL)
gym.register(
    id="Quadrotor-MPC-Bezier-v0",
    entry_point="RL_Bezier_MPC.envs:QuadrotorMPCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "RL_Bezier_MPC.envs:QuadrotorMPCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadrotorMPCPPORunnerCfg",
    },
)

# Quadruped MPC environment (DirectRL)
gym.register(
    id="Quadruped-MPC-Bezier-v0",
    entry_point="RL_Bezier_MPC.envs:QuadrupedMPCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "RL_Bezier_MPC.envs:QuadrupedMPCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadrupedMPCPPORunnerCfg",
    },
)