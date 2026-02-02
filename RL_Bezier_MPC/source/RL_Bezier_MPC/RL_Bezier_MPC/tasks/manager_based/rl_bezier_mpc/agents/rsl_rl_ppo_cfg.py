# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for RL+MPC Bezier environments."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for legacy cartpole environment."""

    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class QuadrotorMPCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Quadrotor MPC Bezier environment.

    This configuration is tuned for the hierarchical RL+MPC architecture:
    - Larger networks to handle the more complex control task
    - More rollout steps for better trajectory learning
    - Observation normalization for stable training
    """

    num_steps_per_env = 24  # Longer rollouts for trajectory-level learning
    max_iterations = 1000
    save_interval = 100
    experiment_name = "quadrotor_mpc_bezier"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,  # Normalize observations
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],  # Deeper networks
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Higher entropy for exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # Standard PPO learning rate
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )