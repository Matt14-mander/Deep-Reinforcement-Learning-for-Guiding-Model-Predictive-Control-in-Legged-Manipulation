"""Probe the real Stage 1 training environment before allocating PPO rollouts."""
import json

import torch


@torch.inference_mode()
def check_stage1_entry(env):
    cfg = env.cfg
    problems = []
    if cfg.action_space != 12 or not cfg.fix_gait_params:
        raise ValueError("Expected the fixed-gait 12-D policy")
    report = dict(backend=cfg.mpc_backend, target_range_m=list(cfg.target_pos_range),
                  goal_radius_m=cfg.target_threshold, episode_s=cfg.episode_length_s)
    if tuple(cfg.target_pos_range[:4]) != (.032, .048, 0., 0.):
        problems.append("Target sampler is not configured for the 32-48 mm forward curriculum")
    if cfg.target_threshold > .004:
        problems.append("Goal radius exceeds the validated 4 mm endpoint tolerance")
    if cfg.episode_length_s < 26.8:
        problems.append("Episode too short for the measured two-cycle crawl plus stopping")
    print(json.dumps({**report, "configuration_problems": problems}), flush=True)
    traces = []
    for distance in (.032, .048):
        env.reset(seed=cfg.seed)
        data = env.robot.data
        start = data.root_link_pos_w.clone()
        initial = torch.cat((data.root_link_pose_w, data.root_link_vel_w,
                             data.joint_pos, data.joint_vel), -1).clone()
        action = torch.zeros((1, 12), device=env.device)
        width = (env.action_high - env.action_low).reshape(-1)[9]
        action[:, 9] = 2 * distance / width
        if action.abs().max() > 1:
            raise ValueError("Endpoint probe exceeds action bounds")
        xs, efforts = [], []
        goal = env.target_positions.clone()
        for _ in range(12):
            obs, reward, term, trunc, _ = env.step(action)
            x = data.root_link_pos_w - start
            tau = env._pending_joint_efforts
            if (obs["policy"].shape != (1, cfg.observation_space)
                    or not all(v.isfinite().all() for v in (obs["policy"], reward, x, tau))):
                raise RuntimeError("Invalid observation, reward, state or torque")
            if (term | trunc).any():
                raise RuntimeError("Unexpected episode end during action probe")
            xs.append(x[0, 0].clone())
            efforts.append(tau.clone())
        traces.append((initial, torch.stack(xs), torch.stack(efforts), goal))
    a, b = traces
    report["initial_delta_max"] = (a[0]-b[0]).abs().max().item()
    report["task_goal_delta_m"] = (a[3]-b[3]).abs().max().item()
    report["motion_delta_max_mm"] = 1000 * (a[1]-b[1]).abs().max().item()
    report["torque_delta_rms_Nm"] = (a[2]-b[2]).square().mean().sqrt().item()
    if max(report["initial_delta_max"], report["task_goal_delta_m"]) > 1e-6:
        problems.append("Initial states or sampled goals differ; comparison inconclusive")
    if report["motion_delta_max_mm"] < .1 or report["torque_delta_rms_Nm"] < .001:
        problems.append("Insufficient motion or torque separation between endpoint actions")
    print(json.dumps({**report, "problems": problems}), flush=True)
    if problems:
        raise RuntimeError("Stage 1 entry rejected: " + " | ".join(problems))
    print("PASS: Stage 1 short action probe; full episode and PPO gates still required", flush=True)
