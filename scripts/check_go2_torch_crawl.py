"""Two cycles of a fixed Go2 crawl in one PhysX episode."""
import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def run(args):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg
    from RL_Bezier_MPC.envs.fixed_crawl_env import FixedCrawlEnv

    cfg = QuadrupedMPCEnvCfg()
    cfg.scene.num_envs = 1
    if args.probe_result is not None or args.endpoint_result is not None:
        cfg.seed = 37
    cfg.episode_length_s = 90.
    cfg.sim.device = args.device
    cfg.mpc_backend = "torch_tracking"
    cfg.torch_foot_plane_height = .022
    cfg.use_mpc_cluster = False
    env = None
    try:
        env = FixedCrawlEnv(cfg=cfg)
        env.reset()
        env.begin_crawl(args.support_shift_fraction, endpoint=args.endpoint_result is not None,
                        shape=args.probe_result is not None, record_probe=args.probe_result is not None)
        print(f"support_shift_fraction={args.support_shift_fraction}", flush=True)
        action = torch.zeros((1,cfg.action_space), device=env.device)
        if args.endpoint_result is not None:
            if cfg.action_space != 12 or args.probe_result is not None:
                raise ValueError("Endpoint trial requires the fixed-gait 12-D action alone")
            env.endpoint_ready = env.stopped = False
            env.hold_ticks = 0
            env.stop_window = []
            action[:, 9] = 2 * args.target_mm * .001 / (env.action_high-env.action_low).reshape(-1)[9]
        if args.probe_result is not None:
            env.probe_ready = False
            env.samples = []
            # Express the desired physical perturbation through the real normalized action bounds.
            action[:, [3, 6]] = 2 * args.probe_mm * .001 / (env.action_high-env.action_low).reshape(-1)[[3, 6]]
        while env.index < len(env.plan.reference)-1 or (args.endpoint_result is not None and not env.stopped):
            _, _, term, trunc, _ = env.step(action)
            if (term | trunc).any():
                raise RuntimeError(f"Episode ended at tick {env.tick}")
        delta = env.robot.data.root_link_pos_w[0].double() - env.tracking_origins[0] - env.base_start
        if env.landings != 8 or not .030 <= delta[0].item() <= .050 or abs(delta[1].item()) > .010:
            raise RuntimeError(f"Final displacement {delta.tolist()} m; recovered landings={env.landings}")
        print(json.dumps(dict(dx_m=delta[0].item(), dy_m=delta[1].item(), landings=env.landings,
                              max_stance_drift_mm=(1000*env.max_drift).tolist())), flush=True)
        print("PASS: single Go2 PhysX two-cycle fixed crawl", flush=True)
        if args.endpoint_result is not None:
            if abs(env.goal - args.target_mm * .001) > 1e-6:
                raise ValueError("Endpoint action clipped or decoded incorrectly")
            xs, speeds, spins, joints = zip(*env.stop_window)
            summary = dict(initial=env.initial, target_mm=args.target_mm, decoded_m=env.goal,
                           dx_m=delta[0].item(), dy_m=delta[1].item(), landings=env.landings,
                           drift_mm=(1000*env.max_drift).tolist(), stable_ticks=len(xs),
                           stop_span_mm=1000*(max(xs)-min(xs)), max_speed_mps=max(speeds),
                           max_spin_rad_s=max(spins), max_joint_rad_s=max(joints))
            if abs(summary["dx_m"]-env.goal) > .004:
                raise RuntimeError(f"Endpoint missed after final physics step: {summary}")
            args.endpoint_result.write_text(json.dumps(summary, allow_nan=False), encoding="utf8")
        if args.probe_result is not None:
            args.probe_result.write_text(json.dumps(dict(initial=env.initial, samples=env.samples,
                controls=env.controls, landings=env.landings, dx_m=delta[0].item(),
                drift_mm=(1000*env.max_drift).tolist()), allow_nan=False), encoding="utf-8")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-result", type=Path)
    parser.add_argument("--target-mm", type=float, choices=(32., 40., 48.), default=40.)
    parser.add_argument("--probe-result", type=Path)
    parser.add_argument("--probe-mm", type=float, default=0.0)
    parser.add_argument("--support-shift-fraction", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    completed = False
    try:
        run(args)
        completed = True
    finally:
        try:
            launcher.app.close()
        finally:
            if not completed:
                raise SystemExit(1)
