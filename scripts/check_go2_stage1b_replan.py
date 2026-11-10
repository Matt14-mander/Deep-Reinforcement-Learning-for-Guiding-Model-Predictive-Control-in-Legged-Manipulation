"""Three continuous Go2 crawl cycles. Change the endpoint during the first FL swing."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def run(args):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg
    from RL_Bezier_MPC.envs.stage1b_cycle_env import Stage1BCycleEnv

    cfg = QuadrupedMPCEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 37
    cfg.sim.device = args.device
    cfg.mpc_backend = "torch_tracking"
    cfg.use_mpc_cluster = False
    cfg.torch_foot_plane_height = .022
    cfg.episode_length_s = 60.
    env = Stage1BCycleEnv(cfg=cfg)
    def trial(target_mm):
        actions = []

        def policy(obs):
            # Leave a small residual to the final cycle. This distinguishes a
            # consumed policy action from the old equal-division replanner.
            mean = obs[:, :1] / obs[:, 1:]
            stride = torch.where(obs[:, 1:] > 1, .020 + .95*(mean-.020), mean)
            actions.append((obs.clone(), stride.item()))
            return stride

        summary = env.run_episode(target_mm, cfg.seed, policy if args.cycle_policy else None)
        print(json.dumps(summary, allow_nan=False), flush=True)
        if args.cycle_policy:
            if len(actions) != 3 or len(env.replans) != 3 or len(env.distances) != 3:
                raise RuntimeError("Expected exactly one stride decision for each of three cycles")
            for i, ((obs, stride), event, measured) in enumerate(zip(actions, env.replans, env.distances)):
                goal = .060 if i == 0 else target_mm / 1000
                if not (event["completed_cycles"] == i and obs[0, 1].item() == 3-i
                        and abs(obs[0, 0].item() - (goal-event["dx_m"])) < 1e-6
                        and abs(event["stride_m"]-stride) < 1e-8 and abs(measured-stride) <= .002):
                    raise RuntimeError(f"Invalid observation, applied stride or measured response: {event}")
            direction = 1 if target_mm > 60 else -1
            if not (env.replans[0]["tick"] < env.request_tick < env.replans[1]["tick"]
                    and all(direction*(a-actions[0][1]) > .002 for _, a in actions[1:])
                    and all(direction*(d-env.distances[0]) > .002 for d in env.distances[1:])):
                raise RuntimeError("Expected signed stride adjustment on the next cycle")
        else:
            if len(env.replans) != 1 or len(env.distances) != 3:
                raise RuntimeError("Expected one replan and three completed cycles")
            event = env.replans[0]
            if not (event["completed_cycles"] == 1 and env.request_tick < event["tick"]
                    and event["stride_m"] > event["old_stride_m"]+.002
                    and all(abs(d-event["stride_m"]) <= .002 for d in env.distances[1:])):
                raise RuntimeError("Endpoint replan did not increase cycle distance")
        if not summary["arrived"]:
            raise RuntimeError("Stage 1B endpoint gate failed")
        return torch.tensor(env.initial, dtype=torch.float64)

    try:
        initial = trial(68)
        if args.cycle_policy:
            if not torch.allclose(initial, trial(52), rtol=0, atol=1e-6):
                raise RuntimeError("Initial states of lengthen and shorten trials differ")
            print("PASS: Stage 1B cycle policy lengthens and shortens stride", flush=True)
        else:
            print("PASS: Stage 1B three-cycle crawl with live endpoint replan", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-policy", action="store_true", help="Run 60->68 and 60->52 mm policy trials")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    success = False
    try:
        run(args)
        success = True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            launcher.app.close()
        finally:
            if not success:
                raise SystemExit(1)
