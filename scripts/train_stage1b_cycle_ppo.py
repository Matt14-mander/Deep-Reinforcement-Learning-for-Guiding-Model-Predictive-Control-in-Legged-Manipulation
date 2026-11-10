"""One Stage 1B PPO batch (3 episodes, 9 strides) and deterministic PhysX evaluation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def run(args):
    import torch
    from torch import nn
    from torch.distributions import Normal

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg
    from RL_Bezier_MPC.envs.stage1b_cycle_env import Stage1BCycleEnv

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
            self.critic = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
            self.log_std = nn.Parameter(torch.tensor([-.7]))

        def distribution(self, x):
            return Normal(self.actor(x), self.log_std.clamp(-4., 1.).exp())

    cfg = QuadrupedMPCEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.mpc_backend = "torch_tracking"
    cfg.use_mpc_cluster = False
    cfg.torch_foot_plane_height = .022
    cfg.episode_length_s = 60.
    out = args.output or Path("logs/stage1b_cycle") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    env = Stage1BCycleEnv(cfg=cfg)
    try:
        torch.manual_seed(args.seed)
        noise = torch.Generator(device=env.device).manual_seed(args.seed)
        policy = Policy().to(env.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        batch = []
        report = {"seed": args.seed, "train": [], "evaluation": []}

        def episode(goal, sampling):
            records = []

            @torch.no_grad()
            def stride(obs):
                x = obs / obs.new_tensor([.060, 3.])
                dist = policy.distribution(x)
                z = dist.mean
                if sampling:
                    z = z + dist.stddev * torch.randn(z.shape, device=z.device, generator=noise)
                    records.append((x, z, dist.log_prob(z).sum(-1)))
                # Bounded in the previously validated forward crawl range.
                return .020 + .004 * z.tanh()

            m = env.run_episode(goal, cfg.seed, stride, require_goal_for_stop=False)
            if len(m["cycle_decisions"]) != 3:
                raise RuntimeError("Expected exactly three cycle decisions")
            for i, d in enumerate(m["cycle_decisions"]):
                expected = (.060 if i == 0 else goal*.001)
                if d["cycles_left"] != 3-i or abs(d["goal_m"]-expected) > 1e-7:
                    raise RuntimeError(f"Incorrect cycle goal: {d}")
            reward = -m["final_error_mm"]/8. + float(m["arrived"])
            m.update(reward=reward)
            print(json.dumps(dict(stage="rollout" if sampling else "evaluation", **m),
                             allow_nan=False), flush=True)
            if sampling:
                if len(records) != 3:
                    raise RuntimeError("Incomplete sampled episode")
                # Undiscounted episodic return: zero intermediate rewards and one terminal reward.
                batch.extend((*row, reward) for row in records)
            return m

        goals = (60, 68, 52)
        path = args.evaluate
        if path is None:
            for goal in goals:
                report["train"].append(episode(goal, True))
            if len(batch) != 9:
                raise RuntimeError("PPO requires nine on-policy decisions")
            obs, z, old, returns = zip(*batch)
            x, z, old = torch.cat(obs), torch.cat(z), torch.cat(old)
            returns = x.new_tensor(returns)
            advantage = returns - policy.critic(x).squeeze(-1).detach()
            before = nn.utils.parameters_to_vector(policy.actor.parameters()).detach().clone()
            for _ in range(4):
                ratio = (policy.distribution(x).log_prob(z).sum(-1)-old).exp()
                loss = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()
                loss += .5*(policy.critic(x).squeeze(-1)-returns).square().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            change = (nn.utils.parameters_to_vector(policy.actor.parameters())-before).norm().item()
            if not 0 < change < float("inf") or not loss.isfinite():
                raise RuntimeError("Invalid PPO update")
            report["update"] = dict(decisions=len(batch), loss=loss.item(), actor_change=change)
            print(json.dumps(dict(stage="ppo_update", **report["update"])), flush=True)
            path = out / "stage1b_0001.pt"
            torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                            iteration=1, seed=args.seed), path)

        loaded = Policy().to(env.device)
        saved = torch.load(path, weights_only=True, map_location=env.device)
        loaded.load_state_dict(saved["policy"])
        restored_optimizer = torch.optim.Adam(loaded.parameters())
        restored_optimizer.load_state_dict(saved["optimizer"])
        if args.evaluate is None and any(not torch.equal(v, loaded.state_dict()[k])
                                         for k, v in policy.state_dict().items()):
            raise RuntimeError("Checkpoint reload differs from updated policy")
        policy = loaded.eval()
        report["checkpoint"] = str(path.resolve())
        print(f"Reloaded Stage 1B checkpoint {path}", flush=True)
        for goal in goals:
            report["evaluation"].append(episode(goal, False))
        report["all_eval_arrived"] = all(m["arrived"] for m in report["evaluation"])
        (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
        print(f"PASS: Stage 1B deterministic evaluation; all_eval_arrived={report['all_eval_arrived']}; {out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", type=Path, help="Evaluate an existing checkpoint without a PPO update")
    parser.add_argument("--output", type=Path, help="New results directory")
    parser.add_argument("--seed", type=int, default=37)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    completed = False
    try:
        run(args)
        completed = True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            launcher.app.close()
        finally:
            if not completed:
                raise SystemExit(1)
