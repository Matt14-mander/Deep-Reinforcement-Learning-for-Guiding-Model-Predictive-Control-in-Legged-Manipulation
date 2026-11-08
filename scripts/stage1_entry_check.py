"""Full-episode gates and one-decision PPO for the fixed-crawl curriculum."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal


def check_stage1_entry(task, seed):
    previous = None
    initial = None
    for goal in task.targets_mm:
        obs = task.reset(goal, seed)
        # Scripted policy used ONLY for the three physical admission episodes.
        action = torch.stack((obs.new_zeros(()), obs[0]))
        reward, metrics = task.step(action)
        print(json.dumps(dict(stage="admission", reward=reward, **metrics)), flush=True)
        if not metrics["success"]:
            raise RuntimeError(f"Full-episode gate failed for {goal} mm")
        state = torch.tensor(task.env.initial, dtype=torch.float64)
        if initial is None:
            initial = state
        elif not torch.allclose(initial, state, atol=1e-6, rtol=0):
            raise RuntimeError("Admission trials did not start at the same state")
        dx = metrics["dx_mm"]
        if previous is not None and dx - previous < 4.:
            raise RuntimeError("Insufficient separation between measured endpoints")
        previous = dx
    print("PASS: Stage 1 training entry, three full PhysX episodes", flush=True)


class TrajectoryPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 2))
        self.critic = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        self.log_std = nn.Parameter(torch.full((2,), -.7))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-4, 1).exp())


def ppo_iteration(task, policy, optimizer, samples, goals=None):
    # Every decision ends an episode, so the return is its measured terminal reward.
    observations, latents, old_logps, returns = [], [], [], []
    completed = 0
    for i in range(samples):
        obs = task.reset() if goals is None else task.reset(goals[i])
        with torch.no_grad():
            dist = policy.distribution(obs)
            latent = dist.sample()
            old_logps.append(dist.log_prob(latent).sum())
        reward, metrics = task.step(latent.tanh())
        print(json.dumps(dict(stage="rollout", reward=reward, **metrics)), flush=True)
        completed += "failure" not in metrics
        observations.append(obs)
        latents.append(latent)
        returns.append(reward)
    if not completed:
        raise RuntimeError("No sampled trajectory completed the physical crawl")
    obs = torch.stack(observations)
    latent = torch.stack(latents)
    old = torch.stack(old_logps)
    target = obs.new_tensor(returns)
    advantage = target - policy.critic(obs).squeeze(-1).detach()
    before = torch.nn.utils.parameters_to_vector(policy.actor.parameters()).detach().clone()
    for _ in range(4):
        # The fixed tanh transform cancels in the old/new likelihood ratio.
        ratio = (policy.distribution(obs).log_prob(latent).sum(-1) - old).exp()
        surrogate = torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)
        value_error = policy.critic(obs).squeeze(-1) - target
        loss = -surrogate.mean() + .5 * value_error.square().mean()
        if not loss.isfinite():
            raise RuntimeError("Nonfinite PPO loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    change = (torch.nn.utils.parameters_to_vector(policy.actor.parameters()) - before).norm().item()
    if not 0 < change < float("inf"):
        raise RuntimeError("PPO actor failed to update")
    print(json.dumps(dict(stage="ppo_update", samples=samples, completed=completed,
                          loss=loss.item(), actor_change=change)), flush=True)


@torch.no_grad()
def evaluate(task, policy, role, seed):
    if policy is not None:
        policy.eval()
    rows = {}
    for goal in task.targets_mm:
        obs = task.reset(goal, seed)
        action = obs.new_zeros(2) if policy is None else policy.actor(obs).tanh()
        reward, metrics = task.step(action)
        rows[goal] = dict(role=role, action=action.tolist(), reward=reward, **metrics)
        print(json.dumps(dict(stage="evaluation", **rows[goal]), allow_nan=False), flush=True)
    return rows


def report_comparison(baseline, before, after, destination, seed):
    rows = []
    for goal in before:
        trials = [group[goal] for group in (baseline, before, after)]
        errors = [r.get("error_mm") for r in trials]
        complete = all("failure" not in r and r["landings"] == 8 and r["stable_ticks"] == 50
                       and max(r["drift_mm"]) <= 10 and r["stop_span_mm"] <= 1 for r in trials)
        baseline_error, pre_error, post_error = errors
        rows.append(dict(target_mm=goal, baseline_error_mm=baseline_error,
                         pre_error_mm=pre_error, post_error_mm=post_error,
                         improvement_mm=pre_error-post_error if complete else None,
                         beats_both=complete and post_error < min(baseline_error, pre_error),
                         physical_checks=complete, post_success=after[goal]["success"]))
    passed = (all(r["physical_checks"] and r["post_success"] for r in rows)
              and all(r["beats_both"] for r in rows if r["target_mm"] in (32, 48)))
    report = dict(seed=seed, baseline=baseline, before=before, after=after,
                  comparison=rows, pilot_criteria_met=passed)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(dict(stage="comparison", goals=rows, pilot_criteria_met=passed,
                          report=str(destination))), flush=True)


def evaluate_saved_run(task, run, seed):
    run = Path(run)
    manifest = json.loads((run / "comparison_checkpoints.json").read_text(encoding="utf8"))
    baseline = evaluate(task, None, "fixed40", seed)
    results = []
    for role in ("before", "after"):
        policy = TrajectoryPolicy().to(task.env.device)
        saved = torch.load(run / manifest[role], map_location=task.env.device, weights_only=True)
        policy.load_state_dict(saved["policy"])
        results.append(evaluate(task, policy, role, seed))
    report_comparison(baseline, *results, run / f"reevaluation_seed{seed}.json", seed)


def train_stage1(task, iterations, checkpoint_dir, *, compare=False, seed=37, start_from=None):
    policy = TrajectoryPolicy().to(task.env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    start = 0
    if start_from:
        saved = torch.load(start_from, map_location=task.env.device, weights_only=True)
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["iteration"]
    if any(directory.glob("stage1_*.pt")):
        raise FileExistsError(f"Choose a fresh checkpoint directory: {directory}")
    if compare:
        pre_path = directory / f"stage1_{start:04d}.pt"
        torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                        iteration=start), pre_path)
        baseline = evaluate(task, None, "fixed40", seed)
        before = evaluate(task, policy, "before", seed)
        policy.train()
    for iteration in range(start + 1, start + iterations + 1):
        if compare:
            # Equal coverage at each update; draw independent exploratory actions.
            goals = [goal for goal in task.targets_mm for _ in range(2)]
            order = torch.randperm(len(goals), generator=task.rng).tolist()
            ppo_iteration(task, policy, optimizer, len(goals), [goals[i] for i in order])
        else:
            ppo_iteration(task, policy, optimizer, samples=4)
        path = directory / f"stage1_{iteration:04d}.pt"
        torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                        iteration=iteration), path)
        saved = torch.load(path, map_location=task.env.device, weights_only=True)
        restored = TrajectoryPolicy().to(task.env.device)
        restored.load_state_dict(saved["policy"])
        resumed = torch.optim.Adam(restored.parameters())
        resumed.load_state_dict(saved["optimizer"])
        goals = torch.tensor([[-1.], [0.], [1.]], device=task.env.device)
        with torch.no_grad():
            if not all(torch.equal(a, b) for a, b in (
                    (policy.actor(goals), restored.actor(goals)),
                    (policy.critic(goals), restored.critic(goals)),
                    (policy.log_std, restored.log_std))):
                raise RuntimeError("Reloaded policy differs on the three goals")
        policy, optimizer = restored, resumed
        print(f"PASS: PPO update and checkpoint reload {path}", flush=True)
    if compare:
        manifest = dict(before=pre_path.name, after=path.name)
        (directory / "comparison_checkpoints.json").write_text(json.dumps(manifest), encoding="utf8")
        after = evaluate(task, policy, "after", seed)
        report_comparison(baseline, before, after, directory / "comparison.json", seed)
