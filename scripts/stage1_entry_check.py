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


def ppo_iteration(task, policy, optimizer, samples):
    # Every decision ends an episode, so the return is its measured terminal reward.
    observations, latents, old_logps, returns = [], [], [], []
    completed = 0
    for _ in range(samples):
        obs = task.reset()
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


def train_stage1(task, iterations, checkpoint_dir):
    policy = TrajectoryPolicy().to(task.env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, iterations + 1):
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
