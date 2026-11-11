"""Iteration-boundary checkpoints and deterministic Stage 2 comparison reports."""
import copy
import json
from pathlib import Path
from statistics import mean

import torch


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sampling_state(noise, goal_rng, bags):
    return dict(torch=torch.get_rng_state(), noise=[g.get_state() for g in noise],
                goals=[g.getstate() for g in goal_rng], bags=copy.deepcopy(bags))


def restore_sampling(state, noise, goal_rng, bags):
    if not (len(state["noise"]) == len(state["goals"]) == len(state["bags"]) == len(noise)):
        raise ValueError("Checkpoint environment count does not match")
    torch.set_rng_state(state["torch"])
    for generator, saved in zip(noise, state["noise"]):
        generator.set_state(saved)
    for generator, saved in zip(goal_rng, state["goals"]):
        generator.setstate(saved)
    bags[:] = copy.deepcopy(state["bags"])


def save_checkpoint(path, policy, optimizer, iteration, task, noise, goal_rng, bags):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                    iteration=iteration, task=task, seed=task["seed"],
                    speed_task=task["speed_task"], period_actions=task["period_actions"],
                    sampling=sampling_state(noise, goal_rng, bags)), temporary)
    temporary.replace(path)


def restore_checkpoint(path, policy, optimizer, task, noise, goal_rng, bags):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    iteration = checkpoint["iteration"]
    if type(iteration) is not int or iteration < 1:
        raise ValueError("Checkpoint must contain a positive completed iteration")
    for key in ("speed_task", "period_actions"):
        if checkpoint.get(key, False) != task[key]:
            raise ValueError(f"Resume task differs in {key}; use --checkpoint for a policy warm start")
    if "task" in checkpoint and checkpoint["task"] != task:
        raise ValueError("Checkpoint task/seed/environment configuration differs")
    policy.load_state_dict(checkpoint["policy"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "sampling" in checkpoint:
        restore_sampling(checkpoint["sampling"], noise, goal_rng, bags)
        mode = "restored"
    else:
        # Legacy checkpoints saved policy and Adam but omitted the sampling streams.
        # Start distinct streams and explicitly record this limitation.
        for row, generator in enumerate(noise):
            generator.manual_seed(task["seed"] + 499*row + 1 + 1000003*iteration)
        for row, generator in enumerate(goal_rng):
            generator.seed(task["seed"] + 1019*row + 1000003*iteration)
        bags[:] = [[] for _ in noise]
        mode = "legacy_reseeded"
    return iteration, mode


def period_probes(policy):
    result = []
    with torch.no_grad():
        for remaining in (64., 40., 20.):
            for previous_gait in (0., 1.):
                for previous_slot in (0, 1, 2):
                    choices = []
                    for speed in (3., 6.):
                        obs = torch.tensor([(remaining-30)/15, previous_gait,
                                            (speed-6)/3, previous_slot-1.])
                        probabilities = policy.period_distribution(obs).probs.tolist()
                        choices.append(dict(desired_mm_s=speed, slot=int(max(range(3), key=probabilities.__getitem__)),
                                            probabilities=probabilities))
                    result.append(dict(remaining_mm=remaining, previous_gait=previous_gait,
                                       previous_slot=previous_slot, choices=choices,
                                       speed_changes_slot=choices[0]["slot"] != choices[1]["slot"]))
    return result


def compare_evaluation(learned, baseline, iteration, speed_task, probes):
    index = {m["identity"][1]: m for m in baseline}
    if len(index) != 8 or len(learned) != 8 or len({m["identity"][1] for m in learned}) != 8:
        raise RuntimeError("Evaluation must contain eight unique rows per controller")
    pairs = []
    for m in sorted(learned, key=lambda item: item["identity"][1]):
        b = index[m["identity"][1]]
        if b["target_mm"] != m["target_mm"] or b.get("desired_mm_s") != m.get("desired_mm_s"):
            raise RuntimeError("Mismatched paired evaluation conditions")
        pairs.append(dict(row=m["identity"][1], target_mm=m["target_mm"], policy=m, rule=b))
    passed = all(pair[mode]["success"] for pair in pairs for mode in ("policy", "rule"))
    groups = []
    if speed_task:
        for mode in ("policy", "rule"):
            for speed in (3., 6.):
                group = [pair[mode] for pair in pairs if pair[mode]["desired_mm_s"] == speed]
                switches = [s for m in group for s in m["switches"]]
                slots = [[d.get("period_slot", 1) for d in m["decisions"]] for m in group]
                valid = len(group) == 4 and all(m["speed_ok"] for m in group)
                # A constant gait can be valid after period control is enabled.
                # Every actual switch is still checked by the physical handoff gate.
                passed = passed and valid
                groups.append(dict(mode=mode, desired_mm_s=speed,
                    measured_mm_s=mean(m["walking_mm_s"] for m in group),
                    speed_error_mm_s=mean(abs(m["walking_mm_s"]-speed) for m in group),
                    arrival_s=[m["arrival_s"] for m in group],
                    endpoint_errors_mm=[m["final_x_error_mm"] for m in group],
                    endpoint_abs_error_mm=mean(abs(m["final_x_error_mm"]) for m in group),
                    gaits=[[c["gait"] for c in m["cycles"]] for m in group],
                    time_scales=[[c["time_scale"] for c in m["cycles"]] for m in group],
                    period_slots=slots,
                    period_counts=[sum(row.count(i) for row in slots) for i in range(3)],
                    switch_count=len(switches),
                    max_anchor_jump_mm=max((s["anchor_jump_mm"] for s in switches), default=0.),
                    max_torque_step_Nm=max((max(s["requested_step_Nm"],s["applied_step_Nm"]) for s in switches), default=0.),
                    max_rotation_step_deg=max((s["rotation_step_deg"] for s in switches), default=0.),
                    max_stance_mm=max(max(m["max_stance_mm"]) for m in group),
                    physical_passed=all(m["success"] for m in group), speed_passed=valid))
            slow, fast = groups[-2:]
            times = slow["arrival_s"] + fast["arrival_s"]
            faster = None not in times and max(fast["arrival_s"]) < min(slow["arrival_s"])
            fast["faster_arrival_than_slow"] = faster
            passed = passed and faster
    nondefault = any(d.get("period_slot", 1) != 1 for m in learned for d in m["decisions"])
    conditioned = any(p["speed_changes_slot"] for p in probes)
    return dict(iteration=iteration, pairs=pairs, groups=groups, passed=passed,
                period_probes=probes, nondefault_period_used=nondefault,
                speed_conditioned_period_choice=conditioned,
                period_adaptation_observed=nondefault and conditioned)


def learning_summary(evaluations, requested_iterations=()):
    milestones = sorted(evaluations, key=lambda m: m["iteration"])
    baseline = milestones[0]
    summaries = []
    for report in milestones:
        comparisons = []
        for group in report["groups"]:
            if group["mode"] != "policy":
                continue
            rule = next(g for g in report["groups"] if g["mode"] == "rule" and g["desired_mm_s"] == group["desired_mm_s"])
            initial = next(g for g in baseline["groups"] if g["mode"] == "policy" and g["desired_mm_s"] == group["desired_mm_s"])
            arrival = None if None in group["arrival_s"]+rule["arrival_s"] else mean(group["arrival_s"])-mean(rule["arrival_s"])
            comparisons.append(dict(desired_mm_s=group["desired_mm_s"],
                period_counts=group["period_counts"], measured_mm_s=group["measured_mm_s"],
                speed_error_change_from_first_mm_s=group["speed_error_mm_s"]-initial["speed_error_mm_s"],
                speed_error_minus_rule_mm_s=group["speed_error_mm_s"]-rule["speed_error_mm_s"],
                arrival_minus_rule_s=arrival, endpoint_abs_error_mm=group["endpoint_abs_error_mm"],
                endpoint_error_minus_rule_mm=group["endpoint_abs_error_mm"]-rule["endpoint_abs_error_mm"],
                max_anchor_jump_mm=group["max_anchor_jump_mm"], max_torque_step_Nm=group["max_torque_step_Nm"],
                max_rotation_step_deg=group["max_rotation_step_deg"], max_stance_mm=group["max_stance_mm"]))
        summaries.append(dict(iteration=report["iteration"], passed=report["passed"],
            period_adaptation_observed=report["period_adaptation_observed"], comparisons=comparisons))
    observed = {m["iteration"] for m in milestones}
    missing = sorted(i for i in set(requested_iterations) if i <= milestones[-1]["iteration"] and i not in observed)
    return dict(reference_iteration=baseline["iteration"], milestones=summaries,
                missing_milestones=missing, milestone_coverage_complete=not missing,
                all_evaluations_passed=all(m["passed"] for m in milestones),
                final_period_adaptation_observed=milestones[-1]["period_adaptation_observed"])
