"""Pure learning-signal summaries; no changes to the training objective."""
import torch


def reward_parts(progress,velocity,desired,effort,action_change,failed,arrived,dt):
    return torch.stack((100*progress,dt*torch.exp(-((velocity-desired)/.01).square()),
        -.001*dt*effort.square().mean(-1),-.02*action_change.square().mean(-1),
        -failed.float(),arrived.float()),-1)


PARTS=("progress","speed","effort","action_change","failure","arrival")


def advantage_summary(adv,mask):
    selected=adv[mask]
    if selected.numel()==0:
        return dict(count=0,mean=None,positive_fraction=None)
    return dict(count=selected.numel(),mean=selected.mean().item(),
                positive_fraction=(selected>0).float().mean().item())


def summarize(batch,events):
    advantage=batch["advantage"]
    normalized=batch.get("normalized_advantage")
    if normalized is None:
        normalized=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
    applied=torch.zeros_like(advantage,dtype=torch.bool)
    for event in events:
        step,row=event["step"],event["env"]
        assert not applied[step,row]
        assert batch["identity"][step,row].tolist()==event["identity"]
        applied[step,row]=True
        event["advantage"]=advantage[step,row].item()
        event["normalized_advantage"]=normalized[step,row].item()
    groups={}
    for gait,name in enumerate(("crawl","trot")):
        mask=applied & (batch["actions"][:,:,0]>0 if gait else batch["actions"][:,:,0]<=0)
        groups[name]=dict(raw=advantage_summary(advantage,mask),normalized=advantage_summary(normalized,mask),
            period_counts=[int((mask&(batch["actions"][:,:,2]==slot-1)).sum()) for slot in range(3)],
            period_advantages=[advantage_summary(normalized,mask&(batch["actions"][:,:,2]==slot-1)) for slot in range(3)])
    # Reward is labelled only for transitions whose MPC ticks all have one mode.
    rewards={}
    for mode,name in enumerate(("settle","crawl","trot","hold")):
        counts=batch["mode_counts"]
        mask=(counts[:,:,mode]>0)&(counts[:,:,mode]==counts.sum(-1))
        rewards[name]=dict(transitions=int(mask.sum()),
            mean_components={part:batch["parts"][:,:,j][mask].mean().item() if mask.any() else None for j,part in enumerate(PARTS)})
    mixed=((batch["mode_counts"]>0).sum(-1)>1)
    return dict(transitions=advantage.numel(),effective_cycle_decisions=int(applied.sum()),
        effective_fraction=applied.float().mean().item(),effective_by_gait=groups,
        unapplied_categorical_change_penalty_total=batch["categorical_change_cost"][~applied].sum().item(),
        applied_categorical_change_penalty_total=batch["categorical_change_cost"][applied].sum().item(),
        unapplied_categorical_requests=advantage_summary(normalized,~applied),
        submitted_trot_count=int((batch["actions"][:,:,0]>0).sum()),
        mixed_mode_transitions=int(mixed.sum()),reward_by_actual_mode=rewards,
        mean_boundary_trot_probability=sum(e["gait_probability"][1] for e in events)/len(events) if events else None,
        gae_weight_one_crawl_cycle=(.99*.95)**(6.9/.25),
        stationary_speed_reward_per_second=float(torch.exp(torch.tensor(-(.006/.01)**2))))
