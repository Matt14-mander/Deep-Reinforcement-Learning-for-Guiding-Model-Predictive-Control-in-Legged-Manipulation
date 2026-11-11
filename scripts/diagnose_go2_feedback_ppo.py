"""Frozen-policy exploration/advantage diagnostics and matched gait reward experiments."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher


def run(args):
    import torch
    from go2_feedback_ppo import FeedbackPolicy
    from go2_feedback_training import load_checkpoint,save_json
    from go2_learning_signal import reward_parts,PARTS,summarize
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import Go2AugMPCEnv,Go2AugMPCEnvCfg
    from RL_Bezier_MPC.envs.go2_feedback import generalized_advantage
    if args.mode=="sample" and (not args.checkpoints or args.steps<8):
        raise ValueError("Sampling needs checkpoints and at least eight steps")
    args.output.mkdir(parents=True,exist_ok=False)

    class DiagnosticEnv(Go2AugMPCEnv):
        def _reset_idx(self,ids):
            super()._reset_idx(ids)
            if self.feedback_ready and len(ids):
                self.commands[ids,0]=torch.tensor([.032,.040,.048],device=self.device)[ids%3]
                self.commands[ids,1]=.006

        def _pre_physics_step(self,actions):
            super()._pre_physics_step(actions)
            self.boundary_events=[]
            self.mode_counts=torch.zeros(self.num_envs,4,device=self.device,dtype=torch.long)

        def _choose_cycle(self,clock,obs,remaining):
            chosen=super()._choose_cycle(clock,obs,remaining)
            row=clock.row_id
            self.boundary_events.append(dict(env=row,identity=self.transition_identity[row].tolist(),
                world_tick=self.world,remaining_mm=remaining,gait=chosen[0],period_slot=chosen[2]))
            return chosen

        def _solve_mpc_control_step(self):
            super()._solve_mpc_control_step()
            for row,clock in enumerate(self.clocks):
                mode=3 if clock.finishing else (0 if not clock.schedule else (1 if clock.schedule[-1][0]=="crawl" else 2))
                self.mode_counts[row,mode]+=1

        def _get_rewards(self):
            reward=super()._get_rewards()
            state=self._state()
            distance=(self.commands[:,0]-state[:,0]).abs().float()
            desired=torch.where(distance<=self.cfg.goal_tolerance_m,0.,self.commands[:,1])
            parts=reward_parts(self.start_distance-distance,state[:,19].float(),desired,
                self._pending_joint_efforts,self.actions-self.previous_actions,self.failed,self.arrived,self.step_dt)
            assert torch.allclose(parts.sum(-1),reward,atol=2e-5,rtol=0)
            self.diag=dict(categorical_change_cost=-.02*(self.actions-self.previous_actions)[:,[0,2]].square().sum(-1)/4,parts=parts.clone(),events=list(self.boundary_events),mode_counts=self.mode_counts.clone(),
                state=state.clone(),failed=self.failed.clone(),arrived=self.arrived.clone(),drift=self.drift.clone())
            return reward

    cfg=Go2AugMPCEnvCfg()
    cfg.seed=args.seed
    cfg.sim.device=args.device
    cfg.episode_length_s=60.
    cfg.scene.num_envs=8 if args.mode=="sample" else 6
    env=DiagnosticEnv(cfg)
    try:
        obs,_=env.reset(seed=args.seed)
        obs=obs["policy"]
        if args.mode=="paired":
            initial=env._state().clone()
            assert torch.allclose(initial[:3],initial[3:],atol=2e-6,rtol=0)
            active=torch.ones(6,device=env.device,dtype=torch.bool)
            total=torch.zeros(6,6,device=env.device)
            discounted=torch.zeros_like(total)
            records=[]
            events=[]
            with torch.no_grad():
                for step in range(env.max_episode_length):
                    actions=torch.zeros(6,4,device=env.device)
                    actions[:,0]=-1
                    remaining=env.commands[3:,0]-env._state()[3:,0]
                    actions[3:,0]=torch.where(remaining>.03,1.,-1.)
                    following,reward,term,trunc,info=env.step(actions)
                    d=env.diag
                    total[active]+=d["parts"][active]
                    discounted[active]+=(.99**step)*d["parts"][active]
                    events.extend([dict(step=step,**e) for e in d["events"] if active[e["env"]]])
                    for row in (active&(term|trunc)).nonzero().flatten().tolist():
                        active[row]=False
                        records.append(dict(controller="crawl_only" if row<3 else "trot_then_crawl",
                            target_mm=[32,40,48][row%3],elapsed_s=(step+1)*.25,
                            dx_mm=1000*(d["state"][row,0]-initial[row,0]).item(),
                            arrived=bool(d["arrived"][row]),failed=bool(d["failed"][row]),truncated=bool(trunc[row]),
                            max_drift_mm=1000*d["drift"][row].max().item(),
                            components=dict(zip(PARTS,total[row].tolist())),
                            discounted_components=dict(zip(PARTS,discounted[row].tolist())),
                            total_reward=total[row].sum().item(),discounted_return=discounted[row].sum().item()))
                    if not active.any():
                        break
                    if (step+1)%40==0:
                        print(json.dumps(dict(stage="paired_progress",sim_s=(step+1)*.25,remaining=int(active.sum()))),flush=True)
            gaps=[]
            for goal in [32,40,48]:
                a=next(r for r in records if r["target_mm"]==goal and r["controller"]=="crawl_only")
                b=next(r for r in records if r["target_mm"]==goal and r["controller"]=="trot_then_crawl")
                gaps.append(dict(target_mm=goal,crawl_minus_trot_reward=a["total_reward"]-b["total_reward"],
                    crawl_minus_trot_discounted_return=a["discounted_return"]-b["discounted_return"],
                    crawl_extra_seconds=a["elapsed_s"]-b["elapsed_s"]))
            assert len(records)==6
            report=dict(mode="matched_rule_counterfactual",seed=args.seed,episodes=records,return_gaps=gaps,
                note="Matched initial states and goals; gait choice changes subsequent states. Not historical PPO advantages.")
            save_json(args.output/"events.json",events)
        else:
            policies={}
            checkpoint_tasks={}
            for path in args.checkpoints:
                checkpoint=load_checkpoint(path)
                policy=FeedbackPolicy().to(env.device)
                policy.load_state_dict(checkpoint["policy"])
                policy.eval()
                policies[checkpoint["iteration"]]=policy
                checkpoint_tasks[checkpoint["iteration"]]=checkpoint.get("training_task",{})
            latest=max(policies)
            policy=policies[latest]
            torch.manual_seed(args.seed)
            rows=[]
            events=[]
            with torch.no_grad():
                for step in range(args.steps):
                    gd,pd,_=policy.distributions(obs)
                    action,latent,log,value=policy.act(obs)
                    following,reward,term,trunc,info=env.step(action)
                    t=info["transition"]
                    next_value=policy.value(t["bootstrap_observation"])
                    d=env.diag
                    for e in d["events"]:
                        row=e["env"]
                        assert e["identity"]==t["identity"][row].tolist()
                        assert (e["gait"]=="trot")==bool(action[row,0]>0)
                        assert e["period_slot"]==int(action[row,2])+1
                        event=dict(step=step,**e,gait_probability=gd.probs[row].tolist(),period_probability=pd.probs[row].tolist(),
                            fixed_observation_probability={})
                        for iteration,probe in policies.items():
                            g,p,_=probe.distributions(obs[row:row+1])
                            event["fixed_observation_probability"][str(iteration)]=dict(gait=g.probs[0].tolist(),period=p.probs[0].tolist())
                        events.append(event)
                    rows.append(tuple(x.clone() for x in (obs,action,latent,log,value,reward,next_value,term,trunc,
                        t["identity"],d["parts"],d["mode_counts"],gd.probs,pd.probs,d["categorical_change_cost"])))
                    obs=following["policy"]
                    if (step+1)%24==0:
                        print(json.dumps(dict(stage="sample_progress",steps=step+1,boundaries=len(events))),flush=True)
            keys=["observations","actions","latent","log_prob","values","reward","next_values","terminated","truncated",
                "identity","parts","mode_counts","gait_probs","period_probs","categorical_change_cost"]
            batch=dict(zip(keys,[torch.stack([r[i] for r in rows]) for i in range(len(keys))]))
            for value in batch.values():
                assert torch.isfinite(value).all()
            block_steps=checkpoint_tasks[latest].get("rollout_steps",96)
            batch["advantage"]=torch.empty_like(batch["reward"])
            batch["returns"]=torch.empty_like(batch["reward"])
            batch["normalized_advantage"]=torch.empty_like(batch["reward"])
            for start in range(0,args.steps,block_steps):
                block=slice(start,min(start+block_steps,args.steps))
                a,r=generalized_advantage(batch["reward"][block],batch["values"][block],batch["next_values"][block],batch["terminated"][block],batch["truncated"][block])
                batch["advantage"][block]=a
                batch["returns"][block]=r
                batch["normalized_advantage"][block]=(a-a.mean())/a.std(unbiased=False).clamp_min(1e-6)
            report=dict(mode="fresh_frozen_checkpoint_rollout",historical_replay=False,seed=args.seed,
                checkpoint_iteration=latest,probe_iterations=sorted(policies),rollout_steps=args.steps,advantage_block_steps=block_steps,
                target_goals_mm=[32,40,48],goal_assignment="env_id_mod_3",desired_speed_mps=.006,
                probe_observations="same request-submission observations at latest-policy effective boundaries",**summarize(batch,events))
            report["boundary_probability_by_checkpoint"]={str(i):dict(
                mean_trot_probability=sum(e["fixed_observation_probability"][str(i)]["gait"][1] for e in events)/len(events),
                mean_period_probability=[sum(e["fixed_observation_probability"][str(i)]["period"][j] for e in events)/len(events) for j in range(3)]) for i in policies} if events else {}
            torch.save({k:v.cpu() for k,v in batch.items()},args.output/"transitions.pt")
            save_json(args.output/"events.json",events)
        save_json(args.output/"summary.json",report)
        print(json.dumps(report),flush=True)
        print("COMPLETE: learning signal diagnosis; no policy update",flush=True)
    finally:
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["sample","paired"],required=True)
    parser.add_argument("--checkpoints",type=Path,nargs="*")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=192)
    parser.add_argument("--seed",type=int,default=37)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    launcher=AppLauncher(args)
    passed=False
    try:
        run(args)
        passed=True
    except Exception:
        traceback.print_exc()
    finally:
        launcher.app.close()
    if not passed:
        raise SystemExit(1)
