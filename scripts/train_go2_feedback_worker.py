"""Continuous on-policy updates within one PhysX process; checkpoint every update."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from isaaclab.app import AppLauncher


def run(args):
    import torch
    from go2_feedback_ppo import FeedbackPolicy,update
    from go2_feedback_training import load_checkpoint,save_checkpoint,restore_rng,validate_task
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import Go2AugMPCEnv,Go2AugMPCEnvCfg
    from RL_Bezier_MPC.envs.go2_feedback import generalized_advantage
    state=load_checkpoint(args.checkpoint)
    if args.until<=state["iteration"] or args.rollout_steps<2:
        raise ValueError("Worker needs at least one new update and two rollout steps")
    task=dict(environments=8,goals_mm=[32,40,48],speed_mps=.006,episode_s=60.,
        rollout_steps=args.rollout_steps,epochs=4,gamma=.99,gae_lambda=.95,policy_hz=4.)
    validate_task(state,task)
    if any((args.output/f"feedback_{i:04d}.pt").exists() for i in range(state["iteration"]+1,args.until+1)):
        raise ValueError("Refusing to overwrite existing update checkpoints")
    goal_rng=torch.Generator(device="cpu").manual_seed(38)
    if "goal_rng" in state:
        goal_rng.set_state(state["goal_rng"].cpu())

    class TrainingEnv(Go2AugMPCEnv):
        def _reset_idx(self,ids):
            super()._reset_idx(ids)
            if not self.feedback_ready or len(ids)==0:
                return
            choices=torch.randint(3,(len(ids),),generator=goal_rng)
            goals=torch.tensor(task["goals_mm"],device=self.device)[choices.to(self.device)]/1000
            self.commands[ids,0]=goals
            self.commands[ids,1]=task["speed_mps"]

    cfg=Go2AugMPCEnvCfg()
    cfg.seed=37
    cfg.sim.device=args.device
    cfg.episode_length_s=task["episode_s"]
    env=TrainingEnv(cfg)
    try:
        obs,_=env.reset(seed=37)
        obs=obs["policy"]
        policy=FeedbackPolicy().to(env.device)
        policy.load_state_dict(state["policy"])
        optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state)
        args.output.mkdir(parents=True,exist_ok=True)
        print(json.dumps(dict(stage="resume",completed_iteration=state["iteration"],until=args.until,
            optimizer_restored=True,environment_reset=True,training_task=task)),flush=True)
        for iteration in range(state["iteration"]+1,args.until+1):
            started=time.perf_counter()
            rows=[]
            completed=successes=failures=timeouts=0
            for step in range(args.rollout_steps):
                identity=torch.stack((torch.arange(8,device=env.device),env.ledger.episode,env.ledger.decision),-1).clone()
                with torch.no_grad():
                    action,latent,log,value=policy.act(obs)
                    following,reward,term,trunc,info=env.step(action)
                    t=info["transition"]
                    next_value=policy.value(t["bootstrap_observation"])
                    assert torch.equal(identity,t["identity"])
                    assert torch.equal(action,t["actions"]) and torch.equal(reward,t["reward"])
                    assert torch.equal(t["bootstrap_mask"],~term)
                    assert torch.equal(t["trace_mask"],~(term|trunc))
                    for v in (obs,action,log,value,reward,next_value,t["mpc_health"]):
                        assert torch.isfinite(v).all()
                    rows.append(tuple(v.clone() for v in (obs,latent,log,value,reward,next_value,term,trunc)))
                    completed+=int((term|trunc).sum())
                    successes+=int(t["terminated_success"].sum())
                    failures+=int(t["terminated_failure"].sum())
                    timeouts+=int(trunc.sum())
                    obs=following["policy"]
                if (step+1)%24==0:
                    print(json.dumps(dict(stage="sampling",iteration=iteration,steps=step+1,
                        completed=completed,failures=failures)),flush=True)
            observations,latent,log,values,rewards,next_values,term,trunc=[torch.stack([r[i] for r in rows]) for i in range(8)]
            advantage,returns=generalized_advantage(rewards,values,next_values,term,trunc)
            stats=update(policy,optimizer,(observations.flatten(0,1),latent.flatten(0,1),log.flatten(),
                advantage.flatten(),returns.flatten()))
            assert all(torch.isfinite(p).all() for p in policy.parameters())
            checkpoint=args.output/f"feedback_{iteration:04d}.pt"
            saved=dict(format_version=2,iteration=iteration,observation_size=57,action_size=4,
                initial_policy=state["initial_policy"],policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                seed=37,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                goal_rng=goal_rng.get_state(),training_task=task,policy_hz=4.,mpc_hz=50.)
            save_checkpoint(checkpoint,saved)
            elapsed=time.perf_counter()-started
            print(json.dumps(dict(stage="ppo_update",iteration=iteration,samples=8*args.rollout_steps,
                completed=completed,successes=successes,failures=failures,timeouts=timeouts,
                mean_reward=rewards.mean().item(),wall_s=elapsed,transitions_per_s=8*args.rollout_steps/elapsed,
                checkpoint=str(checkpoint),**stats)),flush=True)
        print("COMPLETE: feedback PPO training segment",flush=True)
    finally:
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--until",type=int,required=True)
    parser.add_argument("--rollout-steps",type=int,default=96)
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
