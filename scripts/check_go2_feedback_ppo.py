"""Eight-environment rollout, PPO update and exact checkpoint reload acceptance."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher


def run(args):
    import torch
    import gymnasium as gym
    from go2_feedback_ppo import FeedbackPolicy,update
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import Go2AugMPCEnvCfg
    from RL_Bezier_MPC.envs.go2_feedback import generalized_advantage
    cfg=Go2AugMPCEnvCfg()
    cfg.seed=37
    cfg.sim.device=args.device
    if args.steps<84:
        raise ValueError("At least 84 steps required to exercise the 20-second timeout")
    args.output.mkdir(parents=True,exist_ok=False)
    env=gym.make("Isaac-Go2-AugMPC-Direct-v0",cfg=cfg).unwrapped
    try:
        obs,_=env.reset(seed=37)
        env.set_commands([0],0.,0.)
        env.set_commands([6],.5,.003)
        obs=env._get_observations()["policy"]
        torch.manual_seed(37)
        policy=FeedbackPolicy().to(env.device)
        optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
        rows=[]
        natural=truncations=failures=0
        for step in range(args.steps):
            identity=torch.stack((torch.arange(8,device=env.device),env.ledger.episode,env.ledger.decision),-1).clone()
            with torch.no_grad():
                action,latent,log,value=policy.act(obs)
                following,reward,terminated,truncated,info=env.step(action)
                t=info["transition"]
                final=t["bootstrap_observation"]
                next_value=policy.value(final)
                assert torch.equal(identity,t["identity"])
                assert torch.equal(action,t["actions"]),"Executed action mismatch"
                assert torch.equal(reward,t["reward"])
                assert torch.equal(t["bootstrap_mask"],~terminated)
                assert torch.equal(t["trace_mask"],~(terminated|truncated))
                assert t["mpc_ticks"]==(13 if step%2==0 else 12)
                for item in (obs,action,latent,log,value,reward,next_value,final,t["mpc_health"]):
                    assert torch.isfinite(item).all()
                rows.append(tuple(x.clone() for x in (obs,latent,log,value,reward,next_value,terminated,truncated,identity,action,final)))
                natural+=int(terminated.sum())
                truncations+=int(truncated.sum())
                failures+=int(t["terminated_failure"].sum())
                obs=following["policy"]
            if (step+1)%20==0:
                print(json.dumps(dict(stage="rollout",steps=step+1,natural=natural,truncated=truncations,failures=failures)),flush=True)
        arrays=[torch.stack([row[i] for row in rows]) for i in range(11)]
        observations,latent,logs,values,rewards,next_values,terminated,truncated,identities,actions,finals=arrays
        assert natural>0 and truncations>0 and failures==0
        advantage,returns=generalized_advantage(rewards,values,next_values,terminated,truncated)
        # Independent per-environment scalar recursion checks episode-boundary handling.
        for row in range(8):
            carry=0.
            for tick in reversed(range(args.steps)):
                delta=rewards[tick,row]+.99*(not terminated[tick,row])*next_values[tick,row]-values[tick,row]
                carry=delta+.99*.95*(not (terminated[tick,row] or truncated[tick,row]))*carry
                assert torch.allclose(advantage[tick,row],carry,atol=1e-5)
        batch=(observations.flatten(0,1),latent.flatten(0,1),logs.flatten(),advantage.flatten(),returns.flatten())
        with torch.no_grad():
            before=policy.act(batch[0],True)[0].clone()
        stats=update(policy,optimizer,batch)
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        checkpoint=args.output/"feedback_0001.pt"
        state=dict(format_version=1,iteration=1,observation_size=57,action_size=4,
                   policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                   rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   policy_hz=4.,mpc_hz=50.)
        torch.save(state,checkpoint)
        loaded=torch.load(checkpoint,map_location=env.device,weights_only=True)
        restored=FeedbackPolicy().to(env.device)
        restored.load_state_dict(loaded["policy"])
        restored_optimizer=torch.optim.Adam(restored.parameters(),lr=3e-4)
        restored_optimizer.load_state_dict(loaded["optimizer"])
        with torch.no_grad():
            deterministic=policy.act(batch[0],True)
            replay=restored.act(batch[0],True)
            assert all(torch.equal(a,b) for a,b in zip(deterministic,replay))
            action_delta=(deterministic[0]-before).abs().max().item()
        # Identical next update proves Adam moments/step counters were restored.
        probe=(batch[0],batch[1],policy.log_prob(batch[0],batch[1]).detach(),batch[3],batch[4])
        update(policy,optimizer,probe,epochs=1)
        update(restored,restored_optimizer,probe,epochs=1)
        assert all(torch.equal(a,b) for a,b in zip(policy.parameters(),restored.parameters()))
        # Checkpoint remains the single-update model, not the optimizer test probe.
        restored.load_state_dict(loaded["policy"])
        torch.set_rng_state(loaded["rng_cpu"].cpu())
        if loaded["rng_cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in loaded["rng_cuda"]])
        with torch.no_grad():
            sample=restored.act(obs)[0]
        torch.set_rng_state(loaded["rng_cpu"].cpu())
        if loaded["rng_cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in loaded["rng_cuda"]])
        with torch.no_grad():
            assert torch.equal(sample,restored.act(obs)[0])
            # Exercise the reloaded deterministic policy on real PhysX transitions.
            for _ in range(4):
                action=restored.act(obs,True)[0]
                following,reward,term,trunc,info=env.step(action)
                assert not info["transition"]["terminated_failure"].any()
                assert torch.isfinite(following["policy"]).all() and torch.isfinite(reward).all()
                obs=following["policy"]
        torch.save(dict(observations=observations,latent=latent,actions=actions,log_prob=logs,
            rewards=rewards,values=values,next_values=next_values,terminated=terminated,
            truncated=truncated,identities=identities,terminal_observations=finals,
            advantages=advantage,returns=returns),args.output/"rollout.pt")
        report=dict(passed=True,iteration=1,environments=8,steps=args.steps,samples=args.steps*8,
            simulation_s=args.steps*.25,natural_terminations=natural,time_truncations=truncations,
            failures=failures,deterministic_action_delta_max=action_delta,checkpoint=str(checkpoint),
            reload_exact=True,optimizer_resume_exact=True,rng_resume_exact=True,
            reloaded_physx_steps=4,bootstrap_checked=True,**stats)
        (args.output/"acceptance.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        print("PASS: eight Go2 feedback PPO rollout update save reload",flush=True)
    finally:
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=96)
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
