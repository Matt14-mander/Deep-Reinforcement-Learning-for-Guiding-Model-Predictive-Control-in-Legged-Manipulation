"""Matched complete episodes: pre-update, post-update, and rule controllers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher


def run(args):
    import torch
    from go2_feedback_ppo import FeedbackPolicy
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import Go2AugMPCEnv,Go2AugMPCEnvCfg

    class EvaluationEnv(Go2AugMPCEnv):
        def _get_rewards(self):
            reward=super()._get_rewards()
            # DirectRLEnv resets done rows after rewards: capture physical state here.
            self.evaluation_state=dict(state=self._state().clone(),loads=self._loads().clone(),
                drift=self.drift.clone(),failed=self.failed.clone(),arrived=self.arrived.clone(),
                schedules=[list(c.schedule) for c in self.clocks],
                decisions=[[dict(d) for d in c.decisions] for c in self.clocks])
            return reward

    cfg=Go2AugMPCEnvCfg()
    cfg.seed=37
    cfg.sim.device=args.device
    cfg.scene.num_envs=3*len(args.goals_mm)
    cfg.episode_length_s=args.episode_seconds
    if args.episode_seconds<40 or any(g<=0 for g in args.goals_mm):
        raise ValueError("Use positive goals and at least 40 seconds per full episode")
    args.output.mkdir(parents=True,exist_ok=False)
    env=EvaluationEnv(cfg)
    try:
        checkpoint=torch.load(args.checkpoint,map_location=env.device,weights_only=True)
        if checkpoint.get("observation_size")!=57 or checkpoint.get("action_size")!=4:
            raise ValueError("Incompatible continuous-feedback checkpoint")
        post=FeedbackPolicy().to(env.device)
        post.load_state_dict(checkpoint["policy"])
        torch.manual_seed(37)
        pre=FeedbackPolicy().to(env.device)
        initial_source="checkpoint_initial_policy"
        if "initial_policy" in checkpoint:
            pre.load_state_dict(checkpoint["initial_policy"])
        else:
            if checkpoint.get("iteration")!=1:
                raise ValueError("Legacy reconstruction only supports the original iteration-1 smoke checkpoint")
            rollout_path=args.checkpoint.parent/"rollout.pt"
            original=torch.load(rollout_path,map_location=env.device,weights_only=True)
            with torch.no_grad():
                obs=original["observations"].flatten(0,1)
                log=pre.log_prob(obs,original["latent"].flatten(0,1))
                if not torch.allclose(log,original["log_prob"].flatten(),atol=2e-5,rtol=0):
                    raise RuntimeError("Reconstructed initial policy likelihood mismatch")
                if not torch.allclose(pre.value(obs),original["values"].flatten(),atol=2e-5,rtol=0):
                    raise RuntimeError("Reconstructed initial critic mismatch")
            initial_source="seed37_verified_against_original_rollout"
        pre.eval()
        post.eval()
        obs,_=env.reset(seed=37)
        n=len(args.goals_mm)
        targets=torch.tensor(args.goals_mm*3,device=env.device)/1000
        ids=torch.arange(3*n,device=env.device)
        initial=env._state().clone()
        for arm in range(1,3):
            assert torch.allclose(initial[:n],initial[arm*n:(arm+1)*n],atol=2e-6,rtol=0),"Initial states differ"
        env.set_commands(ids,initial[:,0]+targets,args.speed_mps)
        obs=env._get_observations()["policy"]
        active=torch.ones(3*n,device=env.device,dtype=torch.bool)
        histories=[[] for _ in range(3*n)]
        records=[]
        names=("pre","post","rule")
        with torch.no_grad():
            for step in range(env.max_episode_length):
                action=torch.zeros(3*n,4,device=env.device)
                action[:n]=pre.act(obs[:n],True)[0]
                action[n:2*n]=post.act(obs[n:2*n],True)[0]
                remaining=env.commands[2*n:,0]-env._state()[2*n:,0]
                action[2*n:,0]=torch.where((env.commands[2*n:,1]>.0045)&(remaining>.03),1.,-1.)
                following,reward,terminated,truncated,info=env.step(action)
                snapshot=env.evaluation_state
                assert torch.isfinite(following["policy"]).all() and torch.isfinite(reward).all()
                for row in active.nonzero().flatten().tolist():
                    state=snapshot["state"][row]
                    assert torch.isfinite(state).all() and torch.isfinite(snapshot["loads"][row]).all()
                    history=histories[row]
                    history.append(dict(x=state[0].item(),speed=state[19:22].norm().item(),
                        vx=state[19].item(),spin=state[22:25].norm().item(),
                        joint=state[25:].abs().max().item(),load=snapshot["loads"][row].min().item(),
                        drift=snapshot["drift"][row].max().item()))
                    if not (terminated[row] or truncated[row]):
                        continue
                    active[row]=False
                    dx=state[0].item()-initial[row,0].item()
                    signed_error=1000*dx-args.goals_mm[row%n]
                    tail=history[-5:] # one full second between first and last samples
                    stop_span=max(v["x"] for v in tail)-min(v["x"] for v in tail)
                    stopped=(len(tail)==5 and stop_span<.001 and all(v["speed"]<.01 and
                        v["spin"]<.04 and v["joint"]<.1 and v["load"]>5 for v in tail))
                    within=abs(signed_error)<=1000*cfg.goal_tolerance_m
                    failed=bool(snapshot["failed"][row])
                    elapsed=(step+1)*env.step_dt
                    record=dict(controller=names[row//n],target_mm=args.goals_mm[row%n],
                        desired_speed_mps=args.speed_mps,dx_mm=1000*dx,
                        dy_mm=1000*(state[1]-initial[row,1]).item(),signed_error_mm=signed_error,
                        error_mm=abs(signed_error),within_goal=within,stopped=stopped,
                        success=within and stopped and not failed,failed=failed,
                        terminated=bool(terminated[row]),truncated=bool(truncated[row]),
                        elapsed_s=elapsed,arrival_time_s=elapsed if within and stopped and not failed else None,
                        mean_forward_speed_mps=dx/elapsed,
                        measured_vx_mean_mps=sum(v["vx"] for v in history)/len(history),
                        final_speed_mps=history[-1]["speed"],stop_span_mm=1000*stop_span,
                        max_stance_drift_mm=1000*max(v["drift"] for v in history),
                        final_min_foot_load_N=history[-1]["load"],
                        cycles=len(snapshot["schedules"][row]),decisions=snapshot["decisions"][row])
                    records.append(record)
                    print(json.dumps(dict(stage="episode",**record)),flush=True)
                obs=following["policy"]
                if not active.any():
                    break
                if (step+1)%40==0:
                    print(json.dumps(dict(stage="progress",sim_s=(step+1)*env.step_dt,remaining=int(active.sum()))),flush=True)
        assert len(records)==3*n and not active.any(),"Incomplete episodes"
        summary={}
        for name in names:
            subset=[r for r in records if r["controller"]==name]
            summary[name]=dict(episodes=n,arrival_rate=sum(r["success"] for r in subset)/n,
                stopped_rate=sum(r["stopped"] for r in subset)/n,
                mean_error_mm=sum(r["error_mm"] for r in subset)/n,
                mean_forward_speed_mps=sum(r["mean_forward_speed_mps"] for r in subset)/n,
                max_stance_drift_mm=max(r["max_stance_drift_mm"] for r in subset),
                failures=sum(r["failed"] for r in subset))
        comparisons=[]
        for goal in args.goals_mm:
            by={r["controller"]:r for r in records if r["target_mm"]==goal}
            comparisons.append(dict(target_mm=goal,pre_error_mm=by["pre"]["error_mm"],
                post_error_mm=by["post"]["error_mm"],rule_error_mm=by["rule"]["error_mm"],
                improvement_mm=by["pre"]["error_mm"]-by["post"]["error_mm"]))
        report=dict(evaluation_completed=True,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            initial_policy_source=initial_source,seed=37,episode_limit_s=args.episode_seconds,
            policy_sampling=False,discrete_choice="argmax",continuous_choice="tanh_mean",
            initial_states_matched=True,summary=summary,comparisons=comparisons,episodes=records,
            policy_all_arrived=summary["post"]["arrival_rate"]==1.,
            improvement_demonstrated=all(c["improvement_mm"]>.1 for c in comparisons))
        (args.output/"comparison.json").write_text(json.dumps(report,indent=2))
        (args.output/"traces.json").write_text(json.dumps(histories))
        print(json.dumps(dict(stage="comparison",summary=summary,comparisons=comparisons,
            policy_all_arrived=report["policy_all_arrived"],improvement_demonstrated=report["improvement_demonstrated"])),flush=True)
        print("COMPLETE: matched full-episode evaluation (performance reported separately)",flush=True)
    finally:
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--goals-mm",type=int,nargs="+",default=[32,40,48])
    parser.add_argument("--speed-mps",type=float,default=.006)
    parser.add_argument("--episode-seconds",type=float,default=60.)
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
