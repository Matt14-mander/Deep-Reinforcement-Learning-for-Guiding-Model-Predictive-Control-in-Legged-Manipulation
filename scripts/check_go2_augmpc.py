"""60-second, eight-Go2 continuous feedback and transition ownership acceptance."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher


def run(args):
    import torch
    import gymnasium as gym
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import Go2AugMPCEnvCfg
    from RL_Bezier_MPC.envs.go2_feedback import generalized_advantage, dense_reward

    cfg=Go2AugMPCEnvCfg()
    cfg.seed=37
    cfg.sim.device=args.device
    args.output.mkdir(parents=True,exist_ok=False)
    env=gym.make("Isaac-Go2-AugMPC-Direct-v0",cfg=cfg).unwrapped
    transitions=[]
    manual=[]
    target_changes=[]
    natural=truncations=0
    peak_drift=0.
    oracle_checked=False
    def snapshot(peers):
        buffers=(env.actions,env.previous_actions,env.commands,env.mask,env.anchors,env.origins,
            env.residual,env.health,env.drift,env.predicted_velocity,env.prediction_valid,
            env.failed,env.arrived,env.ledger.request_episode,env.pending_cycle_action,env.applied_request,
            env._pending_joint_efforts,env.ledger.episode,env.ledger.decision,
            env.mpc._warm,env.mpc._dual,env.mpc._valid,env.episode_length_buf)
        hardware=(env.robot.data.root_link_pose_w,env.robot.data.root_link_vel_w,
                  env.robot.data.joint_pos,env.robot.data.joint_vel,
                  env.foot_contact_sensor.data.net_forces_w)
        return ([value[peers].clone() for value in buffers],
                [(env.clocks[i].tick,env.clocks[i].boundary,len(env.clocks[i].schedule)) for i in peers],
                [value[peers].clone() for value in hardware])
    try:
        obs,_=env.reset(seed=37)
        env.set_commands([0],0.,0.)
        env.set_commands([6],.5,.003)
        obs=env._get_observations()
        steps=round(args.seconds/env.step_dt)
        if steps*env.step_dt<60.:
            raise ValueError("Acceptance requires at least 60 simulated seconds")
        for step in range(steps):
            if step in (24,48,96,160):
                row=(step//24)%8
                peers=[i for i in range(8) if i!=row]
                before=env._get_observations()["policy"].clone()
                target=env._state()[row,0].item()+(.04 if step==48 else .10)
                env.set_commands([row],target,.006 if row%2 else .003)
                after=env._get_observations()["policy"]
                assert torch.equal(before[peers],after[peers]),"Goal update changed peer observations"
                target_changes.append(dict(step=step,row=row,target=target))
                obs=env._get_observations()
            if len(manual)<2 and step>=(40 if not manual else 140):
                candidates=(~env.mask.all(-1)).nonzero().flatten().tolist()
                if candidates:
                    row=candidates[0]
                    peers=[i for i in range(8) if i!=row]
                    before=snapshot(peers)
                    episode=env.ledger.episode[row].item()
                    obs,event=env.reset_envs([row])
                    assert event["episode_ids"][0]==episode and event["bootstrap_mask"][0]
                    assert not event["trace_mask"][0]
                    assert event["decision_ids"][0]>=0
                    # Explicit resets truncate the previously collected transition.
                    transitions[-1][4][row]=True
                    transitions[-1][2][row]=event["bootstrap_observation"][0,0]
                    after=snapshot(peers)
                    assert all(torch.equal(a,b) for a,b in zip(before[0],after[0]))
                    assert before[1]==after[1]
                    assert all(torch.allclose(a,b,atol=1e-7,rtol=0) for a,b in zip(before[2],after[2]))
                    assert env.ledger.episode[row]==episode+1
                    assert env.clocks[row].tick==0 and not env.clocks[row].schedule
                    assert env.mask[row].all() and not env.mpc._valid[row]
                    assert not env.actions[row].any() and not env.pending_cycle_action[row].any()
                    assert not env._pending_joint_efforts[row].any()
                    assert env.ledger.request_episode[row]==-1 and (env.applied_request[row]==-1).all()
                    manual.append(dict(step=step,row=row,old_episode=episode))
            if step==8:
                mpc=env.mpc
                state=env._state()
                refs=[]
                accs=[]
                for clock in env.clocks:
                    index=min(clock.tick,len(clock.plan[0])-1)
                    refs.append(clock.plan[0][(env.nodes+index).clamp_max(len(clock.plan[0])-1)])
                    accs.append(clock.plan[1][index])
                warm=[v.clone() for v in (mpc._warm,mpc._dual,mpc._valid)]
                try:
                    a,ra=mpc.compute(state,torch.stack(refs),torch.stack(accs),env.mask,env.anchors)
                    peers_after=[v[1:].clone() for v in (mpc._warm,mpc._dual,mpc._valid)]
                    for target,value in zip((mpc._warm,mpc._dual,mpc._valid),warm):
                        target.copy_(value)
                    changed=state.clone()
                    changed[0,0]+=1e-5
                    b,rb=mpc.compute(changed,torch.stack(refs),torch.stack(accs),env.mask,env.anchors)
                    assert torch.allclose(a[1:],b[1:],atol=1e-10,rtol=0)
                    assert torch.equal(ra.status[1:],rb.status[1:])
                    assert all(torch.equal(v[1:],saved) for v,saved in zip((mpc._warm,mpc._dual,mpc._valid),peers_after))
                    oracle_checked=True
                finally:
                    for target,value in zip((mpc._warm,mpc._dual,mpc._valid),warm):
                        target.copy_(value)
                # Policy observations default to measured velocity, independently of MPC predictions.
                before=env._get_observations()["policy"].clone()
                prediction=env.predicted_velocity.clone()
                env.predicted_velocity.add_(.123)
                assert torch.equal(before,env._get_observations()["policy"])
                env.predicted_velocity.copy_(prediction)
            state=env._state()
            remaining=env.commands[:,0]-state[:,0]
            action=torch.zeros(8,4,device=env.device)
            action[:,0]=torch.where((env.commands[:,1]>.0045)&(remaining>.03),1.,-1.)
            action[:,3]=.25*torch.sin(torch.tensor(step*.2,device=env.device)+torch.arange(8,device=env.device))
            old_episode=env.ledger.episode.clone()
            old_decision=env.ledger.decision.clone()
            values=obs["policy"][:,0].clone()
            obs,reward,terminated,truncated,info=env.step(action)
            t=info["transition"]
            assert torch.equal(t["identity"][:,0],torch.arange(8,device=env.device))
            assert torch.equal(t["identity"][:,1],old_episode)
            assert torch.equal(t["identity"][:,2],old_decision)
            assert torch.equal(t["actions"],action)
            applied=t["applied_cycle_request"][:,0]>=0
            assert torch.equal(t["applied_cycle_request"][applied,1],old_episode[applied])
            assert torch.equal(t["reward"],reward)
            distance=t["bootstrap_observation"][:,0].abs()*.1
            expected=dense_reward(values.abs()*.1-distance,t["measured_velocity_b"][:,0],
                torch.where(distance<=cfg.goal_tolerance_m,0.,t["commands"][:,1]),
                t["effort"],t["action_change"],t["terminated_failure"],t["terminated_success"],env.step_dt)
            assert torch.allclose(expected,reward,atol=2e-5,rtol=0),"Reward ownership mismatch"
            assert t["mpc_ticks"]==(13 if step%2==0 else 12)
            assert abs(t["elapsed_s"]-.25)<1e-12
            assert torch.equal(t["bootstrap_mask"],~terminated)
            assert torch.equal(t["trace_mask"],~(terminated|truncated))
            assert not (terminated&truncated).any()
            done=terminated|truncated
            assert torch.equal(env.ledger.episode,old_episode+done.long())
            assert not t["mpc_health"][:,5].any(),"Unexpected physical/solver failure"
            for tensor in (obs["policy"],reward,t["bootstrap_observation"],t["mpc_health"],
                           t["measured_velocity_b"],t["mpc_predicted_velocity_b"]):
                assert torch.isfinite(tensor).all()
            if done.any():
                assert (t["bootstrap_observation"][done]-obs["policy"][done]).abs().sum()>0
                assert env.mask[done].all() and not env.mpc._valid[done].any()
                assert not env.actions[done].any()
            natural+=int(terminated.sum())
            truncations+=int(truncated.sum())
            peak_drift=max(peak_drift,1000*env.drift.max().item())
            transitions.append((reward.clone(),values,t["bootstrap_observation"][:,0].clone(),
                                terminated.clone(),truncated.clone()))
            if step%40==39:
                print(json.dumps(dict(stage="progress",sim_s=(step+1)*env.step_dt,
                    mpc_ticks=env.solve_count,natural=natural,truncated=truncations,
                    manual_resets=len(manual))),flush=True)
        tensors=[torch.stack([t[i] for t in transitions]) for i in range(5)]
        advantage,_=generalized_advantage(*tensors,gamma=.99,lam=0.)
        rewards,values,next_values,terminated,truncated=tensors
        assert torch.allclose(advantage[truncated],(rewards+.99*next_values-values)[truncated])
        assert torch.allclose(advantage[terminated],(rewards-values)[terminated])
        assert env.solve_count==round(steps*env.step_dt/cfg.mpc_dt)
        assert natural>0 and truncations>0 and len(manual)==2 and oracle_checked
        report=dict(passed=True,sim_seconds=steps*env.step_dt,environments=8,
            transitions=steps*8,policy_hz=1/env.step_dt,mpc_hz=1/cfg.mpc_dt,
            mpc_ticks=env.solve_count,natural_terminations=natural,time_truncations=truncations,
            manual_resets=manual,target_changes=target_changes,max_stance_drift_mm=peak_drift,
            bootstrap_checked=True,mpc_row_independence_checked=oracle_checked,
            measured_velocity_observation_independence_checked=True)
        (args.output/"acceptance.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        print("PASS: eight Go2 continuous feedback 60 seconds",flush=True)
    finally:
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=60.)
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
