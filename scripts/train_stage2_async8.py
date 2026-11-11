"""Eight independent Go2 cycle rollouts with episode-local PPO returns and paired evaluation."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback

from isaaclab.app import AppLauncher
from check_stage2_async_reset import async_types
from check_go2_slow_trot import SETTLE, HOLD, evaluate_mixed, make_plan, PERIOD_SCALES
from train_stage2_cycle_ppo import GOALS, make_policy
from stage2_training_state import (
    save_json, save_checkpoint, restore_checkpoint, sampling_state, restore_sampling,
    period_probes, compare_evaluation, learning_summary,
)

N = 8
SPEED_GOAL_MM = 64
SPEEDS = (3., 6.)


def speed_feature(speed):
    return (speed - SPEEDS[1]) / (SPEEDS[1] - SPEEDS[0])


def terminal_reward(m):
    reward = (2. if m["success"] else -2.) - abs(m["final_x_error_mm"]) / 4
    if "desired_mm_s" in m:
        reward -= 3 * abs(m["walking_mm_s"] / m["desired_mm_s"] - 1)
    return reward


def cycle_return(m, tick, dt):
    if "desired_mm_s" in m:
        return m["terminal_reward"]
    return m["terminal_reward"] - .1 * (m["end_tick"] - tick) * dt


def audit_batch(samples,reports):
    owners={tuple(m["identity"]):m for m in reports}
    keys={tuple(p["key"]) for p in samples}
    expected={(*k,i) for k,m in owners.items() for i in range(len(m["decisions"]))}
    if len(samples)!=len(keys) or keys!=expected or len(owners)!=len(reports):
        raise RuntimeError("Duplicate or missing cycle/episode IDs")
    for p in samples:
        m=owners[tuple(p["key"][:3])]
        i=p["key"][3]
        d=m["decisions"][i]
        if "period_slot" in d or "period_slot" in p:
            previous = m["decisions"][i-1]["period_slot"] if i else 1
            slot=p.get("period_slot")
            if (slot not in (0,1,2) or slot!=d["period_slot"] or abs(p["obs"][3]-(previous-1))>1e-6
                or d["time_scale"]!=PERIOD_SCALES[d["gait"]][slot]
                or m["cycles"][i]["time_scale"]!=d["time_scale"]
                or abs(d["duration_s"]-(p["end"]-p["start"])*.02)>1e-9):
                raise RuntimeError(f"Incorrect period action ownership: {p['key']}")
        if (p["goal"]!=m["target_mm"] or abs(p["obs"][0]-(d["remaining_mm"]-30)/15)>1e-5
            or abs(p["obs"][1]-float(i>0 and m["decisions"][i-1]["gait"]=="trot"))>1e-6
            or abs(max(10.,min(24.,min(d["remaining_mm"],21.)+3*math.tanh(p["z"])))-d["stride_mm"])>1e-5
            or p["end"]-p["start"]!=round(d["stride_mm"]/m["cycles"][i]["command_mm_s"]/.02)
            or ("desired_mm_s" in m and (p["desired_mm_s"] != m["desired_mm_s"]
                or abs(p["obs"][2]-speed_feature(m["desired_mm_s"]))>1e-6))
            or abs(m["terminal_reward"]-terminal_reward(m))>1e-6
            or bool(p["g"])!=(d["gait"]=="trot")
            or abs(p["return_to_go"]-cycle_return(m,d["local_tick"],.02))>1e-5):
            raise RuntimeError(f"Cycle/reward assigned to incorrect owner: {p['key']}")


def run(args):
    import torch
    import omni.usd
    from PIL import Image, ImageDraw
    from torch import nn
    torch.manual_seed(args.seed)
    policy = make_policy(speed_task=args.speed_task, period_actions=args.period_actions)
    optimizer = torch.optim.Adam(policy.parameters(),lr=.003)
    start_iteration = 0
    if args.checkpoint:
        loaded = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
        policy.load_compatible_state(loaded["policy"])
        if args.eval_only:
            start_iteration = loaded.get("iteration",0)
    args.output.mkdir(parents=True,exist_ok=False)
    goal_rng = [random.Random(args.seed+1019*r) for r in range(N)]
    bags = [[] for _ in range(N)]
    noise = [torch.Generator().manual_seed(args.seed+499*r+1) for r in range(N)]
    speed_groups = [i % 2 for i in range(N)]
    random.Random(args.seed).shuffle(speed_groups)
    task = dict(seed=args.seed, speed_task=args.speed_task, period_actions=args.period_actions,
                num_envs=N, speed_goal_mm=SPEED_GOAL_MM, speeds=SPEEDS, period_scales=PERIOD_SCALES)
    resume_mode = None
    if args.resume:
        start_iteration,resume_mode = restore_checkpoint(
            args.resume,policy,optimizer,task,noise,goal_rng,bags)
        print(json.dumps(dict(stage="resume",iteration=start_iteration,
                              sampling_state=resume_mode,optimizer_restored=True)),flush=True)
    final_iteration = args.until if args.until is not None else start_iteration+(args.updates or 1)
    if not args.eval_only and not args.evaluate_run and final_iteration<start_iteration:
        raise ValueError("--until precedes the restored iteration")
    def sample_goal(r):
        if not bags[r]:
            bags[r] = list(GOALS)
            goal_rng[r].shuffle(bags[r])
        return bags[r].pop()

    def choose(c,obs,remaining):
        with torch.no_grad():
            if args.speed_task:
                obs = torch.cat((obs,obs.new_tensor([speed_feature(c.desired_mm_s)])))
            if args.period_actions:
                obs = torch.cat((obs,obs.new_tensor([c.period_slot-1.])))
            gd,zd = policy.distributions(obs)
            if c.mode=="sample":
                generator = noise[c.row_id]
                g = torch.bernoulli(gd.probs,generator=generator)
                z = zd.mean+zd.stddev*torch.randn((),generator=generator)
                key = (*c.identity,len(c.schedule))
                c.pending.append(dict(key=key,start=c.tick,obs=obs.tolist(),g=g.item(),z=z.item(),
                                      logp=(gd.log_prob(g)+zd.log_prob(z)).item()))
            elif c.mode=="policy":
                g,z = gd.logits>=0,zd.mean
            else:
                g,z = remaining>=30 and (not args.speed_task or c.desired_mm_s==SPEEDS[1]),torch.tensor(0.)
            gait = "trot" if bool(g) else "crawl"
            if not args.period_actions:
                return gait,z.item()
            pd = policy.period_distribution(obs)
            if c.mode == "sample":
                slot = int(torch.multinomial(pd.probs,1,generator=noise[c.row_id]).item())
                c.pending[-1].update(period_slot=slot,
                    logp=c.pending[-1]["logp"]+pd.log_prob(torch.tensor(slot)).item())
            elif c.mode == "policy":
                slot = int(pd.logits.argmax().item())
            elif c.mode == "period_probe":
                gait = "crawl" if c.row_id%2==0 else "trot"
                slot = (0,2,1)[len(c.schedule)%3]
                z = torch.tensor(math.atanh((20.-min(remaining,21.))/3.))
            else:
                slot = 1
            return gait,z.item(),slot

    Clock,Env = async_types(choose, period_actions=args.period_actions)
    from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg
    def rollout(mode,update,output=None):
        output = args.output if output is None else output
        quota = args.episodes_per_env if mode=="sample" else 1
        cfg = QuadrupedMPCEnvCfg()
        cfg.scene.num_envs,cfg.scene.env_spacing,cfg.seed,cfg.sim.device = N,1.15,args.seed,args.device
        cfg.mpc_backend,cfg.use_mpc_cluster,cfg.torch_foot_plane_height = "torch_tracking",False,.022
        cfg.rl_policy_period,cfg.decimation,cfg.episode_length_s = 1,cfg.mpc_decimation,80.
        cfg.viewer.eye,cfg.viewer.lookat,cfg.viewer.resolution = (1.3,-4.5,3.7),(.05,0,.13),(1152,720)
        record = mode=="policy" and args.video
        cfg.sim.render_interval = cfg.decimation if record else 1000000
        omni.usd.get_context().new_stage()
        env = Env(cfg,render_mode="rgb_array" if record else None)
        reports,batch,counts,peer_checks = [],[],[0]*N,0
        writer=None
        projected_frames=0
        def clock(r,park=False):
            goal = SPEED_GOAL_MM if args.speed_task else (sample_goal(r) if mode=="sample" and not park else GOALS[r%3])
            if mode=="period_probe":
                goal = 60
            c=Clock(env,goal,delay=0)
            c.row_id,c.mode,c.identity,c.pending = r,mode,(update,r,counts[r]),[]
            if args.speed_task:
                group = (speed_groups[r]+counts[r]+update-1)%2 if mode=="sample" else r%2
                c.desired_mm_s = SPEEDS[group]
            c.parked=park
            return c

        def reset_row(r):
            nonlocal peer_checks
            peers = [j for j in range(N) if j!=r]
            buffers = (env.anchors,env.mask,env._pending_joint_efforts,env.robot.data.applied_torque,
                       mpc._warm,mpc._dual,mpc._valid,env.episode_length_buf)
            before = [b[peers].clone() for b in buffers]
            env.allowed_reset_ids=[r]
            env._reset_idx(torch.tensor([r],device=env.device))
            env.allowed_reset_ids=[]
            if mpc._valid[r] or env.episode_length_buf[r]!=0:
                raise RuntimeError(f"Failed reset in row {r}")
            for b,old in zip(buffers,before):
                if not torch.equal(b[peers],old):
                    raise RuntimeError(f"Reset changed peers of row {r}")
            peer_checks+=1
            env.anchors[r]=mpc.model.contact_positions
            env.origins[r]=env.anchors[r]
            env.drift[r].zero_()
            env.mask[r]=True
            env.clocks[r]=clock(r,park=counts[r]>=quota)

        try:
            env.reset(seed=args.seed)
            mpc=env.tracking.controller.mpc
            dt=cfg.mpc_dt
            env.nominal=make_plan(mpc.model,dt,[("crawl",.020)],cycle_decisions=True)
            env.anchors=mpc.model.contact_positions.expand(N,-1,-1).clone()
            env.origins=env.anchors.clone()
            env.drift=env.anchors.new_zeros(N,4)
            env.mask=torch.ones(N,4,device=env.device,dtype=torch.bool)
            env.nodes=torch.arange(mpc.horizon+1,device=env.device)
            env.world,env.live,env.allowed_reset_ids=0,True,[]
            env.clocks=[clock(r) for r in range(N)]
            action=torch.zeros((N,cfg.action_space),device=env.device)
            if record:
                writer=subprocess.Popen(["ffmpeg","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
                    "-s","1152x720","-r",str(1/dt),"-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p",
                    "-movflags","+faststart",str(output/"policy_8go2.mp4")],stdin=subprocess.PIPE)
            if str(env.device).startswith("cuda"):
                torch.cuda.synchronize(env.device)
            start=time.perf_counter()
            while min(counts)<quota:
                # Seal the previously executed action on its own measured boundary.
                # Terminal reward arrives later, after this row has stopped.
                for c in env.clocks:
                    if c.pending and not c.finishing and not c.parked and c.tick==c.boundary:
                        if "end" in c.pending[-1]:
                            raise RuntimeError("Cycle completed twice")
                        c.pending[-1]["end"]=c.tick
                _,_,term,trunc,_=env.step(action)
                if (term|trunc).any() or env.world>3500*quota:
                    raise RuntimeError("Invalid rollout termination")
                if writer:
                    env.sim.render()
                    rgb=env.render()
                    for _ in range(8):
                        if rgb.size and rgb[:600:8,::8,0].std()>8:
                            break
                        env.sim.set_camera_view(cfg.viewer.eye,cfg.viewer.lookat)
                        env.sim.render()
                        rgb=env.render()
                    if rgb.size and rgb[:600:8,::8,0].std()>8:
                        frame=Image.fromarray(rgb).convert("RGB")
                    else:
                        # Keep the exact physics tick visible when RTX returns gray.
                        # Project the measured links of all eight robots together.
                        projected_frames+=1
                        frame=Image.new("RGB",(1152,720),"#eaf0f3")
                        pen=ImageDraw.Draw(frame)
                        def project(xyz):
                            x,y,z=xyz
                            return (int(575+218*x+28*y),int(352-128*y-255*z))
                        for i in range(-5,6):
                            v=i/2
                            pen.line([project((v,-2,0)),project((v,2,0))],fill="#bac8cf")
                            pen.line([project((-2.7,v,0)),project((2.7,v,0))],fill="#bac8cf")
                        for r,xyz in enumerate(env.robot.data.body_link_pos_w.tolist()):
                            links=dict(zip(env.robot.body_names,map(project,xyz)))
                            for leg in ("FL","RL","FR","RR"):
                                pen.line([links[f"{leg}_{part}"] for part in ("hip","thigh","calf","foot")],
                                         fill="#2f6179",width=4)
                            pen.polygon([links[f"{leg}_hip"] for leg in ("FL","FR","RR","RL")],fill="#354858")
                            pen.text(links["FR_hip"],str(r),fill="white")
                        pen.text((12,15),"PhysX measured link positions",fill="#192532")
                    draw=ImageDraw.Draw(frame)
                    draw.rectangle((0,648,1152,720),fill="#192532")
                    draw.text((12,653),f"Stage 2 PPO    8 Go2    deterministic evaluation    {env.world*dt:.2f} s")
                    for r,c in enumerate(env.clocks):
                        gait="episode complete" if c.parked else c.rows[-1]["gait"]
                        speed_label = f"{c.desired_mm_s:g} mm/s   " if args.speed_task else ""
                        draw.text((12+(r%4)*285,675+(r//4)*20),f"Go2 {r}   {c.goal} mm   {speed_label}{gait}")
                    writer.stdin.write(frame.tobytes())
                for r,c in enumerate(env.clocks):
                    if c.parked or not c.finishing:
                        continue
                    timeout=c.tick-c.boundary>=HOLD
                    if c.stable<50 and not timeout:
                        continue
                    m=evaluate_mixed(c.rows,env.drift[r].tolist(),dt,c.schedule,target_mm=c.goal,time_scales=c.time_scales)
                    arrived=c.stable>=50 and all(m["checks"].values())
                    m["success"] = arrived
                    if args.speed_task:
                        walking_s = (c.boundary-SETTLE)*dt
                        walking_mm_s = 1000*(c.rows[c.boundary]["x"]-c.rows[SETTLE]["x"])/walking_s
                        m.update(desired_mm_s=c.desired_mm_s,walking_s=walking_s,walking_mm_s=walking_mm_s,
                                 speed_ok=abs(walking_mm_s/c.desired_mm_s-1)<=.20)
                    terminal=terminal_reward(m)
                    m.update(identity=c.identity,mode=mode,success=arrived,terminal_reward=terminal,
                             decisions=c.decisions,arrival_s=(c.tick-SETTLE)*dt if arrived else None,
                             end_tick=c.tick,reset_world_tick=env.world)
                    reports.append(m)
                    print(json.dumps(dict(stage="episode",mode=mode,identity=c.identity,goal_mm=c.goal,
                                success=arrived,world_tick=env.world,cycles=len(c.schedule))),flush=True)
                    if mode=="sample":
                        for i,p in enumerate(c.pending):
                            # End ticks come from the advancing row's measured clock, never global adjacency.
                            end=c.decisions[i+1]["local_tick"] if i+1<len(c.decisions) else c.boundary
                            if p.get("end")!=end or end<=p["start"] or p["key"]!=(*c.identity,i) or p["start"]!=c.decisions[i]["local_tick"]:
                                raise RuntimeError("Sample identity or cycle boundary mismatch")
                            if args.speed_task:
                                p["desired_mm_s"]=c.desired_mm_s
                            p.update(goal=c.goal,return_to_go=cycle_return(m,p["start"],dt))
                        if len(c.pending)!=len(c.schedule):
                            raise RuntimeError("Missing completed cycle")
                        batch.extend(c.pending)
                    counts[r]+=1
                    reset_row(r)
            if str(env.device).startswith("cuda"):
                torch.cuda.synchronize(env.device)
            elapsed=time.perf_counter()-start
            print(json.dumps(dict(stage=mode,update=update,episodes=len(reports),cycles=sum(len(m["cycles"]) for m in reports),
                    seconds=elapsed,peer_reset_checks=peer_checks,projected_frames=projected_frames,success=sum(m["success"] for m in reports))),flush=True)
            return reports,batch,elapsed
        finally:
            try:
                if writer:
                    writer.stdin.close()
                    if writer.wait():
                        raise RuntimeError("Encoder failed")
            finally:
                env.close()

    def update_policy(samples,reports):
        audit_batch(samples,reports)
        obs,g,z,old,ret=[torch.tensor([p[k] for p in samples]) for k in ("obs","g","z","logp","return_to_go")]
        period = torch.tensor([p["period_slot"] for p in samples]) if args.period_actions else None
        with torch.no_grad():
            if not torch.allclose(policy.action_log_prob(obs,g,z,period),old,atol=1e-5,rtol=0):
                raise RuntimeError("Stored action probability does not match the rollout policy")
        adv=(ret-policy.value(obs).squeeze(-1)).detach()
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
        before=nn.utils.parameters_to_vector(policy.actor.parameters()).detach().clone()
        period_before = (nn.utils.parameters_to_vector(policy.period_actor.parameters()).detach().clone()
                         if args.period_actions else None)
        for _ in range(4):
            ratio=(policy.action_log_prob(obs,g,z,period)-old).exp()
            loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*(policy.value(obs).squeeze(-1)-ret).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(),1.)
            optimizer.step()
        change=(nn.utils.parameters_to_vector(policy.actor.parameters())-before).norm().item()
        if not math.isfinite(change) or change<=0:
            raise RuntimeError("No finite PPO actor update")
        if args.period_actions:
            period_change=(nn.utils.parameters_to_vector(policy.period_actor.parameters())-period_before).norm().item()
            if not math.isfinite(period_change) or period_change<=0:
                raise RuntimeError("Period head did not update")
        else:
            period_change=0.
        return change,period_change

    if args.period_probe_only:
        reports,_,_=rollout("period_probe",0)
        passed=all(m["success"] and [d["period_slot"] for d in m["decisions"]]==[0,2,1]
                   and len(m["switches"])==2 for m in reports)
        (args.output/"comparison.json").write_text(json.dumps(dict(passed=passed,episodes=reports),indent=2))
        if not passed:
            raise RuntimeError("Cycle period switching probe failed")
        print("PASS: eight asynchronous period action probes",flush=True)
        return

    updates,evaluations = [],[]
    baseline = None
    milestones = {start_iteration} if args.eval_only else set(args.eval_iterations)

    def evaluate(iteration, checkpoint):
        nonlocal baseline
        saved_sampling = sampling_state(noise,goal_rng,bags)
        directory = args.output/f"eval_{iteration:04d}"
        directory.mkdir(exist_ok=False)
        try:
            # Evaluate exactly the file that a future process will reload.
            policy.load_compatible_state(torch.load(checkpoint,map_location="cpu",weights_only=True)["policy"])
            policy.eval()
            if baseline is None:
                baseline = rollout("rule",0)[0]
                save_json(args.output/"baseline.json",baseline)
            learned = rollout("policy",iteration,directory)[0]
            probes = period_probes(policy) if args.period_actions else []
            report = compare_evaluation(learned,baseline,iteration,args.speed_task,probes)
            save_json(directory/"comparison.json",report)
            evaluations.append(report)
            save_json(args.output/"comparison.json",report)
            summary = learning_summary(evaluations,milestones)
            summary.update(start_iteration=start_iteration, last_evaluated_iteration=iteration,
                           resume_sampling_state=resume_mode, updates=updates, completed=False)
            save_json(args.output/"summary.json",summary)
            print(json.dumps(dict(stage="milestone",iteration=iteration,passed=report["passed"],
                period_adaptation_observed=report["period_adaptation_observed"],
                groups=report["groups"])),flush=True)
            return report
        finally:
            restore_sampling(saved_sampling,noise,goal_rng,bags)
            policy.train()

    if args.evaluate_run:
        for iteration in sorted(milestones):
            checkpoint=args.evaluate_run/f"async8_{iteration:04d}.pt"
            restore_checkpoint(checkpoint,policy,optimizer,task,noise,goal_rng,bags)
            evaluate(iteration,checkpoint)
    elif args.eval_only:
        evaluate(start_iteration,args.resume or args.checkpoint)
    else:
        if args.resume:
            initial=args.output/f"async8_{start_iteration:04d}.pt"
            save_checkpoint(initial,policy,optimizer,start_iteration,task,noise,goal_rng,bags)
            for earlier in sorted(i for i in milestones if i<start_iteration):
                source=args.resume.parent/f"async8_{earlier:04d}.pt"
                if source.is_file():
                    copied=args.output/source.name
                    shutil.copyfile(source,copied)
                    evaluate(earlier,copied)
            if start_iteration in milestones or start_iteration==final_iteration:
                evaluate(start_iteration,initial)
            # Earlier milestone evaluation may have loaded an older policy.
            policy.load_state_dict(torch.load(initial,map_location="cpu",weights_only=True)["policy"])
        for iteration in range(start_iteration+1,final_iteration+1):
            reports,samples,elapsed=rollout("sample",iteration)
            change,period_change=update_policy(samples,reports)
            metric=dict(update=iteration,completed_decisions=len(samples),collection_wall_s=elapsed,
                decisions_per_wall_s=len(samples)/elapsed,actor_change=change,period_actor_change=period_change)
            updates.append(metric)
            save_json(args.output/f"batch_{iteration:04d}.json",
                      dict(metric=metric,episodes=reports,samples=samples))
            checkpoint=args.output/f"async8_{iteration:04d}.pt"
            save_checkpoint(checkpoint,policy,optimizer,iteration,task,noise,goal_rng,bags)
            print(json.dumps(dict(stage="ppo_update",**metric)),flush=True)
            if iteration in milestones or iteration==final_iteration:
                evaluate(iteration,checkpoint)
    summary = learning_summary(evaluations,milestones)
    summary.update(start_iteration=start_iteration,
                   last_evaluated_iteration=evaluations[-1]["iteration"],
                   resume_sampling_state=resume_mode, updates=updates, completed=True)
    save_json(args.output/"summary.json",summary)
    print(json.dumps(dict(stage="complete",last_iteration=evaluations[-1]["iteration"],
        all_evaluations_passed=summary["all_evaluations_passed"],
        missing_milestones=summary["missing_milestones"],
        final_period_adaptation_observed=summary["final_period_adaptation_observed"])),flush=True)
    if not evaluations[-1]["passed"]:
        raise RuntimeError("Final evaluation failed; training and reports are saved in the output directory")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--period-actions", action="store_true", help="Choose a validated slow/current/fast period each cycle")
    parser.add_argument("--period-probe-only", action="store_true", help="Execute slow/fast/current period actions and verify boundaries")
    parser.add_argument("--speed-task", action="store_true", help="64 mm paired goal at 3 and 6 mm/s walking speed")
    parser.add_argument("--seed",type=int,default=37)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--updates",type=int,help="Number of additional PPO updates (default 1)")
    duration.add_argument("--until",type=int,help="Cumulative completed iteration, inclusive")
    parser.add_argument("--eval-iterations",nargs="+",type=int,default=[1,4,8,12])
    parser.add_argument("--episodes-per-env",type=int,default=2)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint",type=Path,help="Policy-only warm start; resets optimizer and iteration")
    source.add_argument("--resume",type=Path,help="Restore completed iteration, Adam and sampling state")
    source.add_argument("--evaluate-run",type=Path,help="Evaluate saved milestone checkpoints without training")
    parser.add_argument("--eval-only",action="store_true")
    parser.add_argument("--video",action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if ((args.updates is not None and args.updates<1) or (args.until is not None and args.until<1)
        or args.episodes_per_env<1 or any(i<1 for i in args.eval_iterations)):
        parser.error("Rollout sizes and iteration numbers must be positive")
    if args.eval_only and not (args.checkpoint or args.resume):
        parser.error("--eval-only requires --checkpoint or --resume")
    if args.period_probe_only and (args.eval_only or args.evaluate_run or args.resume):
        parser.error("Period probe cannot be combined with resume or evaluation modes")
    if args.evaluate_run:
        missing=[i for i in args.eval_iterations if not (args.evaluate_run/f"async8_{i:04d}.pt").is_file()]
        if missing:
            parser.error(f"Missing milestone checkpoints: {missing}")
    if args.period_actions and not args.speed_task:
        parser.error("--period-actions requires --speed-task")
    if args.period_probe_only and not args.period_actions:
        parser.error("--period-probe-only requires --period-actions")
    if args.video:
        args.enable_cameras=True
    launcher=AppLauncher(args)
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()
