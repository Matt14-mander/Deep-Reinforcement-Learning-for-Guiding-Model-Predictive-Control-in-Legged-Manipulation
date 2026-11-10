"""Eight independent Go2 cycle rollouts with episode-local PPO returns and paired evaluation."""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import time
import traceback

from isaaclab.app import AppLauncher
from check_stage2_async_reset import async_types
from check_go2_slow_trot import SETTLE, HOLD, evaluate_mixed, make_plan
from train_stage2_cycle_ppo import GOALS, make_policy

N = 8


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
        if (p["goal"]!=m["target_mm"] or abs(p["obs"][0]-(d["remaining_mm"]-30)/15)>1e-5
            or abs(p["obs"][1]-float(i>0 and m["decisions"][i-1]["gait"]=="trot"))>1e-6
            or abs(max(10.,min(24.,min(d["remaining_mm"],21.)+3*math.tanh(p["z"])))-d["stride_mm"])>1e-5
            or p["end"]-p["start"]!=round(d["stride_mm"]/m["cycles"][i]["command_mm_s"]/.02)
            or abs(m["terminal_reward"]-((2. if m["success"] else -2.)-abs(m["final_x_error_mm"])/4))>1e-6
            or bool(p["g"])!=(d["gait"]=="trot")
            or abs(p["return_to_go"]-(m["terminal_reward"]-.1*(m["end_tick"]-d["local_tick"])*.02))>1e-5):
            raise RuntimeError(f"Cycle/reward assigned to incorrect owner: {p['key']}")


def run(args):
    import torch
    import omni.usd
    from PIL import Image, ImageDraw
    from torch import nn
    torch.manual_seed(args.seed)
    policy = make_policy()
    optimizer = torch.optim.Adam(policy.parameters(),lr=.003)
    if args.checkpoint:
        policy.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=True)["policy"])
    args.output.mkdir(parents=True,exist_ok=False)
    goal_rng = [random.Random(args.seed+1019*r) for r in range(N)]
    bags = [[] for _ in range(N)]
    noise = [torch.Generator().manual_seed(args.seed+499*r+1) for r in range(N)]
    def sample_goal(r):
        if not bags[r]:
            bags[r] = list(GOALS)
            goal_rng[r].shuffle(bags[r])
        return bags[r].pop()

    def choose(c,obs,remaining):
        with torch.no_grad():
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
                g,z = remaining>=30,torch.tensor(0.)
            return ("trot" if bool(g) else "crawl"),z.item()

    Clock,Env = async_types(choose)
    from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg
    def rollout(mode,update):
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
            goal = sample_goal(r) if mode=="sample" and not park else GOALS[r%3]
            c=Clock(env,goal,delay=0)
            c.row_id,c.mode,c.identity,c.pending = r,mode,(update,r,counts[r]),[]
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
                    "-movflags","+faststart",str(args.output/"policy_8go2.mp4")],stdin=subprocess.PIPE)
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
                        gait="arrived / reset" if c.parked else c.rows[-1]["gait"]
                        draw.text((12+(r%4)*285,675+(r//4)*20),f"Go2 {r}   {GOALS[r%3]} mm   {gait}")
                    writer.stdin.write(frame.tobytes())
                for r,c in enumerate(env.clocks):
                    if c.parked or not c.finishing:
                        continue
                    timeout=c.tick-c.boundary>=HOLD
                    if c.stable<50 and not timeout:
                        continue
                    m=evaluate_mixed(c.rows,env.drift[r].tolist(),dt,c.schedule,target_mm=c.goal)
                    arrived=c.stable>=50 and all(m["checks"].values())
                    terminal=(2. if arrived else -2.)-abs(m["final_x_error_mm"])/4
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
                            p.update(goal=c.goal,return_to_go=terminal-.1*(c.tick-p["start"])*dt)
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
        with torch.no_grad():
            gd,zd=policy.distributions(obs)
            if not torch.allclose(gd.log_prob(g)+zd.log_prob(z),old,atol=1e-5,rtol=0):
                raise RuntimeError("Stored action probability does not match the rollout policy")
        adv=(ret-policy.value(obs).squeeze(-1)).detach()
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
        before=nn.utils.parameters_to_vector(policy.actor.parameters()).detach().clone()
        for _ in range(4):
            gd,zd=policy.distributions(obs)
            ratio=(gd.log_prob(g)+zd.log_prob(z)-old).exp()
            loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*(policy.value(obs).squeeze(-1)-ret).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(),1.)
            optimizer.step()
        change=(nn.utils.parameters_to_vector(policy.actor.parameters())-before).norm().item()
        if not math.isfinite(change) or change<=0:
            raise RuntimeError("No finite PPO actor update")
        return change

    updates=[]
    if not args.eval_only:
        for i in range(1,args.updates+1):
            reports,samples,elapsed=rollout("sample",i)
            change=update_policy(samples,reports)
            metric=dict(update=i,completed_decisions=len(samples),collection_wall_s=elapsed,
                        decisions_per_wall_s=len(samples)/elapsed,actor_change=change)
            print(json.dumps(dict(stage="ppo_update",**metric)),flush=True)
            updates.append(metric)
            (args.output/f"batch_{i:04d}.json").write_text(json.dumps(dict(metric=metric,episodes=reports,samples=samples),allow_nan=False))
            last=args.output/f"async8_{i:04d}.pt"
            torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),iteration=i),last)
        policy.load_state_dict(torch.load(last,map_location="cpu",weights_only=True)["policy"])
    policy.eval()
    learned=rollout("policy",0)[0]
    baseline=rollout("rule",0)[0]
    index={tuple(m["identity"]):m for m in baseline}
    pairs=[]
    for m in sorted(learned,key=lambda v:v["identity"]):
        b=index[tuple(m["identity"])]
        if b["target_mm"]!=m["target_mm"]:
            raise RuntimeError("Mismatched evaluation goals")
        pairs.append(dict(row=m["identity"][1],target_mm=m["target_mm"],policy=m,rule=b))
    passed=all(m[k]["success"] for m in pairs for k in ("rule","policy"))
    (args.output/"comparison.json").write_text(json.dumps(dict(updates=updates,pairs=pairs,passed=passed),indent=2,allow_nan=False))
    if not passed:
        raise RuntimeError("Physical evaluation failed; see comparison.json")
    print("PASS: eight Go2 asynchronous PPO evaluation",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=37)
    parser.add_argument("--updates",type=int,default=1)
    parser.add_argument("--episodes-per-env",type=int,default=2)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--eval-only",action="store_true")
    parser.add_argument("--video",action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.updates<1 or args.episodes_per_env<1 or (args.eval_only and not args.checkpoint):
        parser.error("Positive rollout sizes required; --eval-only requires --checkpoint")
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
