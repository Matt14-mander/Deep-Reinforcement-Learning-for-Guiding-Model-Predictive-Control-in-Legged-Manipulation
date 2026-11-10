"""Single-Go2 slow walking trot: diagonal load transfer, cruise, braking and stop."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from isaaclab.app import AppLauncher

# Foot array order: FL, FR, RL, RR. Each cycle swings both diagonal pairs.
PAIRS = ((0, 3), (1, 2))
STRIDES = (.014, .024, .024, .024, .016, .010)
SWING, TRANSFER = 16, 14
SETTLE, HOLD = 50, 100


CRAWL = ((2,), (0,), (3,), (1,))
MIXED = (("crawl", .020), ("trot", .018), ("trot", .024), ("trot", .024),
         ("trot", .018), ("crawl", .016))
HANDOFF = 10


def swing_groups(gait):
    return CRAWL if gait == "crawl" else PAIRS


def make_plan(model, dt, schedule):
    import torch
    from crocoddyl_batched_mpc.models.go2_walk import go2_leg_ik, go2_center_of_mass, smoothstep
    q0 = model.reference_configuration
    anchors = model.contact_positions.clone()
    body = q0[:3].clone()
    bodies, targets, phases = [], [], []

    def append(base, feet, phase):
        bodies.append(base.clone())
        targets.append(feet.clone())
        phases.append(phase)

    for _ in range(SETTLE):
        append(body, anchors, (-1, -1, False))
    for cycle, (gait, stride) in enumerate(schedule):
        for pair, swing in enumerate(swing_groups(gait)):
            support = [j for j in range(4) if j not in swing]
            desired = anchors[support, :2].mean(0)
            if gait == "crawl":
                leg = swing[0]
                midpoint = (anchors[leg ^ 1,:2] + anchors[leg ^ 2,:2])/2
                desired = torch.lerp(midpoint, desired, .35)
            base = q0.clone()
            base[:3] = body
            for _ in range(4):
                pose = go2_leg_ik(model, base[None], anchors[None])
                base[:2] += desired - go2_center_of_mass(model, pose)[0, :2]
            transfer_ticks = 45 if gait == "crawl" else TRANSFER
            for j in range(1, transfer_ticks+1):
                append(torch.lerp(body, base[:3], smoothstep(j/transfer_ticks)), anchors, (cycle, pair, False))
            body = base[:3].clone()
            for j in range(1, SWING+1):
                u = j / SWING
                feet = anchors.clone()
                feet[list(swing), 0] += stride * smoothstep(u)
                feet[list(swing), 2] += .020 * 64 * (u*(1-u))**3
                append(body, feet, (cycle, pair, True))
            anchors[list(swing), 0] += stride
            if gait == "crawl":
                for _ in range(14):
                    append(body, anchors, (cycle, -1, False))
        changing = cycle+1 < len(schedule) and schedule[cycle+1][0] != gait
        if gait == "crawl" or changing:
            # Compare progress at a centered four-foot stance, after unloading
            # the lateral/fore-aft displacement used to balance the last swing.
            center = q0[:3].clone()
            center[0] += sum(length for _, length in schedule[:cycle+1])
            for j in range(1, 36):
                append(torch.lerp(body,center,smoothstep(j/35)), anchors, (cycle,-1,False))
            body = center
        if changing:
            for _ in range(HANDOFF):
                append(body, anchors, (cycle, -1, False))
    end = q0[:3].clone()
    end[0] += sum(stride for _,stride in schedule)
    for j in range(1, HOLD+1):
        append(torch.lerp(body, end, smoothstep(min(1., j/TRANSFER))), anchors, (len(schedule), -1, False))
    feet = torch.stack(targets)
    bases = q0.expand(len(feet), -1).clone()
    bases[:, :3] = torch.stack(bodies)
    q = go2_leg_ik(model, bases, feet)
    v = q.new_zeros(len(q),18)
    v[1:-1,:3] = (q[2:,:3]-q[:-2,:3])/(2*dt)
    v[1:-1,6:] = (q[2:,7:]-q[:-2,7:])/(2*dt)
    acc = torch.zeros_like(v)
    acc[1:-1] = (v[2:]-v[:-2])/(2*dt)
    return torch.cat((q,v),-1), acc, phases


def run(args):
    import torch
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

    schedule = MIXED if args.mixed_gaits else tuple(("trot", stride) for stride in STRIDES)

    class Trot(QuadrupedMPCEnv):
        active = False

        def _reset_idx(self, ids):
            if self.active:
                raise RuntimeError(f"Episode reset at tick {self.tick}")
            return super()._reset_idx(ids)

        def _solve_mpc_control_step(self):
            i = self.tick
            d, adapter = self.robot.data, self.tracking
            mpc = adapter.controller.mpc
            state = adapter.pack_state(d.root_link_pose_w.double(), d.root_link_vel_w.double(),
                                       d.joint_pos.double(), d.joint_vel.double(), self.tracking_origins)
            cycle, pair, swing = self.phases[i]
            mask = torch.ones((1,4), device=self.device, dtype=torch.bool)
            gait = schedule[cycle][0] if 0 <= cycle < len(schedule) else "stance"
            if swing:
                mask[:, list(swing_groups(gait)[pair])] = False
            feet = d.body_link_pos_w[0,self._foot_body_indices.tolist()].double()-self.tracking_origins[0]
            loads = self.foot_contact_sensor.data.net_forces_w[0,self._contact_sensor_foot_indices,2]
            if not (torch.isfinite(state).all() and torch.isfinite(feet).all() and torch.isfinite(loads).all()):
                raise RuntimeError(f"Nonfinite measurement at {i}")
            changed = mask ^ self.mask
            if changed.any():
                gained = (mask & changed)[0]
                self.anchors[0,gained,:2] = feet[gained,:2]
                self.origin[gained] = feet[gained]
                mpc.reset()
            if i == SETTLE:
                self.origin = feet.clone()
            if i >= SETTLE:
                delta = ((feet-self.origin).norm(dim=-1)*mask[0]).tolist()
                self.drift = [max(a,b) for a,b in zip(self.drift,delta)]
            self.mask = mask
            w,x,y,z = d.root_link_quat_w[0].tolist()
            roll = math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)))
            pitch = math.degrees(math.asin(max(-1., min(1.,2*(w*y-z*x)))))
            yaw = math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
            self.rows.append(dict(tick=i,cycle=cycle,pair=pair,swing=swing,
                x=state[0,0].item(),y=state[0,1].item(),roll=roll,pitch=pitch,yaw=yaw,
                loads=loads.tolist(),speed=state[0,19:22].norm().item(),
                spin=state[0,22:25].norm().item(),joint=state[0,25:].abs().max().item()))
            ids = (torch.arange(mpc.horizon+1,device=self.device)+i).clamp_max(len(self.ref)-1)
            effort,res = mpc.compute(state,self.ref[ids][None],self.acc[i:i+1],mask,self.anchors)
            if not res.usable.all():
                raise RuntimeError(f"MPC rejected {i=} {cycle=} {pair=} {swing=} {self.rows[-1]}")
            self._pending_joint_efforts.copy_(adapter.scatter_action(effort).clamp(
                -self.cfg.max_joint_torque,self.cfg.max_joint_torque))
            if args.mixed_gaits:
                self.rows[-1].update(gait=gait, anchors=self.anchors[0].tolist(),
                    requested=self._pending_joint_efforts[0].tolist(), applied=d.applied_torque[0].tolist(),
                    quat=d.root_link_quat_w[0].tolist())
            self._last_mpc_costs, self._last_mpc_converged = res.cost,res.success
            self._advance_reference_clocks()
            self.tick += 1

    cfg = QuadrupedMPCEnvCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 1,37,args.device
    cfg.mpc_backend, cfg.use_mpc_cluster, cfg.torch_foot_plane_height = "torch_tracking",False,.022
    cfg.episode_length_s = 30.
    cfg.rl_policy_period, cfg.decimation = 1,cfg.mpc_decimation
    cfg.viewer.eye,cfg.viewer.lookat,cfg.viewer.resolution = (.65,-1.,.51),(.06,0,.20),(800,480)
    cfg.sim.render_interval = cfg.decimation
    args.output.mkdir(parents=True,exist_ok=False)
    env = Trot(cfg,render_mode="rgb_array" if args.video else None)
    writer = None
    try:
        env.reset(seed=cfg.seed)
        mpc = env.tracking.controller.mpc
        env.ref,env.acc,env.phases = make_plan(mpc.model,cfg.mpc_dt,schedule)
        env.mask = torch.ones(1,4,device=env.device,dtype=torch.bool)
        env.anchors = mpc.model.contact_positions[None].clone()
        env.origin = env.anchors[0].clone()
        env.rows,env.drift,env.tick,env.active = [],[0.]*4,0,True
        zero = torch.zeros((1,cfg.action_space),device=env.device)
        if args.video:
            writer = subprocess.Popen(["ffmpeg","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
                "-s","800x480","-r",str(1/cfg.mpc_dt),"-i","pipe:0","-c:v","libx264","-pix_fmt",
                "yuv420p","-movflags","+faststart",str(args.output/("mixed_gaits.mp4" if args.mixed_gaits else "trot.mp4"))],stdin=subprocess.PIPE)
        while env.tick < len(env.ref):
            _,_,term,trunc,_ = env.step(zero)
            if (term|trunc).any():
                raise RuntimeError("Trot terminated before stop")
            if writer:
                rgb = env.render()
                for _ in range(8):
                    if rgb.size and rgb[40:430:8,::8,0].std()>8:
                        break
                    env.sim.set_camera_view(cfg.viewer.eye,cfg.viewer.lookat)
                    env.sim.render()
                    rgb = env.render()
                else:
                    raise RuntimeError("Unusable RTX camera frame")
                frame = Image.fromarray(rgb).convert("RGB")
                draw = ImageDraw.Draw(frame)
                row = env.rows[-1]
                gait = schedule[row["cycle"]][0] if 0 <= row["cycle"] < len(schedule) else "stance"
                released = swing_groups(gait)[row["pair"]] if row["swing"] else ()
                label = "+".join(leg for j, leg in enumerate(("FL","FR","RL","RR")) if j not in released)
                draw.rectangle((0,435,800,480), fill="#182532")
                draw.text((12,442),f"GO2   /   {gait.upper()}     {env.tick*cfg.mpc_dt:.2f} s     planned support: {label}")
                draw.text((12,461),"Measured vertical force (N)    "+"    ".join(
                    f"{leg} {load:.0f}" for leg,load in zip(("FL","FR","RL","RR"),row["loads"])))
                writer.stdin.write(frame.tobytes())
        result = (evaluate_mixed if args.mixed_gaits else evaluate)(env.rows,env.drift,cfg.mpc_dt)
        print(json.dumps(result,allow_nan=False),flush=True)
        (args.output/"measurements.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        if not all(result["checks"].values()):
            raise RuntimeError(f"Failed checks: {[k for k,v in result['checks'].items() if not v]}")
    finally:
        try:
            if getattr(env,"rows",None):
                (args.output/"trace.json").write_text(json.dumps(env.rows))
            if writer:
                writer.stdin.close()
                if writer.wait():
                    raise RuntimeError("MP4 encoder failed")
        finally:
            env.close()


def evaluate(rows, drift, dt):
    period = 2*(SWING+TRANSFER)*dt
    fractions, recovered, speeds = [], [], []
    for i,stride in enumerate(STRIDES):
        start = SETTLE+i*2*(SWING+TRANSFER)
        speeds.append((rows[start+2*(SWING+TRANSFER)]["x"]-rows[start]["x"])/period)
        for pair in range(2):
            # Forces at each tick reflect the previous command interval.
            begin = start+pair*(SWING+TRANSFER)+TRANSFER
            middle = rows[begin+SWING//4+1:begin+3*SWING//4+1]
            correct = [all(r["loads"][j]>10 for j in PAIRS[1-pair]) and
                       all(r["loads"][j]<2 for j in PAIRS[pair]) for r in middle]
            fractions.append(sum(correct)/len(correct))
            recovery = rows[begin+SWING+1:begin+SWING+TRANSFER+1]
            recovered.append(any(all(min(r["loads"])>5 for r in recovery[k:k+3])
                                 for k in range(len(recovery)-2)))
    stop = rows[-50:]
    dx = rows[-1]["x"]-rows[SETTLE]["x"]
    dy = rows[-1]["y"]-rows[SETTLE]["y"]
    peak = [max(abs(r[k]) for r in rows[SETTLE:]) for k in ("roll","pitch","yaw")]
    cmd = [s/period for s in STRIDES]
    span = max(r["x"] for r in stop)-min(r["x"] for r in stop)
    return dict(command_mps=cmd,measured_cycle_mps=speeds,dx_m=dx,dy_m=dy,
        diagonal_fraction=fractions,four_foot_recovered=recovered,
        max_roll_pitch_yaw_deg=peak,max_stance_drift_mm=[1000*x for x in drift],
        stop_span_mm=1000*span,max_stop_speed_mps=max(r["speed"] for r in stop),
        checks=dict(diagonal=min(fractions)>=.75,landings=all(recovered),
            cruise=all(abs(speeds[i]-cmd[i])<=.25*cmd[i] for i in (1,2,3)),
            braking=speeds[3]>speeds[4]>speeds[5]>0,
            straight=abs(dx-sum(STRIDES))<.008 and abs(dy)<.01,
            attitude=max(peak)<5,drift=max(drift)<=.010,
            stopped=span<.001 and all(min(r["loads"])>5 and r["speed"]<.01 and
                                      r["spin"]<.04 and r["joint"]<.10 for r in stop)))


def evaluate_mixed(rows, drift, dt):
    from itertools import groupby
    from math import dist

    cycles, switches, contacts = [], [], []
    for i, (gait, stride) in enumerate(MIXED):
        samples = [r for r in rows if r["cycle"] == i]
        first, end = samples[0]["tick"], samples[-1]["tick"]+1
        duration = (end-first)*dt
        cycles.append(dict(gait=gait, command_mm_s=1000*stride/duration,
                           measured_mm_s=1000*(rows[end]["x"]-rows[first]["x"])/duration))
        for (pair, swing), group in groupby(samples, key=lambda r: (r["pair"], r["swing"])):
            if not swing:
                continue
            leg_ids = swing_groups(gait)[pair]
            ticks = list(group)
            start = ticks[0]["tick"]
            middle = rows[start+SWING//4+1:start+3*SWING//4+1]
            correct = [all(f<2 if j in leg_ids else f>10 for j,f in enumerate(r["loads"])) for r in middle]
            recovery = rows[start+SWING+1:start+SWING+TRANSFER+1]
            recovered = any(all(min(r["loads"])>5 for r in recovery[k:k+3]) for k in range(len(recovery)-2))
            contacts.append(dict(cycle=i+1, swing=leg_ids, support_fraction=sum(correct)/len(correct),
                                 recovered=recovered))
        if i and MIXED[i-1][0] != gait:
            # Include the first five commands in the new transfer and their measured response.
            window = rows[first-5:first+7]
            def max_joint_step(key):
                return max(abs(a-b) for prev,cur in zip(window,window[1:])
                           for a,b in zip(prev[key],cur[key]))
            def angle(a,b):
                dot = abs(sum(x*y for x,y in zip(a,b)))
                norms = math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
                return math.degrees(2*math.acos(min(1.,dot/norms)))
            switches.append(dict(change=MIXED[i-1][0]+" -> "+gait,tick=first,
                all_loaded=all(min(r["loads"])>5 for r in window),
                anchor_jump_mm=1000*max(dist(a,b) for r in window for a,b in zip(window[0]["anchors"],r["anchors"])),
                requested_step_Nm=max_joint_step("requested"), applied_step_Nm=max_joint_step("applied"),
                rotation_step_deg=max(angle(a["quat"],b["quat"]) for a,b in zip(window,window[1:]))))
    stop = rows[-50:]
    dx = 1000*(stop[-1]["x"]-rows[SETTLE]["x"])
    dy = 1000*(stop[-1]["y"]-rows[SETTLE]["y"])
    goal = 1000*sum(stride for _,stride in MIXED)
    span = 1000*(max(r["x"] for r in stop)-min(r["x"] for r in stop))
    peaks = [max(abs(r[k]) for r in rows) for k in ("roll","pitch","yaw")]
    return dict(cycles=cycles, switches=switches, contacts=contacts, max_stance_mm=[1000*x for x in drift],
        target_mm=goal, dx_mm=dx, dy_mm=dy, final_x_error_mm=dx-goal,
        max_roll_pitch_yaw_deg=peaks, stop_span_mm=span, stop_ticks=len(stop),
        checks=dict(handoffs=len(switches)==2 and all(s["all_loaded"] and s["anchor_jump_mm"]<.001
            and max(s["requested_step_Nm"],s["applied_step_Nm"])<3 and s["rotation_step_deg"]<1 for s in switches),
            contacts=len(contacts)==16 and all(c["support_fraction"]>=.75 and c["recovered"] for c in contacts),
            cruise=all(abs(c["measured_mm_s"]/c["command_mm_s"]-1)<.25 for c in cycles[2:4]),
            pacing=0<cycles[0]["measured_mm_s"]<min(c["measured_mm_s"] for c in cycles[2:4])
                and 0<cycles[-1]["measured_mm_s"]<cycles[-2]["measured_mm_s"],
            drift=max(drift)<=.010, attitude=max(peaks)<5, arrived=abs(dx-goal)<=4 and abs(dy)<=10,
            stopped=span<1 and all(min(r["loads"])>5 and r["speed"]<.01 and
                                  r["spin"]<.04 and r["joint"]<.10 for r in stop)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mixed-gaits",action="store_true", help="Run a crawl, four trot cycles and a final crawl")
    parser.add_argument("--video",action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video:
        args.enable_cameras = True
    launcher = AppLauncher(args)
    passed = False
    try:
        run(args)
        passed = True
        print("PASS: single Go2 crawl -> trot -> crawl" if args.mixed_gaits else
              "PASS: single Go2 slow diagonal trot",flush=True)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        try:
            launcher.app.close()
        finally:
            if not passed:
                raise SystemExit(1)
