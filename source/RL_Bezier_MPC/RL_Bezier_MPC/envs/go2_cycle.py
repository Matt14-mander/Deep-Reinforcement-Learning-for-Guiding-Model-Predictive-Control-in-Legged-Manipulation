"""Shared Go2 contact schedule and per-environment MPC clock."""
import math
import json
import torch

# Foot array order: FL, FR, RL, RR. Each cycle swings both diagonal pairs.
PAIRS = ((0, 3), (1, 2))
STRIDES = (.014, .024, .024, .024, .016, .010)
SWING, TRANSFER = 16, 14
SETTLE, HOLD = 50, 100


CRAWL = ((2,), (0,), (3,), (1,))
MIXED = (("crawl", .020), ("trot", .018), ("trot", .024), ("trot", .024),
         ("trot", .018), ("crawl", .016))
HANDOFF = 10
PERIOD_SCALES = {"crawl": (1.2, 1., .8), "trot": (1.2, 1., .9)}


def swing_groups(gait):
    return CRAWL if gait == "crawl" else PAIRS


def cycle_timing(scale=1.0):
    if not math.isfinite(scale) or not .5 <= scale <= 2.:
        raise ValueError("Cycle time scale must be finite and between 0.5 and 2")
    return dict(swing=round(SWING*scale), transfer=round(TRANSFER*scale),
                crawl_transfer=round(45*scale), center=round(35*scale),
                recovery=max(14,round(14*scale)), handoff=max(HANDOFF,round(HANDOFF*scale)))


def make_plan(model, dt, schedule, *, cycle_decisions=False, time_scale=1.0, time_scales=None):
    import torch
    from crocoddyl_batched_mpc.models.go2_walk import go2_leg_ik, go2_center_of_mass, smoothstep
    scales = [time_scale]*len(schedule) if time_scales is None else list(time_scales)
    if len(scales) != len(schedule):
        raise ValueError("One time scale is required per cycle")
    timings = [cycle_timing(scale) for scale in scales]
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
        timing = timings[cycle]
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
            transfer_ticks = timing["crawl_transfer"] if gait == "crawl" else timing["transfer"]
            for j in range(1, transfer_ticks+1):
                append(torch.lerp(body, base[:3], smoothstep(j/transfer_ticks)), anchors, (cycle, pair, False))
            body = base[:3].clone()
            for j in range(1, timing["swing"]+1):
                u = j / timing["swing"]
                feet = anchors.clone()
                feet[list(swing), 0] += stride * smoothstep(u)
                feet[list(swing), 2] += .020 * 64 * (u*(1-u))**3
                append(body, feet, (cycle, pair, True))
            anchors[list(swing), 0] += stride
            if gait == "crawl":
                for _ in range(timing["recovery"]):
                    append(body, anchors, (cycle, -1, False))
        changing = cycle+1 < len(schedule) and schedule[cycle+1][0] != gait
        if gait == "crawl" or changing or cycle_decisions:
            # Compare progress at a centered four-foot stance, after unloading
            # the lateral/fore-aft displacement used to balance the last swing.
            center = q0[:3].clone()
            center[0] += sum(length for _, length in schedule[:cycle+1])
            for j in range(1, timing["center"]+1):
                append(torch.lerp(body,center,smoothstep(j/timing["center"])), anchors, (cycle,-1,False))
            body = center
        if changing or cycle_decisions:
            for _ in range(timing["handoff"]):
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


class CycleClock:
    def __init__(self,env,goal,delay=0,*,choose,period_actions=False,verbose=False,max_cycles=7):
        self.choose,self.period_actions,self.verbose,self.max_cycles = choose,period_actions,verbose,max_cycles
        self.env = env
        self.goal = goal
        self.tick = -delay
        self.boundary = SETTLE
        self.schedule, self.decisions, self.rows = [], [], []
        self.time_scales = []
        self.period_slot = 1
        self.plan = env.nominal
        self.start_x = None
        self.finishing = self.parked = False
        self.stable = 0

    def prepare(self,x,loads):
        if self.parked or self.finishing or self.tick != self.boundary:
            return
        if not (loads>5).all():
            raise RuntimeError(f"Missing four-foot recovery at decision for {self.goal} mm")
        if self.start_x is None:
            self.start_x = x
        remaining = self.goal-1000*(x-self.start_x)
        if self.schedule and remaining <= 4:
            self.finishing = True
            return
        if self.max_cycles is not None and len(self.schedule)>=self.max_cycles:
            raise RuntimeError("Cycle budget exceeded")
        obs = torch.tensor([(remaining-30)/15,float(bool(self.schedule and self.schedule[-1][0]=="trot"))])
        action = self.choose(self,obs,remaining)
        gait,z = action[:2]
        if self.period_actions:
            slot = action[2]
            if isinstance(slot, bool) or not isinstance(slot, int) or slot not in range(3):
                raise ValueError("Period slot must be 0, 1 or 2")
            self.period_slot = slot
            self.time_scales.append(PERIOD_SCALES[gait][slot])
        else:
            self.time_scales.append(1.)
        stride = max(10.,min(24.,min(remaining,21.)+3*math.tanh(z)))
        self.schedule.append((gait,stride/1000))
        reference,acceleration,phases = make_plan(
            self.env.tracking.controller.mpc.model,self.env.cfg.mpc_dt,self.schedule,
            cycle_decisions=True,time_scales=self.time_scales)
        # Keep all previously executed samples, including boundary derivatives.
        self.plan = (torch.cat((self.plan[0][:self.tick],reference[self.tick:])),
                     torch.cat((self.plan[1][:self.tick],acceleration[self.tick:])),
                     self.plan[2][:self.tick]+phases[self.tick:])
        self.boundary = len(self.plan[0])-HOLD
        event = dict(world_tick=self.env.world,local_tick=self.tick,remaining_mm=remaining,gait=gait,stride_mm=stride)
        if self.period_actions:
            event.update(period_slot=self.period_slot,time_scale=self.time_scales[-1],
                         duration_s=(self.boundary-self.tick)*self.env.cfg.mpc_dt)
        self.decisions.append(event)
        if self.verbose:
            print(json.dumps(dict(stage="decision",goal_mm=self.goal,**event)),flush=True)
