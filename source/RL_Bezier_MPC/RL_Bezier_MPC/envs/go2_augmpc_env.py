"""Continuous high-level transitions with independent Go2 episode ownership."""
import torch

from .quadruped_mpc_env import QuadrupedMPCEnv
from .go2_cycle import CycleClock, SETTLE, make_plan, swing_groups
from .go2_feedback import EpisodeLedger, build_observation, dense_reward


class Go2AugMPCEnv(QuadrupedMPCEnv):
    def __init__(self,cfg,render_mode=None,**kwargs):
        self.feedback_ready=False
        super().__init__(cfg,render_mode,**kwargs)
        self.mpc=self.tracking.controller.mpc
        self.nominal=make_plan(self.mpc.model,cfg.mpc_dt,[("crawl",.020)],cycle_decisions=True)
        self.ledger=EpisodeLedger(self.num_envs,self.device)
        self.actions=torch.zeros(self.num_envs,4,device=self.device)
        self.previous_actions=torch.zeros_like(self.actions)
        self.commands=torch.zeros(self.num_envs,2,device=self.device)
        self.mask=torch.ones(self.num_envs,4,device=self.device,dtype=torch.bool)
        self.anchors=self.mpc.model.contact_positions.expand(self.num_envs,-1,-1).clone()
        self.origins=self.anchors.clone()
        self.drift=self.anchors.new_zeros(self.num_envs,4)
        self.residual=self.anchors.new_zeros(self.num_envs,2)
        self.predicted_velocity=self.anchors.new_zeros(self.num_envs,3)
        self.prediction_valid=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.health=torch.zeros(self.num_envs,6,device=self.device)
        self.failed=torch.zeros_like(self.prediction_valid)
        self.arrived=torch.zeros_like(self.prediction_valid)
        self.pending_cycle_action=torch.zeros_like(self.actions)
        self.applied_request=torch.full((self.num_envs,3),-1,device=self.device,dtype=torch.long)
        self.clocks=[self._new_clock() for _ in range(self.num_envs)]
        self.nodes=torch.arange(self.mpc.horizon+1,device=self.device)
        self.physics_ticks=0
        self.world=0
        self.solve_count=0
        self.feedback_ready=True
        self.action_low=torch.full((4,),-1.,device=self.device)
        self.action_high=torch.ones(4,device=self.device)
        print("[AugMPC] 4-D actions: gait, stride, period slot, forward CoM residual")

    def _new_clock(self):
        clock=CycleClock(self,64.,choose=self._choose_cycle,period_actions=True,max_cycles=None)
        clock.row_id=-1
        return clock

    def _choose_cycle(self,clock,obs,remaining):
        row=clock.row_id
        if self.ledger.request_episode[row]!=self.ledger.episode[row]:
            raise RuntimeError("Attempted to execute an action from a previous episode")
        action=self.actions[row]
        gait="trot" if action[0]>0 else "crawl"
        slot=int(torch.round(action[2]+1).clamp(0,2).item())
        self.pending_cycle_action[row]=action
        self.applied_request[row]=self.transition_identity[row]
        return gait,action[1].item(),slot

    def set_commands(self,ids,target_x,speed):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        target=torch.as_tensor(target_x,device=self.device,dtype=self.commands.dtype).detach().expand(len(ids))
        desired=torch.as_tensor(speed,device=self.device,dtype=self.commands.dtype).detach().expand(len(ids))
        if not (torch.isfinite(target).all() and torch.isfinite(desired).all() and (desired>=0).all()):
            raise ValueError("Commands must be finite; speed must be nonnegative")
        self.commands[ids,0]=target
        self.commands[ids,1]=desired

    def reset_envs(self,ids):
        """Return a truncation event for the collector to close the interrupted trajectory."""
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        terminal,_=self._observation()
        event=dict(env_ids=ids.clone(),episode_ids=self.ledger.episode[ids].clone(),
                   decision_ids=self.ledger.decision[ids].clone()-1,
                   bootstrap_observation=terminal[ids].clone(),
                   bootstrap_mask=torch.ones(len(ids),device=self.device,dtype=torch.bool),
                   trace_mask=torch.zeros(len(ids),device=self.device,dtype=torch.bool),
                   terminated=torch.zeros(len(ids),device=self.device,dtype=torch.bool),
                   truncated=torch.ones(len(ids),device=self.device,dtype=torch.bool),
                   reason="external_reset")
        self._reset_idx(ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        return self._get_observations(),event

    def _reset_idx(self,ids):
        super()._reset_idx(ids)
        if not self.feedback_ready or len(ids)==0:
            return
        self.ledger.reset(ids)
        self.actions[ids]=0
        self.previous_actions[ids]=0
        self.pending_cycle_action[ids]=0
        self.applied_request[ids]=-1
        self.residual[ids]=0
        self.predicted_velocity[ids]=0
        self.prediction_valid[ids]=False
        self.health[ids]=0
        self.failed[ids]=False
        self.arrived[ids]=False
        self.mask[ids]=True
        self.anchors[ids]=self.mpc.model.contact_positions
        self.origins[ids]=self.anchors[ids]
        self.drift[ids]=0
        fraction=(ids%4).float()/3
        self.commands[ids,0]=self.cfg.target_min_m+fraction*(self.cfg.target_max_m-self.cfg.target_min_m)
        self.commands[ids,1]=torch.where(ids%2==0,.003,.006)
        for row in ids.tolist():
            self.clocks[row]=self._new_clock()
            self.clocks[row].row_id=row

    def _state(self):
        d=self.robot.data
        return self.tracking.pack_state(d.root_link_pose_w.double(),d.root_link_vel_w.double(),
            d.joint_pos.double(),d.joint_vel.double(),self.tracking_origins)

    def _loads(self):
        return self.foot_contact_sensor.data.net_forces_w[:,self._contact_sensor_foot_indices,2]

    def _pre_physics_step(self,actions):
        if actions.shape!=self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Expected finite [num_envs,4] actions")
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.detach().clamp(-1,1))
        self.transition_identity=self.ledger.begin()
        self.start_distance=(self.commands[:,0]-self._state()[:,0]).abs().float()
        self.step_solve_start=self.solve_count

    def _apply_action(self):
        # 50 physics steps per policy step; the MPC phase continues across calls.
        if self.physics_ticks%self.cfg.mpc_decimation==0:
            self._solve_mpc_control_step()
            self.solve_count+=1
        self.robot.set_joint_effort_target(self._pending_joint_efforts)
        self.physics_ticks+=1

    def _solve_mpc_control_step(self):
        state=self._state()
        feet=(self.robot.data.body_link_pos_w[:,self._foot_body_indices.tolist()].double()
              -self.tracking_origins[:,None])
        loads=self._loads()
        finite=torch.isfinite(state).all(-1)&torch.isfinite(feet).all((-1,-2))&torch.isfinite(loads).all(-1)
        self.failed |= ~finite
        safe_state=torch.where(finite[:,None],state,self.mpc.model.standing_state)
        refs,accs=[],[]
        mask=torch.ones_like(self.mask)
        for row,clock in enumerate(self.clocks):
            origin=clock.start_x if clock.start_x is not None else safe_state[row,0].item()
            clock.goal=1000*(self.commands[row,0].item()-origin)
            remaining=self.commands[row,0].item()-safe_state[row,0].item()
            if clock.finishing and remaining>self.cfg.goal_tolerance_m:
                clock.finishing=False
                clock.boundary=clock.tick
                clock.stable=0
            if not clock.schedule and clock.tick==SETTLE and remaining<=self.cfg.goal_tolerance_m:
                clock.finishing=True
            if not self.failed[row] and self.ledger.request_episode[row]==self.ledger.episode[row]:
                clock.prepare(safe_state[row,0].item(),loads[row])
            reference,acceleration,phases=clock.plan
            index=min(max(clock.tick,0),len(reference)-1)
            cycle,pair,swing=phases[index]
            if swing:
                mask[row,list(swing_groups(clock.schedule[cycle][0])[pair])]=False
            refs.append(reference[(self.nodes+index).clamp_max(len(reference)-1)])
            accs.append(acceleration[index])
            if clock.tick==SETTLE:
                self.origins[row]=feet[row]
            steady=(clock.finishing and abs(remaining)<=self.cfg.goal_tolerance_m
                    and loads[row].min()>5 and safe_state[row,19:22].norm()<.01
                    and safe_state[row,22:25].norm()<.04 and safe_state[row,25:].abs().max()<.1)
            clock.stable=clock.stable+1 if steady else 0
            if not clock.finishing:
                clock.tick+=1
        changed=mask^self.mask
        if changed.any():
            gained=mask&changed
            self.anchors[:,:,:2][gained]=feet[:,:,:2][gained]
            self.origins[gained]=feet[gained]
            self.mpc.reset(changed.any(-1))
        self.mask.copy_(mask)
        reference=torch.stack(refs)
        acceleration=torch.stack(accs)
        # Critically damped residual filter gives fixed-frequency actions an immediate,
        # bounded effect while contact timing stays with the validated scheduler.
        dt=self.cfg.mpc_dt
        residual_acc=100*(self.cfg.residual_limit_m*self.actions[:,3]-self.residual[:,0])-20*self.residual[:,1]
        self.residual[:,1]+=dt*residual_acc
        self.residual[:,0]+=dt*self.residual[:,1]
        horizon=self.nodes*dt
        reference[:,:,0]+=self.residual[:,0,None]+horizon*self.residual[:,1,None]+.5*horizon.square()*residual_acc[:,None]
        reference[:,:,19]+=self.residual[:,1,None]+horizon*residual_acc[:,None]
        acceleration[:,0]+=residual_acc
        tau,result=self.mpc.compute(safe_state,reference,acceleration,self.mask,self.anchors)
        self.failed |= ~result.usable
        self.prediction_valid.copy_(result.usable)
        self.predicted_velocity.copy_(torch.nan_to_num(result.xs[:,1,19:22]))
        cost=torch.nan_to_num(result.cost,nan=1e6,posinf=1e6,neginf=1e6).clamp(0,1e6)
        self.health.copy_(torch.stack((result.usable.float(),self.prediction_valid.float(),
            result.status.float()/4,result.iterations.float()/16,torch.log1p(cost)/10,self.failed.float()),-1))
        action=self.tracking.scatter_action(tau).clamp(-self.cfg.max_joint_torque,self.cfg.max_joint_torque)
        self._pending_joint_efforts.copy_(torch.where(self.failed[:,None],0.,action))
        drift=(feet-self.origins).norm(dim=-1)*self.mask
        self.drift=torch.maximum(self.drift,torch.nan_to_num(drift))
        self.world+=1

    def _get_dones(self):
        state=self._state()
        up=self.robot.data.projected_gravity_b[:,2]
        finite=torch.isfinite(state).all(-1)&torch.isfinite(up)&torch.isfinite(self._loads()).all(-1)
        self.failed |= ~finite | (state[:,2]<.15) | (state[:,2]>.6) | (up>-.5)
        self.arrived.copy_(torch.tensor([c.stable>=self.cfg.stop_ticks for c in self.clocks],
                                       device=self.device)&~self.failed)
        self.health[:,5]=self.failed.float()
        terminated=self.failed|self.arrived
        truncated=(self.episode_length_buf>=self.max_episode_length)&~terminated
        self._transition_terminated=terminated.clone()
        self._transition_truncated=truncated.clone()
        return terminated,truncated

    def _observation(self):
        state=self._state()
        remaining=self.commands[:,0]-state[:,0]
        command=torch.stack((remaining/.1,self.commands[:,1]/.01),-1)
        phases=[]
        for clock in self.clocks:
            index=min(max(clock.tick,0),len(clock.plan[0])-1)
            cycle=clock.plan[2][index][0]
            gait=clock.schedule[cycle][0] if 0<=cycle<len(clock.schedule) else None
            start=clock.decisions[-1]["local_tick"] if clock.decisions else 0
            fraction=max(0.,min(1.,(clock.tick-start)/max(1,clock.boundary-start)))
            phases.append((fraction,float(gait=="crawl"),float(gait=="trot"),float(clock.period_slot-1)))
        plan=torch.cat((self.mask.float(),(self._loads()>5).float(),
                        torch.tensor(phases,device=self.device)),-1)
        obs=build_observation(command,self.robot.data.projected_gravity_b,state[:,22:25],
            state[:,7:19]-self.mpc.model.reference_configuration[7:19],state[:,25:],
            self.actions,state[:,19:22],self.predicted_velocity,plan,self.health,
            self.cfg.policy_velocity_source)
        return torch.nan_to_num(obs),state

    def _get_observations(self):
        obs,_=self._observation()
        return {"policy":obs}

    def _get_rewards(self):
        obs,state=self._observation()
        distance=(self.commands[:,0]-state[:,0]).abs().float()
        desired=torch.where(distance<=self.cfg.goal_tolerance_m,0.,self.commands[:,1])
        progress=self.start_distance-distance
        reward=dense_reward(progress,state[:,19].float(),desired,self._pending_joint_efforts,
                            self.actions-self.previous_actions,self.failed,self.arrived,self.step_dt)
        reward=torch.nan_to_num(reward,nan=-1.,posinf=-1.,neginf=-1.)
        self.extras["time_outs"]=self._transition_truncated.clone()
        self.extras["transition"]=dict(
            identity=self.transition_identity.clone(),bootstrap_observation=obs.clone(),
            terminated_success=self.arrived.clone(),terminated_failure=self.failed.clone(),
            bootstrap_mask=(~self._transition_terminated).clone(),
            trace_mask=(~(self._transition_terminated|self._transition_truncated)).clone(),
            measured_velocity_b=state[:,19:22].float().clone(),
            mpc_predicted_velocity_b=self.predicted_velocity.float().clone(),
            mpc_prediction_valid=self.prediction_valid.clone(),
            mpc_health=self.health.clone(),actions=self.actions.clone(),
            applied_cycle_request=self.applied_request.clone(),contact_plan=self.mask.clone(),
            reward=reward.clone(),elapsed_s=self.step_dt,commands=self.commands.clone(),
            effort=self._pending_joint_efforts.clone(),
            action_change=(self.actions-self.previous_actions).clone(),
            residual_state=self.residual.clone(),
            mpc_ticks=self.solve_count-self.step_solve_start,
            measured_time_s=self.physics_ticks*self.cfg.sim.dt,
            mpc_prediction_time_s=self.solve_count*self.cfg.mpc_dt,
            velocity_kind="one_step_mpc_prediction_from_measured_state")
        return reward
