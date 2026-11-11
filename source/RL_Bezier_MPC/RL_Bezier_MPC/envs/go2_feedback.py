"""Episode ownership, observation layout and time-limit aware advantage targets."""
import torch

OBSERVATION_FIELDS = (
    ("command",2), ("gravity_b",3), ("angular_velocity_b",3),
    ("joint_position_delta",12), ("joint_velocity",12), ("last_action",4),
    ("linear_velocity_b",3), ("contact_plan",12), ("mpc_health",6),
)
OBSERVATION_SIZE = sum(size for _,size in OBSERVATION_FIELDS)


class EpisodeLedger:
    def __init__(self, count, device):
        self.episode = torch.full((count,),-1,dtype=torch.long,device=device)
        self.decision = torch.zeros_like(self.episode)
        self.request_episode = torch.full_like(self.episode,-1)

    def reset(self, ids):
        self.episode[ids] += 1
        self.decision[ids] = 0
        self.request_episode[ids] = -1

    def begin(self):
        identity = torch.stack((torch.arange(len(self.episode),device=self.episode.device),
                                self.episode,self.decision),dim=-1)
        self.request_episode.copy_(self.episode)
        self.decision += 1
        return identity


def build_observation(command,gravity,angular,joint_delta,joint_velocity,last_action,
                      measured_velocity,mpc_velocity,contact_plan,health,velocity_source):
    if velocity_source not in ("measured","mpc_prediction"):
        raise ValueError("Unknown policy velocity source")
    velocity = measured_velocity if velocity_source=="measured" else mpc_velocity
    values=(command,gravity,angular,joint_delta,joint_velocity/10.,last_action,
            velocity/.01,contact_plan,health)
    for value,(_,width) in zip(values,OBSERVATION_FIELDS):
        if value.shape != (len(command),width):
            raise ValueError("Observation feature shape mismatch")
    return torch.cat(values,dim=-1).float()


def dense_reward(progress,velocity,desired,effort,action_change,failed,arrived,dt):
    tracking=torch.exp(-((velocity-desired)/.01).square())
    return (100*progress + dt*tracking - .001*dt*effort.square().mean(-1)
            - .02*action_change.square().mean(-1) - failed.float() + arrived.float())


def generalized_advantage(rewards,values,next_values,terminated,truncated,gamma=.99,lam=.95):
    """next_values must evaluate pre-reset terminal observations, including time limits."""
    if not (rewards.shape==values.shape==next_values.shape==terminated.shape==truncated.shape):
        raise ValueError("GAE expects matching [time, environment] tensors")
    if terminated.dtype!=torch.bool or truncated.dtype!=torch.bool:
        raise ValueError("Termination masks must be boolean")
    delta=rewards+gamma*(~terminated)*next_values-values
    advantage=torch.empty_like(delta)
    carry=torch.zeros_like(delta[0])
    for tick in range(len(delta)-1,-1,-1):
        carry=delta[tick]+gamma*lam*(~(terminated[tick]|truncated[tick]))*carry
        advantage[tick]=carry
    return advantage,advantage+values
