"""Small mixed-action PPO for the 57-D continuous-feedback contract."""
import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F


class FeedbackPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(57,64),nn.Tanh(),nn.Linear(64,7))
        self.critic=nn.Sequential(nn.Linear(57,64),nn.Tanh(),nn.Linear(64,1))
        self.log_std=nn.Parameter(torch.full((2,),-1.5))
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.tensor([1.5,0.,0.,2.,0.,0.,0.]))

    def distributions(self,obs):
        out=self.actor(obs)
        return Categorical(logits=out[:,:2]),Categorical(logits=out[:,2:5]),Normal(out[:,5:],self.log_std.clamp(-5,1).exp())

    def value(self,obs):
        return self.critic(obs).squeeze(-1)

    def log_prob(self,obs,latent):
        gait,period,normal=self.distributions(obs)
        z=latent[:,2:]
        jacobian=2*(0.6931471805599453-z-F.softplus(-2*z))
        return gait.log_prob(latent[:,0].long())+period.log_prob(latent[:,1].long())+(normal.log_prob(z)-jacobian).sum(-1)

    def act(self,obs,deterministic=False):
        gait,period,normal=self.distributions(obs)
        g=gait.logits.argmax(-1) if deterministic else gait.sample()
        p=period.logits.argmax(-1) if deterministic else period.sample()
        z=normal.loc if deterministic else normal.sample()
        latent=torch.cat((g[:,None],p[:,None],z),-1)
        action=torch.stack((2*g-1,z[:,0].tanh(),p-1,z[:,1].tanh()),-1)
        return action,latent,self.log_prob(obs,latent),self.value(obs)


def update(policy,optimizer,batch,epochs=4):
    obs,latent,old_log,adv,returns=batch
    adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    with torch.no_grad():
        initial_ratio=(policy.log_prob(obs,latent)-old_log).exp()
        if not torch.allclose(initial_ratio,torch.ones_like(initial_ratio),atol=1e-5,rtol=0):
            raise RuntimeError("Rollout likelihood does not match executed policy")
    before=torch.cat([p.detach().flatten().clone() for p in policy.actor.parameters()])
    for _ in range(epochs):
        log=policy.log_prob(obs,latent)
        ratio=(log-old_log).exp()
        actor=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        value=.5*(policy.value(obs)-returns).square().mean()
        loss=actor+.5*value
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite PPO loss")
        optimizer.zero_grad()
        loss.backward()
        norm=nn.utils.clip_grad_norm_(policy.parameters(),1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
    after=torch.cat([p.detach().flatten() for p in policy.actor.parameters()])
    change=(after-before).norm().item()
    if change<=0:
        raise RuntimeError("Actor did not update")
    return dict(loss=loss.item(),actor_change=change,gradient_norm=norm.item(),epochs=epochs)
