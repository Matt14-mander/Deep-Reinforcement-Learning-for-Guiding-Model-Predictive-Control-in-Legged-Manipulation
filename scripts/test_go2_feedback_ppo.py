"""Probability and optimizer acceptance without Isaac Sim."""
import io
import unittest
import torch
from go2_feedback_ppo import FeedbackPolicy,update


class FeedbackPPOTests(unittest.TestCase):
    def test_mixed_actions_and_tanh_density(self):
        torch.manual_seed(3)
        policy=FeedbackPolicy()
        obs=torch.randn(32,57)
        with torch.no_grad():
            action,latent,log,value=policy.act(obs)
            self.assertTrue((action.abs()<=1).all())
            self.assertTrue(torch.equal(action[:,0],2*latent[:,0]-1))
            self.assertTrue(torch.equal(action[:,2],latent[:,1]-1))
            self.assertTrue(torch.equal(action[:,[1,3]],latent[:,2:].tanh()))
            gait,period,normal=policy.distributions(obs)
            expected=gait.log_prob(latent[:,0])+period.log_prob(latent[:,1])+(normal.log_prob(latent[:,2:])-torch.log1p(-action[:,[1,3]].square())).sum(-1)
            self.assertTrue(torch.allclose(log,expected,atol=1e-6))
            latent[:,2:]=100.
            self.assertTrue(torch.isfinite(policy.log_prob(obs,latent)).all())

    def test_update_and_optimizer_reload(self):
        torch.manual_seed(5)
        policy=FeedbackPolicy()
        optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
        obs=torch.randn(32,57)
        with torch.no_grad():
            _,latent,logs,values=policy.act(obs)
        advantage=torch.randn(32)
        batch=(obs,latent,logs,advantage,values+advantage)
        self.assertGreater(update(policy,optimizer,batch)["actor_change"],0)
        stream=io.BytesIO()
        torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict()),stream)
        stream.seek(0)
        state=torch.load(stream,weights_only=True)
        replay=FeedbackPolicy()
        replay.load_state_dict(state["policy"])
        replay_optimizer=torch.optim.Adam(replay.parameters())
        replay_optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            self.assertTrue(torch.equal(policy.act(obs,True)[0],replay.act(obs,True)[0]))
        batch=(obs,latent,policy.log_prob(obs,latent).detach(),advantage,values+advantage)
        update(policy,optimizer,batch,epochs=1)
        update(replay,replay_optimizer,batch,epochs=1)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(policy.parameters(),replay.parameters())))


if __name__=="__main__":
    unittest.main()
