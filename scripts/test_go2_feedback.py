"""Standalone checks for observation isolation and terminal-value semantics."""
import importlib.util
from pathlib import Path
import unittest
import torch

path=Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC/RL_Bezier_MPC/envs/go2_feedback.py"
spec=importlib.util.spec_from_file_location("go2_feedback",path)
feedback=importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback)


class FeedbackTests(unittest.TestCase):
    def test_reset_invalidates_only_selected_owner(self):
        ledger=feedback.EpisodeLedger(8,"cpu")
        ledger.reset(torch.arange(8))
        identity=ledger.begin()
        self.assertTrue((identity[:,1:]==0).all())
        peer_episode=ledger.episode[[0,2,3,4,5,6,7]].clone()
        ledger.reset(torch.tensor([1]))
        self.assertEqual(ledger.request_episode[1],-1)
        self.assertEqual(ledger.episode[1],1)
        self.assertEqual(ledger.decision[1],0)
        self.assertTrue(torch.equal(peer_episode,ledger.episode[[0,2,3,4,5,6,7]]))

    def test_velocity_sources_and_reward_are_row_local(self):
        features=[torch.zeros(8,n) for n in (2,3,3,12,12,4,3,3,12,6)]
        a=feedback.build_observation(*features,"measured")
        features[7][2]=1.
        self.assertTrue(torch.equal(a,feedback.build_observation(*features,"measured")))
        b=feedback.build_observation(*features,"mpc_prediction")
        peers=[0,1,3,4,5,6,7]
        self.assertTrue(torch.equal(a[peers],b[peers]))
        self.assertFalse(torch.equal(a[2],b[2]))
        inputs=[torch.zeros(8),torch.zeros(8),torch.zeros(8),torch.zeros(8,12),
                torch.zeros(8,4),torch.zeros(8,dtype=torch.bool),torch.zeros(8,dtype=torch.bool)]
        a=feedback.dense_reward(*inputs,.25)
        inputs[0][2]=.02
        inputs[3][2]=3.
        inputs[4][2]=1.
        b=feedback.dense_reward(*inputs,.25)
        self.assertTrue(torch.equal(a[peers],b[peers]))
        self.assertNotEqual(a[2],b[2])

    def test_time_limit_bootstraps_but_cuts_recursion(self):
        rewards=torch.tensor([[1.],[2.],[100.]])
        values=torch.zeros_like(rewards)
        next_values=torch.tensor([[10.],[20.],[30.]])
        terminated=torch.zeros_like(rewards,dtype=torch.bool)
        truncated=terminated.clone()
        truncated[1]=True
        advantage,_=feedback.generalized_advantage(rewards,values,next_values,terminated,truncated,gamma=.9,lam=1.)
        torch.testing.assert_close(advantage,torch.tensor([[28.],[20.],[127.]]))
        terminated[1]=True
        truncated[1]=False
        advantage,_=feedback.generalized_advantage(rewards,values,next_values,terminated,truncated,gamma=.9,lam=1.)
        torch.testing.assert_close(advantage,torch.tensor([[11.8],[2.],[127.]]))


if __name__=="__main__":
    unittest.main()
