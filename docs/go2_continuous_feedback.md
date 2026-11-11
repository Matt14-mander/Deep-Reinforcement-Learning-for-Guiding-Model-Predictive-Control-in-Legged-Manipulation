# Go2 continuous feedback environment

Gym ID: `Isaac-Go2-AugMPC-Direct-v0`. The environment reuses the existing
robot, sensors, state packing, Torch MPC and effort actuator integration.
The old cycle-level PPO scripts remain separate experiments; their checkpoints
are not compatible with this new observation/action interface.

## Timing and actions

Default physics is 200 Hz, MPC 50 Hz, and policy 4 Hz. A policy step advances
50 physics ticks. The global MPC phase persists across policy steps and
individual resets, giving alternating 13 and 12 MPC solves per policy step.

Four normalized actions are clipped to [-1,1]: gait (positive=trot, otherwise
crawl), stride residual, period slot (round(action+1)), and forward CoM residual.
The first three are requests consumed at the next safe cycle boundary. The
fourth drives a critically damped filter at 50 Hz, bounded to a 1 mm target,
so policy actions also affect control within the active cycle. Existing swing
schedules are retained. This is a feedback interface acceptance, not proof of
successful PPO learning or large-batch throughput.

## Observation and transition contract

The 57-D observation contains remaining distance and desired speed, projected
gravity, body angular velocity, joint displacement/velocity, last action,
body linear velocity, planned/measured contacts and phase summary, and MPC
health. See `go2_feedback.py` for field order and scaling. Measured velocity is
the default policy input. MPC velocity is explicitly a one-step prediction
initialized from measured state, not an independent state estimator. Both are
logged with their own timestamps and a prediction-valid flag. Switching to
`mpc_prediction` is a configuration option, not an estimator validation.

Each transition has [environment, episode, decision] identity. Automatic reset
returns the new episode observation; `info['transition']['bootstrap_observation']`
is the final pre-reset observation for value estimation. Natural termination
(success or failure) suppresses bootstrap. Time truncation keeps bootstrap but
cuts GAE recursion. `generalized_advantage` implements both masks. Do not use
the automatically reset observation as the terminal next value.

Explicit `reset_envs(ids)` returns `(observations, event)`. The collector must
mark the identified last transition as truncated, replace its next value with
that from the event's pre-reset observation, and cut recursion. A decision ID
of -1 means no action has occurred in that episode, so there is no transition
to modify. Reset clears only those environments' action requests, filtered
residuals, warm starts, contact schedules and episode counters.

## Server acceptance

```bash
conda activate limx
cd ~/workspace/Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
git switch go2-torch-integration
git pull --ff-only origin go2-torch-integration
export PYTHONPATH="$HOME/workspace/Crocoddyl-batched-mpc/src${PYTHONPATH:+:$PYTHONPATH}"
RUN="logs/go2_augmpc/check_$(date +%Y%m%d_%H%M%S)"
python -u scripts/check_go2_augmpc.py --headless --device cuda:0 --seconds 60 --output "$RUN"
python -m json.tool "$RUN/acceptance.json"
```

The acceptance executes at least 60 seconds of world simulation with eight
environments, command changes, natural termination, time truncation and two
individual resets during swing. It checks row-isolated MPC/cache responses,
physical peer continuity on reset, action/reward identity, finite outputs,
exact frequencies, and terminal-value semantics. Stance drift is reported as
a diagnostic; this test does not introduce a new locomotion drift threshold.
Pure helper tests: `python scripts/test_go2_feedback.py`.

## Local acceptance result (2026-11-11)

Eight Go2s completed 60 s in PhysX: 1,920 policy transitions, 3,000 MPC ticks,
12 natural terminations, 11 time truncations, and two explicit resets during
swing. Four target changes were exercised. All isolation, finite-value,
frequency and bootstrap assertions passed. Reported maximum stance drift was
5.2115 mm (diagnostic). Three feedback helper tests and five existing Stage 2
training-state tests passed. Extracted planner outputs matched the previous
planner for crawl/trot across all three period scales.

## Eight-environment PPO smoke acceptance

```bash
conda activate limx
(
  set -e
  cd ~/workspace/Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
  git switch go2-torch-integration
  git pull --ff-only origin go2-torch-integration
  export PYTHONPATH="$HOME/workspace/Crocoddyl-batched-mpc/src${PYTHONPATH:+:$PYTHONPATH}"
  RUN="logs/go2_augmpc/ppo_$(date +%Y%m%d_%H%M%S)"
  python -u scripts/check_go2_feedback_ppo.py --headless --device cuda:0 --steps 96 --output "$RUN"
  python -m json.tool "$RUN/acceptance.json"
)
```

This collects 96 steps per environment (768 transitions, 24 simulated seconds),
then performs ONE PPO update with four full-batch optimization epochs. The actor
uses categorical gait/period choices and tanh-transformed Gaussian stride/CoM
residuals. Likelihood includes the tanh Jacobian; latent samples are retained
rather than reconstructing them from clipped actions. The sampled categorical
choices map exactly to the environment's gait threshold and period slots.

The first three action components are pending requests, consumed only at safe
cycle boundaries, while the fourth affects ongoing control. PPO records requests
at their submission time; it does not claim all components execute immediately.

The acceptance checks rollout identities, executed actions, finite signals, and
termination masks. GAE uses pre-reset final observations for time-limit bootstrap
and cuts recursion for all episode boundaries. It is independently checked per
row. Environment 0 starts at an already-reached goal and environment 6 at a far
goal to guarantee coverage of natural termination and timeout; these are test
fixtures, not a training target distribution.

Outputs are `rollout.pt`, `feedback_0001.pt`, and `acceptance.json`. The checkpoint
stores model, Adam state, CPU/CUDA RNG state, iteration and interface dimensions.
Exact deterministic outputs after reload, an identical next Adam update, and
identical stochastic samples after RNG restoration are checked. Four real PhysX
steps then exercise the reloaded mean policy. The optimizer comparison is a
throwaway probe; the saved checkpoint remains iteration 1. This checkpoint does
not serialize PhysX state or in-flight episodes, so restarting collection must
reset the environment and begin a fresh rollout.

This acceptance is not a full-episode policy-vs-rule evaluation or evidence of
learned arrival behavior. Next, add matched-seed complete-episode evaluations
before starting longer training. Unit tests: `python scripts/test_go2_feedback_ppo.py`.

Local PPO acceptance (2026-11-11): 768 transitions / 24 s, 3 natural
terminations, 6 time truncations, zero failures. One update (4 epochs) changed
actor parameters by L2 0.048533 and deterministic actions by maximum 0.029177.
Loss was finite (2.030428). Model/optimizer/RNG reload checks and four post-reload
PhysX steps passed. Two probability/optimizer tests and three feedback helper
tests passed. These changes establish a working training pipeline, not measured
improvement in goal-reaching performance.

## Deterministic full-episode comparison

Run the evaluator on the actual iteration-1 checkpoint from the smoke test:

```bash
conda activate limx
(
  set -e
  cd ~/workspace/Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
  git switch go2-torch-integration
  git pull --ff-only origin go2-torch-integration
  export PYTHONPATH="$HOME/workspace/Crocoddyl-batched-mpc/src${PYTHONPATH:+:$PYTHONPATH}"
  CHECKPOINT=$(python -c 'from pathlib import Path; p=list(Path("logs/go2_augmpc").rglob("feedback_0001.pt")); assert p, "No feedback_0001.pt found; run the PPO smoke acceptance first"; print(max(p,key=lambda x:x.stat().st_mtime).resolve())')
  RUN="logs/go2_augmpc/evaluation_$(date +%Y%m%d_%H%M%S)"
  echo "Checkpoint: $CHECKPOINT"
  python -u scripts/evaluate_go2_feedback_ppo.py --headless --device cuda:0 \
    --checkpoint "$CHECKPOINT" --output "$RUN"
  python -m json.tool "$RUN/comparison.json"
)
```

Defaults: 32/40/48 mm targets, desired speed 0.006 m/s, 60 s timeout. Nine
independent robots run three goals for each of pre-update, post-update, and rule
controllers in the same simulation. Their local initial states are checked equal.
No action sampling occurs: categories use argmax and continuous actions use tanh
of Gaussian means. The rule uses trot when desired speed exceeds 0.0045 m/s and
remaining distance exceeds 30 mm, otherwise crawl; stride/period/residual inputs
are zero. It is a fixed reference controller, not a separately optimized baseline.

New smoke checkpoints include initial weights. Legacy iteration-1 checkpoints
reconstruct the seed-37 model and MUST match both log probabilities and critic
values in the sibling `rollout.pt`; missing/mismatching evidence aborts evaluation.
Keep the original checkpoint and rollout together when transferring them.

Each arm records exactly the first complete episode, using a physical snapshot
before automatic reset. Later auto-reset episodes of completed rows are excluded.
Stopping is independent of arrival: over the last 1 s, position span must be below
1 mm, body speed below 0.01 m/s, angular speed below 0.04 rad/s, joint speeds below
0.1 rad/s, and every foot load above 5 N. Arrival additionally requires endpoint
error at most 4 mm and no failure. Signed error, time to arrival, total-episode
mean forward speed, sampled body velocity, final speed, and maximum stance drift
are reported. Drift is the per-stance-anchor maximum retained at MPC frequency;
it is not a cumulative path-length measurement. The fixed target set gives a
small acceptance-set arrival rate, not a statistical generalization estimate.

`comparison.json` distinguishes evaluation completion, policy_all_arrived, and
improvement_demonstrated (every goal improves endpoint error by more than 0.1 mm).
A completed evaluation is not a performance PASS. `traces.json` stores sampled
physical signals. The script never changes the input checkpoint or trains a model.

Local full-episode result (2026-11-11, iteration 1): all nine episodes arrived
within 4 mm and stopped, with zero failures. Mean absolute endpoint errors were
1.3002 mm (pre), 1.3022 mm (post), and 1.1833 mm (rule). Maximum stance drift was
5.2726 / 5.2781 / 5.2696 mm respectively. Mean whole-episode forward speeds were
0.002239 / 0.002236 / 0.003139 m/s. Pre/post mean policies both selected crawl;
the rule used trot for the first cycle and crawl thereafter. Error changes were
-0.0414, -0.0336, +0.0689 mm improvements at 32, 40, 48 mm. Thus the pipeline and
complete-episode acceptance pass, but one PPO update has not demonstrated
performance improvement. This closes the feedback-environment/training-pipeline
milestone, not the learned-controller-performance milestone.
