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
