# RL-Guided MPC with Bezier Trajectory Parameterization for Legged Locomotion

> A research framework that combines **Reinforcement Learning** as a high-level *trajectory shaper* with a **Crocoddyl FDDP MPC** as a low-level *whole-body controller*. The RL policy outputs Bezier control points for the CoM reference, the MPC tracks it through full-body dynamics with contact, and the resulting joint torques drive a quadruped in **IsaacLab / PhysX 5** at scale.

**Current focus (Stage 1):** Unitree **Go2** quadruped, forward trotting with fixed gait timing; RL learns the 12-D CoM Bezier curve only.
**Roadmap (Stage 2+):** Unlock gait modulation (15-D action), omnidirectional locomotion, terrain, sim-to-real on B1 / Go2.

---

## 1. Motivation & Idea

Pure model-free RL on legged robots is sample-hungry and produces jittery, hard-to-deploy policies.
Pure MPC needs a *good reference* — the operator must hand-craft a CoM trajectory and gait pattern, which is brittle across tasks and terrains.

**Our idea: let RL design the reference, let MPC execute it.**

- The RL **Actor** runs at **5 Hz** and emits four 3-D Bezier control points `a_t = [P0, P1, P2, P3]` shaping the next ~1.5 s of CoM motion.
- A **Bezier Trajectory Generator** expands the control points into dense 50-Hz waypoints `x_ref`.
- A **Gait Scheduler** + **Foothold Planner** provide the contact schedule `Σ_c`.
- A **Crocoddyl FDDP MPC** solves an OCP (horizon = 25 × 0.02 s) over full-body dynamics + holonomic contacts and returns the optimal `(q*, v*, τ*)`.
- A **PD tracking layer** at 200 Hz follows the optimal trajectory inside the IsaacLab/PhysX-5 simulator.
- Rewards flow back to a **Critic** (PPO, RSL-RL) which trains the policy on the **GPU**, while the MPC fleet runs in batch on the **CPU**.

This decouples *what to do* (RL: where the CoM should go) from *how to do it* (MPC: feasible, contact-aware joint torques). The policy stays small, low-frequency, and physically grounded; the MPC stays warm-started and bounded.

---

## 2. System Architecture

The high-level structure is a three-loop hierarchy. The RL Guide Layer reshapes the reference every 200 ms; the MPC Optimizing Layer re-plans every 20 ms; the Controller and Environment run at 200 Hz.

![System Framework](images/System%20framework.png)

| Layer | Rate | Role | Output |
|---|---|---|---|
| **RL Guide Layer** | 5 Hz | PPO actor π_θ(a_t \| o_t) shapes the CoM Bezier | `a_t = [P0, P1, P2, P3]` (12-D) |
| **MPC Optimizing Layer** | 50 Hz | Bezier planner + Gait scheduler + Crocoddyl FDDP | `(q*, v*, τ*)` over 25 nodes |
| **Controller Layer** | 200 Hz | PD tracking `τ = τ_ff + Kp·Δq + Kd·Δv` | Joint torques |
| **Environment** | 200 Hz | IsaacLab (PhysX 5) sim + reset + rollout buffer | `x_t = (q, v, R_b, ω_b)` |

The advantage signal `A_t = R_t + γ V_φ(s_{t+1}) − V_φ(s_t)` feeds the Critic, closing the standard PPO loop on top of the MPC-conditioned dynamics.

---

## 3. MPC Internals (Whole-Body OCP)

Each MPC tick builds and solves a discrete OCP via Crocoddyl's `ShootingProblem` + FDDP. The pipeline below is what happens inside the **MPC Optimizing Layer** block above.

![MPC Work Flow](images/work%20flow.png)

**Inputs**
- `x_ref` — dense CoM reference (from the Bezier planner).
- `Σ_c` — contact schedule (from the Gait scheduler).

**OCP Factory** (`source/RL_Bezier_MPC/.../gait/ocp_factory.py`)
- *Full-body dynamics* — Pinocchio `ABA(q, v, τ)` with floating base.
- *Contact models* — holonomic 3-D point contacts on active feet (`Σ_c[k]`).
- *Cost function* — CoM track / foot track / state reg / control reg / friction cone / state bounds / orientation track (terminal weights ×3).

**FDDP Solver**
- *Backward pass* — Riccati / value iteration on the LQ approximation.
- *Forward pass* — nonlinear rollout with line-search.
- *Warm-start* — `setCandidate(shift(prev_xs), shift(prev_us), False)`; cold-start uses gravity-comp + `problem.rollout` (see [§7 Lessons learned](#7-lessons-learned-stage-1)).

**Outputs**
- *Optimal trajectory* `x* = (q*, v*)` and feed-forward `τ_ff` → PD tracking layer at 200 Hz.
- *State estimate* `q̂, v̂` flows back from the simulator at 50 Hz to re-initialize `x_0` for the next solve.

The framework instantiates **N independent OCPs** in parallel (one per environment). MPC outputs are batched back into IsaacLab as actions.

---

## 4. CPU / GPU Parallelization

IsaacLab parallelizes the simulator across thousands of envs on the GPU, but Crocoddyl is a CPU C++/Eigen solver. The training loop is staged so the CPU MPC pool, the GPU physics step, and the GPU policy forward pass overlap in time:

![CPU/GPU Time Slicing](images/CPU%26GPU.png)

| Stage | Device | What runs |
|---|---|---|
| **Apply Action (MPC & Policy)** | **CPU** | N Crocoddyl MPC controllers solve in a worker pool |
| **Isaac Lab Simulation Step** | **GPU** | IsaacLab API + PhysX 5 stepping |
| **Observation & Reward Calc** | **GPU** | Env logic, observation assembly, reward |
| **Deep Policy Forward Pass** | **GPU** | PPO actor & critic forward pass |

This lets the MPC fleet (the bottleneck) run while the GPU is busy with physics and inference, instead of serializing them.

---

## 5. Repository Layout

```
RL_Bezier_MPC/
├── scripts/
│   ├── train_quadruped_mpc.py          # Stage-1 entry point (PPO on Go2)
│   ├── play_quadruped_mpc.py           # Eval / visualization
│   ├── test_mpc_quadruped_standalone.py  # MPC sanity check, no IsaacLab
│   ├── test_gait_standalone.py         # Gait scheduler check
│   ├── test_foothold_planner.py        # Raibert-style foothold check
│   └── (quadrotor variants kept for ablation)
│
├── source/RL_Bezier_MPC/RL_Bezier_MPC/
│   ├── envs/
│   │   ├── quadruped_mpc_env.py        # DirectRL env: action → MPC → step → reward
│   │   └── quadruped_mpc_env_cfg.py    # Frequencies, spaces, rewards, terminations
│   ├── controllers/
│   │   ├── base_mpc.py                 # Abstract MPC interface
│   │   └── crocoddyl_quadruped_mpc.py  # Crocoddyl FDDP wrapper + warm-start
│   ├── gait/
│   │   ├── gait_scheduler.py           # Phase-driven contact sequence (trot, …)
│   │   ├── foothold_planner.py         # Raibert heuristic + safety
│   │   ├── contact_sequence.py         # Σ_c data structures
│   │   └── ocp_factory.py              # Pinocchio + Crocoddyl OCP builder
│   ├── trajectory/
│   │   ├── bezier_trajectory.py        # 12-D control points → 50-Hz CoM waypoints
│   │   └── bezier_foot_trajectory.py   # Swing-foot Bezier
│   ├── robots/
│   │   └── quadruped_cfg.py            # Go2 ArticulationCfg
│   └── tasks/manager_based/            # RSL-RL task registry
│
└── images/                              # Architecture diagrams (used above)
```

---

## 6. Stage 1 — What Works Today

**Goal:** robust forward trotting on flat ground, RL learns only the CoM Bezier (gait timing is fixed).

| Item | Value | Notes |
|---|---|---|
| Robot | Unitree **Go2** (nq=19, nv=18, nu=12) | Pinocchio floating-base |
| Physics rate | 200 Hz | `sim.dt = 0.005 s` |
| MPC rate | 50 Hz | `decimation = 4`, horizon = 25 nodes (0.5 s) |
| RL policy rate | 5 Hz | `rl_policy_period = 10` MPC steps |
| Bezier horizon | 1.5–3.0 s | 75–151 waypoints; only first 25 enter the OCP |
| **Action space** | **12-D** | Four 3-D Bezier control point offsets (`fix_gait_params = True`) |
| Observation | 45-D | base 13 + joints 24 + foot contact 4 + target 3 + gait phase 1 |
| Algorithm | PPO via RSL-RL | Asymmetric A-C optional |
| Default gait | Trot (step 0.25 s, support 0.10 s, height 0.15 m) | |

**Key engineering fixes that make Stage 1 stable** (full notes in `memory/`):
1. Foot–ground static/dynamic friction set to **1.5** — prevents slip that was destabilising the MPC.
2. **Selective warm-start** — only commit `_prev_xs/_prev_us` when `cost < 1e4`; otherwise cold-start with gravity-comp rollout.
3. **Initial-feasibility guard** — if shifted warm-start gives `ffeas > 5`, fall back to cold-start (FDDP cannot recover otherwise).
4. **MPC divergence guard** — when a solve diverges (`cost > 5e4` and not converged), replay the last good solution instead of injecting garbage torques.
5. **Gait clock** — `_gait_clock` accumulates across solves so the MPC sees the correct gait phase (fixes the "always-stance node-0" bug).
6. **Trajectory padding** — constant-height linear decel during pad, no "instant-stop wall".
7. **Height-termination grace period** — 5 consecutive too-low steps before terminating, so a normal swing dip doesn't kill the episode.

**Stage 1 result (representative run, see memory item #16):**
500/500 sim steps survived (full 10 s episode), +0.5 m forward, pitch ≤ 8°, MPC convergence ≈ 28 %, mean torque ≈ 3 N·m.

---

## 7. Lessons Learned (Stage 1)

A few non-obvious traps documented during Stage 1:

- `state.lb / state.ub` have dim **nx = 37**, not ndx = 36. For `ActivationBounds`, use `lb[1:nv+1] + lb[-nv:]`.
- `solver.solve([], [], …, isFeasible=True, …)` **silently** discards `setCandidate` and reinits from zeros. Always pass `isFeasible=False`.
- `problem.rollout(shifted_prev_us)` corrupts Crocoddyl's contact-Jacobian cache when the gait phase changed mid-trajectory. Only roll out gravity-comp on cold-start.
- A naive `_prev_xs = xs` after every solve cascades divergence: a diverged xs becomes next step's warm-start. Gate on `cost < 1e4`.
- Body-frame `vz` fed raw into the MPC creates positive feedback during a fall. Apply an EMA filter (α = 0.3) and clip to ±0.15 m/s.

---

## 8. Stage 2+ — Roadmap

| Stage | Action Space | New Capability |
|---|---|---|
| **1 (current)** | 12-D Bezier only | Stable forward trot, fixed gait |
| **2** | 15-D (+ step length / height / frequency) | Adaptive gait, speed envelope |
| **3** | 15-D + target heading | Omnidirectional walking, turning |
| **4** | + terrain randomization, perceptive obs | Rough terrain, obstacles |
| **5** | Sim-to-real | B1 / Go2 hardware deployment |

---

## 9. Installation

### Prerequisites
- Python 3.10+
- IsaacLab 2.1.0+ (PhysX 5 / Isaac Sim 4.5)
- NVIDIA GPU with CUDA
- Crocoddyl ≥ 2.x with Pinocchio ≥ 2.7

### Install Isaac Lab
Follow the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). The conda or uv install is recommended.

### Install this extension
```bash
cd RL_Bezier_MPC
python -m pip install -e source/RL_Bezier_MPC
pip install crocoddyl pinocchio       # MPC stack
pip install rsl-rl-lib>=3.0.1         # PPO trainer
```

### Verify
```bash
python scripts/list_envs.py
# should list:  Quadruped-MPC-Bezier-v0
```

---

## 10. Quick Start

**Stage 1 — Standalone MPC sanity check** (no Isaac Sim required):
```bash
python scripts/test_mpc_quadruped_standalone.py
python scripts/test_gait_standalone.py
python scripts/test_foothold_planner.py
```

**Stage 1 — Random / zero policy in simulator:**
```bash
python scripts/random_agent.py --task Quadruped-MPC-Bezier-v0 --num_envs 4
python scripts/zero_agent.py   --task Quadruped-MPC-Bezier-v0 --num_envs 4
```

**Stage 1 — Train PPO:**
```bash
python scripts/train_quadruped_mpc.py --num_envs 32 --max_iterations 2000
python scripts/train_quadruped_mpc.py --video    # record evaluation video
```

**Stage 1 — Play a checkpoint:**
```bash
python scripts/play_quadruped_mpc.py --checkpoint logs/quadruped_mpc/<run>/model_<it>.pt
```

---

## 11. Key Configuration

All in [`quadruped_mpc_env_cfg.py`](source/RL_Bezier_MPC/RL_Bezier_MPC/envs/quadruped_mpc_env_cfg.py):

| Parameter | Default | Description |
|---|---|---|
| `sim.dt` | 0.005 | Physics step (200 Hz) |
| `decimation` | 4 | Steps per control (50 Hz MPC) |
| `mpc_horizon_steps` | 25 | OCP nodes (0.5 s lookahead) |
| `mpc_max_iterations` | 50 | FDDP iters per solve |
| `bezier_horizon` | 3.0 | Reference duration (s) |
| `rl_policy_period` | 10 | MPC steps per policy update (→ 5 Hz) |
| `fix_gait_params` | **True** | Stage 1: gait timing locked, action = 12-D |
| `default_step_duration` | 0.25 | Swing duration (trot) |
| `default_support_duration` | 0.10 | Double-support duration |
| `default_step_height` | 0.15 | Swing apex height (m) |
| `min_body_height_ratio` | 0.55 | Termination floor (× standing height) |

---

## 12. Citation / Acknowledgements

This project builds on:
- **IsaacLab** (NVIDIA) — GPU-parallel robot simulation.
- **Crocoddyl** (LAAS-CNRS) — multi-contact whole-body OCP.
- **Pinocchio** (LAAS-CNRS) — rigid-body dynamics.
- **RSL-RL** (ETH RSL) — PPO implementation.

The architecture diagrams in this README are taken from the author's MSc thesis on RL-guided MPC for legged manipulation.

---

## 13. Troubleshooting

**`Quadruped-MPC-Bezier-v0` not listed** — `pip install -e source/RL_Bezier_MPC` was skipped or failed.
**`ImportError: crocoddyl`** — install with `pip install crocoddyl pinocchio`.
**MPC diverges immediately on episode start** — check ground friction (must be ≥ 1.0 for Go2), see Stage 1 fix #1.
**Robot walks backward** — `max_bezier_displacement` is too small for the forward-velocity bias `_v_fwd`; see env cfg comments.
**FDDP `preg` escalates to 1e7** — your warm-start is infeasible. Check the `ffeas > 5` guard is active (Stage 1 fix #3).
