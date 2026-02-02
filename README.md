<<<<<<< HEAD
# RL+MPC Bezier Trajectory Control System

A research project integrating Reinforcement Learning (RL) with Model Predictive Control (MPC) for robot trajectory control. The system uses Bezier curves as the trajectory parameterization, where the RL policy outputs Bezier control points and Crocoddyl MPC tracks the resulting trajectory.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RL Policy (5-10 Hz)                         │
│                   Outputs: Bezier control points                    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ 12D: 4 control points × 3D position
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Trajectory Generator                             │
│              Bezier params → Dense waypoints (50 Hz)                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ Reference trajectory: T × 3D positions
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Crocoddyl MPC (50 Hz)                           │
│         Input: current state + reference trajectory                 │
│         Output: thrust + torques (4D)                               │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ 4D: [thrust, τx, τy, τz]
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   IsaacLab Physics (200 Hz)                         │
│                    Quadrotor simulation                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Frequency Hierarchy

| Component | Frequency | Notes |
|-----------|-----------|-------|
| Physics simulation | 200 Hz | `sim.dt = 0.005s` |
| MPC control rate | 50 Hz | `decimation = 4` |
| RL policy rate | 5-10 Hz | Every 5-10 MPC cycles |
| Bezier horizon | 1.5 s | 75 waypoints at 50 Hz |

## Project Structure

```
RL_Bezier_MPC/
├── scripts/
│   ├── train_quadrotor_mpc.py      # RL training entry point
│   ├── play_quadrotor_mpc.py       # Evaluation and visualization
│   └── test_mpc_standalone.py      # Test MPC without IsaacLab
│
└── source/RL_Bezier_MPC/RL_Bezier_MPC/
    ├── envs/                       # DirectRL environments
    ├── controllers/                # MPC controllers
    ├── trajectory/                 # Trajectory generators
    ├── robots/                     # Robot configurations
    └── utils/                      # Helper functions
```

**Keywords:** rl, mpc, bezier, trajectory, quadrotor, isaaclab

## Installation

### Prerequisites

- Python 3.10+
- IsaacLab 2.1.0+
- NVIDIA GPU with CUDA support

### Install Isaac Lab

Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

### Install This Extension

Using a python interpreter that has Isaac Lab installed, install the library in editable mode:

```bash
# Navigate to the project directory
cd RL_Bezier_MPC

# Install the extension
python -m pip install -e source/RL_Bezier_MPC

# Install Crocoddyl for MPC (optional but recommended)
pip install crocoddyl

# Install RSL-RL for training
pip install rsl-rl-lib>=3.0.1
```

### Verify Installation

List available tasks:

```bash
python scripts/list_envs.py
```

You should see `Quadrotor-MPC-Bezier-v0` in the list.

## Quick Start

### Phase 1: Standalone MPC Test (No Simulation)

Test trajectory generation and MPC tracking:

```bash
python scripts/test_mpc_standalone.py
```

### Phase 2: Simulation Test

Run the environment with a random policy:

```bash
python scripts/play_quadrotor_mpc.py --random --num_envs 4
```

### Phase 3: RL Training

Train the RL policy:

```bash
# Default: 64 parallel environments
python scripts/train_quadrotor_mpc.py

# Custom settings
python scripts/train_quadrotor_mpc.py --num_envs 32 --max_iterations 2000

# With video recording
python scripts/train_quadrotor_mpc.py --video
```

### Evaluate Trained Policy

```bash
python scripts/play_quadrotor_mpc.py --checkpoint logs/quadrotor_mpc/*/model_*.pt
```

## Key Components

### BezierTrajectoryGenerator

Converts 12D Bezier control point offsets to dense 3D position waypoints.

### CrocoddylQuadrotorMPC

Optimal control solver using FDDP for trajectory tracking.

### QuadrotorMPCEnv

IsaacLab DirectRL environment integrating RL, trajectory generation, and MPC.

## Configuration

Key parameters in `QuadrotorMPCEnvCfg`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sim.dt` | 0.005 | Physics timestep (200 Hz) |
| `decimation` | 4 | Steps per control (50 Hz MPC) |
| `mpc_horizon_steps` | 25 | MPC lookahead (0.5s) |
| `bezier_horizon` | 1.5 | Trajectory duration (s) |
| `rl_policy_period` | 5 | MPC steps per RL update (10 Hz) |

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/RL_Bezier_MPC/RL_Bezier_MPC/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/RL_Bezier_MPC"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
=======
# Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
Focus on developing RL framework with MPC to generate contact policies for loco-manipulation tasks.
>>>>>>> 2705fb71d876657687c28af6a799525af2578f4a
