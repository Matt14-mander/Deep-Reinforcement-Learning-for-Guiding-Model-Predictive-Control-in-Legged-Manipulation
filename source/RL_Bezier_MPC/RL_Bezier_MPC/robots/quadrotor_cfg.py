# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadrotor robot configuration for RL+MPC control.

This module defines the configuration for quadrotor robots used in the
RL+MPC Bezier trajectory tracking environment. The default configuration
is based on the Crazyflie 2.x nano-quadrotor.

Physical Parameters (Crazyflie 2.x):
    - Mass: 0.027 kg
    - Arm length: 0.046 m
    - Inertia: Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5 kg.m²
    - Max thrust: ~0.6 N (total, all motors)
    - Max torque: ~0.01 N.m per axis
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# Try to import IsaacLab components (optional for standalone testing)
try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.utils import configclass

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False
    sim_utils = None
    RigidObjectCfg = None

    # Dummy decorator for when IsaacLab is not available
    def configclass(cls):
        return dataclass(cls)


@configclass
class QuadrotorCfg:
    """Configuration for a quadrotor robot.

    This configuration defines the physical properties and control limits
    for a quadrotor used in the RL+MPC environment.
    """

    # Physical properties
    mass: float = 0.027  # kg (Crazyflie mass)
    arm_length: float = 0.046  # meters

    # Inertia tensor (diagonal approximation)
    inertia_xx: float = 1.4e-5  # kg.m²
    inertia_yy: float = 1.4e-5  # kg.m²
    inertia_zz: float = 2.17e-5  # kg.m²

    # Control limits
    thrust_min: float = 0.0  # N (minimum collective thrust)
    thrust_max: float = 0.6  # N (maximum collective thrust, ~2x hover)
    torque_max: float = 0.01  # N.m (maximum body torque per axis)

    # Motor properties (for reference/future use)
    num_motors: int = 4
    motor_thrust_coefficient: float = 2.88e-8  # N/(rad/s)²
    motor_torque_coefficient: float = 7.24e-10  # N.m/(rad/s)²
    motor_max_rpm: float = 21000  # Maximum motor RPM

    # Aerodynamic properties
    drag_coefficient: float = 0.0  # Simplified: no drag

    @property
    def inertia(self) -> np.ndarray:
        """Return 3x3 inertia tensor."""
        return np.diag([self.inertia_xx, self.inertia_yy, self.inertia_zz])

    @property
    def hover_thrust(self) -> float:
        """Return thrust required for hover."""
        return self.mass * 9.81

    @property
    def control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return control bounds as (lower, upper)."""
        lower = np.array([
            self.thrust_min,
            -self.torque_max,
            -self.torque_max,
            -self.torque_max
        ])
        upper = np.array([
            self.thrust_max,
            self.torque_max,
            self.torque_max,
            self.torque_max
        ])
        return lower, upper


# Default Crazyflie-like quadrotor configuration
CRAZYFLIE_CFG = QuadrotorCfg()


# =============================================================================
# IsaacLab-specific configurations (only when IsaacLab is available)
# =============================================================================

if ISAACLAB_AVAILABLE:
    # IsaacLab RigidObjectCfg for spawning quadrotor in simulation
    # This uses a simple box as placeholder - replace with actual USD model
    QUADROTOR_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Quadrotor",
        spawn=sim_utils.UsdFileCfg(
            # Use a simple cube as placeholder
            # Replace with actual Crazyflie USD path when available
            usd_path=None,  # Will need actual USD path
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.027,  # Crazyflie mass
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),  # Start 1m above ground
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    @configclass
    class QuadrotorSpawnCfg:
        """Configuration for spawning a quadrotor as a simple rigid body.

        Uses a programmatically created box mesh since we may not have
        access to a Crazyflie USD model.
        """

        # Spawn as simple cube (visual placeholder)
        spawn: sim_utils.SpawnerCfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.03),  # Approximate quadrotor size
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.027,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.8),  # Blue color
            ),
        )

        # Initial state
        init_state: RigidObjectCfg.InitialStateCfg = field(
            default_factory=lambda: RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            )
        )

else:
    # Placeholders when IsaacLab is not available
    QUADROTOR_CFG = None
    QuadrotorSpawnCfg = None
