# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadruped robot configuration for RL+MPC control.

This module defines the configuration for quadruped robots used in the
RL+MPC Bezier trajectory tracking environment. Multiple quadruped platforms
are supported:

1. ANYmal series (ANYbotics)
2. Go1/Unitree series
3. Solo/Open Dynamic Robot Initiative

The configuration includes:
- Physical properties (mass, inertia, dimensions)
- Joint limits and control gains
- Hip offsets for foothold planning
- Frame naming conventions for Pinocchio/Crocoddyl

The default uses robot data from example-robot-data package for Crocoddyl
compatibility, with IsaacLab articulation configs for simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import IsaacLab components (optional for standalone testing)
try:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.utils import configclass

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False

    # Dummy decorator for when IsaacLab is not available
    def configclass(cls):
        return dataclass(cls)


@configclass
class QuadrupedPhysicsCfg:
    """Physical properties configuration for a quadruped robot.

    These properties are used by both the Crocoddyl MPC and the IsaacLab
    simulation environment.
    """

    # Basic properties
    mass: float = 12.0  # kg (typical for small quadrupeds)
    body_length: float = 0.4  # meters (front to back)
    body_width: float = 0.2  # meters (left to right)
    body_height: float = 0.1  # meters
    standing_height: float = 0.35  # meters (CoM height when standing)

    # Leg segment lengths
    hip_length: float = 0.08  # hip offset from body (lateral)
    thigh_length: float = 0.2  # upper leg length
    calf_length: float = 0.2  # lower leg length

    # Inertia (approximate for box-like body)
    inertia_xx: float = 0.07  # kg.m²
    inertia_yy: float = 0.26  # kg.m²
    inertia_zz: float = 0.24  # kg.m²

    @property
    def inertia(self) -> np.ndarray:
        """Return 3x3 inertia tensor."""
        return np.diag([self.inertia_xx, self.inertia_yy, self.inertia_zz])


@configclass
class QuadrupedJointsCfg:
    """Joint configuration for a quadruped robot.

    Standard 12-DOF configuration with 3 joints per leg:
    - HAA (Hip Abduction/Adduction): lateral swing
    - HFE (Hip Flexion/Extension): forward/backward swing
    - KFE (Knee Flexion/Extension): knee bend
    """

    # Number of joints
    num_joints: int = 12
    joints_per_leg: int = 3

    # Joint order (matches typical URDF convention)
    joint_names: List[str] = field(default_factory=lambda: [
        "LF_HAA", "LF_HFE", "LF_KFE",  # Left Front
        "RF_HAA", "RF_HFE", "RF_KFE",  # Right Front
        "LH_HAA", "LH_HFE", "LH_KFE",  # Left Hind
        "RH_HAA", "RH_HFE", "RH_KFE",  # Right Hind
    ])

    # Joint limits (radians)
    haa_limits: Tuple[float, float] = (-0.8, 0.8)  # Hip abduction
    hfe_limits: Tuple[float, float] = (-1.0, 1.0)  # Hip flexion
    kfe_limits: Tuple[float, float] = (-2.6, -0.4)  # Knee (usually negative)

    # Default standing pose (radians)
    default_haa: float = 0.0
    default_hfe: float = 0.4
    default_kfe: float = -0.8

    # Torque limits
    max_torque: float = 20.0  # N.m per joint

    # PD gains for position control
    kp: float = 50.0
    kd: float = 1.0

    @property
    def default_joint_positions(self) -> np.ndarray:
        """Return default standing joint positions for all 12 joints."""
        leg_config = [self.default_haa, self.default_hfe, self.default_kfe]
        return np.array(leg_config * 4)  # 4 legs

    @property
    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return joint position limits as (lower, upper)."""
        leg_lower = [self.haa_limits[0], self.hfe_limits[0], self.kfe_limits[0]]
        leg_upper = [self.haa_limits[1], self.hfe_limits[1], self.kfe_limits[1]]
        lower = np.array(leg_lower * 4)
        upper = np.array(leg_upper * 4)
        return lower, upper


@configclass
class QuadrupedFramesCfg:
    """Frame naming configuration for Pinocchio/Crocoddyl integration.

    Maps logical foot names to actual frame names in the URDF/model.
    """

    # Foot frame names (as they appear in URDF)
    lf_foot: str = "LF_FOOT"
    rf_foot: str = "RF_FOOT"
    lh_foot: str = "LH_FOOT"
    rh_foot: str = "RH_FOOT"

    # Base frame name
    base_link: str = "base_link"

    @property
    def foot_frame_names(self) -> Dict[str, str]:
        """Return mapping from logical foot name to frame name."""
        return {
            "LF": self.lf_foot,
            "RF": self.rf_foot,
            "LH": self.lh_foot,
            "RH": self.rh_foot,
        }


@configclass
class QuadrupedCfg:
    """Complete configuration for a quadruped robot.

    Combines physical, joint, and frame configurations.
    Also includes hip offsets for foothold planning.
    """

    # Sub-configurations
    physics: QuadrupedPhysicsCfg = field(default_factory=QuadrupedPhysicsCfg)
    joints: QuadrupedJointsCfg = field(default_factory=QuadrupedJointsCfg)
    frames: QuadrupedFramesCfg = field(default_factory=QuadrupedFramesCfg)

    # Robot name/identifier
    name: str = "generic_quadruped"

    # URDF path (for Pinocchio/Crocoddyl)
    urdf_path: Optional[str] = None

    # Hip offsets in body frame for FootholdPlanner
    # These are the nominal foot positions relative to CoM when standing
    hip_offsets: Dict[str, np.ndarray] = field(default_factory=lambda: {
        "LF": np.array([+0.2, +0.1, 0.0]),  # front-left
        "RF": np.array([+0.2, -0.1, 0.0]),  # front-right
        "LH": np.array([-0.2, +0.1, 0.0]),  # hind-left
        "RH": np.array([-0.2, -0.1, 0.0]),  # hind-right
    })

    # Gait defaults
    default_gait: str = "trot"
    default_step_height: float = 0.05  # meters
    default_step_duration: float = 0.15  # seconds
    default_support_duration: float = 0.05  # seconds

    def get_foot_frame_names(self) -> Dict[str, str]:
        """Return foot frame names for MPC."""
        return self.frames.foot_frame_names


# =============================================================================
# Pre-defined robot configurations
# =============================================================================

# ANYmal C configuration
ANYMAL_C_CFG = QuadrupedCfg(
    name="anymal_c",
    physics=QuadrupedPhysicsCfg(
        mass=30.0,
        body_length=0.53,
        body_width=0.33,
        standing_height=0.5,
        thigh_length=0.25,
        calf_length=0.25,
    ),
    joints=QuadrupedJointsCfg(
        max_torque=40.0,
        kp=80.0,
        kd=2.0,
    ),
    frames=QuadrupedFramesCfg(
        lf_foot="LF_FOOT",
        rf_foot="RF_FOOT",
        lh_foot="LH_FOOT",
        rh_foot="RH_FOOT",
    ),
    hip_offsets={
        "LF": np.array([+0.26, +0.17, 0.0]),
        "RF": np.array([+0.26, -0.17, 0.0]),
        "LH": np.array([-0.26, +0.17, 0.0]),
        "RH": np.array([-0.26, -0.17, 0.0]),
    },
)

# Unitree Go1 configuration
GO1_CFG = QuadrupedCfg(
    name="go1",
    physics=QuadrupedPhysicsCfg(
        mass=12.0,
        body_length=0.366,
        body_width=0.094,
        standing_height=0.35,
        thigh_length=0.2,
        calf_length=0.2,
    ),
    joints=QuadrupedJointsCfg(
        max_torque=23.0,
        kp=50.0,
        kd=1.0,
    ),
    hip_offsets={
        "LF": np.array([+0.183, +0.047, 0.0]),
        "RF": np.array([+0.183, -0.047, 0.0]),
        "LH": np.array([-0.183, +0.047, 0.0]),
        "RH": np.array([-0.183, -0.047, 0.0]),
    },
)

# Unitree B1 configuration
B1_CFG = QuadrupedCfg(
    name="b1",
    physics=QuadrupedPhysicsCfg(
        mass=50.0,
        body_length=0.6,
        body_width=0.2,
        standing_height=0.45,
        thigh_length=0.35,
        calf_length=0.35,
    ),
    joints=QuadrupedJointsCfg(
        max_torque=55.0,
        kp=100.0,
        kd=3.0,
    ),
    frames=QuadrupedFramesCfg(
        lf_foot="FL_foot",
        rf_foot="FR_foot",
        lh_foot="RL_foot",
        rh_foot="RR_foot",
    ),
    hip_offsets={
        "LF": np.array([+0.3, +0.1, 0.0]),
        "RF": np.array([+0.3, -0.1, 0.0]),
        "LH": np.array([-0.3, +0.1, 0.0]),
        "RH": np.array([-0.3, -0.1, 0.0]),
    },
    default_step_height=0.15,
)

# Solo 12 configuration (Open Dynamic Robot Initiative)
SOLO12_CFG = QuadrupedCfg(
    name="solo12",
    physics=QuadrupedPhysicsCfg(
        mass=2.5,
        body_length=0.3,
        body_width=0.1,
        standing_height=0.24,
        thigh_length=0.16,
        calf_length=0.16,
    ),
    joints=QuadrupedJointsCfg(
        max_torque=3.0,
        kp=3.0,
        kd=0.1,
    ),
    hip_offsets={
        "LF": np.array([+0.15, +0.05, 0.0]),
        "RF": np.array([+0.15, -0.05, 0.0]),
        "LH": np.array([-0.15, +0.05, 0.0]),
        "RH": np.array([-0.15, -0.05, 0.0]),
    },
)

# Default configuration
DEFAULT_QUADRUPED_CFG = B1_CFG


# =============================================================================
# IsaacLab ArticulationCfg (only when IsaacLab is available)
# =============================================================================

if ISAACLAB_AVAILABLE:

    @configclass
    class QuadrupedArticulationCfg:
        """IsaacLab ArticulationCfg wrapper for quadruped robots.

        This provides the IsaacLab-specific spawning and articulation
        configuration for simulation.
        """

        # Quadruped physics config
        quadruped_cfg: QuadrupedCfg = field(default_factory=lambda: B1_CFG)

        # USD/URDF path for spawning
        # Set to None to use procedural generation (placeholder)
        usd_path: Optional[str] = None

        # Articulation configuration for IsaacLab
        # This will be created based on the USD/URDF
        @property
        def articulation_cfg(self) -> ArticulationCfg:
            """Create IsaacLab ArticulationCfg from quadruped config."""
            cfg = self.quadruped_cfg

            # Actuator configuration
            actuator_cfg = ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # All joints
                stiffness=cfg.joints.kp,
                damping=cfg.joints.kd,
            )

            if self.usd_path is not None:
                spawn_cfg = sim_utils.UsdFileCfg(
                    usd_path=self.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=False,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                    ),
                )
            else:
                # Placeholder: spawn as simple box
                # In practice, you would use an actual quadruped USD
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(
                        cfg.physics.body_length,
                        cfg.physics.body_width,
                        cfg.physics.body_height,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=cfg.physics.mass,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.6, 0.4, 0.2),  # Brown color
                    ),
                )

            return ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=spawn_cfg,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, cfg.physics.standing_height),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos=dict(zip(
                        cfg.joints.joint_names,
                        cfg.joints.default_joint_positions.tolist(),
                    )),
                ),
                actuators={"legs": actuator_cfg},
            )


# =============================================================================
# Pinocchio model loading utilities
# =============================================================================

def load_pinocchio_model(urdf_path: Optional[str] = None, robot_name: str = "solo12"):
    """Load a Pinocchio model for MPC.

    Tries to load from:
    1. Provided URDF path
    2. example-robot-data package (if installed)
    3. Raises error if neither available

    Args:
        urdf_path: Path to URDF file. If None, tries example-robot-data.
        robot_name: Name of robot in example-robot-data ("solo12", "anymal", etc.)

    Returns:
        Tuple of (pinocchio.Model, urdf_path_used).
    """
    try:
        import pinocchio
    except ImportError:
        raise ImportError("Pinocchio is required for loading robot models.")

    if urdf_path is not None:
        model = pinocchio.buildModelFromUrdf(urdf_path)
        return model, urdf_path

    # Try example-robot-data
    try:
        import example_robot_data

        # Try generic loading first (works for all robots including b1)
        robot = example_robot_data.load(robot_name)
        return robot.model, robot.urdf_path

    except ImportError:
        raise ImportError(
            f"Could not load robot model. Either provide a URDF path or "
            f"install example-robot-data: pip install example-robot-data"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load robot '{robot_name}' from example-robot-data: {e}")


def get_foot_frame_ids(model, foot_frame_names: Dict[str, str]) -> Dict[str, int]:
    """Get Pinocchio frame IDs for foot frames.

    Args:
        model: Pinocchio model.
        foot_frame_names: Dict mapping foot name to frame name.

    Returns:
        Dict mapping foot name to frame ID.
    """
    frame_ids = {}
    for foot_name, frame_name in foot_frame_names.items():
        try:
            frame_id = model.getFrameId(frame_name)
            if frame_id < model.nframes:
                frame_ids[foot_name] = frame_id
            else:
                raise ValueError(f"Frame '{frame_name}' not found in model")
        except Exception as e:
            raise ValueError(f"Could not find frame '{frame_name}' for foot '{foot_name}': {e}")

    return frame_ids
