#!/usr/bin/env python3
"""
ROS2 controller node for Go2 quadruped — supports two modes:

  --mode baseline   Pure Crocoddyl MPC with straight-line CoM trajectory.
                    No RL policy; serves as the comparison baseline.

  --mode rl         Trained RL policy (RSL-RL) shapes the Bezier CoM
                    trajectory, which is then tracked by Crocoddyl MPC.

Both modes use the IDENTICAL Crocoddyl backend, making the comparison fair:
the only difference is how the CoM reference trajectory is generated.

Dependencies (no IsaacLab required):
    pip install torch numpy scipy pinocchio
    conda install -c conda-forge crocoddyl
    sudo apt install ros-jazzy-gazebo-ros-pkgs ros-jazzy-ros2-control

Subscribed topics:
    /joint_states              sensor_msgs/JointState    50 Hz
    /imu/data                  sensor_msgs/Imu           200 Hz
    /gazebo/model_states       gazebo_msgs/ModelStates   50 Hz   (pos + vel)

Published topics:
    /joint_group_effort_controller/commands   std_msgs/Float64MultiArray   50 Hz

Parameters:
    mode            str     'baseline' or 'rl'           default: 'baseline'
    checkpoint      str     path to .pt file             required for 'rl' mode
    urdf_path       str     path to go2.urdf             required
    target_x        float   goal x position (m)          default: 2.0
    target_y        float   goal y position (m)          default: 0.0
    target_z        float   goal z position (m)          default: 0.0
    rl_period       int     RL inference every N steps   default: 10  (=5 Hz)
    max_torque      float   joint torque clamp (N·m)     default: 23.5

Usage:
    # Baseline (pure MPC)
    ros2 run rl_bezier_mpc go2_mpc_node --ros-args \
        -p mode:=baseline \
        -p urdf_path:=/path/to/go2.urdf

    # RL + MPC
    ros2 run rl_bezier_mpc go2_mpc_node --ros-args \
        -p mode:=rl \
        -p checkpoint:=/path/to/model_1098.pt \
        -p urdf_path:=/path/to/go2.urdf
"""

from __future__ import annotations

import sys
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray

try:
    from gazebo_msgs.msg import ModelStates
    GAZEBO_AVAILABLE = True
except ImportError:
    GAZEBO_AVAILABLE = False
    print("[go2_mpc_node] WARNING: gazebo_msgs not available. Using IMU-only mode.")

# ---------------------------------------------------------------------------
# Add project source to path (no IsaacLab needed)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SOURCE_DIR = _PROJECT_ROOT / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(_SOURCE_DIR))

from RL_Bezier_MPC.controllers.crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC
from RL_Bezier_MPC.trajectory.bezier_trajectory import BezierTrajectoryGenerator

from policy_loader import load_policy
from observation_builder import (
    build_ros_to_isaac_mapping,
    build_ros_to_pinocchio_mapping,
    world_to_body_velocity,
    build_mpc_state,
    build_observation,
    PINOCCHIO_JOINT_ORDER,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MPC_DT        = 0.02      # 50 Hz
MPC_HORIZON   = 25        # 0.5 s lookahead
BEZIER_HORIZON = 3.0      # s  (151 waypoints)
MAX_BEZIER_DISP = 0.5     # m  (matches cfg)
FWD_BIAS_VEL  = 0.3       # m/s forward bias added to Bezier
STANDING_HEIGHT = 0.35    # m  target CoM height
BASELINE_SPEED  = 0.3     # m/s for straight-line baseline trajectory


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------

def generate_baseline_trajectory(
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    horizon: int = MPC_HORIZON,
    dt: float = MPC_DT,
    speed: float = BASELINE_SPEED,
) -> np.ndarray:
    """Straight-line CoM trajectory at constant speed toward target.

    This is the baseline: no RL, pure geometric planning.

    Args:
        current_pos: (3,) current CoM position.
        target_pos:  (3,) goal position.
        horizon:     number of waypoints.
        dt:          time step (s).
        speed:       maximum linear speed (m/s).

    Returns:
        trajectory: (horizon, 3) CoM waypoints.
    """
    direction = target_pos[:2] - current_pos[:2]
    distance  = np.linalg.norm(direction)

    trajectory = np.zeros((horizon, 3), dtype=np.float32)

    for i in range(horizon):
        t = (i + 1) * dt
        step = min(speed * t, distance) if distance > 0.01 else 0.0
        waypoint = current_pos.copy()
        if distance > 0.01:
            waypoint[:2] += direction / distance * step
        waypoint[2] = STANDING_HEIGHT  # maintain nominal height
        trajectory[i] = waypoint

    return trajectory


def apply_forward_bias(
    bezier_params: np.ndarray,
    horizon: float = BEZIER_HORIZON,
    v_fwd: float = FWD_BIAS_VEL,
) -> np.ndarray:
    """Add a forward velocity bias to raw RL Bezier parameters.

    Prevents the policy from outputting a backward trajectory when actions
    are near zero. Matches the _pre_physics_step logic in training env.

    Args:
        bezier_params: (12,) raw RL action output.
        horizon:       Bezier time horizon in seconds.
        v_fwd:         Forward bias velocity (m/s).

    Returns:
        biased_params: (12,) with forward offset applied.
    """
    fwd_bias = np.array([
        0.0, 0.0, 0.0,                          # P0
        v_fwd * horizon / 3, 0.0, 0.0,          # P1
        v_fwd * horizon * 2 / 3, 0.0, 0.0,      # P2
        v_fwd * horizon, 0.0, 0.0,               # P3
    ], dtype=np.float32)
    return bezier_params + fwd_bias


def denormalize_bezier(
    action: np.ndarray,
    max_displacement: float = MAX_BEZIER_DISP,
) -> np.ndarray:
    """Map tanh action [-1, 1]^12 → displacement offsets [-max, max]^12."""
    return np.tanh(action) * max_displacement


def generate_rl_trajectory(
    bezier_params: np.ndarray,
    current_pos: np.ndarray,
    horizon_steps: int,
    bezier_horizon: float = BEZIER_HORIZON,
    dt: float = MPC_DT,
) -> np.ndarray:
    """Convert RL Bezier params to CoM waypoints.

    Args:
        bezier_params: (12,) raw params (P0..P3 offsets, after bias).
        current_pos:   (3,) current CoM position.
        horizon_steps: number of waypoints to return (≤ total waypoints).

    Returns:
        trajectory: (horizon_steps, 3).
    """
    generator = BezierTrajectoryGenerator(
        degree=3,
        state_dim=3,
        max_displacement=MAX_BEZIER_DISP,
    )
    num_waypoints = int(bezier_horizon / dt) + 1
    # Evaluate Bezier
    all_waypoints = generator.generate(
        start_state=current_pos,
        params=bezier_params,
        num_waypoints=num_waypoints,
    )
    # Keep only first horizon_steps
    trajectory = all_waypoints[:horizon_steps]
    # Force Z = standing height to avoid vertical drift
    trajectory[:, 2] = STANDING_HEIGHT
    return trajectory.astype(np.float32)


# ---------------------------------------------------------------------------
# Main ROS2 Node
# ---------------------------------------------------------------------------

class Go2MPCNode(Node):
    """ROS2 node that runs Crocoddyl MPC at 50 Hz.

    In 'baseline' mode, the CoM reference trajectory is a straight line
    to the target — no learning involved.

    In 'rl' mode, a trained RSL-RL policy outputs Bezier control points
    every 10 steps (5 Hz) which shape the CoM reference trajectory.
    """

    def __init__(self):
        super().__init__("go2_mpc_node")

        # ---- ROS2 parameters ----
        self.declare_parameter("mode", "baseline")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("target_x", 2.0)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", 0.0)
        self.declare_parameter("rl_period", 10)
        self.declare_parameter("max_torque", 23.5)
        self.declare_parameter("robot_name", "go2")

        self.mode        = self.get_parameter("mode").value
        checkpoint       = self.get_parameter("checkpoint").value
        urdf_path        = self.get_parameter("urdf_path").value
        self.target_pos  = np.array([
            self.get_parameter("target_x").value,
            self.get_parameter("target_y").value,
            self.get_parameter("target_z").value,
        ], dtype=np.float32)
        self.rl_period   = self.get_parameter("rl_period").value
        self.max_torque  = self.get_parameter("max_torque").value
        self.robot_name  = self.get_parameter("robot_name").value

        self.get_logger().info(
            f"[go2_mpc_node] mode={self.mode}  target={self.target_pos}"
        )

        # ---- Validate parameters ----
        if not urdf_path:
            self.get_logger().fatal("Parameter 'urdf_path' is required!")
            raise RuntimeError("urdf_path not set")

        if self.mode == "rl":
            if not checkpoint:
                self.get_logger().fatal("Parameter 'checkpoint' required for rl mode!")
                raise RuntimeError("checkpoint not set")
            self.actor, self.obs_mean, self.obs_std = load_policy(checkpoint)
            self.get_logger().info(f"[go2_mpc_node] RL policy loaded from {checkpoint}")
        else:
            self.actor = None
            self.obs_mean = None
            self.obs_std = None
            self.get_logger().info("[go2_mpc_node] Baseline mode: no RL policy loaded")

        # ---- Crocoddyl MPC (shared by both modes) ----
        self.mpc = CrocoddylQuadrupedMPC(
            urdf_path=urdf_path,
            robot_name=self.robot_name,
            dt=MPC_DT,
            horizon_steps=MPC_HORIZON,
        )
        self.get_logger().info("[go2_mpc_node] CrocoddylQuadrupedMPC initialised")

        # ---- State variables ----
        self._lock = threading.Lock()

        # Sensor buffers (updated by callbacks)
        self._joint_pos_ros  = np.zeros(12, dtype=np.float32)  # ROS order
        self._joint_vel_ros  = np.zeros(12, dtype=np.float32)  # ROS order
        self._q_wxyz         = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._ang_vel_world  = np.zeros(3, dtype=np.float32)
        self._root_pos_w     = np.zeros(3, dtype=np.float32)
        self._root_pos_w[2]  = STANDING_HEIGHT
        self._lin_vel_world  = np.zeros(3, dtype=np.float32)

        # Joint ordering maps (built on first /joint_states)
        self._ros_to_isaac: Optional[np.ndarray] = None
        self._ros_to_pin:   Optional[np.ndarray] = None
        self._joint_names_received = False

        # Z-velocity filter state
        self._z_vel_filter = 0.0

        # Gait / RL counters
        self._gait_clock  = 0.0
        self._step_count  = 0
        self._bezier_params = np.zeros(12, dtype=np.float32)

        # Foot height buffer (updated via Pinocchio FK in control loop)
        self._foot_heights = np.array([STANDING_HEIGHT] * 4, dtype=np.float32)

        # Last good torques (for guard fallback)
        self._last_torques = np.zeros(12, dtype=np.float32)

        # ---- Subscribers ----
        self.create_subscription(JointState, "/joint_states",
                                 self._cb_joint_states, 10)
        self.create_subscription(Imu, "/imu/data",
                                 self._cb_imu, 10)
        if GAZEBO_AVAILABLE:
            self.create_subscription(ModelStates, "/gazebo/model_states",
                                     self._cb_model_states, 10)

        # ---- Publisher ----
        self._pub_effort = self.create_publisher(
            Float64MultiArray,
            "/joint_group_effort_controller/commands",
            10,
        )

        # ---- Control timer (50 Hz) ----
        self._timer = self.create_timer(MPC_DT, self._control_loop)
        self.get_logger().info("[go2_mpc_node] Ready — 50 Hz control timer started")

    # -----------------------------------------------------------------------
    # Sensor callbacks
    # -----------------------------------------------------------------------

    def _cb_joint_states(self, msg: JointState):
        with self._lock:
            if not self._joint_names_received:
                # Build ordering maps once from the actual joint name list
                try:
                    self._ros_to_isaac = build_ros_to_isaac_mapping(list(msg.name))
                    self._ros_to_pin   = build_ros_to_pinocchio_mapping(list(msg.name))
                    self._joint_names_received = True
                    self.get_logger().info(
                        f"[go2_mpc_node] Joint mapping built from: {list(msg.name)}"
                    )
                except ValueError as e:
                    self.get_logger().error(str(e))
                    return

            pos = np.array(msg.position, dtype=np.float32)
            vel = np.array(msg.velocity, dtype=np.float32)
            if len(pos) >= 12:
                self._joint_pos_ros = pos[:12]
            if len(vel) >= 12:
                self._joint_vel_ros = vel[:12]

    def _cb_imu(self, msg: Imu):
        with self._lock:
            q = msg.orientation
            # ROS uses (x,y,z,w); convert to Isaac Lab (w,x,y,z)
            self._q_wxyz = np.array(
                [q.w, q.x, q.y, q.z], dtype=np.float32
            )
            av = msg.angular_velocity
            self._ang_vel_world = np.array([av.x, av.y, av.z], dtype=np.float32)

    def _cb_model_states(self, msg: ModelStates):
        """Extract robot position and linear velocity from Gazebo."""
        with self._lock:
            robot_name = self.robot_name
            try:
                idx = msg.name.index(robot_name)
            except ValueError:
                # Try partial match (e.g. 'go2' in 'go2::base')
                idx = next(
                    (i for i, n in enumerate(msg.name) if robot_name in n),
                    None,
                )
                if idx is None:
                    return

            p = msg.pose[idx].position
            self._root_pos_w = np.array([p.x, p.y, p.z], dtype=np.float32)

            v = msg.twist[idx].linear
            self._lin_vel_world = np.array([v.x, v.y, v.z], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------

    def _control_loop(self):
        """Main 50 Hz control loop — runs both modes."""
        with self._lock:
            if not self._joint_names_received:
                return  # wait for first joint_states

            # Snapshot sensor data
            q_wxyz        = self._q_wxyz.copy()
            ang_vel_world = self._ang_vel_world.copy()
            root_pos_w    = self._root_pos_w.copy()
            lin_vel_world = self._lin_vel_world.copy()
            joint_pos_ros = self._joint_pos_ros.copy()
            joint_vel_ros = self._joint_vel_ros.copy()

        # ---- Reorder joints ----
        joint_pos_isaac = joint_pos_ros[self._ros_to_isaac]
        joint_vel_isaac = joint_vel_ros[self._ros_to_isaac]
        joint_pos_pin   = joint_pos_ros[self._ros_to_pin]
        joint_vel_pin   = joint_vel_ros[self._ros_to_pin]

        # ---- Body-frame velocities (for MPC state) ----
        lin_vel_body, ang_vel_body = world_to_body_velocity(
            lin_vel_world, ang_vel_world, q_wxyz
        )

        # ---- Build 37D Pinocchio MPC state ----
        state_37d, self._z_vel_filter = build_mpc_state(
            root_pos_w    = root_pos_w,
            q_wxyz        = q_wxyz,
            joint_pos_pin = joint_pos_pin,
            lin_vel_body  = lin_vel_body,
            ang_vel_body  = ang_vel_body,
            joint_vel_pin = joint_vel_pin,
            z_vel_filter_state = self._z_vel_filter,
        )

        # ---- Foot heights via Pinocchio FK ----
        try:
            self._foot_heights = self.mpc.compute_foot_heights(state_37d)
        except Exception:
            pass  # keep previous estimate

        # ---- Generate CoM reference trajectory ----
        if self.mode == "baseline":
            com_traj = generate_baseline_trajectory(
                current_pos = root_pos_w,
                target_pos  = self.target_pos,
                horizon     = MPC_HORIZON,
            )
        else:
            # RL inference every rl_period steps (5 Hz)
            if self._step_count % self.rl_period == 0:
                obs = build_observation(
                    root_pos_w     = root_pos_w,
                    q_wxyz         = q_wxyz,
                    lin_vel_world  = lin_vel_world,
                    ang_vel_world  = ang_vel_world,
                    joint_pos_isaac= joint_pos_isaac,
                    joint_vel_isaac= joint_vel_isaac,
                    foot_heights   = self._foot_heights,
                    target_pos     = self.target_pos,
                    gait_phase     = (self._gait_clock % 0.7) / 0.7,
                )
                obs_norm = (obs - self.obs_mean) / self.obs_std
                with torch.no_grad():
                    raw_action = self.actor(
                        torch.FloatTensor(obs_norm)
                    ).numpy()
                bezier_denorm = denormalize_bezier(raw_action)
                self._bezier_params = apply_forward_bias(bezier_denorm)

            com_traj = generate_rl_trajectory(
                bezier_params  = self._bezier_params,
                current_pos    = root_pos_w,
                horizon_steps  = MPC_HORIZON,
            )

        # ---- Crocoddyl MPC solve ----
        try:
            solution = self.mpc.solve(
                state       = state_37d,
                com_reference = com_traj,
            )

            if solution is not None and hasattr(solution, 'control'):
                torques_pin = np.array(solution.control, dtype=np.float32)
                torques_pin = np.clip(torques_pin, -self.max_torque, self.max_torque)

                # Reorder Pinocchio → ROS/Isaac joint order for publishing
                torques_isaac = np.zeros(12, dtype=np.float32)
                for pin_idx in range(12):
                    # pin_idx → isaac_idx via inverse of ros_to_pin map
                    ros_idx = self._ros_to_pin[pin_idx]
                    isaac_idx = int(np.where(self._ros_to_isaac == ros_idx)[0][0])
                    torques_isaac[isaac_idx] = torques_pin[pin_idx]

                self._last_torques = torques_isaac
            else:
                torques_isaac = self._last_torques

        except Exception as e:
            self.get_logger().warn(f"[MPC] solve failed: {e}")
            torques_isaac = self._last_torques

        # ---- Publish torques ----
        msg = Float64MultiArray()
        msg.data = torques_isaac.tolist()
        self._pub_effort.publish(msg)

        # ---- Update counters ----
        self._gait_clock += MPC_DT
        self._step_count += 1

        # ---- Periodic status log (every 5s) ----
        if self._step_count % 250 == 0:
            dist = float(np.linalg.norm(root_pos_w[:2] - self.target_pos[:2]))
            self.get_logger().info(
                f"[go2_mpc_node] mode={self.mode}  "
                f"step={self._step_count}  "
                f"pos=({root_pos_w[0]:.2f},{root_pos_w[1]:.2f},{root_pos_w[2]:.2f})  "
                f"dist_to_target={dist:.2f}m  "
                f"|τ|_max={float(np.abs(torques_isaac).max()):.1f}N·m"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = Go2MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
