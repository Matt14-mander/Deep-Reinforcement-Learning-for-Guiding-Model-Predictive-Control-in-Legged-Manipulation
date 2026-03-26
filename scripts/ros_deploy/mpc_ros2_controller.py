#!/usr/bin/env python3
"""
mpc_ros2_controller.py — Gazebo / ROS2 主控制节点

将 IsaacLab 训练的 RL+MPC 策略部署到 ROS2 / Gazebo 环境中。

控制频率层次：
    仿真 (Gazebo)：通常 1000Hz
    MPC 控制循环：50Hz (每 20ms)
    RL 策略推理：5Hz  (每 10 个 MPC 步)

启动方式（直接）：
    ros2 run rl_bezier_mpc mpc_ros2_controller \
        --ros-args \
        -p checkpoint:=/path/to/model_1098.pt \
        -p urdf:=/path/to/go2.urdf \
        -p target_x:=1.5 -p target_y:=0.0

启动方式（launch 文件）：
    ros2 launch rl_bezier_mpc go2_mpc_controller.launch.py \
        checkpoint:=/path/to/model_1098.pt \
        urdf:=/path/to/go2.urdf

话题订阅：
    /joint_states            sensor_msgs/msg/JointState     — 关节角度/速度
    /imu/data                sensor_msgs/msg/Imu            — IMU 姿态/角速度
    /gazebo/model_states     gazebo_msgs/msg/ModelStates    — 位姿/速度（GT）
    /mpc_controller/goal     geometry_msgs/msg/PointStamped — 动态更新目标（可选）

话题发布：
    /joint_group_effort_controller/command  std_msgs/msg/Float64MultiArray — 12 关节力矩

依赖：
    pip install torch numpy scipy
    pip install pin         # pinocchio
    pip install crocoddyl   # or conda
    # ROS2 包：ros-jazzy-gazebo-msgs  ros-jazzy-sensor-msgs  ros-jazzy-std-msgs
"""

import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

# ── 加入项目模块路径 ──────────────────────────────────────────────────────────
# 文件结构: scripts/ros_deploy/mpc_ros2_controller.py → 上两级为项目根
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SOURCE_PATH  = os.path.join(_PROJECT_ROOT, "source", "RL_Bezier_MPC")
if _SOURCE_PATH not in sys.path:
    sys.path.insert(0, _SOURCE_PATH)
# ros_deploy 模块自身（policy_loader, observation_builder）
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from RL_Bezier_MPC.controllers.crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC
from RL_Bezier_MPC.trajectory.bezier_trajectory import BezierTrajectoryGenerator
from RL_Bezier_MPC.utils.math_utils import blend_trajectories

from observation_builder import build_mpc_state, build_observation, world_vel_to_body
from policy_loader import load_policy

# ── Go2 关节名称（URDF / Pinocchio 顺序，按腿分组） ──────────────────────────
GO2_JOINT_NAMES_PIN = [
    "FL_hip_joint",  "FL_thigh_joint",  "FL_calf_joint",
    "FR_hip_joint",  "FR_thigh_joint",  "FR_calf_joint",
    "RL_hip_joint",  "RL_thigh_joint",  "RL_calf_joint",
    "RR_hip_joint",  "RR_thigh_joint",  "RR_calf_joint",
]

# ── 动作反归一化参数（与 env_cfg.py 的 max_bezier_displacement=0.5m 一致） ──
_MAX_BEZIER_DISP = 0.5          # meters
_ACTION_LOW  = np.full(12, -_MAX_BEZIER_DISP, dtype=np.float32)
_ACTION_HIGH = np.full(12,  _MAX_BEZIER_DISP, dtype=np.float32)
# P0 固定为零（从当前位置出发）
_ACTION_LOW[0:3]  = 0.0
_ACTION_HIGH[0:3] = 0.0
# Z 方向约束（与 bezier_trajectory.get_param_bounds() 一致）
for _cp in range(1, 4):
    _z = _cp * 3 + 2
    _ACTION_LOW[_z]  = -0.05   # -5cm
    _ACTION_HIGH[_z] =  0.10   # +10cm

# ── 前进偏置（对应 _pre_physics_step 中 _v_fwd=0.15 m/s, horizon=3.0s） ──────
_V_FWD          = 0.15
_BEZIER_HORIZON = 3.0
_FWD_BIAS = np.array([
    0.0, 0.0, 0.0,
    _V_FWD * _BEZIER_HORIZON / 3,      0.0, 0.0,
    _V_FWD * _BEZIER_HORIZON * 2 / 3,  0.0, 0.0,
    _V_FWD * _BEZIER_HORIZON,           0.0, 0.0,
], dtype=np.float32)

# ── MPC 参数（与 env_cfg.py 一致） ──────────────────────────────────────────
_MPC_DT        = 0.02   # 50 Hz
_MPC_HORIZON   = 25     # 0.5s lookahead
_RL_PERIOD     = 10     # RL 每 10 步推理一次（即 5 Hz）
_MAX_TORQUE    = 23.5   # N·m
_MPC_MAX_ITERS = 50

# Go2 trot 步态周期（step=0.25s, support=0.10s → cycle = 2*(0.25+0.10) = 0.70s）
_CYCLE_DURATION = 0.70

# MPC Guard：代价超过此值时回退
_GUARD_COST_THRESHOLD = 50000.0


def _denormalize_action(action_norm: np.ndarray) -> np.ndarray:
    """将 [-1, 1] 归一化动作转换为实际 Bezier 参数。

    对应 env.py: 0.5 * (a + 1) * (high - low) + low
    """
    a = np.clip(action_norm, -1.0, 1.0)
    return 0.5 * (a + 1.0) * (_ACTION_HIGH - _ACTION_LOW) + _ACTION_LOW


class QuadrupedMPCNode(Node):
    """Go2 RL+MPC ROS2 控制节点。"""

    def __init__(self):
        super().__init__("quadruped_mpc_controller")

        # ── 声明并读取 ROS2 参数 ──────────────────────────────────────────────
        self.declare_parameter("checkpoint",  "")
        self.declare_parameter("urdf",        "")
        self.declare_parameter("robot_name",  "go2")
        self.declare_parameter("target_x",    1.5)
        self.declare_parameter("target_y",    0.0)
        self.declare_parameter("max_torque",  _MAX_TORQUE)
        self.declare_parameter("action_dim",  12)

        checkpoint  = self.get_parameter("checkpoint").value
        urdf_path   = self.get_parameter("urdf").value
        robot_name  = self.get_parameter("robot_name").value
        target_x    = float(self.get_parameter("target_x").value)
        target_y    = float(self.get_parameter("target_y").value)
        max_torque  = float(self.get_parameter("max_torque").value)
        action_dim  = int(self.get_parameter("action_dim").value)

        if not checkpoint:
            raise RuntimeError("参数 'checkpoint' 未指定！请传入 .pt 文件路径。")
        if not urdf_path:
            raise RuntimeError("参数 'urdf' 未指定！请传入 go2.urdf 路径。")

        self._robot_name = robot_name
        self._max_torque = max_torque
        self._target_pos = np.array([target_x, target_y, 0.0], dtype=np.float32)

        # ── 加载 RL 策略 ─────────────────────────────────────────────────────
        self.get_logger().info(f"[PolicyLoader] 加载检查点: {checkpoint}")
        self._policy_fn, _, _ = load_policy(
            checkpoint, obs_dim=45, action_dim=action_dim
        )

        # ── Bezier 轨迹生成器 ────────────────────────────────────────────────
        self._traj_gen = BezierTrajectoryGenerator(
            degree=3, state_dim=3, max_displacement=_MAX_BEZIER_DISP
        )

        # ── Crocoddyl MPC 控制器 ─────────────────────────────────────────────
        self.get_logger().info(f"[MPC] 初始化 Crocoddyl，URDF: {urdf_path}")
        self._mpc = CrocoddylQuadrupedMPC(
            urdf_path=urdf_path,
            dt=_MPC_DT,
            horizon_steps=_MPC_HORIZON,
            max_iterations=_MPC_MAX_ITERS,
            verbose=False,
        )

        # ── 传感器状态缓冲区（回调写入，控制循环读取） ───────────────────────
        self._lock = threading.Lock()

        self._joint_pos_pin   = np.zeros(12)
        self._joint_vel_pin   = np.zeros(12)
        self._root_pos_w      = np.array([0.0, 0.0, 0.4])
        self._root_quat_wxyz  = np.array([1.0, 0.0, 0.0, 0.0])  # (w,x,y,z)
        self._root_lin_vel_w  = np.zeros(3)
        self._root_ang_vel_b  = np.zeros(3)   # IMU 体坐标系角速度

        # 关节名称 → Pinocchio 索引映射
        self._joint_name_to_pin: dict[str, int] = {
            name: i for i, name in enumerate(GO2_JOINT_NAMES_PIN)
        }

        # ── 控制器内部状态 ────────────────────────────────────────────────────
        self._gait_clock    = 0.0
        self._rl_step_count = 0
        self._bezier_params: np.ndarray = _FWD_BIAS.copy()
        self._com_trajectory: Optional[np.ndarray] = None   # (N, 3)
        self._traj_phase    = 0
        self._z_vel_filter  = np.array([0.0])
        self._last_good_solution = None

        # ── QoS 配置（与 Gazebo 话题兼容） ───────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 订阅传感器话题 ────────────────────────────────────────────────────
        self.create_subscription(
            JointState, "/joint_states", self._cb_joints, sensor_qos
        )
        self.create_subscription(
            Imu, "/imu/data", self._cb_imu, sensor_qos
        )
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._cb_model_states, sensor_qos
        )
        self.create_subscription(
            PointStamped, "/mpc_controller/goal", self._cb_goal, reliable_qos
        )

        # ── 发布关节力矩 ─────────────────────────────────────────────────────
        self._pub_torque = self.create_publisher(
            Float64MultiArray, "/joint_group_effort_controller/command", reliable_qos
        )

        # ── 50Hz 控制定时器 ───────────────────────────────────────────────────
        self._timer = self.create_timer(_MPC_DT, self._control_loop)

        self.get_logger().info(
            f"[MPC] 控制器启动完成 (50Hz)，目标位置: "
            f"({target_x:.2f}, {target_y:.2f}, 0.0)"
        )

    # ─────────────────────────── 传感器回调 ──────────────────────────────────

    def _cb_joints(self, msg: JointState):
        """处理 /joint_states，将 12 个关节角度/速度存为 Pinocchio 顺序。"""
        pos = np.zeros(12)
        vel = np.zeros(12)
        for i, name in enumerate(msg.name):
            if name in self._joint_name_to_pin:
                pin_idx = self._joint_name_to_pin[name]
                if i < len(msg.position):
                    pos[pin_idx] = msg.position[i]
                if i < len(msg.velocity):
                    vel[pin_idx] = msg.velocity[i]
        with self._lock:
            self._joint_pos_pin = pos
            self._joint_vel_pin = vel

    def _cb_imu(self, msg: Imu):
        """处理 /imu/data，提取四元数（转为 w,x,y,z）和体坐标系角速度。"""
        q  = msg.orientation
        av = msg.angular_velocity
        with self._lock:
            # ROS geometry_msgs/Quaternion 格式: x,y,z,w → 转为 IsaacLab 约定 w,x,y,z
            self._root_quat_wxyz = np.array([q.w, q.x, q.y, q.z])
            self._root_ang_vel_b = np.array([av.x, av.y, av.z])

    def _cb_model_states(self, msg: ModelStates):
        """处理 /gazebo/model_states，提取机身世界系位置和线速度（ground truth）。"""
        if self._robot_name not in msg.name:
            return
        idx = msg.name.index(self._robot_name)
        p = msg.pose[idx].position
        v = msg.twist[idx].linear
        with self._lock:
            self._root_pos_w     = np.array([p.x, p.y, p.z])
            self._root_lin_vel_w = np.array([v.x, v.y, v.z])

    def _cb_goal(self, msg: PointStamped):
        """动态更新目标位置（可选，通过 /mpc_controller/goal 话题）。"""
        with self._lock:
            self._target_pos = np.array(
                [msg.point.x, msg.point.y, 0.0], dtype=np.float32
            )
        self.get_logger().info(
            f"[MPC] 目标更新: ({msg.point.x:.2f}, {msg.point.y:.2f})"
        )

    # ─────────────────────────── 主控制循环 ──────────────────────────────────

    def _control_loop(self):
        """50Hz 主循环：RL 推理 → 轨迹生成 → MPC 求解 → 发布力矩。"""
        # 复制传感器状态（短暂持锁）
        with self._lock:
            root_pos_w    = self._root_pos_w.copy()
            root_quat     = self._root_quat_wxyz.copy()
            root_lin_vel_w = self._root_lin_vel_w.copy()
            root_ang_vel_b = self._root_ang_vel_b.copy()
            joint_pos_pin = self._joint_pos_pin.copy()
            joint_vel_pin = self._joint_vel_pin.copy()
            target_pos    = self._target_pos.copy()

        # ── 步态相位（与 gait_clock 同步） ───────────────────────────────────
        gait_phase = (self._gait_clock % _CYCLE_DURATION) / _CYCLE_DURATION

        # ── 体坐标系线速度（MPC 需要） ────────────────────────────────────────
        root_lin_vel_b = world_vel_to_body(root_lin_vel_w, root_quat)

        # ── RL 策略推理（5Hz）─────────────────────────────────────────────────
        if self._rl_step_count % _RL_PERIOD == 0:
            obs = build_observation(
                root_pos_w     = root_pos_w,
                root_quat_wxyz = root_quat,
                root_lin_vel_w = root_lin_vel_w,
                root_ang_vel_w = root_ang_vel_b,  # 训练时观测使用世界系，差异较小
                joint_pos_pin  = joint_pos_pin,
                joint_vel_pin  = joint_vel_pin,
                target_pos     = target_pos,
                gait_phase     = gait_phase,
            )
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # (1, 45)
            with torch.no_grad():
                action_norm = self._policy_fn(obs_tensor).squeeze(0).numpy()

            # 反归一化 + 前进偏置
            bezier_raw = _denormalize_action(action_norm[:12])
            self._bezier_params = bezier_raw + _FWD_BIAS

            # 生成新的 CoM 轨迹
            new_traj = self._traj_gen.params_to_waypoints(
                params        = self._bezier_params,
                dt            = _MPC_DT,
                horizon       = _BEZIER_HORIZON,
                start_position= root_pos_w,
            )

            # 平滑融合新旧轨迹（避免轨迹突变）
            if self._com_trajectory is not None and self._traj_phase > 0:
                old_shifted = np.roll(self._com_trajectory, -self._traj_phase, axis=0)
                self._com_trajectory = blend_trajectories(
                    old_trajectory = old_shifted,
                    new_trajectory = new_traj,
                    blend_steps    = 5,
                )
            else:
                self._com_trajectory = new_traj
            self._traj_phase = 0

        # ── 轨迹还未生成时跳过 ───────────────────────────────────────────────
        if self._com_trajectory is None:
            self._step_counters()
            return

        # ── 截取 MPC 参考轨迹片段 ────────────────────────────────────────────
        phase  = self._traj_phase
        end_ph = min(phase + _MPC_HORIZON, len(self._com_trajectory))
        com_ref = self._com_trajectory[phase:end_ph]

        # 补全到 MPC horizon 长度（平滑停止）
        if len(com_ref) < _MPC_HORIZON:
            pad_len = _MPC_HORIZON - len(com_ref)
            if len(com_ref) > 1:
                step_delta = (com_ref[-1] - com_ref[-2]).copy()
                step_delta[2] = 0.0   # 保持高度恒定
                last = com_ref[-1]
                pad  = np.array([
                    last + step_delta * max(0.0, 1.0 - (i + 1) / pad_len)
                    for i in range(pad_len)
                ])
            else:
                pad = np.repeat(com_ref[-1:], pad_len, axis=0)
            com_ref = np.concatenate([com_ref, pad], axis=0)

        # ── 构建 37D Pinocchio 状态（含 Z 速度 EMA 滤波） ─────────────────────
        mpc_state, self._z_vel_filter = build_mpc_state(
            root_pos_w        = root_pos_w,
            root_quat_wxyz    = root_quat,
            root_lin_vel_b    = root_lin_vel_b,
            root_ang_vel_b    = root_ang_vel_b,
            joint_pos_pin     = joint_pos_pin,
            joint_vel_pin     = joint_vel_pin,
            z_vel_filter_state= self._z_vel_filter,
        )

        # ── 获取足端位置（Pinocchio FK） ──────────────────────────────────────
        foot_positions = self._mpc.get_foot_positions(mpc_state)

        # ── Crocoddyl MPC 求解 ────────────────────────────────────────────────
        try:
            solution = self._mpc.solve(
                current_state         = mpc_state,
                com_reference         = com_ref,
                current_foot_positions= foot_positions,
                gait_params           = {
                    "step_length":    1.0,
                    "step_height":    1.0,
                    "step_frequency": 1.0,
                },
                warm_start = True,
            )

            # MPC Guard：代价爆炸时回退
            if solution.cost > _GUARD_COST_THRESHOLD or np.isnan(solution.cost):
                self.get_logger().warn(
                    f"[MPC Guard] cost={solution.cost:.0f} > {_GUARD_COST_THRESHOLD:.0f}，回退",
                    throttle_duration_sec=1.0,
                )
                if self._last_good_solution is not None:
                    solution = self._last_good_solution
                else:
                    self._publish_zero_torques()
                    self._step_counters()
                    return
            else:
                self._last_good_solution = solution

        except Exception as e:
            self.get_logger().error(
                f"[MPC] 求解异常: {e}",
                throttle_duration_sec=1.0,
            )
            self._publish_zero_torques()
            self._step_counters()
            return

        # ── 钳制并发布关节力矩 ────────────────────────────────────────────────
        torques = np.clip(solution.control, -self._max_torque, self._max_torque)
        msg = Float64MultiArray()
        msg.data = torques.tolist()
        self._pub_torque.publish(msg)

        self._step_counters()

    def _step_counters(self):
        """推进步态时钟、RL 计数器和轨迹相位。"""
        self._gait_clock    += _MPC_DT
        self._rl_step_count += 1
        if self._com_trajectory is not None:
            self._traj_phase = min(
                self._traj_phase + 1, len(self._com_trajectory) - 1
            )

    def _publish_zero_torques(self):
        """紧急回退：发布零力矩（让腿自然下垂）。"""
        msg = Float64MultiArray()
        msg.data = [0.0] * 12
        self._pub_torque.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = QuadrupedMPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
