#!/usr/bin/env python3
"""
mpc_ros_controller.py — Gazebo/ROS 主控制节点

将 IsaacLab 训练的 RL+MPC 策略部署到 ROS/Gazebo 环境中。

控制频率层次：
    仿真 (Gazebo)：通常 1000Hz
    MPC 控制循环：50Hz (每 20ms)
    RL 策略推理：5Hz  (每 10 个 MPC 步)

启动方式：
    rosrun rl_bezier_mpc mpc_ros_controller.py \
        _checkpoint:=/path/to/model_1098.pt \
        _urdf:=/path/to/go2.urdf \
        _target_x:=1.5 _target_y:=0.0

话题订阅：
    /joint_states         sensor_msgs/JointState      — 关节角度/速度
    /imu/data             sensor_msgs/Imu             — IMU 姿态/角速度
    /gazebo/model_states  gazebo_msgs/ModelStates     — 世界位姿/速度（ground truth）
    /mpc_controller/goal  geometry_msgs/PointStamped  — 动态更新目标点（可选）

话题发布：
    /joint_group_effort_controller/command  std_msgs/Float64MultiArray  — 12 关节力矩

参数：
    ~checkpoint   str   — .pt 模型文件路径（必填）
    ~urdf         str   — Go2 URDF 文件路径（必填）
    ~robot_name   str   — Gazebo 模型名称（默认 "go2"）
    ~target_x     float — 目标 x 坐标（默认 1.5m）
    ~target_y     float — 目标 y 坐标（默认 0.0m）
    ~max_torque   float — 最大关节力矩 N·m（默认 23.5）
    ~action_dim   int   — 动作维度 12(Stage1) 或 15(Stage2)（默认 12）
"""

import os
import sys
import time
import threading
from typing import Optional

import numpy as np
import torch

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

# ── 加入项目模块路径 ──────────────────────────────────────────────────────────
# 假设本文件位于 scripts/ros_deploy/，项目根目录在上两级
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SOURCE_PATH = os.path.join(_PROJECT_ROOT, "source", "RL_Bezier_MPC")
if _SOURCE_PATH not in sys.path:
    sys.path.insert(0, _SOURCE_PATH)

from RL_Bezier_MPC.controllers.crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC
from RL_Bezier_MPC.trajectory.bezier_trajectory import BezierTrajectoryGenerator
from RL_Bezier_MPC.utils.math_utils import blend_trajectories

from observation_builder import (
    build_mpc_state,
    build_observation,
    world_vel_to_body,
)
from policy_loader import load_policy

# ── Go2 关节名称（URDF 顺序，Pinocchio 顺序）────────────────────────────────
GO2_JOINT_NAMES_PIN = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# action 反归一化参数（对应 env_cfg.py max_bezier_displacement=0.5）
_MAX_BEZIER_DISP = 0.5   # meters
_ACTION_LOW  = np.full(12, -_MAX_BEZIER_DISP, dtype=np.float32)
_ACTION_HIGH = np.full(12,  _MAX_BEZIER_DISP, dtype=np.float32)

# 前进偏置（与 _pre_physics_step 中一致，_v_fwd=0.15 m/s, horizon=3.0s）
_V_FWD = 0.15
_BEZIER_HORIZON = 3.0
_FWD_BIAS = np.array([
    0.0, 0.0, 0.0,
    _V_FWD * _BEZIER_HORIZON / 3,       0.0, 0.0,
    _V_FWD * _BEZIER_HORIZON * 2 / 3,   0.0, 0.0,
    _V_FWD * _BEZIER_HORIZON,            0.0, 0.0,
], dtype=np.float32)

# MPC 参数（与 env_cfg.py 一致）
_MPC_DT          = 0.02   # 50Hz
_MPC_HORIZON     = 25     # 0.5s lookahead
_RL_PERIOD       = 10     # RL 每 10 个 MPC 步推理一次
_MAX_TORQUE      = 23.5   # N·m
_MPC_MAX_ITERS   = 50

# MPC Guard 阈值
_GUARD_COST_THRESHOLD = 50000.0


def denormalize_action(action_norm: np.ndarray) -> np.ndarray:
    """将 [-1, 1] 归一化动作转换为实际 Bezier 参数空间。

    对应 env.py: 0.5 * (a + 1) * (high - low) + low
    """
    action_clamped = np.clip(action_norm, -1.0, 1.0)
    return 0.5 * (action_clamped + 1.0) * (_ACTION_HIGH - _ACTION_LOW) + _ACTION_LOW


class QuadrupedMPCController:
    """Go2 RL+MPC ROS 控制器。"""

    def __init__(self):
        rospy.init_node("quadruped_mpc_controller", anonymous=False)

        # ── 参数 ────────────────────────────────────────────────────────────
        checkpoint = rospy.get_param("~checkpoint")
        urdf_path  = rospy.get_param("~urdf")
        self._robot_name  = rospy.get_param("~robot_name", "go2")
        self._max_torque  = rospy.get_param("~max_torque",  _MAX_TORQUE)
        self._action_dim  = rospy.get_param("~action_dim",  12)

        target_x = rospy.get_param("~target_x", 1.5)
        target_y = rospy.get_param("~target_y", 0.0)
        self._target_pos = np.array([target_x, target_y, 0.0], dtype=np.float32)

        # ── 加载 RL 策略 ────────────────────────────────────────────────────
        self._policy_fn, _, _ = load_policy(
            checkpoint, obs_dim=45, action_dim=self._action_dim
        )

        # ── Bezier 轨迹生成器 ───────────────────────────────────────────────
        self._traj_gen = BezierTrajectoryGenerator(
            degree=3, state_dim=3, max_displacement=_MAX_BEZIER_DISP
        )

        # ── Crocoddyl MPC 控制器 ────────────────────────────────────────────
        rospy.loginfo(f"[MPC] 初始化 Crocoddyl MPC，URDF: {urdf_path}")
        self._mpc = CrocoddylQuadrupedMPC(
            urdf_path=urdf_path,
            dt=_MPC_DT,
            horizon_steps=_MPC_HORIZON,
            max_iterations=_MPC_MAX_ITERS,
            verbose=False,
        )

        # ── 状态缓冲区（由回调线程写入，控制循环读取） ──────────────────────
        self._lock = threading.Lock()

        self._joint_pos_pin = np.zeros(12)   # Pinocchio 顺序
        self._joint_vel_pin = np.zeros(12)
        self._root_pos_w    = np.array([0.0, 0.0, 0.4])
        self._root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # (w,x,y,z)
        self._root_lin_vel_w = np.zeros(3)
        self._root_ang_vel_b = np.zeros(3)   # IMU 直接给出体坐标系角速度

        # 关节名称 → Pinocchio 索引映射（初始化时建立）
        self._joint_name_to_pin: dict[str, int] = {
            name: i for i, name in enumerate(GO2_JOINT_NAMES_PIN)
        }

        # ── 控制器内部状态 ───────────────────────────────────────────────────
        self._gait_clock     = 0.0
        self._rl_step_count  = 0
        self._bezier_params  = _FWD_BIAS.copy()  # 初始：纯前进
        self._com_trajectory: Optional[np.ndarray] = None  # (N, 3)
        self._traj_phase     = 0

        # EMA z 速度滤波器状态
        self._z_vel_filter = np.array([0.0])

        # MPC Guard：上一个好的解
        self._last_good_solution = None

        # ── 发布 / 订阅 ──────────────────────────────────────────────────────
        self._pub_torque = rospy.Publisher(
            "/joint_group_effort_controller/command",
            Float64MultiArray,
            queue_size=1,
        )

        rospy.Subscriber("/joint_states",        JointState,   self._cb_joints,      queue_size=1)
        rospy.Subscriber("/imu/data",            Imu,          self._cb_imu,         queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates,  self._cb_model_states, queue_size=1)
        rospy.Subscriber("/mpc_controller/goal", PointStamped, self._cb_goal,        queue_size=1)

        rospy.loginfo("[MPC] 控制器初始化完成，等待传感器数据...")
        rospy.sleep(1.0)  # 等待第一批传感器消息到达

        # ── 50Hz 控制定时器 ─────────────────────────────────────────────────
        rospy.Timer(rospy.Duration(_MPC_DT), self._control_loop)
        rospy.loginfo(f"[MPC] 控制循环启动 (50Hz)，目标: {self._target_pos}")

    # ────────────────────────── 传感器回调 ──────────────────────────────────

    def _cb_joints(self, msg: JointState):
        """处理 /joint_states 消息，提取 12 个 Go2 关节的角度/速度。"""
        pos = np.zeros(12)
        vel = np.zeros(12)
        for i, name in enumerate(msg.name):
            if name in self._joint_name_to_pin:
                pin_idx = self._joint_name_to_pin[name]
                pos[pin_idx] = msg.position[i] if i < len(msg.position) else 0.0
                vel[pin_idx] = msg.velocity[i] if i < len(msg.velocity) else 0.0
        with self._lock:
            self._joint_pos_pin = pos
            self._joint_vel_pin = vel

    def _cb_imu(self, msg: Imu):
        """处理 /imu/data 消息，提取四元数和体坐标系角速度。"""
        q = msg.orientation
        with self._lock:
            # ROS IMU: geometry_msgs/Quaternion 是 (x,y,z,w)
            # 转为 IsaacLab 约定 (w,x,y,z)
            self._root_quat_wxyz = np.array([q.w, q.x, q.y, q.z])
            av = msg.angular_velocity
            self._root_ang_vel_b = np.array([av.x, av.y, av.z])

    def _cb_model_states(self, msg: ModelStates):
        """处理 /gazebo/model_states，提取机身位置和线速度（world frame）。"""
        if self._robot_name not in msg.name:
            return
        idx = msg.name.index(self._robot_name)
        p = msg.pose[idx].position
        t = msg.twist[idx].linear
        with self._lock:
            self._root_pos_w    = np.array([p.x, p.y, p.z])
            self._root_lin_vel_w = np.array([t.x, t.y, t.z])

    def _cb_goal(self, msg: PointStamped):
        """动态更新目标位置。"""
        with self._lock:
            self._target_pos = np.array([msg.point.x, msg.point.y, 0.0], dtype=np.float32)
        rospy.loginfo(f"[MPC] 目标更新: {self._target_pos}")

    # ────────────────────────── 主控制循环 ──────────────────────────────────

    def _control_loop(self, event):
        """50Hz 控制循环：RL 推理 → 轨迹生成 → MPC 求解 → 发布力矩。"""
        # 复制传感器状态（尽量短暂持锁）
        with self._lock:
            root_pos_w    = self._root_pos_w.copy()
            root_quat     = self._root_quat_wxyz.copy()
            root_lin_vel_w = self._root_lin_vel_w.copy()
            root_ang_vel_b = self._root_ang_vel_b.copy()
            joint_pos_pin = self._joint_pos_pin.copy()
            joint_vel_pin = self._joint_vel_pin.copy()
            target_pos    = self._target_pos.copy()

        # ── 计算步态相位 ─────────────────────────────────────────────────────
        # 使用 gait_clock / cycle_duration（与 env 中 generate_from_phase_offset 一致）
        # Go2 trot：step=0.25s, support=0.10s → cycle = 2*(0.25+0.10) = 0.70s
        cycle_duration = 0.70
        gait_phase = (self._gait_clock % cycle_duration) / cycle_duration

        # ── 体坐标系线速度（Crocoddyl 需要） ────────────────────────────────
        root_lin_vel_b = world_vel_to_body(root_lin_vel_w, root_quat)

        # ── RL 策略推理（5Hz）────────────────────────────────────────────────
        if self._rl_step_count % _RL_PERIOD == 0:
            obs = build_observation(
                root_pos_w=root_pos_w,
                root_quat_wxyz=root_quat,
                root_lin_vel_w=root_lin_vel_w,
                root_ang_vel_w=root_ang_vel_b,  # 训练时用 world frame 但差别很小
                joint_pos_pin=joint_pos_pin,
                joint_vel_pin=joint_vel_pin,
                target_pos=target_pos,
                gait_phase=gait_phase,
            )
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_norm = self._policy_fn(obs_tensor).squeeze(0).numpy()
            bezier_raw = denormalize_action(action_norm[:12])

            # 加入前进偏置（与训练环境一致）
            self._bezier_params = bezier_raw + _FWD_BIAS

            # 生成新的 CoM 轨迹
            new_traj = self._traj_gen.params_to_waypoints(
                params=self._bezier_params,
                dt=_MPC_DT,
                horizon=_BEZIER_HORIZON,
                start_position=root_pos_w,
            )

            # 平滑融合新旧轨迹
            if self._com_trajectory is not None and self._traj_phase > 0:
                old_shifted = np.roll(self._com_trajectory, -self._traj_phase, axis=0)
                self._com_trajectory = blend_trajectories(
                    old_trajectory=old_shifted,
                    new_trajectory=new_traj,
                    blend_steps=5,
                )
            else:
                self._com_trajectory = new_traj
            self._traj_phase = 0

        # ── 构建 MPC 参考轨迹切片 ───────────────────────────────────────────
        if self._com_trajectory is None:
            self._gait_clock    += _MPC_DT
            self._rl_step_count += 1
            return

        phase    = self._traj_phase
        end_ph   = min(phase + _MPC_HORIZON, len(self._com_trajectory))
        com_ref  = self._com_trajectory[phase:end_ph]

        # 不足 horizon 时平滑补零
        if len(com_ref) < _MPC_HORIZON:
            pad_len = _MPC_HORIZON - len(com_ref)
            if len(com_ref) > 1:
                step_delta = (com_ref[-1] - com_ref[-2]).copy()
                step_delta[2] = 0.0
                last = com_ref[-1]
                pad = np.array([
                    last + step_delta * max(0.0, 1.0 - (i + 1) / pad_len)
                    for i in range(pad_len)
                ])
            elif len(com_ref) == 1:
                pad = np.repeat(com_ref, pad_len, axis=0)
            else:
                pad = np.zeros((pad_len, 3))
            com_ref = np.concatenate([com_ref, pad], axis=0)

        # ── 构建 37D Pinocchio 状态 ──────────────────────────────────────────
        mpc_state, self._z_vel_filter = build_mpc_state(
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat,
            root_lin_vel_b=root_lin_vel_b,
            root_ang_vel_b=root_ang_vel_b,
            joint_pos_pin=joint_pos_pin,
            joint_vel_pin=joint_vel_pin,
            z_vel_filter_state=self._z_vel_filter,
        )

        # ── 获取足端位置（Pinocchio FK） ─────────────────────────────────────
        foot_positions = self._mpc.get_foot_positions(mpc_state)

        # ── Crocoddyl MPC 求解 ───────────────────────────────────────────────
        try:
            solution = self._mpc.solve(
                current_state=mpc_state,
                com_reference=com_ref,
                current_foot_positions=foot_positions,
                gait_params={"step_length": 1.0, "step_height": 1.0, "step_frequency": 1.0},
                warm_start=True,
            )

            # MPC Guard：代价爆炸时回退到上一个好的解
            if solution.cost > _GUARD_COST_THRESHOLD or np.isnan(solution.cost):
                rospy.logwarn_throttle(
                    1.0,
                    f"[MPC Guard] 代价爆炸 cost={solution.cost:.0f}，回退到上一个好的解"
                )
                if self._last_good_solution is not None:
                    solution = self._last_good_solution
                else:
                    self._publish_standing_torques()
                    self._gait_clock    += _MPC_DT
                    self._rl_step_count += 1
                    self._traj_phase = min(self._traj_phase + 1, len(self._com_trajectory) - 1)
                    return
            else:
                self._last_good_solution = solution

        except Exception as e:
            rospy.logerr_throttle(1.0, f"[MPC] 求解异常: {e}")
            self._publish_standing_torques()
            self._gait_clock    += _MPC_DT
            self._rl_step_count += 1
            return

        # ── 提取并发布关节力矩 ───────────────────────────────────────────────
        torques = np.clip(solution.control, -self._max_torque, self._max_torque)
        msg = Float64MultiArray()
        msg.data = torques.tolist()
        self._pub_torque.publish(msg)

        # ── 推进计数器 ───────────────────────────────────────────────────────
        self._gait_clock    += _MPC_DT
        self._rl_step_count += 1
        self._traj_phase = min(self._traj_phase + 1, len(self._com_trajectory) - 1)

    def _publish_standing_torques(self):
        """发布零力矩（紧急情况回退）。"""
        msg = Float64MultiArray()
        msg.data = [0.0] * 12
        self._pub_torque.publish(msg)


def main():
    try:
        node = QuadrupedMPCController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
