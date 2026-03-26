"""
go2_mpc_controller.launch.py — ROS2 launch file for Go2 RL+MPC controller

使用方式:
    ros2 launch rl_bezier_mpc go2_mpc_controller.launch.py \
        checkpoint:=/path/to/model_1098.pt \
        urdf:=/path/to/go2.urdf

或自定义目标:
    ros2 launch rl_bezier_mpc go2_mpc_controller.launch.py \
        checkpoint:=... urdf:=... \
        target_x:=2.0 target_y:=0.5

参数说明:
    checkpoint  — .pt 检查点路径（必填）
    urdf        — Go2 URDF 路径（必填）
    robot_name  — Gazebo 中模型名称（默认 go2）
    target_x    — 目标 x 坐标，单位 m（默认 1.5）
    target_y    — 目标 y 坐标，单位 m（默认 0.0）
    max_torque  — 关节力矩上限 N·m（默认 23.5）
    action_dim  — 动作维度 12(Stage1/fixed-gait) 或 15(Stage2)（默认 12）
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Launch 参数声明 ───────────────────────────────────────────────────────
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint", default_value="",
        description="Path to trained .pt checkpoint file (required)"
    )
    urdf_arg = DeclareLaunchArgument(
        "urdf", default_value="",
        description="Path to Go2 URDF file (required)"
    )
    robot_name_arg = DeclareLaunchArgument(
        "robot_name", default_value="go2",
        description="Gazebo model name"
    )
    target_x_arg = DeclareLaunchArgument(
        "target_x", default_value="1.5",
        description="Target X position in meters"
    )
    target_y_arg = DeclareLaunchArgument(
        "target_y", default_value="0.0",
        description="Target Y position in meters"
    )
    max_torque_arg = DeclareLaunchArgument(
        "max_torque", default_value="23.5",
        description="Maximum joint torque in N·m"
    )
    action_dim_arg = DeclareLaunchArgument(
        "action_dim", default_value="12",
        description="Action dimension: 12 (Stage1, fixed gait) or 15 (Stage2)"
    )

    # ── 控制器节点 ────────────────────────────────────────────────────────────
    controller_node = Node(
        package="rl_bezier_mpc",
        executable="mpc_ros2_controller",
        name="quadruped_mpc_controller",
        output="screen",
        parameters=[{
            "checkpoint":  LaunchConfiguration("checkpoint"),
            "urdf":        LaunchConfiguration("urdf"),
            "robot_name":  LaunchConfiguration("robot_name"),
            "target_x":    LaunchConfiguration("target_x"),
            "target_y":    LaunchConfiguration("target_y"),
            "max_torque":  LaunchConfiguration("max_torque"),
            "action_dim":  LaunchConfiguration("action_dim"),
        }],
    )

    return LaunchDescription([
        checkpoint_arg,
        urdf_arg,
        robot_name_arg,
        target_x_arg,
        target_y_arg,
        max_torque_arg,
        action_dim_arg,
        controller_node,
    ])
