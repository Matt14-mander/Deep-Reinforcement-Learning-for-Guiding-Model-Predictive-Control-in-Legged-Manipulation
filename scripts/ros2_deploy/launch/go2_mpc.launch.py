"""
ROS2 Launch file for Go2 MPC controller.

Launches one of two controller modes:
  - baseline : Pure Crocoddyl MPC (no RL), straight-line CoM trajectory
  - rl        : Trained RSL-RL policy + Crocoddyl MPC

Usage:
    # Baseline (pure MPC)
    ros2 launch ros2_deploy go2_mpc.launch.py \
        mode:=baseline \
        urdf_path:=/path/to/go2.urdf \
        target_x:=2.0

    # RL + MPC
    ros2 launch ros2_deploy go2_mpc.launch.py \
        mode:=rl \
        urdf_path:=/path/to/go2.urdf \
        checkpoint:=/path/to/model_1098.pt \
        target_x:=2.0

Prerequisites:
    1. Gazebo Classic running with Go2 model:
       ros2 launch go2_description go2_gazebo.launch.py
    2. ros2_control effort controllers loaded:
       ros2 control load_controller --set-state active joint_group_effort_controller
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # ---- Declare arguments ----
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="baseline",
        description="Controller mode: 'baseline' (pure MPC) or 'rl' (MPC + RL policy)",
    )
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint",
        default_value="",
        description="Path to trained RSL-RL checkpoint (.pt). Required for mode:=rl",
    )
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value="",
        description="Path to go2.urdf file (required)",
    )
    target_x_arg = DeclareLaunchArgument(
        "target_x", default_value="2.0",
        description="Goal X position (m)",
    )
    target_y_arg = DeclareLaunchArgument(
        "target_y", default_value="0.0",
        description="Goal Y position (m)",
    )
    target_z_arg = DeclareLaunchArgument(
        "target_z", default_value="0.0",
        description="Goal Z position (m)",
    )
    rl_period_arg = DeclareLaunchArgument(
        "rl_period", default_value="10",
        description="RL inference interval (MPC steps). 10 = 5 Hz at 50 Hz control",
    )
    max_torque_arg = DeclareLaunchArgument(
        "max_torque", default_value="23.5",
        description="Maximum joint torque (N·m). Go2 HV peak = 23.5 N·m",
    )
    robot_name_arg = DeclareLaunchArgument(
        "robot_name", default_value="go2",
        description="Robot model name in Gazebo (must match /gazebo/model_states)",
    )

    # ---- Controller node ----
    # Find the go2_mpc_node.py script relative to this launch file
    _launch_dir = os.path.dirname(os.path.realpath(__file__))
    _script_dir = os.path.dirname(_launch_dir)
    _node_script = os.path.join(_script_dir, "go2_mpc_node.py")

    controller_node = Node(
        package="rl_bezier_mpc",       # ROS2 package name (or use exec_name directly)
        executable=_node_script,        # absolute path works without installing
        name="go2_mpc_node",
        output="screen",
        parameters=[{
            "mode":       LaunchConfiguration("mode"),
            "checkpoint": LaunchConfiguration("checkpoint"),
            "urdf_path":  LaunchConfiguration("urdf_path"),
            "target_x":   LaunchConfiguration("target_x"),
            "target_y":   LaunchConfiguration("target_y"),
            "target_z":   LaunchConfiguration("target_z"),
            "rl_period":  LaunchConfiguration("rl_period"),
            "max_torque": LaunchConfiguration("max_torque"),
            "robot_name": LaunchConfiguration("robot_name"),
        }],
    )

    return LaunchDescription([
        mode_arg,
        checkpoint_arg,
        urdf_path_arg,
        target_x_arg,
        target_y_arg,
        target_z_arg,
        rl_period_arg,
        max_torque_arg,
        robot_name_arg,
        LogInfo(msg=["[go2_mpc] Starting controller in mode: ", LaunchConfiguration("mode")]),
        controller_node,
    ])
