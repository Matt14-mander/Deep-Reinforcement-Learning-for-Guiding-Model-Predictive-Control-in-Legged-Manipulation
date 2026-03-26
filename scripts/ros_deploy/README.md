# ROS2 / Gazebo 部署说明

将 IsaacLab 训练的 RL+MPC 策略部署到 ROS2 + Gazebo 环境中运行。
**不需要 IsaacLab 环境**，只需 Python + ROS2 + 依赖库。

---

## 系统要求

| 组件 | 版本 |
|------|------|
| Ubuntu | 24.04 |
| ROS2 | Jazzy |
| Gazebo | Harmonic (gz-harmonic) 或 Classic |
| Python | 3.10+ |
| CUDA | 可选（CPU 推理已足够）|

---

## 依赖安装

```bash
# 1. Python 依赖（非 ROS）
pip install torch numpy scipy

# 2. Pinocchio（运动学/动力学）
pip install pin
# 或 conda: conda install -c conda-forge pinocchio

# 3. Crocoddyl（MPC 求解器）
pip install crocoddyl
# 或 conda: conda install -c conda-forge crocoddyl

# 4. ROS2 包
sudo apt install ros-jazzy-gazebo-msgs \
                 ros-jazzy-sensor-msgs \
                 ros-jazzy-geometry-msgs \
                 ros-jazzy-std-msgs \
                 ros-jazzy-launch-ros
```

---

## Go2 Gazebo 仿真环境准备

```bash
# 方法一：Unitree 官方 ROS2 SDK（推荐）
git clone https://github.com/unitreerobotics/unitree_ros2
cd unitree_ros2 && colcon build

# 方法二：社区 Go2 Gazebo 包
git clone https://github.com/unitreerobotics/go2_ros2_sdk

# 启动 Gazebo 仿真（以 unitree_ros2 为例）
ros2 launch go2_description gazebo.launch.py
```

启动后确认以下话题存在：
```bash
ros2 topic list | grep -E "joint_states|imu|model_states"
# 应该看到:
# /joint_states
# /imu/data
# /gazebo/model_states
```

---

## 运行控制器

### 方法一：直接运行（无需 ROS2 package）

```bash
cd /path/to/project/scripts/ros_deploy

python mpc_ros2_controller.py \
    --ros-args \
    -p checkpoint:=/path/to/logs/quadruped_mpc/2026-03-16_01-06-07/model_1098.pt \
    -p urdf:=/path/to/go2_description/urdf/go2.urdf \
    -p target_x:=1.5 \
    -p target_y:=0.0
```

### 方法二：作为 ROS2 package 安装运行

```bash
# 在项目根目录创建 setup.py（或 CMakeLists.txt）
# 然后 colcon build 后使用 launch 文件启动

ros2 launch scripts/ros_deploy/launch/go2_mpc_controller.launch.py \
    checkpoint:=/path/to/model_1098.pt \
    urdf:=/path/to/go2.urdf \
    target_x:=2.0 target_y:=0.0
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `policy_loader.py` | 加载 RSL-RL 训练的 .pt 检查点，纯 PyTorch，无 ROS 依赖 |
| `observation_builder.py` | 将 ROS2 传感器数据组装为 45D 观测向量；含关节顺序转换 |
| `mpc_ros2_controller.py` | ROS2 主控制节点（rclpy），50Hz 控制循环 |
| `mpc_ros_controller.py` | ROS1 版本（rospy），仅供参考 |
| `launch/go2_mpc_controller.launch.py` | ROS2 launch 文件 |
| `launch/go2_mpc_controller.launch` | ROS1 launch 文件，仅供参考 |

---

## 关键设计说明

### 关节顺序转换

Gazebo（URDF/Pinocchio，按腿分组）↔ IsaacLab（按关节类型分组）：

```
Gazebo/Pinocchio 顺序:    IsaacLab 顺序:
FL_hip(0)   →             hip:   FL(0) FR(1) RL(2) RR(3)
FL_thigh(1) →             thigh: FL(4) FR(5) RL(6) RR(7)
FL_calf(2)  →             calf:  FL(8) FR(9) RL(10) RR(11)
FR_hip(3)   → ...
...
```

`observation_builder.py` 中的 `PIN_TO_ISAAC` 数组自动完成此转换。

### 速度约定

- **观测向量（RL）**：使用世界系速度（`root_lin_vel_w`, `root_ang_vel_w`）
- **MPC 状态（Crocoddyl）**：使用体坐标系速度（`root_lin_vel_b`, `root_ang_vel_b`）
- IMU 的 `angular_velocity` 直接是体坐标系，可直接使用
- `root_lin_vel_w` 从 `/gazebo/model_states` 的 `twist.linear` 获取，再由 `world_vel_to_body()` 转换为体坐标系

### Z 轴速度滤波

MPC 状态构建中包含与训练一致的 EMA 低通滤波：
```python
z_vel_filtered = 0.3 * z_vel_raw + 0.7 * z_vel_filtered_prev
z_vel_clipped  = clip(z_vel_filtered, -0.15, 0.15)
```
防止足端冲击脉冲引发 MPC 代价爆炸。

---

## 动态更新目标

可在运行时通过 ROS2 话题更新目标位置：

```bash
ros2 topic pub /mpc_controller/goal geometry_msgs/msg/PointStamped \
    "{header: {frame_id: 'world'}, point: {x: 3.0, y: 0.5, z: 0.0}}"
```

---

## 常见问题

**Q: `/gazebo/model_states` 不存在**
Gazebo Harmonic（ROS2 Jazzy 配套）默认不发布此话题，需要安装 `ros-jazzy-gazebo-ros` 插件并在 SDF 中添加 `gazebo_ros_state` 插件。

**Q: 找不到 `crocoddyl` 模块**
`pip install crocoddyl` 如果失败，用 conda：
```bash
conda install -c conda-forge crocoddyl pinocchio
```

**Q: MPC 一直触发 Guard（cost > 50000）**
说明机器狗初始姿态与站立姿态偏差过大。在 Gazebo 中确认机器狗以标准站立姿态（CoM 高度约 0.4m）初始化。
