# crocoddyl_iface/base_refs.py
import numpy as np
import pinocchio as pin

def yaw_to_quat(yaw):
    # z轴旋转
    return pin.Quaternion(np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)]))  # [w, x, y, z]

def build_base_refs(x, y, yaw, h=0.42):
    """
    返回每个 knot 的参考 SE3（位姿）与近似速度（twist）。高度h可按机器人身高调。
    """
    N = len(x)
    frames_SE3 = []
    twists = []

    for k in range(N):
        qz = yaw_to_quat(yaw[k])
        R = qz.toRotationMatrix()
        p = np.array([x[k], y[k], h])
        frames_SE3.append(pin.SE3(R, p))

    # 速度（差分近似），仅用于参考，可选
    for k in range(N):
        if k == 0:
            vx = (x[1] - x[0])
            vy = (y[1] - y[0])
            wz = (yaw[1] - yaw[0])
        else:
            vx = (x[k] - x[k-1])
            vy = (y[k] - y[k-1])
            wz = (yaw[k] - yaw[k-1])
        twists.append(np.array([vx, vy, 0., 0., 0., wz]))  # [vx,vy,vz,wx,wy,wz]
    return frames_SE3, twists
