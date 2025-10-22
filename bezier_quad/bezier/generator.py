# bezier/generator.py
import numpy as np

def bezier_cubic(P0, P1, P2, P3, t):
    """Cubic Bézier position and first derivative (for yaw/velocity)."""
    t = np.asarray(t)
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    p  = (B0[:,None]*P0 + B1[:,None]*P1 + B2[:,None]*P2 + B3[:,None]*P3)

    dB0 = -3 * (1 - t) ** 2
    dB1 = 3 * (1 - t) ** 2 - 6 * (1 - t) * t
    dB2 = 6 * (1 - t) * t - 3 * t ** 2
    dB3 = 3 * t ** 2
    dp = (dB0[:,None]*P0 + dB1[:,None]*P1 + dB2[:,None]*P2 + dB3[:,None]*P3)

    return p, dp  # p:(N,2), dp:(N,2)

def sample_xy_yaw(P0, P1, P2, P3, N, yaw_mode="tangent"):
    """Sample (x,y) & yaw along the cubic Bézier."""
    t = np.linspace(0.0, 1.0, N)
    pos, dpos = bezier_cubic(np.array(P0), np.array(P1), np.array(P2), np.array(P3), t)
    x, y = pos[:,0], pos[:,1]
    if yaw_mode == "tangent":
        yaw = np.arctan2(dpos[:,1], dpos[:,0] + 1e-9)
    else:
        # constant yaw towards goal
        dyaw = np.arctan2(P3[1] - P0[1], P3[0] - P0[0])
        yaw = np.full_like(x, dyaw)
    return x, y, yaw
