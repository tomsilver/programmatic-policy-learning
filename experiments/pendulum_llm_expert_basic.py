import numpy as np


def expert_policy(obs: "np.ndarray") -> "np.ndarray":
    x, y, theta_dot = obs
    theta = np.arctan2(y, x)
    kp = 10.0
    kd = 1.0
    torque = -kp * theta - kd * theta_dot
    torque = np.clip(torque, -2.0, 2.0)
    return np.array([torque], dtype=np.float32)

