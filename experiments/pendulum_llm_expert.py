import numpy as np


def expert_policy(obs: "np.ndarray") -> "np.ndarray":
    x = obs[0]
    y = obs[1]
    theta_dot = obs[2]
    theta = np.arctan2(y, x)
    Kp = 10.0
    Kd = 2.0
    torque = -Kp * theta - Kd * theta_dot
    torque = np.clip(torque, -2.0, 2.0)
    return np.array([torque], dtype=np.float32)
