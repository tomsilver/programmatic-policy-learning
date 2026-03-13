import numpy as np


def expert_policy(obs: "np.ndarray") -> "np.ndarray":
    obs = np.asarray(obs, dtype=np.float32)
    x = float(obs[0]); y = float(obs[1]); angvel = float(obs[2])
    theta = float(np.arctan2(y, x))

    HANG_THRESH = 0.1
    TOP_THRESH = 0.1
    VEL_SMALL = 0.5
    VEL_FAST = 1.0
    BOOST = 0.5
    kp_top = 5.0
    kd_top = 1.0
    kp_mid = 3.0
    kd_mid = 0.5

    is_hanging_down = abs(abs(theta) - np.pi) < HANG_THRESH
    is_near_top = abs(theta) < TOP_THRESH

    if is_hanging_down:
        if abs(angvel) > VEL_SMALL:
            torque = 2.0 * np.sign(angvel)
        else:
            torque = 2.0 * np.sign(theta)
    elif is_near_top:
        torque = -kp_top * theta - kd_top * angvel
    else:
        torque = -kp_mid * theta - kd_mid * angvel
        if abs(angvel) > VEL_FAST:
            torque += BOOST * np.sign(angvel)

    torque = float(np.clip(torque, -2.0, 2.0))
    return np.array([torque], dtype=np.float32)

