import numpy as np


def expert_policy(obs: "np.ndarray") -> int:
    obs = np.asarray(obs)
    if obs.ndim != 1 or obs.size < 128: return 1
    ram = obs.astype(np.int32, copy=False)
    paddle_x = int(ram[72])
    ball_x = int(ram[99])
    ball_y = int(ram[101])
    
    if not hasattr(expert_policy, "step_counter"):
        expert_policy.step_counter = 0
    
    if expert_policy.step_counter < 10:
        expert_policy.step_counter += 1
        return 1  # FIRE
    
    dx = ball_x - paddle_x
    deadband_px = 2.0
    if dx > deadband_px:
        return 2  # RIGHT
    elif dx < -deadband_px:
        return 3  # LEFT
    else:
        return 0  # NOOP

