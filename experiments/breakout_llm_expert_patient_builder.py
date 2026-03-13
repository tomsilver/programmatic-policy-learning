import numpy as np


def expert_policy(obs: "np.ndarray") -> int:
    obs = np.asarray(obs)
    if obs.ndim != 1 or obs.size < 128: return 1
    ram = obs.astype(np.int32, copy=False)
    paddle_x = int(ram[72]); ball_x = int(ram[99]); ball_y = int(ram[101])
    
    if not hasattr(expert_policy, "_initialized"):
        expert_policy._initialized = True
        expert_policy._steps = 0
        expert_policy._prev_ball_x = ball_x

    if expert_policy._steps < 5:
        expert_policy._steps += 1
        return 1

    if ball_y < 108:
        return 2 if ball_x > paddle_x else 3

    if ball_x < paddle_x - 1:
        return 2
    elif ball_x > paddle_x + 1:
        return 3

    if ball_x == expert_policy._prev_ball_x:
        return 1

    expert_policy._prev_ball_x = ball_x
    return 0

