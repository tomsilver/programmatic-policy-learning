import numpy as np


def expert_policy(obs: "np.ndarray") -> int:
    obs = np.asarray(obs)
    if obs.ndim != 1 or obs.size < 128: return 1
    ram = obs.astype(np.int32, copy=False)
    paddle_x = int(ram[72])
    ball_x = int(ram[99])
    ball_y = int(ram[101])
    
    if not hasattr(expert_policy, '_steps'):
        expert_policy._steps = 0
        expert_policy._prev_ball_x = ball_x
        expert_policy._fired_once = False

    if expert_policy._steps < 10:
        expert_policy._steps += 1
        return 1

    if ball_y < 80 and expert_policy._prev_ball_x == ball_x:
        expert_policy._fired_once = True

    if ball_y >= 80 and expert_policy._fired_once:
        if paddle_x > ball_x:
            action = 3
        elif paddle_x < ball_x:
            action = 2
        else:
            action = 0
    else:
        action = 0

    expert_policy._prev_ball_x = ball_x
    return action

