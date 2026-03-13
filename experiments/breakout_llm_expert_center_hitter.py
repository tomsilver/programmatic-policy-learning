import numpy as np


def expert_policy(obs: "np.ndarray") -> int:
    obs = np.asarray(obs)
    if obs.ndim != 1 or obs.size < 128: return 1
    ram = obs.astype(np.int32, copy=False)
    paddle_x = int(ram[72])
    ball_x = int(ram[99])
    ball_y = int(ram[101])

    if not hasattr(expert_policy, "_steps"):
        expert_policy._steps = 0
        expert_policy._prev_ball_y = ball_y

    if expert_policy._steps < 5:
        expert_policy._steps += 1
        return 1

    if ball_y < 110 and ball_y > 100 and expert_policy._prev_ball_y <= ball_y:
        expert_policy._prev_ball_y = ball_y
        if paddle_x < ball_x:
            return 2
        elif paddle_x > ball_x:
            return 3
        else:
            return 0
    elif ball_y > 110:
        expert_policy._prev_ball_y = ball_y
        if paddle_x < ball_x - 1:
            return 2
        elif paddle_x > ball_x + 1:
            return 3
        else:
            return 0
    else:
        expert_policy._prev_ball_y = ball_y
        return 0

