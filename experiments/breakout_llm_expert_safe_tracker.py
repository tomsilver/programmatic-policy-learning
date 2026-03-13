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
        expert_policy._prev_ball_x = ball_x
        expert_policy._serving = True

    if expert_policy._serving:
        expert_policy._steps += 1
        if expert_policy._steps < 10:
            return 1
        else:
            expert_policy._serving = False
            expert_policy._steps = 0

    if ball_y >= 104: 
        if ball_x < paddle_x: 
            return 2
        elif ball_x > paddle_x: 
            return 3
        else:
            return 0
    
    if expert_policy._prev_ball_x == ball_x:
        expert_policy._steps += 1
        if expert_policy._steps > 30:
            expert_policy._serving = True
            expert_policy._steps = 0
            return 1

    expert_policy._prev_ball_x = ball_x
    expert_policy._steps = 0

    if ball_x < paddle_x:
        return 2
    elif ball_x > paddle_x:
        return 3
    else:
        return 0

