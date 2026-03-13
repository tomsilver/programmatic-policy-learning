import numpy as np


def expert_policy(obs: "np.ndarray") -> int:
    obs = np.asarray(obs)
    if obs.ndim != 1 or obs.size < 128: return 1
    ram = obs.astype(np.int32, copy=False)
    paddle_x = int(ram[72])
    ball_x = int(ram[99])
    ball_y = int(ram[101])
    
    if not hasattr(expert_policy, "_side"):
        expert_policy._side = 2
        expert_policy._last_paddle_x = paddle_x
        expert_policy._fire_count = 0

    if ball_y == 0 and expert_policy._fire_count < 5:
        expert_policy._fire_count += 1
        return 1

    if ball_y >= 20:
        expert_policy._fire_count = 0
        
        if (paddle_x < ball_x - 1) and expert_policy._side == 2:
            action = 2
        elif (paddle_x > ball_x + 1) and expert_policy._side == 3:
            action = 3
        elif paddle_x < ball_x:
            action = 2
        elif paddle_x > ball_x:
            action = 3
        else:
            action = 0
        
        if (expert_policy._last_paddle_x == paddle_x) and (action == 0):
            expert_policy._side = 3 if expert_policy._side == 2 else 2
        
        expert_policy._last_paddle_x = paddle_x
        return action
    return 1

