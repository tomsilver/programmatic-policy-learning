import numpy as np


def expert_policy(obs: "np.ndarray") -> int:
    obs = np.asarray(obs)
    if obs.ndim != 1 or obs.size < 128: return 1
    ram = obs.astype(np.int32, copy=False)
    paddle_x = int(ram[72]); ball_x = int(ram[99]); ball_y = int(ram[101])
    
    if not hasattr(expert_policy, "_initialized"):
        expert_policy._initialized = True
        expert_policy._fire_timer = 0
        expert_policy._prev_ball_x = ball_x
        
    if expert_policy._fire_timer < 5:
        expert_policy._fire_timer += 1
        return 1
        
    if abs(ball_y - 103) < 10 and (ball_x < paddle_x or ball_x > paddle_x + 2):
        if ball_x < paddle_x:
            return 3
        else:
            return 2
    elif ball_y >= 103:
        expert_policy._fire_timer = 1
        return 1
    else:
        if ball_x < paddle_x:
            return 3
        elif ball_x > paddle_x + 2:
            return 2
        else:
            return 0

