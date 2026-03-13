import numpy as np


def expert_policy(obs: "np.ndarray") -> int:
    obs = np.asarray(obs)
    if obs.ndim != 1 or obs.size < 128: return 1
    ram = obs.astype(np.int32, copy=False)
    paddle_x = int(ram[72]); ball_x = int(ram[99]); ball_y = int(ram[101])
    
    if not hasattr(expert_policy, '_steps'):
        expert_policy._steps = 0
        expert_policy._prev_ball_x = ball_x
        expert_policy._ball_served = False
        
    if not expert_policy._ball_served:
        expert_policy._steps += 1
        if expert_policy._steps < 5:
            return 1
        expert_policy._ball_served = True
    
    if ball_y > 100:
        if ball_x < paddle_x:
            return 3
        elif ball_x > paddle_x:
            return 2
        else:
            return 0
    else:
        if ball_y < 50:
            if abs(ball_x - paddle_x) > 5:
                return 2 if ball_x > paddle_x else 3
            else:
                return 0
        else:
            if abs(ball_x - expert_policy._prev_ball_x) == 0:
                return 1
            expert_policy._prev_ball_x = ball_x
            
    return 0

