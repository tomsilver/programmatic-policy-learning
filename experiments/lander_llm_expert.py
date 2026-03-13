import numpy as np


def expert_policy(obs: "np.ndarray") -> "np.ndarray":
    obs = np.asarray(obs, dtype=np.float32)
    x = float(obs[0]); y = float(obs[1])
    vx = float(obs[2]); vy = float(obs[3])
    angle = float(obs[4]); angvel = float(obs[5])
    leg_l = float(obs[6]); leg_r = float(obs[7])
    
    Y_THRESH = 1.0
    main = 0.0
    side = 0.0
    
    if y < Y_THRESH:
        main = -vy * 0.5
        main += (0 - y) * 0.1
        if abs(x) > 0.1:
            side = -np.sign(x) * (0.5 * (1 - abs(angle) / np.pi))
        side += -vx * 0.1
    else:
        side = -np.sign(x) * 0.1
    
    main = float(np.clip(main, -1.0, 1.0))
    side = float(np.clip(side, -1.0, 1.0))
    return np.array([main, side], dtype=np.float32)

