import time
import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

ENV_ID = "ALE/Breakout-v5"

# Your current candidates (edit these if needed)
PADDLE_X_IDX = 90
BALL_X_IDX = 70
BALL_Y_IDX = 72

NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

def main():
    env = gym.make(ENV_ID, obs_type="ram", render_mode="human")
    ram, info = env.reset(seed=0)

    # Serve
    for _ in range(5):
        ram, r, term, trunc, info = env.step(FIRE)

    print("t  act  paddle_x  ball_x  ball_y   dx")
    print("-- ---- -------- -------- -------- -----")

    # Move RIGHT for 40 steps, then LEFT for 40, then NOOP
    action_schedule = ([RIGHT] * 40) + ([LEFT] * 40) + ([NOOP] * 40)

    prev = None
    for t, a in enumerate(action_schedule):
        ram, r, term, trunc, info = env.step(a)

        paddle_x = int(ram[PADDLE_X_IDX])
        ball_x = int(ram[BALL_X_IDX])
        ball_y = int(ram[BALL_Y_IDX])
        dx = ball_x - paddle_x

        # show velocity-ish too
        if prev is None:
            vx = vy = 0
        else:
            vx = ball_x - prev[0]
            vy = ball_y - prev[1]
        prev = (ball_x, ball_y)

        print(f"{t:02d}  {a:4d}  {paddle_x:8d} {ball_x:8d} {ball_y:8d} {dx:5d}   vx={vx:3d} vy={vy:3d}")

        if term or trunc:
            break

        time.sleep(0.02)

    env.close()

if __name__ == "__main__":
    main()
