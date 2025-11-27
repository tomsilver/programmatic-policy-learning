from openai import OpenAI
import numpy as np
from PIL import Image
import io
import base64

client = OpenAI()

##############################
# 1. Render grid → image
##############################

def grid_to_image(grid):
    """
    grid: np.array of shape (H, W), with small ints (e.g. EMPTY=0, TOKEN=1, AGENT=2)
    converts grid to a simple PNG image
    """
    H, W = grid.shape
    scale = 40
    img = Image.new("RGB", (W * scale, H * scale), (255, 255, 255))
    pixels = img.load()

    colors = {
        0: (255,255,255),   # empty
        1: (0,0,0),         # token
        2: (200,0,0),       # agent
        3: (0,0,200),       # enemy
    }

    for i in range(H):
        for j in range(W):
            c = colors.get(grid[i, j], (128,128,128))
            for x in range(scale):
                for y in range(scale):
                    pixels[j*scale + x, i*scale + y] = c

    return img


##############################
# 2. VLAM query
##############################

import base64
import io

def query_vlam(img, task_description):
    # Convert PIL image → base64 PNG string
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You choose an action in a grid environment."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_description},
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:image/png;base64,{img_b64}" }
                    },
                    {"type": "text", "text": "Return ONLY 'x,y'."}
                ],
            }
        ]
    )

    # ✔ FIX: message is an object, not a dict
    txt = r.choices[0].message.content.strip()
    return parse_action(txt)


##############################
# 3. Parse action
##############################

def parse_action(txt):
    """
    Expect output like: '3,1'
    """
    try:
        x, y = txt.split(",")
        return int(x), int(y)
    except:
        return None


##############################
# 4. One-step evaluation
##############################

def evaluate_once(env, task_description):
    state, _ = env.reset()
    img = grid_to_image(state)

    action = query_vlam(img, task_description)
    if action is None:
        return False

    next_state, reward, done, info, _ = env.step(action)
    return done and reward > 0


##############################
# 5. Run evaluation
##############################

def run_baseline(env, episodes=10):
    successes = 0
    for ep in range(episodes):
        task_desc = "You are playing the game Nim. The grid shows piles of tokens.On your turn, you must click on exactly one cell containing tokens to remove a token from that pile.Your goal is to make the winning move. Choose the grid cell to click.Return the action as 'x,y'."
        if evaluate_once(env, task_desc):
            successes += 1

    print(f"VLAM success rate: {successes}/{episodes}")

