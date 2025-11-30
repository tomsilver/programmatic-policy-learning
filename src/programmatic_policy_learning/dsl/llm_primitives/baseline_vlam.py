"""VLM BASELINE."""

import base64
import io
import logging
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

client = OpenAI()

##############################
# Render grid → image
##############################


def grid_to_image(grid: np.ndarray) -> Image.Image:
    """Convert a grid to a simple PNG image.

    Args:
        grid (np.ndarray): A 2D array representing the grid.

    Returns:
        Image.Image: The generated image.
    """
    H, W = grid.shape
    scale = 40
    img = Image.new("RGB", (W * scale, H * scale), (255, 255, 255))
    pixels = img.load()

    colors = {
        0: (255, 255, 255),  # empty
        1: (0, 0, 0),  # token
        2: (200, 0, 0),  # agent
        3: (0, 0, 200),  # enemy
    }

    for i in range(H):
        for j in range(W):
            c = colors.get(grid[i, j], (128, 128, 128))
            if pixels is not None:
                for x in range(scale):
                    for y in range(scale):
                        pixels[j * scale + x, i * scale + y] = c

    return img


##############################
# VLM query
##############################


def query_vlam(img: Image.Image, task_description: str) -> tuple[int, int] | None:
    """Query the VLAM model with an image and task description.

    Args:
        img (Image.Image): The input image.
        task_description (str): The task description.

    Returns:
        tuple[int, int] | None: The parsed action as (x, y) or None if parsing fails.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You choose an action in a grid environment.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_description},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": "Return ONLY 'x,y'."},
                ],
            },
        ],
    )

    txt = r.choices[0].message.content.strip() if r.choices[0].message.content else None
    return parse_action(txt) if txt else None


##############################
# 3. Parse action
##############################


def parse_action(txt: str) -> tuple[int, int] | None:
    """Parse an action from a string.

    Args:
        txt (str): The input string, e.g., '3,1'.

    Returns:
        tuple[int, int] | None: The parsed action as (x, y) or None if parsing fails.
    """
    try:
        x, y = txt.split(",")
        return int(x), int(y)
    except ValueError:
        return None


##############################
# 4. One-step evaluation
##############################


def evaluate_once(env: Any, task_description: str) -> bool:
    """Evaluate a single episode in the environment.

    Args:
        env (Any): The environment to evaluate.
        task_description (str): The task description.

    Returns:
        bool: True if the evaluation is successful, False otherwise.
    """
    state, _ = env.reset()
    img = grid_to_image(state)

    action = query_vlam(img, task_description)
    if action is None:
        return False

    _, reward, done, _, _ = env.step(action)
    return done and reward > 0


##############################
# 5. Run evaluation
##############################


def run_baseline(env: Any, episodes: int = 10) -> None:
    """Run the baseline evaluation for a number of episodes.

    Args:
        env (Any): The environment to evaluate.
        episodes (int): The number of episodes to run.
    """
    successes = 0
    for _ in range(episodes):
        task_desc = (
            "You are playing the game Nim. The grid shows piles of tokens."
            "On your turn, you must click on exactly one cell containing tokens "
            "to remove a token from that pile. Your goal is to make the winning move. "
            "Choose the grid cell to click. Return the action as 'x,y'."
        )
        if evaluate_once(env, task_desc):
            successes += 1

    logging.info(f"VLAM success rate: {successes}/{episodes}")
