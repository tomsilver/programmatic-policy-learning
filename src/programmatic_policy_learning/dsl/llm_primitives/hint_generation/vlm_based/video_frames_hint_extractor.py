"""This module provides utilities for extracting hints from demonstration
videos.

It includes functions for processing video frames, converting them to
data URLs, and sending them to a vision-capable model for analysis.
"""

import base64
import logging
import os
import time
from typing import Any, cast

import cv2
from openai import OpenAI


def extract_frames(
    video_path: str,
    max_frames: int = 12,
    sample_every_n_frames: int = 15,
    resize_width: int = 512,
) -> list[Any]:
    """Extract up to max_frames frames from a video, sampling every N frames.

    Returns a list of BGR images (numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % sample_every_n_frames == 0:
            # Resize to reduce payload size
            h, w = frame.shape[:2]
            if w > resize_width:
                new_h = int(h * (resize_width / w))
                frame = cv2.resize(
                    frame, (resize_width, new_h), interpolation=cv2.INTER_AREA
                )

            frames.append(frame)

            if len(frames) >= max_frames:
                break

        idx += 1

    cap.release()
    return frames


def bgr_to_jpeg_bytes(frame_bgr: Any) -> bytes:
    """Convert an OpenCV BGR frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Failed to encode frame as JPEG.")
    return buf.tobytes()


def jpeg_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    """Create a base64 data URL for the OpenAI vision input."""
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def video_to_data_urls(
    video_path: str,
    max_frames: int = 12,
    sample_every_n_frames: int = 15,
) -> list[str]:
    """Convert video frames to data URLs."""
    frames = extract_frames(
        video_path,
        max_frames=max_frames,
        sample_every_n_frames=sample_every_n_frames,
    )
    data_urls = []
    for fr in frames:
        jpeg = bgr_to_jpeg_bytes(fr)
        data_urls.append(jpeg_bytes_to_data_url(jpeg))
    return data_urls


# pylint: disable=line-too-long
def build_demo_hint_prompt(num_traj: int) -> str:
    """Build the VLM prompt for extracting DSL-relevant hints from visual
    demonstrations."""
    return f"""
You are given expert demonstrations from a grid-based environment as IMAGE FRAMES.

Each demonstration consists of:
- An image of the state s_t (before action)
- The action a_t, given as a clicked cell location (row, col)
- An image of the state s_t+1 (after deterministic transition)

You will be shown {num_traj} demonstrations in sequence.

----
Your task:
Explain the decision making rule, how action a is selected in state s.

NOTE:
- just infer rule from the demos, not from the nature of the game, and the way it is usually played.
"""


# From the visual observations ONLY, extract **abstract, reusable decision-time predicates**
# that could serve as **atomic building blocks** in a domain-specific language (DSL).

# CONSTRAINTS:

# OBSERVATION SCOPE
# - Use ONLY information visible in the current image (s_t).
# - Use ONLY information available at decision time.
# - Do NOT use future frames or changes after the action.

# ABSTRACTION LEVEL
# - Each predicate MUST be atomic and independent.
# - Each predicate MUST correspond to a single boolean-valued property.
# - Each predicate MUST be implementable as a function of the form:
#     (Cell, Observation) -> Bool

# PROHIBITED CONTENT
# - Do NOT describe strategies, policies, or goals.
# - Do NOT describe game rules or mechanics.
# - Do NOT name or label objects, symbols, colors, or semantics.
# - Do NOT describe multi-step logic, compositions, or conjunctions.
# - Do NOT describe shapes, patterns, regions, or global configurations.
# - Do NOT describe counts, extrema, uniqueness, or optimality.
# - Do NOT include examples, explanations, or justification.

# REFERENCE CELLS
# - If a predicate depends on another cell, refer to it only as
#   a “reference cell derived from the current observation”.
# - Do NOT define how the reference cell is chosen.

# OUTPUT FORMAT
# - Return ONLY a list of predicates
# - One predicate per line
# - No introduction, explanation, or extra text
# """


def build_structural_prompt(num_frames: int) -> str:
    """Build the VLM prompt for structural hints from video frames."""
    return f"""
You are given expert demonstrations from a grid-based environment as IMAGE FRAMES.

Each demonstration consists of:
- An image of the state s_t (before action)
- The action a_t, given as a clicked cell location (row, col), 0-indexed
- An image of the state s_t+1 (after deterministic transition)

You will be shown {num_frames} demonstrations in sequence.

----
Your task:
Infer the expert’s strategy for selecting action a in state s from the demonstrations.

IMPORTANT RULES
- Use ONLY evidence from the demonstrations. Do NOT use outside knowledge about the game.
- Do NOT narrate individual steps. Extract general rules that hold across demos.
- Express rules as testable predicates over (s, a).
- Prefer rules that generalize; ignore demo-specific coincidences.
- When you are uncertain, say so and give the simplest consistent rule.

Helpful language:
- Use relative positions to the action cell: same row/col, above/below, left/right, diagonals, adjacency.
- If a rule depends on patterns along a direction, describe it using phrases like:
  “first hit”, “blocked by”, “before reaching”, “until”, “closest in that direction”.

Output EXACTLY the following 3 sections and nothing else:

1) POLICY RULES (8–15 bullets):
- Write each rule as a predicate over (s, a).
- Use relative positions anchored at the action cell (r,c).
- After each bullet, add “(seen in ~N/{num_frames} demos)”.
- If multiple modes exist, include separate bullets for each mode.

2) COUNTERFACTUAL DISTRACTORS (5 bullets):
- Write patterns that often appear near the chosen action but where the expert does NOT click.
- Same predicate style.
- After each bullet, add “(seen in ~N/{num_frames} demos)”.

3) GAME DYNAMICS (5–10 bullets):
- Describe deterministic transition patterns you can infer from s_t → s_t+1.
- Focus on consistent mechanisms: movement, falling, blocking, collisions, redraw/erase, spawning, etc.
- Keep each bullet short and testable.
"""


def describe_videos(
    video_paths: list[str],
    model: str = "gpt-4.1",
    num_traj: int = 4,
    max_frames_per_video: int = 12,
    sample_every_n_frames: int = 15,
) -> str:
    """Process expert demo videos and generate a hint summary using a vision
    model."""
    client = OpenAI()

    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": build_demo_hint_prompt(num_traj)}
        # {"type": "input_text", "text": build_structural_prompt(num_traj)}
    ]

    # Build multi-video content
    for i, vp in enumerate(video_paths, start=1):
        data_urls = video_to_data_urls(
            vp,
            max_frames=max_frames_per_video,
            sample_every_n_frames=sample_every_n_frames,
        )

        # Add a small header per video
        content.append(
            {
                "type": "input_text",
                "text": f"\n\n--- Video {i}: {os.path.basename(vp)} ---\n",
            }
        )

        # Attach frames as images
        for url in data_urls:
            content.append(
                {
                    "type": "input_image",
                    "image_url": url,
                }
            )

    # Make the request
    input_payload = [
        {
            "role": "user",
            "content": content,
        }
    ]

    resp = client.responses.create(
        model=model,
        input=cast(Any, input_payload),
        max_output_tokens=800,
        temperature=0.2,
    )

    return resp.output_text


if __name__ == "__main__":
    env_name = "StopTheFall"
    num_demos = 4
    paths = [
        f"videos/expert_demonstration_{env_name}_{i}.mp4" for i in range(num_demos)
    ]

    text = describe_videos(
        paths,
        model="gpt-4.1",
        num_traj=num_demos,
        max_frames_per_video=10,
        sample_every_n_frames=12,
    )

    # Generate a timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join("videos", "output", env_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{timestamp}_{num_demos}_demos_{env_name}.txt"

    # Write the text to the file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    logging.info(f"Hints written to {output_file}")
