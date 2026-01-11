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
def build_demo_hint_prompt(num_frames: int) -> str:
    """Build the VLM prompt for extracting DSL-relevant hints from visual demonstrations."""
    return f"""
You are given expert demonstrations from a grid-based environment as IMAGE FRAMES.

Each demonstration consists of:
- An image of the state s_t (before action)
- The action a_t, given as a clicked cell location (row, col)
- An image of the state s_t+1 (after deterministic transition)

You will be shown {num_frames} demonstrations in sequence.

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


def describe_five_videos(
    video_paths: list[str],
    model: str = "gpt-4.1",
    max_frames_per_video: int = 12,
    sample_every_n_frames: int = 15,
) -> str:
    """Process expert demo videos and generate a hint summary using a vision
    model."""
    client = OpenAI()

    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": build_demo_hint_prompt(4)}
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
    env_name = "TwoPileNim"
    paths = [f"videos/expert_demonstration_{env_name}_{i}.mp4" for i in range(4)]

    text = describe_five_videos(
        paths,
        model="gpt-4.1",
        max_frames_per_video=10,
        sample_every_n_frames=12,
    )

    # Generate a timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    num_demos = len(paths)
    output_dir = "videos/output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{timestamp}_{num_demos}_demos_{env_name}.txt"

    # Write the text to the file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    logging.info(f"Hints written to {output_file}")
