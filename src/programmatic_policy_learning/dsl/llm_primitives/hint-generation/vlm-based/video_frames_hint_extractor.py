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
def build_demo_hint_prompt() -> str:
    """Prompt that asks for behavior/task description WITHOUT inventing DSL
    primitives."""
    return (
        "You are analyzing expert demonstrations from a grid-based Chase-like environment.\n"
        "We ONLY control the agent, not the target.\n\n"
        "Your job:\n"
        "1) Infer the high-level objective of the agent.\n"
        "2) Extract RECURRING SPATIAL–RELATIONAL patterns that the agent seems to use.\n"
        "3) Identify asymmetries, directional relations, and distance-based cues.\n"
        "4) Extract NON-CHEATY hints about decision structure that could inspire DSL primitives later.\n\n"
        "Hard constraints:\n"
        "- Do NOT propose DSL primitives, function names, or grammar rules.\n"
        "- Use descriptive *relational* language only (e.g., 'agent is north of target', 'closing distance along x-axis', 'target nearer to boundary on that side').\n"
        "- Focus on *specific spatial relations*, not general strategies.\n"
        "- If uncertain, say so.\n\n"
        "Output ONLY this template:\n\n"
        "## DEMONSTRATION-INFERRED FEATURES (HINTS)\n\n"
        "Below is a summary of patterns extracted from a set of expert trajectories.\n"
        "These are NOT DSL primitives, but they describe spatial relations frequently\n"
        "relevant to decision-making.\n\n"
        "### High-frequency relational patterns:\n"
        "{HINT_PATTERNS}\n\n"
        "### Useful directional / asymmetry relations:\n"
        "{HINT_DIRECTIONAL}\n\n"
        "### Example state–action correlations:\n"
        "{HINT_CAUSAL}\n\n"
        "### Frequently observed local spatial configurations:\n"
        "{HINT_LOCAL}\n\n"
        "### Observed distance thresholds or step ranges:\n"
        "{HINT_DISTANCES}\n\n"
    )


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
        {"type": "input_text", "text": build_demo_hint_prompt()}
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
    paths = [f"videos/expert_demonstration_{env_name}_{i}.mp4" for i in range(11)]

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
