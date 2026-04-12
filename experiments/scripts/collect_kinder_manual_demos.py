"""Collect manual KinDER demonstrations and save them as repo-native demos.

This script uses the environment's ``get_action_from_gui_input`` hook, similar
to the upstream kindergarden collector, but writes out
``programmatic_policy_learning.data.demo_io.DemoRecord`` files containing a
``Trajectory`` that is directly reusable in this repo.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kinder
import numpy as np

from programmatic_policy_learning.data.demo_io import DemoRecord, save_demo_record
from programmatic_policy_learning.data.demo_types import Trajectory

try:
    import pygame
except ImportError:
    print("Error: pygame is required for manual demo collection.")
    print("Install it with: uv add pygame")
    sys.exit(1)


def _sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "__").replace(":", "_")


def _parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seed list or simple inclusive range."""
    text = raw.strip()
    if ".." in text:
        lo_text, hi_text = text.split("..", maxsplit=1)
        lo = int(lo_text)
        hi = int(hi_text)
        if hi < lo:
            raise ValueError(f"Invalid seed range: {raw!r}")
        return list(range(lo, hi + 1))
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


@dataclass
class AnalogStick:
    """Simple on-screen analog stick for mouse input."""

    center_x: int
    center_y: int
    radius: int = 30
    x: float = 0.0
    y: float = 0.0
    is_active: bool = False

    def update_from_mouse(
        self, mouse_pos: tuple[int, int], mouse_pressed: bool
    ) -> None:
        """Update the stick state from the current mouse input."""
        if not mouse_pressed:
            self.is_active = False
            self.x = 0.0
            self.y = 0.0
            return
        if not self.is_mouse_over(mouse_pos):
            self.is_active = False
            return
        self.is_active = True
        mouse_x, mouse_y = mouse_pos
        dx = mouse_x - self.center_x
        dy = mouse_y - self.center_y
        self.x = max(-1.0, min(1.0, dx / self.radius))
        self.y = max(-1.0, min(1.0, -dy / self.radius))

    def is_mouse_over(self, mouse_pos: tuple[int, int]) -> bool:
        """Return whether the pointer is inside the stick circle."""
        mouse_x, mouse_y = mouse_pos
        dx = mouse_x - self.center_x
        dy = mouse_y - self.center_y
        return math.sqrt(dx * dx + dy * dy) <= self.radius

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the analog stick control."""
        pygame.draw.circle(
            screen, (100, 100, 100), (self.center_x, self.center_y), self.radius, 2
        )
        pygame.draw.line(
            screen,
            (150, 150, 150),
            (self.center_x - 5, self.center_y),
            (self.center_x + 5, self.center_y),
            1,
        )
        pygame.draw.line(
            screen,
            (150, 150, 150),
            (self.center_x, self.center_y - 5),
            (self.center_x, self.center_y + 5),
            1,
        )
        if self.is_active:
            stick_x = self.center_x + int(self.x * self.radius)
            stick_y = self.center_y - int(self.y * self.radius)
            pygame.draw.circle(screen, (255, 255, 255), (stick_x, stick_y), 8)
        else:
            pygame.draw.circle(screen, (200, 200, 200), (self.center_x, self.center_y), 8)


class ManualKinderDemoCollector:
    """Pygame-based manual demo collector for KinDER tasks."""

    def __init__(
        self,
        env_id: str,
        output_dir: Path,
        seeds: list[int],
        *,
        screen_width: int = 1200,
        screen_height: int = 700,
        render_fps: int = 20,
        font_size: int = 24,
    ) -> None:
        self.env_id = env_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = list(seeds)
        if not self.seeds:
            raise ValueError("At least one seed is required.")

        kinder.register_all_environments()
        self.env = kinder.make(env_id, render_mode="rgb_array")
        self.unwrapped_env = self.env.unwrapped
        if not hasattr(self.unwrapped_env, "get_action_from_gui_input"):
            raise RuntimeError(
                f"Environment {env_id} does not implement get_action_from_gui_input."
            )

        pygame.init()
        pygame.joystick.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Manual Demo Collection - {env_id}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, font_size)
        self.render_fps = render_fps
        self.controller = None
        if pygame.joystick.get_count() > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print(f"Controller connected: {self.controller.get_name()}")
        else:
            print("No controller detected; using mouse + keyboard controls.")

        stick_radius = 40
        side_panel_width = 170
        self.left_stick = AnalogStick(side_panel_width // 2, screen_height // 2, stick_radius)
        self.right_stick = AnalogStick(
            screen_width - side_panel_width // 2, screen_height // 2, stick_radius
        )
        self.side_panel_width = side_panel_width

        self.keys_pressed: set[str] = set()
        self.current_seed_index = 0
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.terminated = False
        self.truncated = False
        self.reset_env()

    @property
    def current_seed(self) -> int:
        """Return the active seed being collected."""
        return self.seeds[self.current_seed_index]

    def reset_env(self) -> None:
        """Reset the environment for the current seed."""
        obs, _ = self.env.reset(seed=self.current_seed)
        self.observations = [np.asarray(obs, dtype=np.float32).copy()]
        self.actions = []
        self.rewards = []
        self.terminated = False
        self.truncated = False
        self.keys_pressed.clear()
        print(f"\nReset env for seed={self.current_seed}")

    def _save_current_demo(self) -> Path | None:
        """Save the current trajectory if it contains actions."""
        if not self.actions:
            print("No actions recorded; skipping save.")
            return None
        steps = [
            (obs.copy(), action.copy())
            for obs, action in zip(self.observations[:-1], self.actions)
        ]
        record = DemoRecord(
            env_id=self.env_id,
            seed=self.current_seed,
            trajectory=Trajectory(steps=steps),
            rewards=list(self.rewards),
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            metadata={
                "num_observations": len(self.observations),
                "num_actions": len(self.actions),
                "saved_at_unix": int(time.time()),
            },
        )
        out_path = (
            self.output_dir
            / _sanitize_env_id(self.env_id)
            / f"seed_{self.current_seed:04d}.pkl"
        )
        save_demo_record(out_path, record)
        print(f"Saved demo for seed={self.current_seed} to {out_path}")
        return out_path

    def _advance_seed(self) -> bool:
        """Move to the next seed, returning False when collection is done."""
        self.current_seed_index += 1
        if self.current_seed_index >= len(self.seeds):
            print("Finished all requested seeds.")
            return False
        self.reset_env()
        return True

    def _build_input_data(self) -> dict[str, Any]:
        return {
            "keys": self.keys_pressed,
            "left_stick": (self.left_stick.x, self.left_stick.y),
            "right_stick": (self.right_stick.x, self.right_stick.y),
        }

    def step_env(self) -> None:
        """Advance the environment by one manual control step."""
        action = self.unwrapped_env.get_action_from_gui_input(self._build_input_data())
        action_arr = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, _ = self.env.step(action_arr)
        self.actions.append(action_arr.copy())
        self.rewards.append(float(reward))
        self.observations.append(np.asarray(obs, dtype=np.float32).copy())
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

    def _update_analog_inputs(self) -> bool:
        """Update controller or mouse stick state and return whether input changed."""
        some_action_input = False
        if self.controller:
            left_x = self.controller.get_axis(0)
            left_y = -self.controller.get_axis(1)
            right_x = self.controller.get_axis(2)
            right_y = -self.controller.get_axis(3)
            deadzone = 0.1
            if abs(left_x) < deadzone:
                left_x = 0.0
            if abs(left_y) < deadzone:
                left_y = 0.0
            if abs(right_x) < deadzone:
                right_x = 0.0
            if abs(right_y) < deadzone:
                right_y = 0.0
            self.left_stick.x = left_x
            self.left_stick.y = left_y
            self.left_stick.is_active = abs(left_x) > 0 or abs(left_y) > 0
            self.right_stick.x = right_x
            self.right_stick.y = right_y
            self.right_stick.is_active = abs(right_x) > 0 or abs(right_y) > 0
            some_action_input = self.left_stick.is_active or self.right_stick.is_active
            return some_action_input

        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        left_clicked = self.left_stick.is_mouse_over(mouse_pos) and mouse_pressed
        right_clicked = self.right_stick.is_mouse_over(mouse_pos) and mouse_pressed
        self.left_stick.update_from_mouse(mouse_pos, left_clicked)
        self.right_stick.update_from_mouse(mouse_pos, right_clicked)
        return left_clicked or right_clicked

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False when the app should quit."""
        if self.terminated or self.truncated:
            if self.terminated:
                print("Goal reached; auto-saving successful demo.")
                self._save_current_demo()
            else:
                print("Episode truncated; not saving current demo.")
            return self._advance_seed()

        some_action_input = self._update_analog_inputs()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_env()
                elif event.key == pygame.K_g:
                    self._save_current_demo()
                elif event.key == pygame.K_n:
                    if self._save_current_demo() is not None:
                        return self._advance_seed()
                elif event.key == pygame.K_q:
                    return False
                else:
                    key_name = pygame.key.name(event.key)
                    if key_name not in {"r", "g", "n", "q"}:
                        self.keys_pressed.add(key_name)
                        some_action_input = True
            if event.type == pygame.KEYUP:
                key_name = pygame.key.name(event.key)
                self.keys_pressed.discard(key_name)
            if event.type == pygame.JOYBUTTONDOWN and self.controller:
                if event.button == 0:
                    self.keys_pressed.add("space")
                    some_action_input = True
                elif event.button == 1:
                    self.reset_env()
                elif event.button == 2:
                    if self._save_current_demo() is not None:
                        return self._advance_seed()
                elif event.button == 3:
                    self._save_current_demo()
                elif event.button == 11:
                    self.keys_pressed.add("w")
                    some_action_input = True
                elif event.button == 12:
                    self.keys_pressed.add("s")
                    some_action_input = True
            if event.type == pygame.JOYBUTTONUP and self.controller:
                if event.button == 0:
                    self.keys_pressed.discard("space")
                elif event.button == 11:
                    self.keys_pressed.discard("w")
                elif event.button == 12:
                    self.keys_pressed.discard("s")

        if some_action_input:
            self.step_env()
        return True

    def render(self) -> None:
        """Render the env frame and control overlay inside the pygame window."""
        img: np.ndarray = self.env.render()
        img_surface = pygame.surfarray.make_surface(img[:, :, :3].swapaxes(0, 1))
        img_rect = img_surface.get_rect()
        center_width = self.screen_width - 2 * self.side_panel_width
        scale = min(center_width / img_rect.width, self.screen_height / img_rect.height)
        new_width = int(img_rect.width * scale)
        new_height = int(img_rect.height * scale)
        img_surface = pygame.transform.scale(img_surface, (new_width, new_height))
        img_rect = img_surface.get_rect()
        img_rect.center = (self.screen_width // 2, self.screen_height // 2)

        self.screen.fill((0, 0, 0))
        self.screen.blit(img_surface, img_rect)
        self.left_stick.draw(self.screen)
        self.right_stick.draw(self.screen)

        left_label = self.font.render("Left Stick", True, (255, 255, 255))
        right_label = self.font.render("Right Stick", True, (255, 255, 255))
        self.screen.blit(left_label, (self.left_stick.center_x - 40, self.left_stick.center_y - 60))
        self.screen.blit(right_label, (self.right_stick.center_x - 45, self.right_stick.center_y - 60))

        top_lines = [
            f"{self.env_id}",
            f"seed={self.current_seed} ({self.current_seed_index + 1}/{len(self.seeds)})",
            f"steps={len(self.actions)}",
        ]
        for idx, line in enumerate(top_lines):
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10 + idx * 26))

        instructions = [
            "Mouse/controller sticks: move + rotate",
            "W/S or d-pad up/down: arm out/in",
            "Space or controller A: toggle vacuum",
            "R: reset seed",
            "G: save current demo",
            "N: save and advance to next seed",
            "Q: quit",
        ]
        for idx, line in enumerate(instructions):
            text_surface = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(text_surface, (10, self.screen_height - 25 * (len(instructions) - idx)))

        pygame.display.flip()

    def run(self) -> None:
        """Run the interactive demo collection loop."""
        print("Starting manual KinDER demo collection.")
        print("This env renders through pygame from `rgb_array`; it does not need Gym `human` mode.")
        running = True
        while running:
            running = self.handle_events()
            self.render()
            self.clock.tick(self.render_fps)
        self.env.close()
        pygame.quit()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default="kinder/PushPullHook2D-v0",
        help="KinDER environment id to collect demos for.",
    )
    parser.add_argument(
        "--seeds",
        default="0..9",
        help="Comma-separated seeds or inclusive range like 0..9.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manual_demos"),
        help="Directory where demo pickle files will be saved.",
    )
    parser.add_argument("--render-fps", type=int, default=20)
    args = parser.parse_args()

    collector = ManualKinderDemoCollector(
        args.env_id,
        args.output_dir,
        _parse_seeds(args.seeds),
        render_fps=args.render_fps,
    )
    collector.run()


if __name__ == "__main__":
    _main()
