"""Custom Maze environment provider with outer void region."""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import Env, spaces
from omegaconf import DictConfig


class MazeEnv(gym.Env):
    """A Gymnasium-compatible maze environment with an outer void region.

    Coordinate system uses NumPy convention:
        (0,0) is top-left, row increases downward, col increases rightward.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTIONS = {
        0: np.array([-1, 0]),  # N (up)
        1: np.array([1, 0]),  # S (down)
        2: np.array([0, 1]),  # E (right)
        3: np.array([0, -1]),  # W (left)
    }

    def __init__(
        self,
        inner_maze: Optional[np.ndarray] = None,
        outer_margin: int = 3,
        enable_render: bool = True,
    ) -> None:
        super().__init__()

        # --- Maze setup ---
        assert outer_margin >= 1, "outer_margin must be >= 1"
        self.outer_margin = outer_margin
        self.enable_render = enable_render

        if inner_maze is None:
            inner_maze = np.zeros((10, 10), dtype=np.int8)

        self.inner_maze: np.ndarray = inner_maze
        self.inner_h, self.inner_w = inner_maze.shape

        # Global coordinate bounds
        # Maze covers [0..inner_h-1] × [0..inner_w-1]
        # Add wall layer around (-1..inner_h) and outer void beyond that
        self.row_min = -outer_margin - 1
        self.row_max = self.inner_h + outer_margin
        self.col_min = -outer_margin - 1
        self.col_max = self.inner_w + outer_margin

        self.grid_h = self.row_max - self.row_min + 1
        self.grid_w = self.col_max - self.col_min + 1

        # --- Gymnasium spaces ---
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([self.row_min, self.col_min]),
            high=np.array([self.row_max, self.col_max]),
            dtype=np.int64,
        )

        # --- Grid setup ---
        self._build_grid()

        # --- Start and goal positions (NumPy convention) ---
        self.start_pos = np.array([0, 0])  # top-left
        self.goal_pos = np.array([self.inner_h - 1, self.inner_w - 1])  # bottom-right
        self.agent_pos: np.ndarray = np.zeros(
            2, dtype=np.int64
        )  # properly initialized in reset()

        # --- Rendering setup ---
        if enable_render:
            pygame.init()
            self.cell_size = 30
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode(
                (self.grid_w * self.cell_size, self.grid_h * self.cell_size)
            )
            pygame.display.set_caption("OuterMazeEnv")

    # --------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------
    def _to_grid_idx(self, pos: np.ndarray | tuple[int, int]) -> tuple[int, int]:
        """Map (row, col) world coords to array indices for rendering."""
        r = pos[0] - self.row_min
        c = pos[1] - self.col_min
        return r, c

    def _build_grid(self) -> None:
        """Build the grid with maze, wall border, and outer void."""
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int8)

        # Add walls around inner maze
        for r in range(-1, self.inner_h + 1):
            for c in [-1, self.inner_w]:
                gr, gc = self._to_grid_idx((r, c))
                self.grid[gr, gc] = 1
        for c in range(-1, self.inner_w + 1):
            for r in [-1, self.inner_h]:
                gr, gc = self._to_grid_idx((r, c))
                self.grid[gr, gc] = 1

        # Insert the inner maze itself
        for r in range(self.inner_h):
            for c in range(self.inner_w):
                gr, gc = self._to_grid_idx((r, c))
                self.grid[gr, gc] = self.inner_maze[r, c]

        # Create north entrance above (0,0): (-1, 0)
        gr, gc = self._to_grid_idx((-1, 0))
        self.grid[gr, gc] = 0  # open passage

    # --------------------------------------------------------------
    # Gymnasium API
    # --------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Start in outer void above the maze
        r = np.random.randint(self.row_min, -1)
        c = np.random.randint(self.col_min, self.col_max + 1)
        self.agent_pos = np.array([r, c])
        return self.agent_pos.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        move = self.ACTIONS[action]
        new_pos = self.agent_pos + move
        reward, terminated, truncated = -0.01, False, False

        if (
            self.row_min <= new_pos[0] <= self.row_max
            and self.col_min <= new_pos[1] <= self.col_max
        ):
            gr, gc = self._to_grid_idx(new_pos)
            if self.grid[gr, gc] == 0:
                self.agent_pos = new_pos

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward, terminated = 1.0, True

        return self.agent_pos.copy(), reward, terminated, truncated, {}

    # --------------------------------------------------------------
    # Rendering
    # --------------------------------------------------------------
    def render(self) -> None:
        if not self.enable_render:
            return

        # keep macOS window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((230, 230, 230))
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                color = (0, 0, 0) if self.grid[r, c] == 1 else (220, 220, 220)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        c * self.cell_size,
                        r * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

        # Agent
        gr, gc = self._to_grid_idx(self.agent_pos)
        pygame.draw.rect(
            self.screen,
            (0, 0, 255),
            (gc * self.cell_size, gr * self.cell_size, self.cell_size, self.cell_size),
        )

        # Goal
        gr, gc = self._to_grid_idx(self.goal_pos)
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (gc * self.cell_size, gr * self.cell_size, self.cell_size, self.cell_size),
        )

        pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        if self.enable_render:
            pygame.quit()


def create_maze_env(env_config: DictConfig) -> Env:
    """Create OuterMaze environment with optional custom inner maze."""
    outer_margin = env_config.make_kwargs.get("outer_margin", 3)
    enable_render = env_config.make_kwargs.get("enable_render", True)
    inner_maze_path = env_config.make_kwargs.get("inner_maze_path", None)
    if inner_maze_path is not None:
        inner_maze = np.load(inner_maze_path)
    else:
        inner_maze = None

    env = MazeEnv(
        inner_maze=inner_maze, outer_margin=outer_margin, enable_render=enable_render
    )

    return env


# Simple test run
# if __name__ == "__main__":
#     ACTIONS = {
#         0: np.array([-1, 0]),  # N (up)
#         1: np.array([1, 0]),  # S (down)
#         2: np.array([0, 1]),  # E (right)
#         3: np.array([0, -1]),  # W (left)
#     }

#     maze = np.array(
#         [
#             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#             [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
#             [1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
#             [1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
#             [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#             [1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
#             [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#             [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
#             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#         ]
#     )

#     env = MazeEnv(inner_maze=maze, outer_margin=3, enable_render=True)
#     obs, _ = env.reset()
#     print(f"Start obs: {obs}")
#     previous_obs = obs
#     for _ in range(4000):
#         action = env.action_space.sample()
#         obs, reward, done, _, _ = env.step(action)
#         print(
#             f"Prev: {previous_obs}, Action: {ACTIONS[action]},
#               New: {obs}, Reward: {reward}"
#         )
#         previous_obs = obs
#         env.render()
#         pygame.display.set_caption("OuterMazeEnv")
#         if done:
#             print("Goal reached!")
#             break
#     env.close()
