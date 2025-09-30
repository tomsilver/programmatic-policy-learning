import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D, Maze
from omegaconf import DictConfig
from prpl_utils.gym_utils import GymToGymnasium
import pygame

class OuterMazeEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    ACTION = ["N", "S", "E", "W"]

    def __init__(self, env_id, enable_render=True, **inner_kwargs):
        self.env_id = env_id
        self.enable_render = enable_render

        # --- Build inner maze ---
        inner_env = gym.make(env_id, **inner_kwargs)
        inner_maze_view = inner_env.maze_view
        inner_cells = inner_maze_view.maze.maze_cells
        H, W = inner_cells.shape

        # --- Expand with void border ---
        outer_size = (H + 2, W + 2)
        outer_cells = np.zeros(outer_size, dtype=int)
        outer_cells[1:H+1, 1:W+1] = inner_cells

        # --- Define entrance (bottom middle) ---
        ex = W // 2 + 1
        ey = H + 1
        self.entrance = np.array([ex, ey])

        # --- Carve passage into maze ---
        # Break "N" wall of entrance cell
        outer_cells[ex, ey] = Maze._Maze__break_walls(outer_cells[ex, ey], "N")
        # Break "S" wall of adjacent inner cell
        outer_cells[ex, ey-1] = Maze._Maze__break_walls(outer_cells[ex, ey-1], "S")

        # --- Define goal (inner maze goal = bottom-right corner of inner grid) ---
        self.goal = np.array([W, H])

        # --- Wrap MazeView2D ---
        from gym_maze.envs.maze_view_2d import MazeView2D
        self.maze_view = MazeView2D(
            maze_name=f"OuterMaze({env_id})",
            maze_size=outer_size,
            enable_render=enable_render,
        )
        # Replace the maze with our augmented cells
        self.maze_view._MazeView2D__maze = Maze(maze_cells=outer_cells)

        # --- Spaces ---
        self.action_space = spaces.Discrete(4)
        low = np.zeros(2, dtype=int)
        high = np.array(outer_size) - np.ones(2, dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        self.seed()
        self.reset()
    
    def step(self, action):
        self.maze_view.move_robot(self.ACTION[action])
        self.state = self.maze_view.robot

        done = np.array_equal(self.state, self.goal)
        reward = 1 if done else -0.01

        # Debug print
        print(f"Step: state={self.state}, action={self.ACTION[action]}, reward={reward}, done={done}")

        return self.state, reward, done, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.maze_view.reset_robot()
        # Place at entrance instead of (0,0)
        self.state = self.entrance.copy()
        self.maze_view._MazeView2D__robot = self.state.copy()
        return self.state, {}

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()
            return
        try:
            return self.maze_view.update(mode)
        except pygame.error as e:
            print("[OuterMazeEnv] Render skipped:", e)
            return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

def create_outer_maze_env(env_config: DictConfig) -> GymToGymnasium:
    """Create Outer-Maze environment with legacy gym compatibility."""
    import gym as legacy_gym  # legacy Gym compatibility

    base_id = env_config.make_kwargs.id
    enable_render = getattr(env_config.make_kwargs, "enable_render", False)

    env = OuterMazeEnv(env_id=base_id, enable_render=enable_render)

    while hasattr(env, "env"):
        env = env.env

    return GymToGymnasium(env)