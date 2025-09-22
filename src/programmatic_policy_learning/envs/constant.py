"""Environment constants imported from generalization_grid_games."""

from generalization_grid_games.envs import chase as ec
from generalization_grid_games.envs import checkmate_tactic as ct
from generalization_grid_games.envs import reach_for_the_star as rfts
from generalization_grid_games.envs import stop_the_fall as stf
from generalization_grid_games.envs import two_pile_nim as tpn

# Re-export for easier access
__all__ = ["tpn", "ct", "stf", "ec", "rfts"]
