"""Expert policy implementations for different environments."""

from typing import Callable, Tuple

import numpy as np

from programmatic_policy_learning.envs.constant import ct, ec, rfts, stf, tpn


class ExpertPolicies:
    """Collection of expert policies for different environments."""

    @staticmethod
    def expert_nim_policy(layout: np.ndarray) -> Tuple[int, int]:
        """Expert policy for TwoPileNim environment."""
        r1 = np.max(np.argwhere(layout == tpn.EMPTY)[:, 0])
        if layout[r1, 0] == tpn.TOKEN:
            c1 = 0
        elif layout[r1, 1] == tpn.TOKEN:
            c1 = 1
        else:
            r1 += 1
            c1 = 0
        return (r1, c1)

    @staticmethod
    def expert_checkmate_tactic_policy(layout: np.ndarray) -> Tuple[int, int]:
        """Expert policy for CheckmateTactic environment."""
        if np.any(layout == ct.WHITE_QUEEN):
            return tuple(np.argwhere(layout == ct.WHITE_QUEEN)[0])

        black_king_pos = np.argwhere(layout == ct.BLACK_KING)[0]
        white_king_pos = np.argwhere(layout == ct.WHITE_KING)[0]

        return (
            (black_king_pos[0] + white_king_pos[0]) // 2,
            (black_king_pos[1] + white_king_pos[1]) // 2,
        )

    @staticmethod
    def expert_stf_policy(layout: np.ndarray) -> Tuple[int, int]:
        """Expert policy for StopTheFall environment."""
        r, c = np.argwhere(layout == stf.FALLING)[0]

        while True:
            if layout[r + 1, c] in [stf.STATIC, stf.DRAWN]:
                break
            r += 1

        if stf.RED in (layout[r, c - 1], layout[r, c + 1]):
            return (r, c)

        r, c = np.argwhere(layout == stf.ADVANCE)[0]
        return (r, c)

    @staticmethod
    def expert_ec_policy(layout: np.ndarray) -> Tuple[int, int]:
        """Expert policy for Chase environment."""
        r, c = np.argwhere(layout == ec.TARGET)[0]
        ra, ca = np.argwhere(layout == ec.AGENT)[0]

        left_arrow = tuple(np.argwhere(layout == ec.LEFT_ARROW)[0])
        right_arrow = tuple(np.argwhere(layout == ec.RIGHT_ARROW)[0])
        up_arrow = tuple(np.argwhere(layout == ec.UP_ARROW)[0])
        down_arrow = tuple(np.argwhere(layout == ec.DOWN_ARROW)[0])

        # Top left corner
        if layout[r - 1, c] == ec.WALL and layout[r, c - 1] == ec.WALL:
            # Draw on right
            if layout[r, c + 1] == ec.EMPTY:
                return (r, c + 1)
            # Move to left
            if layout[ra, ca - 1] == ec.EMPTY:
                return left_arrow
            # Move up
            return up_arrow

        # Top right corner
        if layout[r - 1, c] == ec.WALL and layout[r, c + 1] == ec.WALL:
            # Draw on left
            if layout[r, c - 1] == ec.EMPTY:
                return (r, c - 1)
            # Move to right
            if layout[ra, ca + 1] == ec.EMPTY:
                return right_arrow
            # Move up
            return up_arrow

        # Bottom left corner
        if layout[r + 1, c] == ec.WALL and layout[r, c - 1] == ec.WALL:
            # Draw on right
            if layout[r, c + 1] == ec.EMPTY:
                return (r, c + 1)
            # Move to left
            if layout[ra, ca - 1] == ec.EMPTY:
                return left_arrow
            # Move down
            return down_arrow

        # Bottom right corner
        if layout[r + 1, c] == ec.WALL and layout[r, c + 1] == ec.WALL:
            # Draw on left
            if layout[r, c - 1] == ec.EMPTY:
                return (r, c - 1)
            # Move to right
            if layout[ra, ca + 1] == ec.EMPTY:
                return right_arrow
            # Move down
            return down_arrow

        # Wait
        return (0, 0)

    @staticmethod
    def expert_rfts_policy(layout: np.ndarray) -> Tuple[int, int]:
        """Expert policy for ReachForTheStar environment."""
        agent_r, agent_c = np.argwhere(layout == rfts.AGENT)[0]
        star_r, star_c = np.argwhere(layout == rfts.STAR)[0]
        right_arrow = tuple(np.argwhere(layout == rfts.RIGHT_ARROW)[0])
        left_arrow = tuple(np.argwhere(layout == rfts.LEFT_ARROW)[0])

        height_to_star = agent_r - star_r

        # gonna climb up from the left
        if agent_c <= star_c:
            # move to the left more
            if abs(agent_c - star_c) < height_to_star:
                return left_arrow

            # stairs do not exist
            sr, sc = star_r + 1, star_c
            while sc > agent_c:
                if sr >= layout.shape[0] - 2:
                    break
                if layout[sr, sc] != rfts.DRAWN:
                    return (sr, sc)
                sr += 1
                sc -= 1

            # move to the right
            return right_arrow

        # gonna climb up from the right
        # move to the right more
        if abs(agent_c - star_c) < height_to_star:
            return right_arrow

        # stairs do not exist
        sr, sc = star_r + 1, star_c
        while sc < agent_c:
            if sr >= layout.shape[0] - 2:
                break
            if layout[sr, sc] != rfts.DRAWN:
                return (sr, sc)
            sr += 1
            sc += 1

        # move to the left
        return left_arrow

    @classmethod
    def get_policy(cls, env_name: str) -> Callable:
        """Get expert policy for environment name.

        Args:
            env_name: Name of the environment

        Returns:
            Expert policy function
        """
        policy_map = {
            "TwoPileNim": cls.expert_nim_policy,
            "CheckmateTactic": cls.expert_checkmate_tactic_policy,
            "StopTheFall": cls.expert_stf_policy,
            "Chase": cls.expert_ec_policy,
            "ReachForTheStar": cls.expert_rfts_policy,
        }

        if env_name not in policy_map:
            raise ValueError(f"Unknown environment: {env_name}")

        return policy_map[env_name]
