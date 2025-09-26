"""Continuous space primitives for PRBench environments."""

from typing import Any

import numpy as np


def state_dimension(obs: np.ndarray, dim: int) -> float:
    """Get a specific dimension from the observation vector."""
    if dim < 0 or dim >= len(obs):
        return 0.0
    return float(obs[dim])


def state_magnitude(obs: np.ndarray) -> float:
    """Get the magnitude (L2 norm) of the observation vector."""
    return float(np.linalg.norm(obs))


def state_sum(obs: np.ndarray) -> float:
    """Get the sum of all state dimensions."""
    return float(np.sum(obs))


def state_mean(obs: np.ndarray) -> float:
    """Get the mean of all state dimensions."""
    return float(np.mean(obs))


def state_max(obs: np.ndarray) -> float:
    """Get the maximum value in the state vector."""
    return float(np.max(obs))


def state_min(obs: np.ndarray) -> float:
    """Get the minimum value in the state vector."""
    return float(np.min(obs))


def dimension_greater_than(obs: np.ndarray, dim: int, threshold: float) -> bool:
    """Check if a dimension is greater than a threshold."""
    if dim < 0 or dim >= len(obs):
        return False
    return float(obs[dim]) > threshold


def dimension_less_than(obs: np.ndarray, dim: int, threshold: float) -> bool:
    """Check if a dimension is less than a threshold."""
    if dim < 0 or dim >= len(obs):
        return False
    return float(obs[dim]) < threshold


def dimension_in_range(obs: np.ndarray, dim: int, low: float, high: float) -> bool:
    """Check if a dimension is within a range."""
    if dim < 0 or dim >= len(obs):
        return False
    val = float(obs[dim])
    return low <= val <= high


def euclidean_distance_to_point(obs: np.ndarray, target: np.ndarray) -> float:
    """Calculate Euclidean distance from current state to target point."""
    return float(np.linalg.norm(obs - target))


def manhattan_distance_to_point(obs: np.ndarray, target: np.ndarray) -> float:
    """Calculate Manhattan distance from current state to target point."""
    return float(np.sum(np.abs(obs - target)))


def angle_to_point(obs: np.ndarray, target: np.ndarray) -> float:
    """Calculate angle from current position to target point (2D only)."""
    if len(obs) < 2 or len(target) < 2:
        return 0.0
    
    diff = target[:2] - obs[:2]
    return float(np.arctan2(diff[1], diff[0]))


def is_near_point(obs: np.ndarray, target: np.ndarray, tolerance: float = 0.1) -> bool:
    """Check if current state is near a target point."""
    return euclidean_distance_to_point(obs, target) <= tolerance


def velocity_magnitude(obs: np.ndarray, vel_start_idx: int = 2) -> float:
    """Get velocity magnitude assuming velocity starts at a specific index."""
    if vel_start_idx >= len(obs) or vel_start_idx + 1 >= len(obs):
        return 0.0
    
    vel = obs[vel_start_idx:vel_start_idx+2]
    return float(np.linalg.norm(vel))


def is_moving_towards(obs: np.ndarray, target: np.ndarray, vel_start_idx: int = 2) -> bool:
    """Check if velocity is pointing towards target (2D position + velocity)."""
    if len(obs) < 4 or len(target) < 2:
        return False
    
    pos = obs[:2]
    vel = obs[vel_start_idx:vel_start_idx+2]
    
    # Vector from position to target
    to_target = target[:2] - pos
    
    # Check if velocity and direction to target have positive dot product
    return float(np.dot(vel, to_target)) > 0