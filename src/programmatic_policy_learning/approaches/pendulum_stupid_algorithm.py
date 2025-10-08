"""A simple hardcoded approach for pendulum balancing."""
from typing import TypeVar
import numpy as np
from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")  # Keeping your original naming
_ActType = TypeVar("_ActType")

class PendulumStupidAlgorithm(BaseApproach[_ObsType, _ActType]):
    """A hardcoded approach that tries to balance the pendulum at the top."""
    
    def _get_action(self) -> _ActType:  # Keeping your original method signature
        currobs = self._last_observation
        
        # Safety check
        if currobs is None:
            return np.array([0.0], dtype=self._action_space.dtype)
        
        # Parse observation: [cos(θ), sin(θ), angular_velocity]
        x = currobs[0]  # cos(theta)
        y = currobs[1]  # sin(theta)  
        angvel = currobs[2]  # angular velocity
        
        # Calculate approximate angle (-π to π)
        theta = np.arctan2(y, x)
        
        print(f"angle={theta:.3f}, angvel={angvel:.3f}")
        
        # Two-phase controller: Swing-up + Balance
        # Target is theta = 0 (pendulum pointing up)
        
        # Check if pendulum is hanging down (theta near ±π)
        is_hanging_down = abs(abs(theta) - np.pi) < 1.0  # Near bottom
        is_near_top = abs(theta) < 0.5  # Near top
        
        if is_hanging_down:
            # SWING-UP PHASE: Add energy to the system
            # Pump energy by applying torque in direction of velocity
            if abs(angvel) > 0.1:
                torque = 2.0 * np.sign(angvel)  # Push in direction of motion
            else:
                torque = 2.0 * np.sign(theta)   # Small kick to start motion
                
        elif is_near_top:
            # BALANCE PHASE: Gentle PD control when near upright
            kp = 12.0
            kd = 3.0
            torque = -kp * theta - kd * angvel
            
        else:
            # TRANSITION PHASE: Medium energy swing-up
            kp = 5.0
            kd = 1.0
            torque = -kp * theta - kd * angvel
            # Add some energy pumping
            if abs(angvel) > 1.0:
                torque += 1.0 * np.sign(angvel)
        
        # Clip to action space bounds  
        low, high = float(self._action_space.low[0]), float(self._action_space.high[0])
        torque = float(np.clip(torque, low, high))
        
        print(f"torque={torque:.3f}")
        return np.array([torque], dtype=self._action_space.dtype)