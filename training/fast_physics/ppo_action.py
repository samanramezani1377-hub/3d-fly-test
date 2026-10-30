"""PPO action normalization centered on nominal per-vehicle hover throttle.

The policy emits symmetric motor deltas in [-1, 1]. Zero means nominal hover;
the mapping uses only declared vehicle-profile parameters, never randomized
episode ground truth. The inner physics environment still receives [0, 1].
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .parameters import VehicleParameters
from .vehicle_family import profile_to_parameters


def nominal_hover_commands(params: VehicleParameters) -> np.ndarray:
    effectiveness = np.asarray(params.motor_effectiveness, dtype=np.float64)
    omega = np.sqrt(
        params.mass_kg * 9.80665
        / (4.0 * params.thrust_coefficient_n_per_rad_s2 * effectiveness)
    )
    denominator = params.max_rotor_speed_rad_s * params.voltage_factor
    if denominator <= 0 or not np.all(np.isfinite(omega)):
        raise ValueError("invalid nominal hover parameters")
    return np.clip(omega / denominator, 1e-4, 1.0 - 1e-4)


def policy_action_to_motor_commands(action, params: VehicleParameters) -> np.ndarray:
    """Map symmetric policy deltas to physical motor commands; zero is hover."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (4,):
        raise ValueError("policy action must contain four motor deltas")
    action = np.clip(np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    hover = nominal_hover_commands(params).astype(np.float32)
    return np.where(action >= 0.0, hover + action * (1.0 - hover), hover * (1.0 + action)).astype(np.float32)


class SymmetricHoverActionWrapper(gym.ActionWrapper):
    """Expose PPO-friendly symmetric actions while preserving the plant API."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def action(self, action):
        profile = self.env.active_profile
        if profile is None:
            raise RuntimeError("environment must be reset before mapping PPO actions")
        params = profile_to_parameters(profile)
        return policy_action_to_motor_commands(action, params)
