"""Body-wrench PPO actions and nominal, non-privileged motor allocation.

The policy controls collective thrust and body torques rather than four
unrelated motor commands. Allocation uses only the declared vehicle profile;
episode-randomized plant parameters remain hidden from the policy.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .parameters import VehicleParameters
from .vehicle_family import profile_to_parameters


def _allocation_matrix(params: VehicleParameters) -> np.ndarray:
    positions = np.asarray(params.rotor_positions_m, dtype=np.float64)
    cg = np.asarray(params.center_of_mass_m, dtype=np.float64)
    relative = positions - cg
    # Wrench = [collective thrust, roll torque, pitch torque, yaw torque].
    matrix = np.empty((4, 4), dtype=np.float64)
    matrix[0, :] = 1.0
    matrix[1, :] = relative[:, 1]
    matrix[2, :] = -relative[:, 0]
    matrix[3, :] = np.asarray(params.spin_signs, dtype=np.float64) * params.torque_to_thrust_m
    if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < 4:
        raise ValueError("vehicle rotor geometry does not provide a full-rank 4-axis mixer")
    return matrix


def nominal_hover_commands(params: VehicleParameters) -> np.ndarray:
    """Per-motor normalized command for nominal level hover."""
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
    """Map [collective, roll, pitch, yaw] in [-1,1] to physical motor commands."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (4,):
        raise ValueError("policy action must contain collective, roll, pitch and yaw")
    # Keep PPO's sampled Gaussian action unbounded so its log-probability
    # matches the action actually passed to the environment; tanh provides
    # smooth bounded physical controls without hard clipping at the boundary.
    action = np.tanh(np.nan_to_num(action, nan=0.0, posinf=20.0, neginf=-20.0))

    weight = params.mass_kg * 9.80665
    rotor_xy = np.asarray(params.rotor_positions_m, dtype=np.float64)[:, :2]
    arm = max(float(np.mean(np.linalg.norm(rotor_xy, axis=1))), 1e-3)
    # Zero action is nominal hover. Collective spans [0, 2W]; torque
    # authority is a conservative fraction of the nominal full-thrust envelope.
    desired_wrench = np.array([
        weight * (1.0 + action[0]),
        action[1] * 0.20 * weight * arm,
        action[2] * 0.20 * weight * arm,
        action[3] * 0.20 * weight * params.torque_to_thrust_m,
    ], dtype=np.float64)
    rotor_thrust = np.linalg.solve(_allocation_matrix(params), desired_wrench)
    max_per_rotor = (
        params.thrust_coefficient_n_per_rad_s2
        * np.asarray(params.motor_effectiveness, dtype=np.float64)
        * (params.max_rotor_speed_rad_s * params.voltage_factor) ** 2
    )
    if np.any(max_per_rotor <= 0) or not np.all(np.isfinite(max_per_rotor)):
        raise ValueError("invalid per-rotor thrust authority")
    rotor_thrust = np.clip(rotor_thrust, 0.0, max_per_rotor)
    omega = np.sqrt(rotor_thrust / (
        params.thrust_coefficient_n_per_rad_s2
        * np.asarray(params.motor_effectiveness, dtype=np.float64)
    ))
    commands = omega / (params.max_rotor_speed_rad_s * params.voltage_factor)
    return np.clip(commands, 0.0, 1.0).astype(np.float32)


class SymmetricHoverActionWrapper(gym.ActionWrapper):
    """Expose normalized body-wrench actions while preserving the plant API."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def action(self, action):
        profile = self.env.active_profile
        if profile is None:
            raise RuntimeError("environment must be reset before mapping PPO actions")
        return policy_action_to_motor_commands(action, profile_to_parameters(profile))
