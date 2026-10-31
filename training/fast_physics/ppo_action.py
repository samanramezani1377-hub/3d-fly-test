"""Body-wrench PPO actions, nominal allocation and staged reference guidance.

The reference controller is an oracle teacher that may inspect simulator truth.
Only its requested body wrench is blended with the PPO request; no privileged
state, randomized parameter or teacher diagnostic is appended to the observation.
The teacher is removed completely by curriculum stage five.
"""
from __future__ import annotations

from functools import lru_cache

import gymnasium as gym
import numpy as np

from .hover_reference import HoverReferenceController
from .velocity_reference import VelocityYawReferenceController
from .parameters import VehicleParameters
from .vehicle_family import profile_to_parameters
from .propeller import BladeElementPropeller


REFERENCE_WEIGHT_BY_STAGE = {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.20, 5: 0.0}


def _allocation_matrix(params: VehicleParameters) -> np.ndarray:
    positions = np.asarray(params.rotor_positions_m, dtype=np.float64)
    cg = np.asarray(params.center_of_mass_m, dtype=np.float64)
    relative = positions - cg
    matrix = np.empty((4, 4), dtype=np.float64)
    matrix[0, :] = 1.0
    matrix[1, :] = relative[:, 1]
    matrix[2, :] = -relative[:, 0]
    matrix[3, :] = np.asarray(params.spin_signs, dtype=np.float64) * params.torque_to_thrust_m
    if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < 4:
        raise ValueError("vehicle rotor geometry does not provide a full-rank 4-axis mixer")
    return matrix


# VehicleParameters may contain a dict-valued BEMT config and is therefore
# intentionally not hashable. The 4x4 inverse is cheap; do not cache by params.
def _allocation_inverse(params: VehicleParameters) -> np.ndarray:
    return np.linalg.inv(_allocation_matrix(params))


def policy_action_to_wrench(action, params: VehicleParameters) -> np.ndarray:
    """Map raw Gaussian [collective, roll, pitch, yaw] samples to a body wrench."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (4,):
        raise ValueError("policy action must contain collective, roll, pitch and yaw")
    action = np.tanh(np.nan_to_num(action, nan=0.0, posinf=20.0, neginf=-20.0))
    weight = params.mass_kg * 9.80665
    rotor_xy = np.asarray(params.rotor_positions_m, dtype=np.float64)[:, :2]
    arm = max(float(np.mean(np.linalg.norm(rotor_xy, axis=1))), 1e-3)
    return np.array([
        weight * (1.0 + action[0]),
        action[1] * 0.20 * weight * arm,
        action[2] * 0.20 * weight * arm,
        action[3] * 0.20 * weight * params.torque_to_thrust_m,
    ], dtype=np.float64)


@lru_cache(maxsize=16)
def _bemt_static_lookup(config_json: str, max_omega: float, thrust_scale: float):
    """Precompute a monotonic static BEMT thrust curve once per Smooth2 setup."""
    import json
    model = BladeElementPropeller(json.loads(config_json))
    omega_grid = np.linspace(0.0, float(max_omega), 129)
    thrust_grid = np.asarray([model.evaluate(float(w), 0.0)[0] for w in omega_grid]) * float(thrust_scale)
    if not np.all(np.isfinite(thrust_grid)) or np.any(np.diff(thrust_grid) < -1e-8):
        raise ValueError("Smooth2 BEMT static thrust lookup is not finite and monotonic")
    return omega_grid, np.maximum.accumulate(thrust_grid)


def _rotor_thrust_limits_and_inverse(params: VehicleParameters):
    effectiveness = np.asarray(params.motor_effectiveness, dtype=np.float64)
    max_omega = params.max_rotor_speed_rad_s * params.voltage_factor
    if params.propeller_bemt:
        import json
        reference_kf = float(params.propeller_bemt.get(
            "reference_thrust_coefficient_n_per_rad_s2", params.thrust_coefficient_n_per_rad_s2
        ))
        scale = params.thrust_coefficient_n_per_rad_s2 / reference_kf
        omega_grid, base_thrust = _bemt_static_lookup(
            json.dumps(params.propeller_bemt, sort_keys=True, separators=(",", ":")),
            max_omega, scale,
        )
        return base_thrust[-1] * effectiveness, omega_grid, base_thrust, effectiveness
    return params.thrust_coefficient_n_per_rad_s2 * effectiveness * max_omega ** 2, None, None, effectiveness


def wrench_to_motor_commands(wrench, params: VehicleParameters) -> np.ndarray:
    """Allocate wrench and invert the same Smooth2 BEMT static thrust curve used by the plant."""
    wrench = np.asarray(wrench, dtype=np.float64)
    if wrench.shape != (4,) or not np.all(np.isfinite(wrench)):
        raise ValueError("body wrench must be a finite four-value vector")
    max_thrust, bemt_omega, bemt_base_thrust, effectiveness = _rotor_thrust_limits_and_inverse(params)
    if np.any(max_thrust <= 0) or not np.all(np.isfinite(max_thrust)):
        raise ValueError("invalid per-rotor thrust authority")
    inverse = _allocation_inverse(params)
    collective = float(np.clip(wrench[0], 0.0, float(max_thrust.sum())))

    def allocate(scale_collective: float, scale_torque: float) -> np.ndarray:
        request = np.array([
            collective * scale_collective,
            wrench[1] * scale_torque,
            wrench[2] * scale_torque,
            wrench[3] * scale_torque,
        ], dtype=np.float64)
        return inverse @ request

    # First find a feasible collective-only operating point. This is normally
    # scale 1 for symmetric quad layouts; the fallback handles asymmetric arms.
    base = allocate(1.0, 0.0)
    if np.any(base < -1e-8) or np.any(base > max_thrust + 1e-8):
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            candidate = allocate(mid, 0.0)
            if np.all(candidate >= -1e-8) and np.all(candidate <= max_thrust + 1e-8):
                lo = mid
            else:
                hi = mid
        collective_scale = lo
    else:
        collective_scale = 1.0

    # Keep the requested collective and uniformly desaturate body torques.
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        candidate = allocate(collective_scale, mid)
        if np.all(candidate >= -1e-8) and np.all(candidate <= max_thrust + 1e-8):
            lo = mid
        else:
            hi = mid
    rotor_thrust = np.clip(allocate(collective_scale, lo), 0.0, max_thrust)
    if bemt_omega is not None:
        omega = np.interp(
            rotor_thrust / np.maximum(effectiveness, 1e-12),
            bemt_base_thrust, bemt_omega,
        )
    else:
        omega = np.sqrt(rotor_thrust / (params.thrust_coefficient_n_per_rad_s2 * effectiveness))
    commands = omega / (params.max_rotor_speed_rad_s * params.voltage_factor)
    return np.clip(commands, 0.0, 1.0).astype(np.float32)


def policy_action_to_motor_commands(action, params: VehicleParameters) -> np.ndarray:
    """Standalone PPO-to-motor mapping used by tools and non-guided inference."""
    return wrench_to_motor_commands(policy_action_to_wrench(action, params), params)


class BodyWrenchActionWrapper(gym.ActionWrapper):
    """Blend PPO body-wrench requests with a teacher that fades across stages."""
    def __init__(self, env, disable_teacher: bool = False, reference_weight_override: float | None = None):
        super().__init__(env)
        # SB3 PPO requires finite Box bounds. Keep a deliberately wide
        # symmetric envelope because the policy emits raw Gaussian samples;
        # policy_action_to_wrench() performs the smooth tanh squashing. A
        # narrow [-1, 1] Box would clip those raw samples before that mapping
        # and unintentionally reduce the available control authority.
        self.action_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(4,), dtype=np.float32
        )
        self._cached_profile_id = None
        self._cached_params = None
        self._reference = None
        self._velocity_reference = None
        stage = getattr(env, "curriculum_stage", None)
        if reference_weight_override is not None and not 0.0 <= float(reference_weight_override) <= 1.0:
            raise ValueError("reference_weight_override must be between 0.0 and 1.0")
        if disable_teacher:
            self.reference_weight = 0.0
        elif reference_weight_override is not None:
            self.reference_weight = float(reference_weight_override)
        else:
            self.reference_weight = REFERENCE_WEIGHT_BY_STAGE.get(int(stage), 0.0) if stage else 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        profile = self.env.active_profile
        if profile is None or self.env.inner is None:
            raise RuntimeError("vehicle-family environment did not initialize a profile and plant")
        profile_id = profile["vehicle_id"]
        if profile_id != self._cached_profile_id:
            self._cached_params = profile_to_parameters(profile)
            self._cached_profile_id = profile_id
        self._reference = None
        self._velocity_reference = None
        if self.reference_weight > 0.0:
            plant = self.env.inner.plant
            if bool(getattr(self.env.inner, "velocity_control", False)):
                self._velocity_reference = VelocityYawReferenceController(
                    plant.params, wind_m_s=plant.wind, dt_s=self.env.inner.dt,
                )
            else:
                self._reference = HoverReferenceController(
                    plant.params, target_position=self.env.inner.target,
                    wind_m_s=plant.wind, dt_s=self.env.inner.dt,
                )
        info = dict(info)
        info["reference_teacher_weight"] = self.reference_weight
        info["reference_teacher_uses_privileged_simulator_state"] = self.reference_weight > 0.0
        return observation, info

    def action(self, action):
        if self._cached_params is None or self.env.inner is None:
            raise RuntimeError("environment must be reset before mapping PPO actions")
        policy_wrench = policy_action_to_wrench(action, self._cached_params)
        if self._velocity_reference is not None and self.reference_weight > 0.0:
            inner = self.env.inner
            teacher_wrench = self._velocity_reference.desired_wrench(
                inner.plant.state(), target_position=inner.target,
                desired_velocity_body=inner.desired_velocity_body,
                desired_yaw_rate_rad_s=inner.desired_yaw_rate_rad_s,
            )
            wrench = (
                (1.0 - self.reference_weight) * policy_wrench
                + self.reference_weight * teacher_wrench
            )
        elif self._reference is not None and self.reference_weight > 0.0:
            plant = self.env.inner.plant
            teacher_wrench = self._reference.desired_wrench(plant.state())
            wrench = (
                (1.0 - self.reference_weight) * policy_wrench
                + self.reference_weight * teacher_wrench
            )
        else:
            wrench = policy_wrench
        return wrench_to_motor_commands(wrench, self._cached_params)


# Backwards-compatible import name for old tooling; semantics are body-wrench.
SymmetricHoverActionWrapper = BodyWrenchActionWrapper
