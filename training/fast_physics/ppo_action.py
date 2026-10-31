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


def _allocation_matrix(
    params: VehicleParameters,
    rotor_thrust_n: np.ndarray | None = None,
    *,
    bemt_base_thrust: np.ndarray | None = None,
    bemt_base_torque: np.ndarray | None = None,
    effectiveness: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.asarray(params.rotor_positions_m, dtype=np.float64)
    cg = np.asarray(params.center_of_mass_m, dtype=np.float64)
    relative = positions - cg
    matrix = np.empty((4, 4), dtype=np.float64)
    matrix[0, :] = 1.0
    matrix[1, :] = relative[:, 1]
    matrix[2, :] = -relative[:, 0]
    if bemt_base_thrust is not None and bemt_base_torque is not None:
        motor_scale = np.asarray(effectiveness, dtype=np.float64)
        requested = np.asarray(rotor_thrust_n, dtype=np.float64)
        base_thrust = requested / np.maximum(motor_scale, 1e-12)
        thrust_at_point = np.interp(base_thrust, bemt_base_thrust, bemt_base_thrust)
        torque_at_point = np.interp(base_thrust, bemt_base_thrust, bemt_base_torque)
        torque_ratio = torque_at_point / np.maximum(thrust_at_point, 1e-9)
    else:
        torque_ratio = np.full(4, params.torque_to_thrust_m, dtype=np.float64)
    matrix[3, :] = np.asarray(params.spin_signs, dtype=np.float64) * torque_ratio
    if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < 4:
        raise ValueError("vehicle rotor geometry does not provide a full-rank 4-axis mixer")
    return matrix


def _allocation_inverse(params: VehicleParameters, **kwargs) -> np.ndarray:
    return np.linalg.inv(_allocation_matrix(params, **kwargs))

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
def _bemt_static_lookup(config_json: str, max_omega: float, thrust_scale: float, torque_scale: float):
    """Precompute static BEMT thrust and shaft-torque curves for feed-forward allocation."""
    import json
    model = BladeElementPropeller(json.loads(config_json))
    omega_grid = np.linspace(0.0, float(max_omega), 129)
    values = np.asarray([model.evaluate(float(w), 0.0) for w in omega_grid], dtype=np.float64)
    thrust_grid = values[:, 0] * float(thrust_scale)
    torque_grid = values[:, 1] * float(torque_scale)
    if (
        not np.all(np.isfinite(values))
        or np.any(np.diff(thrust_grid) < -1e-8)
        or np.any(np.diff(torque_grid) < -1e-8)
    ):
        raise ValueError("Smooth2 BEMT static thrust/torque lookup is not finite and monotonic")
    return omega_grid, np.maximum.accumulate(thrust_grid), np.maximum.accumulate(torque_grid)


def _rotor_thrust_limits_and_inverse(params: VehicleParameters):
    effectiveness = np.asarray(params.motor_effectiveness, dtype=np.float64)
    max_omega = params.max_rotor_speed_rad_s * params.voltage_factor
    if params.propeller_bemt:
        import json
        reference_kf = float(params.propeller_bemt.get(
            "reference_thrust_coefficient_n_per_rad_s2", params.thrust_coefficient_n_per_rad_s2
        ))
        reference_kq_kf = float(params.propeller_bemt.get(
            "reference_torque_to_thrust_m", params.torque_to_thrust_m
        ))
        thrust_scale = params.thrust_coefficient_n_per_rad_s2 / reference_kf
        torque_scale = params.torque_to_thrust_m / reference_kq_kf
        omega_grid, base_thrust, base_torque = _bemt_static_lookup(
            json.dumps(params.propeller_bemt, sort_keys=True, separators=(",", ":")),
            max_omega, thrust_scale, torque_scale,
        )
        return base_thrust[-1] * effectiveness, omega_grid, base_thrust, base_torque, effectiveness
    return params.thrust_coefficient_n_per_rad_s2 * effectiveness * max_omega ** 2, None, None, None, effectiveness


def wrench_to_motor_commands(wrench, params: VehicleParameters) -> np.ndarray:
    """Allocate wrench and invert the same Smooth2 BEMT static thrust curve used by the plant."""
    wrench = np.asarray(wrench, dtype=np.float64)
    if wrench.shape != (4,) or not np.all(np.isfinite(wrench)):
        raise ValueError("body wrench must be a finite four-value vector")
    max_thrust, bemt_omega, bemt_base_thrust, bemt_base_torque, effectiveness = _rotor_thrust_limits_and_inverse(params)
    if np.any(max_thrust <= 0) or not np.all(np.isfinite(max_thrust)):
        raise ValueError("invalid per-rotor thrust authority")
    collective = float(np.clip(wrench[0], 0.0, float(max_thrust.sum())))

    def allocate(scale_collective: float, scale_torque: float) -> np.ndarray:
        request = np.array([
            collective * scale_collective,
            wrench[1] * scale_torque,
            wrench[2] * scale_torque,
            wrench[3] * scale_torque,
        ], dtype=np.float64)
        estimate = np.full(4, max(collective * scale_collective / 4.0, 1e-8), dtype=np.float64)
        candidate = estimate
        # BEMT reaction torque is nonlinear in rotor speed. Re-linearize its
        # local Q/T ratio around the current per-rotor thrust estimate.
        for _ in range(6):
            inverse = _allocation_inverse(
                params, rotor_thrust_n=estimate,
                bemt_base_thrust=bemt_base_thrust,
                bemt_base_torque=bemt_base_torque,
                effectiveness=effectiveness,
            )
            candidate = inverse @ request
            estimate = np.clip(candidate, 1e-8, max_thrust)
        return candidate

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
