"""Hardware-conditioned vehicle family environment.

Declared nominal vehicle features are visible to the policy. Episode-level
randomized mass, CG, inertia and actuator disturbances remain hidden.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import AdaptiveDroneEnv
from .parameters import VehicleParameters

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VEHICLE_FAMILY = ROOT / "vehicle_models" / "family" / "vehicle_profiles.json"
DEFAULT_RANDOMIZATION = ROOT / "vehicle_models" / "smooth2_parametric" / "domain_randomization.json"
VEHICLE_FEATURE_SIZE = 10


def profile_to_parameters(profile: dict[str, Any]) -> VehicleParameters:
    physical, propulsion = profile["physical"], profile["propulsion"]
    rotors = profile["rotors"]
    return VehicleParameters(
        mass_kg=float(physical["mass_kg"]),
        center_of_mass_m=tuple(float(v) for v in physical.get("center_of_mass_m", [0, 0, 0])),
        inertia_kg_m2=tuple(float(v) for v in np.asarray(physical["inertia_kg_m2"], dtype=np.float64).reshape(-1)),
        rotor_positions_m=tuple(tuple(float(v) for v in r["position_m"]) for r in rotors),
        spin_signs=tuple(float(r["spin_sign"]) for r in rotors),
        max_rotor_speed_rad_s=float(propulsion["max_rotor_speed_rad_s"]),
        thrust_coefficient_n_per_rad_s2=float(propulsion["thrust_coefficient_n_per_rad_s2"]),
        torque_to_thrust_m=float(propulsion["torque_to_thrust_m"]),
        motor_time_constant_s=float(propulsion["motor_time_constant_s"]),
        linear_drag_kg_s=float(physical.get("linear_drag_kg_s", 0.08)),
        quadratic_drag_kg_m=float(physical.get("quadratic_drag_kg_m", 0.015)),
        angular_damping_nms=float(physical.get("angular_damping_nms", 0.002)),
    )


def load_vehicle_profiles(path: str | Path = DEFAULT_VEHICLE_FAMILY) -> list[dict[str, Any]]:
    """Load a family library or a single filled vehicle profile and validate it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if profiles is None and isinstance(data, dict) and "vehicle_id" in data:
        profiles = [data]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profile file must contain a non-empty profiles list or one profile object")
    seen: set[str] = set()
    for p in profiles:
        if not isinstance(p, dict):
            raise ValueError("each vehicle profile must be an object")
        vid = str(p.get("vehicle_id", "")).strip()
        if not vid or vid in seen:
            raise ValueError("vehicle_id values must be non-empty and unique")
        seen.add(vid)
        g, ph, pr = p.get("geometry", {}), p.get("physical", {}), p.get("propulsion", {})
        required = {
            "mass_kg": ph.get("mass_kg"),
            "inertia_kg_m2": ph.get("inertia_kg_m2"),
            "arm_span_diagonal_m": g.get("arm_span_diagonal_m"),
            "propeller_diameter_m": g.get("propeller_diameter_m"),
            "max_rotor_speed_rad_s": pr.get("max_rotor_speed_rad_s"),
            "thrust_coefficient_n_per_rad_s2": pr.get("thrust_coefficient_n_per_rad_s2"),
            "torque_to_thrust_m": pr.get("torque_to_thrust_m"),
            "motor_time_constant_s": pr.get("motor_time_constant_s"),
            "nominal_battery_voltage_v": pr.get("nominal_battery_voltage_v"),
            "battery_capacity_ah": pr.get("battery_capacity_ah"),
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"{vid}: missing required configuration fields: {', '.join(missing)}")
        if int(g.get("rotor_count", 0)) != 4 or not isinstance(p.get("rotors"), list) or len(p["rotors"]) != 4:
            raise ValueError(f"{vid}: v1 supports exactly four rotor definitions")
        params = profile_to_parameters(p)
        params.validate()
        if min(float(g["arm_span_diagonal_m"]), float(g["propeller_diameter_m"]),
               float(pr["nominal_battery_voltage_v"]), float(pr["battery_capacity_ah"])) <= 0:
            raise ValueError(f"{vid}: geometry and battery values must be positive")
        if p.get("calibration_status") not in {"provisional_estimate", "user_supplied", "measured"}:
            raise ValueError(f"{vid}: calibration_status must be provisional_estimate, user_supplied or measured")
    return profiles


def vehicle_feature_vector(profile: dict[str, Any]) -> np.ndarray:
    """Ten bounded, dimensionless hardware features supplied to the policy."""
    ph, g, pr = profile["physical"], profile["geometry"], profile["propulsion"]
    positions = np.asarray([r["position_m"] for r in profile["rotors"]], dtype=np.float64)
    arm_radius = float(np.mean(np.linalg.norm(positions[:, :2], axis=1)))
    values = np.asarray([
        float(ph["mass_kg"]) / 2.0,
        arm_radius / 0.25,
        float(g["propeller_diameter_m"]) / 0.25,
        float(pr["max_rotor_speed_rad_s"]) / 2000.0,
        float(pr["thrust_coefficient_n_per_rad_s2"]) / 1.0e-5,
        float(pr["torque_to_thrust_m"]) / 0.05,
        float(pr["motor_time_constant_s"]) / 0.2,
        float(pr["nominal_battery_voltage_v"]) / 25.0,
        float(pr["battery_capacity_ah"]) / 10.0,
        float(pr.get("kv_rpm_per_volt") or 0.0) / 3000.0,
    ], dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{profile.get('vehicle_id')}: invalid vehicle descriptor")
    return np.clip(values, 0.0, 5.0)


class VehicleFamilyEnv(gym.Env):
    """Sample a vehicle each episode and condition one shared PPO policy on its specs."""
    metadata = {"render_modes": []}

    def __init__(
        self, *, vehicle_family_path: str | Path = DEFAULT_VEHICLE_FAMILY,
        vehicle_id: str | None = None, randomization_path: str | Path = DEFAULT_RANDOMIZATION,
        dt: float = 0.02, max_steps: int = 1500, history_length: int = 4,
        randomize: bool = True, curriculum_stage: int | None = None,
    ):
        super().__init__()
        self.profiles = load_vehicle_profiles(vehicle_family_path)
        self.profile_by_id = {p["vehicle_id"]: p for p in self.profiles}
        if vehicle_id is not None and vehicle_id not in self.profile_by_id:
            raise ValueError(f"unknown vehicle_id {vehicle_id!r}; available: {sorted(self.profile_by_id)}")
        self.pinned_vehicle_id = vehicle_id
        self.randomization_path = Path(randomization_path)
        self.dt, self.max_steps, self.history_length = float(dt), int(max_steps), int(history_length)
        self.randomize, self.curriculum_stage = bool(randomize), curriculum_stage
        self.active_profile = None
        self.active_features = np.zeros(VEHICLE_FEATURE_SIZE, dtype=np.float32)
        self.inner: AdaptiveDroneEnv | None = None
        self.action_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.history_length * 16 + 2 + VEHICLE_FEATURE_SIZE,), dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        requested = (options or {}).get("vehicle_id", self.pinned_vehicle_id)
        if requested is None:
            self.active_profile = self.profiles[int(self.np_random.integers(0, len(self.profiles)))]
        else:
            if requested not in self.profile_by_id:
                raise ValueError(f"unknown vehicle_id {requested!r}")
            self.active_profile = self.profile_by_id[requested]
        self.active_features = vehicle_feature_vector(self.active_profile)
        params = profile_to_parameters(self.active_profile)
        metadata = {
            "vehicle_id": self.active_profile["vehicle_id"],
            "calibration_status": self.active_profile["calibration_status"],
            "parameter_source": self.active_profile.get("parameter_source", "unspecified"),
            "synthetic_calibration_used": self.active_profile["calibration_status"] == "provisional_estimate",
            "profile_is_measured": self.active_profile["calibration_status"] == "measured",
            "missing_fields": [],
        }
        if self.inner is None:
            self.inner = AdaptiveDroneEnv(
                params=params, randomization_path=self.randomization_path, dt=self.dt,
                max_steps=self.max_steps, history_length=self.history_length,
                randomize=self.randomize, curriculum_stage=self.curriculum_stage,
            )
        else:
            self.inner.base_params = params
            self.inner.model_metadata = metadata
        obs, info = self.inner.reset(seed=seed, options=options)
        info = {
            **info, **metadata,
            "policy_observation_includes_vehicle_configuration": True,
            "vehicle_configuration_features": self.active_features.tolist(),
            "vehicle_profile": self.active_profile["display_name"],
            "policy_observation_excludes_episode_randomization_ground_truth": True,
        }
        return np.concatenate((obs, self.active_features)).astype(np.float32), info

    def step(self, action):
        if self.inner is None:
            raise RuntimeError("reset must be called before step")
        obs, reward, terminated, truncated, info = self.inner.step(action)
        info = {**info, "vehicle_id": self.active_profile["vehicle_id"],
                "calibration_status": self.active_profile["calibration_status"]}
        return np.concatenate((obs, self.active_features)).astype(np.float32), reward, terminated, truncated, info

    def close(self):
        if self.inner is not None:
            self.inner.close()
