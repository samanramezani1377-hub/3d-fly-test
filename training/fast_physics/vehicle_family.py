"""Hardware-conditioned vehicle family environment.

Known nominal configuration is exposed to the policy. Unknown but necessary
plant values may be initialized with explicit engineering estimates; those
estimates are recorded in metadata and must be replaced by measurements.
Episode-level randomized physical parameters remain hidden from the policy.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import AdaptiveDroneEnv
from .parameters import (
    VehicleParameters,
    cad_export_inertia_to_body_flu,
    cad_export_vector_to_body_flu,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VEHICLE_FAMILY = ROOT / "vehicle_models" / "family" / "vehicle_profiles.json"
DEFAULT_RANDOMIZATION = ROOT / "vehicle_models" / "family" / "domain_randomization.json"
VEHICLE_FEATURE_SIZE = 12
DERIVED_DEFAULTS = {
    "torque_to_thrust_m": 0.015,
    "motor_time_constant_s": 0.05,
    "assumed_max_thrust_to_weight": 2.0,
}


def resolve_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Fill missing secondary physics with documented estimates, never silently."""
    p = copy.deepcopy(profile)
    physical, geometry, propulsion = p.setdefault("physical", {}), p.setdefault("geometry", {}), p.setdefault("propulsion", {})
    derived = list(p.get("derived_estimates", []))
    mass = physical.get("mass_kg")
    if mass is None or not np.isfinite(float(mass)) or float(mass) <= 0:
        raise ValueError(f"{p.get('vehicle_id')}: a measured or explicitly chosen positive takeoff mass is required")

    cg = physical.get("center_of_mass_m")
    if cg is None:
        physical["center_of_mass_m"] = [0.0, 0.0, 0.0]
        derived.append("center_of_mass_m:geometric-center-assumption")
    else:
        cg_array = np.asarray(cg, dtype=np.float64)
        if cg_array.shape != (3,) or not np.all(np.isfinite(cg_array)):
            raise ValueError(f"{p.get('vehicle_id')}: center_of_mass_m must contain three finite values")
        physical["center_of_mass_m"] = cg_array.tolist()

    positions = p.get("rotors")
    if not isinstance(positions, list) or len(positions) != 4:
        raise ValueError(f"{p.get('vehicle_id')}: exactly four rotor positions are required")
    rotor_xyz = []
    for i, rotor in enumerate(positions):
        point = np.asarray(rotor.get("position_m"), dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{p.get('vehicle_id')}: rotor {i} needs three finite position coordinates in metres")
        if rotor.get("spin_sign") not in (-1, 1, -1.0, 1.0):
            raise ValueError(f"{p.get('vehicle_id')}: rotor {i} spin_sign must be +1 or -1")
        rotor_xyz.append(point)
    rotor_xyz = np.asarray(rotor_xyz)
    source_frame = p.get("coordinate_frame", "BODY_FORWARD_LEFT_UP")
    if source_frame == "CAD_EXPORT_Z_UP":
        rotor_xyz = np.asarray([cad_export_vector_to_body_flu(point) for point in rotor_xyz])
        p["rotors"] = [
            {**rotor, "position_m": rotor_xyz[index].tolist()}
            for index, rotor in enumerate(positions)
        ]
        physical["center_of_mass_m"] = cad_export_vector_to_body_flu(
            physical["center_of_mass_m"]
        ).tolist()
        p["source_coordinate_frame"] = source_frame
        p["coordinate_frame"] = "BODY_FORWARD_LEFT_UP"
    elif source_frame != "BODY_FORWARD_LEFT_UP":
        raise ValueError(f"{p.get('vehicle_id')}: unsupported coordinate frame {source_frame!r}")
    arm_radius_sq = float(np.mean(np.sum(rotor_xyz[:, :2] ** 2, axis=1)))

    inertia = physical.get("inertia_kg_m2")
    if inertia is None:
        # Four point masses at the rotor radius: a rough rigid-body initializer.
        # This is not a measured inertia tensor and ignores body mass distribution.
        ixy = max(float(mass) * arm_radius_sq * 0.5, 1e-6)
        izz = max(float(mass) * arm_radius_sq, 1e-6)
        physical["inertia_kg_m2"] = [ixy, ixy, izz]
        derived.append("inertia_kg_m2:point-mass-arm-estimate")
    else:
        arr = np.asarray(inertia, dtype=np.float64)
        if arr.shape not in ((3,), (3, 3), (9,)) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{p.get('vehicle_id')}: inertia must be a diagonal 3-vector or finite 3x3 matrix")
        physical["inertia_kg_m2"] = arr.reshape(-1).tolist()

    if source_frame == "CAD_EXPORT_Z_UP":
        physical["inertia_kg_m2"] = cad_export_inertia_to_body_flu(
            physical["inertia_kg_m2"]
        ).reshape(-1).tolist()

    voltage = propulsion.get("nominal_battery_voltage_v")
    kv = propulsion.get("kv_rpm_per_volt")
    if voltage is None or kv is None or float(voltage) <= 0 or float(kv) <= 0:
        raise ValueError(f"{p.get('vehicle_id')}: nominal battery voltage and motor KV are required to initialize propulsion")

    omega = propulsion.get("max_rotor_speed_rad_s")
    if omega is None:
        # Approximate loaded speed as 80% of KV no-load speed at nominal voltage.
        omega = float(kv) * float(voltage) * (2.0 * np.pi / 60.0) * 0.80
        propulsion["max_rotor_speed_rad_s"] = omega
        derived.append("max_rotor_speed_rad_s:0.80x-kv-no-load-estimate")

    kf = propulsion.get("thrust_coefficient_n_per_rad_s2")
    if kf is None:
        twr = float(propulsion.get("assumed_max_thrust_to_weight", DERIVED_DEFAULTS["assumed_max_thrust_to_weight"]))
        if twr <= 1.0:
            raise ValueError(f"{p.get('vehicle_id')}: assumed max thrust-to-weight must exceed 1")
        kf = float(mass) * 9.80665 * twr / (4.0 * float(omega) ** 2)
        propulsion["thrust_coefficient_n_per_rad_s2"] = kf
        derived.append(f"thrust_coefficient_n_per_rad_s2:derived-from-{twr:g}-max-TWR")

    if propulsion.get("torque_to_thrust_m") is None:
        propulsion["torque_to_thrust_m"] = DERIVED_DEFAULTS["torque_to_thrust_m"]
        derived.append("torque_to_thrust_m:generic-estimate")
    if propulsion.get("motor_time_constant_s") is None:
        propulsion["motor_time_constant_s"] = DERIVED_DEFAULTS["motor_time_constant_s"]
        derived.append("motor_time_constant_s:generic-estimate")
    geometry.setdefault("rotor_count", 4)
    p["derived_estimates"] = sorted(set(derived))
    return p


def profile_to_parameters(profile: dict[str, Any]) -> VehicleParameters:
    p = resolve_profile(profile)
    physical, propulsion = p["physical"], p["propulsion"]
    return VehicleParameters(
        mass_kg=float(physical["mass_kg"]),
        center_of_mass_m=tuple(float(v) for v in physical.get("center_of_mass_m", [0, 0, 0])),
        inertia_kg_m2=tuple(float(v) for v in np.asarray(physical["inertia_kg_m2"], dtype=np.float64).reshape(-1)),
        rotor_positions_m=tuple(tuple(float(v) for v in r["position_m"]) for r in p["rotors"]),
        spin_signs=tuple(float(r["spin_sign"]) for r in p["rotors"]),
        max_rotor_speed_rad_s=float(propulsion["max_rotor_speed_rad_s"]),
        thrust_coefficient_n_per_rad_s2=float(propulsion["thrust_coefficient_n_per_rad_s2"]),
        torque_to_thrust_m=float(propulsion["torque_to_thrust_m"]),
        motor_time_constant_s=float(propulsion["motor_time_constant_s"]),
        linear_drag_kg_s=float(physical.get("linear_drag_kg_s", 0.08)),
        quadratic_drag_kg_m=float(physical.get("quadratic_drag_kg_m", 0.015)),
        angular_damping_nms=float(physical.get("angular_damping_nms", 0.002)),
        linear_drag_body_kg_s=tuple(physical["linear_drag_body_kg_s"]) if physical.get("linear_drag_body_kg_s") is not None else None,
        quadratic_drag_body_kg_m=tuple(physical["quadratic_drag_body_kg_m"]) if physical.get("quadratic_drag_body_kg_m") is not None else None,
        rotor_inertia_kg_m2=float(propulsion.get("rotor_inertia_kg_m2") or 0.0),
        battery_nominal_voltage_v=float(propulsion["nominal_battery_voltage_v"]) if propulsion.get("nominal_battery_voltage_v") is not None else None,
        battery_capacity_ah=float(propulsion["battery_capacity_ah"]) if propulsion.get("battery_capacity_ah") is not None else None,
        motor_efficiency_upper_bound=float(propulsion.get("motor_efficiency_upper_bound", 0.8)),
    )


def load_vehicle_profiles(path: str | Path = DEFAULT_VEHICLE_FAMILY) -> list[dict[str, Any]]:
    """Load a family library or a single partially specified vehicle profile."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if profiles is None and isinstance(data, dict) and "vehicle_id" in data:
        profiles = [data]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profile file must contain a non-empty profiles list or one profile object")
    seen: set[str] = set()
    resolved = []
    for raw in profiles:
        if not isinstance(raw, dict):
            raise ValueError("each vehicle profile must be an object")
        vid = str(raw.get("vehicle_id", "")).strip()
        if not vid or vid in seen:
            raise ValueError("vehicle_id values must be non-empty and unique")
        seen.add(vid)
        p = resolve_profile(raw)
        g, pr = p["geometry"], p["propulsion"]
        if int(g.get("rotor_count", 0)) != 4:
            raise ValueError(f"{vid}: v1 supports exactly four rotors")
        if g.get("arm_span_diagonal_m") is None or g.get("propeller_diameter_m") is None:
            raise ValueError(f"{vid}: arm span and propeller diameter are required in metres")
        if min(float(g["arm_span_diagonal_m"]), float(g["propeller_diameter_m"]),
               float(pr["nominal_battery_voltage_v"]), float(pr.get("battery_capacity_ah") or 1.0)) <= 0:
            raise ValueError(f"{vid}: geometry and battery values must be positive")
        if p.get("calibration_status") not in {"provisional_estimate", "user_supplied", "measured"}:
            raise ValueError(f"{vid}: calibration_status must be provisional_estimate, user_supplied or measured")
        profile_to_parameters(p).validate()
        resolved.append(p)
    return resolved


def vehicle_feature_vector(profile: dict[str, Any]) -> np.ndarray:
    """Twelve bounded dimensionless hardware features, including provenance confidence."""
    p = resolve_profile(profile)
    ph, g, pr = p["physical"], p["geometry"], p["propulsion"]
    positions = np.asarray([r["position_m"] for r in p["rotors"]], dtype=np.float64)
    arm_radius = float(np.mean(np.linalg.norm(positions[:, :2], axis=1)))
    derived_fraction = len(p.get("derived_estimates", [])) / 6.0
    status_confidence = {"provisional_estimate": 0.25, "user_supplied": 0.60, "measured": 1.0}[
        p["calibration_status"]
    ]
    values = np.asarray([
        float(ph["mass_kg"]) / 2.0,
        arm_radius / 0.25,
        float(g["propeller_diameter_m"]) / 0.25,
        float(pr["max_rotor_speed_rad_s"]) / 2000.0,
        float(pr["thrust_coefficient_n_per_rad_s2"]) / 1.0e-5,
        float(pr["torque_to_thrust_m"]) / 0.05,
        float(pr["motor_time_constant_s"]) / 0.2,
        float(pr["nominal_battery_voltage_v"]) / 25.0,
        float(pr.get("battery_capacity_ah") or 0.0) / 10.0,
        float(pr.get("kv_rpm_per_volt") or 0.0) / 3000.0,
        min(derived_fraction, 1.0),
        status_confidence,
    ], dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{p.get('vehicle_id')}: invalid vehicle descriptor")
    return np.clip(values, 0.0, 5.0)


class VehicleFamilyEnv(gym.Env):
    """Sample a vehicle per episode and condition one shared PPO policy on its specs."""
    metadata = {"render_modes": []}

    def __init__(
        self, *, vehicle_family_path: str | Path = DEFAULT_VEHICLE_FAMILY,
        vehicle_id: str | None = None, randomization_path: str | Path = DEFAULT_RANDOMIZATION,
        dt: float = 0.02, max_steps: int = 1500, history_length: int = 4,
        randomize: bool = True, curriculum_stage: int | None = None,
        sampling_weights_path: str | Path | None = None,
        velocity_control: bool = False,
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
        self.velocity_control = bool(velocity_control)
        self.sampling_weights_path = Path(sampling_weights_path) if sampling_weights_path else None
        self.active_profile = None
        self.active_features = np.zeros(VEHICLE_FEATURE_SIZE, dtype=np.float32)
        self.inner: AdaptiveDroneEnv | None = None
        self.action_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.history_length * 16 + 2 + (4 if self.velocity_control else 0) + VEHICLE_FEATURE_SIZE,), dtype=np.float32,
        )

    def _combine_observation(self, observation: np.ndarray) -> np.ndarray:
        # Keep the complete legacy observation byte-for-byte at the front.
        # This lets checkpoint migration append four zero-initialized command
        # inputs without shifting the twelve hardware-conditioning features.
        if self.velocity_control:
            legacy_state = observation[:-4]
            command = observation[-4:]
            return np.concatenate((legacy_state, self.active_features, command)).astype(np.float32)
        return np.concatenate((observation, self.active_features)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        requested = (options or {}).get("vehicle_id", self.pinned_vehicle_id)
        if requested is None:
            weights = np.ones(len(self.profiles), dtype=np.float64)
            if self.sampling_weights_path is not None and self.sampling_weights_path.is_file():
                raw_weights = json.loads(self.sampling_weights_path.read_text(encoding="utf-8"))
                weights = np.asarray(
                    [float(raw_weights.get(p["vehicle_id"], 1.0)) for p in self.profiles],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                    raise ValueError("profile sampling weights must be finite and positive")
            probabilities = weights / weights.sum()
            self.active_profile = self.profiles[int(self.np_random.choice(len(self.profiles), p=probabilities))]
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
            "derived_estimates": list(self.active_profile.get("derived_estimates", [])),
            "synthetic_calibration_used": self.active_profile["calibration_status"] == "provisional_estimate",
            "profile_is_measured": self.active_profile["calibration_status"] == "measured",
            "missing_fields": [],
        }
        if self.inner is None:
            self.inner = AdaptiveDroneEnv(
                params=params, randomization_path=self.randomization_path, dt=self.dt,
                max_steps=self.max_steps, history_length=self.history_length,
                randomize=self.randomize, curriculum_stage=self.curriculum_stage,
                velocity_control=self.velocity_control,
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
        return self._combine_observation(obs), info

    def step(self, action):
        if self.inner is None:
            raise RuntimeError("reset must be called before step")
        obs, reward, terminated, truncated, info = self.inner.step(action)
        info = {**info, "vehicle_id": self.active_profile["vehicle_id"],
                "calibration_status": self.active_profile["calibration_status"]}
        return self._combine_observation(obs), reward, terminated, truncated, info

    def close(self):
        if self.inner is not None:
            self.inner.close()
