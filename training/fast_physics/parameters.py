"""Validated vehicle parameters and reproducible domain randomization.

Unknown physical values stay unknown in the airframe JSON. An explicitly
opted-in synthetic calibration exists only for software smoke tests and
algorithm development; it is never represented as measured aircraft data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# CAD export axes are X=lateral, Y=longitudinal (nose is -Y), Z=up.
# The rigid-body equations use the conventional body frame X=forward,
# Y=left, Z=up. This proper +90 degree rotation preserves handedness.
CAD_EXPORT_TO_BODY_FLU = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def cad_export_vector_to_body_flu(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("coordinate vector must contain three values")
    return CAD_EXPORT_TO_BODY_FLU @ value


def cad_export_inertia_to_body_flu(inertia) -> np.ndarray:
    value = np.asarray(inertia, dtype=np.float64)
    matrix = np.diag(value) if value.shape == (3,) else value.reshape(3, 3)
    return CAD_EXPORT_TO_BODY_FLU @ matrix @ CAD_EXPORT_TO_BODY_FLU.T

DEFAULT_AIRFRAME = ROOT / "vehicle_models" / "smooth2_parametric" / "base_airframe.json"
DEFAULT_CALIBRATION = ROOT / "vehicle_models" / "smooth2_parametric" / "physics_calibration.json"


@dataclass(frozen=True)
class VehicleParameters:
    mass_kg: float
    center_of_mass_m: tuple[float, float, float]
    inertia_kg_m2: tuple[float, ...]
    rotor_positions_m: tuple[tuple[float, float, float], ...]
    spin_signs: tuple[float, ...]
    max_rotor_speed_rad_s: float
    thrust_coefficient_n_per_rad_s2: float
    torque_to_thrust_m: float
    motor_time_constant_s: float = 0.045
    linear_drag_kg_s: float = 0.08
    quadratic_drag_kg_m: float = 0.015
    angular_damping_nms: float = 0.002
    voltage_factor: float = 1.0
    motor_effectiveness: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    linear_drag_body_kg_s: tuple[float, float, float] | None = None
    quadratic_drag_body_kg_m: tuple[float, float, float] | None = None
    rotor_inertia_kg_m2: float = 0.0

    def validate(self) -> None:
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass_kg must be finite and positive")
        if len(self.center_of_mass_m) != 3 or not np.all(np.isfinite(self.center_of_mass_m)):
            raise ValueError("center_of_mass_m must contain three finite values")
        rotor_positions = np.asarray(self.rotor_positions_m, dtype=np.float64)
        if rotor_positions.shape != (4, 3) or not np.all(np.isfinite(rotor_positions)):
            raise ValueError("rotor_positions_m must be a finite 4x3 array")
        if len(self.inertia_kg_m2) not in (3, 9) or not np.all(np.isfinite(self.inertia_kg_m2)):
            raise ValueError("inertia must be a finite diagonal 3-vector or full 3x3 tensor")
        matrix = self.inertia_matrix
        if not np.allclose(matrix, matrix.T, atol=1e-9):
            raise ValueError("inertia tensor must be symmetric")
        if np.min(np.linalg.eigvalsh(matrix)) <= 0:
            raise ValueError("inertia tensor must be positive definite")
        if len(self.rotor_positions_m) != 4 or len(self.spin_signs) != 4:
            raise ValueError("the v1 smooth2 airframe requires exactly four rotor definitions")
        if not np.all(np.isfinite([
            self.max_rotor_speed_rad_s, self.thrust_coefficient_n_per_rad_s2,
            self.torque_to_thrust_m, self.motor_time_constant_s,
            self.linear_drag_kg_s, self.quadratic_drag_kg_m,
            self.angular_damping_nms, self.voltage_factor,
        ])):
            raise ValueError("motor, drag, damping and voltage parameters must be finite")
        if self.max_rotor_speed_rad_s <= 0 or self.thrust_coefficient_n_per_rad_s2 <= 0:
            raise ValueError("motor calibration must be positive")
        if self.torque_to_thrust_m <= 0 or self.motor_time_constant_s <= 0:
            raise ValueError("torque ratio and motor time constant must be positive")
        if min(self.linear_drag_kg_s, self.quadratic_drag_kg_m, self.angular_damping_nms) < 0:
            raise ValueError("drag and angular damping coefficients must be non-negative")
        if len(self.motor_effectiveness) != 4 or not np.all(np.isfinite(self.motor_effectiveness)) or any(v <= 0 for v in self.motor_effectiveness):
            raise ValueError("motor_effectiveness must contain four finite positive values")
        if not 0 < self.voltage_factor <= 1.2:
            raise ValueError("voltage_factor must be in (0, 1.2]")
        if any(sign not in (-1, 1, -1.0, 1.0) for sign in self.spin_signs):
            raise ValueError("spin_signs must contain exactly four +1/-1 directions")
        for name in ("linear_drag_body_kg_s", "quadratic_drag_body_kg_m"):
            values = getattr(self, name)
            if values is not None and (len(values) != 3 or not np.all(np.isfinite(values)) or np.any(np.asarray(values) < 0)):
                raise ValueError(f"{name} must be three finite non-negative body-axis coefficients")
        if not np.isfinite(self.rotor_inertia_kg_m2) or self.rotor_inertia_kg_m2 < 0:
            raise ValueError("rotor_inertia_kg_m2 must be finite and non-negative")

    @property
    def inertia_matrix(self) -> np.ndarray:
        values = np.asarray(self.inertia_kg_m2, dtype=np.float64)
        return np.diag(values) if values.size == 3 else values.reshape(3, 3)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vehicle_parameters(
    airframe_path: str | Path = DEFAULT_AIRFRAME,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    *,
    allow_synthetic_calibration: bool = False,
) -> tuple[VehicleParameters, dict[str, Any]]:
    """Load the fixed CAD airframe and separate physical calibration.

    The mass/inertia baseline in physics_calibration.json is marked unknown
    until measured. Synthetic motor values are only available by explicit opt-in.
    """
    airframe = _load_json(Path(airframe_path))
    calibration = _load_json(Path(calibration_path))
    physical = calibration.get("physical_properties", {})
    motor = calibration.get("motor_propeller", {})
    missing = [
        key for key in ("takeoff_mass_kg", "center_of_mass_m", "inertia_tensor_kg_m2")
        if physical.get(key) is None
    ]
    missing += [
        key for key in ("max_rotor_speed_rad_s", "thrust_coefficient_n_per_rad_s2", "torque_to_thrust_m")
        if motor.get(key) is None
    ]
    if missing and not allow_synthetic_calibration:
        raise ValueError(
            "Physical calibration is incomplete: " + ", ".join(missing)
            + ". Add traceable measurements/manufacturer data, or pass "
              "allow_synthetic_calibration=True for software-only experiments."
        )

    synthetic = {
        "mass_kg": 0.6,
        "center_of_mass_m": [0.0, 0.0, 0.0],
        "inertia_kg_m2": [0.004, 0.006, 0.004],
        "max_rotor_speed_rad_s": 1100.0,
        "thrust_coefficient_n_per_rad_s2": 2.0e-6,
        "torque_to_thrust_m": 0.012,
    }
    mass = physical.get("takeoff_mass_kg")
    cg = physical.get("center_of_mass_m")
    inertia = physical.get("inertia_tensor_kg_m2")
    if isinstance(inertia, dict):
        if "matrix" in inertia:
            inertia = np.asarray(inertia["matrix"], dtype=np.float64).reshape(-1).tolist()
        else:
            inertia = [inertia["xx"], inertia["yy"], inertia["zz"]]
    if inertia is not None and np.asarray(inertia).shape == (3, 3):
        inertia = np.asarray(inertia, dtype=np.float64).reshape(-1).tolist()
    rotor_data = airframe["rotors"]
    scale = float(airframe["geometry"]["model_scale"])
    source_frame = airframe.get("coordinate_frame", {}).get("frame", "BODY_FORWARD_LEFT_UP")
    positions_array = np.asarray([r["position_m"] for r in rotor_data], dtype=np.float64)
    cg_array = np.asarray(cg if cg is not None else synthetic["center_of_mass_m"], dtype=np.float64)
    inertia_array = np.asarray(inertia if inertia is not None else synthetic["inertia_kg_m2"], dtype=np.float64)
    if source_frame == "CAD_EXPORT_Z_UP":
        positions_array = (CAD_EXPORT_TO_BODY_FLU @ positions_array.T).T
        cg_array = cad_export_vector_to_body_flu(cg_array)
        inertia_array = cad_export_inertia_to_body_flu(inertia_array)
    elif source_frame != "BODY_FORWARD_LEFT_UP":
        raise ValueError(f"unsupported airframe coordinate frame: {source_frame}")
    positions = tuple(tuple(float(x) for x in row) for row in positions_array)
    params = VehicleParameters(
        mass_kg=float(mass if mass is not None else synthetic["mass_kg"]),
        center_of_mass_m=tuple(float(x) for x in cg_array),
        inertia_kg_m2=tuple(float(x) for x in inertia_array.reshape(-1)),
        rotor_positions_m=positions,
        spin_signs=tuple(float(r["spin_sign"]) for r in rotor_data),
        max_rotor_speed_rad_s=float(motor.get("max_rotor_speed_rad_s") or synthetic["max_rotor_speed_rad_s"]),
        thrust_coefficient_n_per_rad_s2=float(motor.get("thrust_coefficient_n_per_rad_s2") or synthetic["thrust_coefficient_n_per_rad_s2"]),
        torque_to_thrust_m=float(motor.get("torque_to_thrust_m") or synthetic["torque_to_thrust_m"]),
        motor_time_constant_s=float(motor.get("motor_time_constant_s") or 0.045),
        linear_drag_kg_s=float(physical.get("linear_drag_kg_s", 0.08)),
        quadratic_drag_kg_m=float(physical.get("quadratic_drag_kg_m", 0.015)),
        angular_damping_nms=float(physical.get("angular_damping_nms", 0.002)),
        linear_drag_body_kg_s=tuple(physical["linear_drag_body_kg_s"]) if physical.get("linear_drag_body_kg_s") is not None else None,
        quadratic_drag_body_kg_m=tuple(physical["quadratic_drag_body_kg_m"]) if physical.get("quadratic_drag_body_kg_m") is not None else None,
        rotor_inertia_kg_m2=float(motor.get("rotor_inertia_kg_m2") or 0.0),
    )
    params.validate()
    metadata = {
        "vehicle_id": airframe["vehicle_id"],
        "airframe_source": airframe["source"],
        "geometry_scale": scale,
        "calibration_status": calibration.get("status", "unknown"),
        "synthetic_calibration_used": bool(missing),
        "missing_fields": missing,
        "units": "SI",
        "source_coordinate_frame": source_frame,
        "body_coordinate_frame": "BODY_FORWARD_LEFT_UP",
    }
    return params, metadata


def sample_scenario(
    base: VehicleParameters,
    rng: np.random.Generator,
    randomization: dict[str, Any],
) -> tuple[VehicleParameters, dict[str, Any]]:
    """Sample a physically bounded episode variant around the fixed airframe."""
    def uniform(section: str, key: str, default: tuple[float, float]) -> float:
        bounds = randomization.get(section, {}).get(key, default)
        return float(rng.uniform(float(bounds[0]), float(bounds[1])))

    mass_scale = uniform("mass", "scale", (0.80, 1.25))
    cg_cfg = randomization.get("center_of_mass", {})
    if "offset_arm_fraction" in cg_cfg:
        arm_radius = float(np.mean(np.linalg.norm(np.asarray(base.rotor_positions_m)[:, :2], axis=1)))
        cg_bound = float(cg_cfg["offset_arm_fraction"])
        cg_offset = rng.uniform(-cg_bound, cg_bound, size=3) * arm_radius
    else:
        cg_bounds = cg_cfg.get("offset_m", [-0.015, 0.015])
        cg_offset = rng.uniform(float(cg_bounds[0]), float(cg_bounds[1]), size=3)
    inertia_bounds = randomization.get("inertia", {}).get("scale", [0.70, 1.40])
    inertia_scale = rng.uniform(float(inertia_bounds[0]), float(inertia_bounds[1]), size=3)
    motor_cfg = randomization.get("motor", {})
    motor_bounds = motor_cfg.get("effectiveness", [0.90, 1.10])
    motor_effectiveness = rng.uniform(float(motor_bounds[0]), float(motor_bounds[1]), size=4)
    motor_response_scale = uniform("motor", "time_constant_scale", (1.0, 1.0))
    max_rotor_speed_scale = uniform("motor", "max_rotor_speed_scale", (1.0, 1.0))
    thrust_coefficient_scale = uniform("thrust", "coefficient_scale", (1.0, 1.0))
    torque_to_thrust_scale = uniform("motor", "torque_to_thrust_scale", (1.0, 1.0))
    voltage_factor = uniform("battery", "voltage_factor", (0.82, 1.0))
    drag_scale = uniform("aerodynamics", "drag_scale", (0.70, 1.40))

    sampled = replace(
        base,
        mass_kg=base.mass_kg * mass_scale,
        center_of_mass_m=tuple(np.asarray(base.center_of_mass_m) + cg_offset),
        inertia_kg_m2=tuple(
            (np.asarray(base.inertia_kg_m2) * inertia_scale).tolist()
            if len(base.inertia_kg_m2) == 3
            else (
                np.diag(np.sqrt(inertia_scale))
                @ np.asarray(base.inertia_kg_m2, dtype=np.float64).reshape(3, 3)
                @ np.diag(np.sqrt(inertia_scale))
            ).reshape(-1).tolist()
        ),
        motor_effectiveness=tuple(float(v) for v in motor_effectiveness),
        voltage_factor=voltage_factor,
        max_rotor_speed_rad_s=base.max_rotor_speed_rad_s * max_rotor_speed_scale,
        thrust_coefficient_n_per_rad_s2=base.thrust_coefficient_n_per_rad_s2 * thrust_coefficient_scale,
        torque_to_thrust_m=base.torque_to_thrust_m * torque_to_thrust_scale,
        motor_time_constant_s=base.motor_time_constant_s * motor_response_scale,
        linear_drag_kg_s=base.linear_drag_kg_s * drag_scale,
        quadratic_drag_kg_m=base.quadratic_drag_kg_m * drag_scale,
        linear_drag_body_kg_s=(
            tuple((np.asarray(base.linear_drag_body_kg_s) * drag_scale).tolist())
            if base.linear_drag_body_kg_s is not None else None
        ),
        quadratic_drag_body_kg_m=(
            tuple((np.asarray(base.quadratic_drag_body_kg_m) * drag_scale).tolist())
            if base.quadratic_drag_body_kg_m is not None else None
        ),
    )
    sampled.validate()
    details = {
        "mass_scale": mass_scale,
        "center_of_mass_offset_m": cg_offset.tolist(),
        "inertia_scale_xyz": inertia_scale.tolist(),
        "motor_effectiveness": motor_effectiveness.tolist(),
        "motor_response_scale": motor_response_scale,
        "max_rotor_speed_scale": max_rotor_speed_scale,
        "thrust_coefficient_scale": thrust_coefficient_scale,
        "torque_to_thrust_scale": torque_to_thrust_scale,
        "voltage_factor": voltage_factor,
        "drag_scale": drag_scale,
    }
    return sampled, details
