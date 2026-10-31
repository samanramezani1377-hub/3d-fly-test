"""Sanity tests for the Smooth2 propeller design (run: python -m pytest cad/parametric or python this file)."""
import json
import math
from pathlib import Path

import prop_bemt as pb

PARAMS = json.loads(Path(__file__).with_name("model.params.json").read_text(encoding="utf-8"))
SCALE = PARAMS["model_scale"]
RADIUS = PARAMS["prop_radius"] * SCALE / 1000.0
HUB = PARAMS["prop_hub_radius"] * SCALE / 1000.0
ROWS = PARAMS["prop_blade_stations"]


def test_bemt_matches_report_for_original_blade():
    """The original constant-pitch blade must give ~0.20 N at 9979 RPM, as the report states."""
    old = [(0.05, 6.96), (0.14, 8.36), (0.23, 9.21), (0.33, 9.43), (0.44, 8.79), (0.56, 7.82),
           (0.68, 6.86), (0.80, 6.0), (0.90, 5.14), (0.96, 4.71), (1.0, 4.28)]
    stations = [(x, c * SCALE / 1000.0) for x, c in old]
    old_radius = 26.45 * SCALE / 1000.0  # radius of the first design, before the rotor was resized
    betas = pb.constant_pitch_betas(old_radius, stations, 0.053086, cap_deg=42)
    result = pb.analyse(old_radius, stations, betas, 9979, hub_radius=0.0)
    assert abs(result["thrust_N"] - 0.200) < 0.005


def test_new_stations_are_designed_not_capped():
    betas = [r[2] for r in ROWS]
    assert all(len(r) == 3 for r in ROWS)
    assert all(b2 <= b1 for b1, b2 in zip(betas, betas[1:])), "pitch must decrease outward"
    assert min(betas) > 5.0
    assert ROWS[0][0] < PARAMS["prop_hub_radius"] / PARAMS["prop_radius"], "root starts inside the hub"


def _motor_point(voltage=None):
    stations = [(r[0], r[1] * SCALE / 1000.0) for r in ROWS]
    alpha0 = pb.thin_airfoil_alpha0(PARAMS["prop_airfoil_camber"], PARAMS["prop_airfoil_camber_position"])
    motor = dict(pb.A2212_ASSUMED, **PARAMS["prop_design"]["motor"])
    return pb.motor_operating_point(RADIUS, stations, [r[2] for r in ROWS], HUB,
                                    PARAMS["prop_blade_count"], alpha0, motor, voltage=voltage)


def test_full_throttle_thrust_meets_target_within_current_limit():
    design = PARAMS["prop_design"]
    full = _motor_point()
    weight = design["vehicle_mass_kg"] * 9.80665 / 4.0
    assert full["thrust_N"] >= design["target_thrust_to_weight"] * weight
    assert full["current_a"] <= design["max_motor_current_a"] + 1e-6


def test_hover_needs_less_than_full_throttle():
    design = PARAMS["prop_design"]
    weight = design["vehicle_mass_kg"] * 9.80665 / 4.0
    half = _motor_point(voltage=0.6 * design["motor"]["voltage_v"])
    assert half["thrust_N"] > weight, "60% throttle must already lift the 0.6 kg vehicle"


def test_zero_lift_angle_is_negative_for_positive_camber():
    assert pb.thin_airfoil_alpha0(0.04, 0.4) < math.radians(-3.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
