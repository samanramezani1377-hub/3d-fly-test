"""Fast non-training contract checks for the Smooth2 BEMT simulator path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.fast_physics.dynamics import FastQuadrotor
from training.fast_physics.ppo_action import wrench_to_motor_commands
from training.fast_physics.propeller import BladeElementPropeller
from training.fast_physics.vehicle_family import load_vehicle_profiles, profile_to_parameters

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "vehicle_models/family/vehicle_profiles.json"
SMOOTH2_ID = "smooth2-a2212-custom-prop"


def smooth2_profile():
    profiles = load_vehicle_profiles(PROFILE_PATH)
    selected = [p for p in profiles if p["vehicle_id"] == SMOOTH2_ID]
    assert len(selected) == 1, f"expected exactly one {SMOOTH2_ID} profile"
    return selected[0]


def test_smooth2_profile_has_traceable_provisional_physics_contract():
    profile = smooth2_profile()
    assert profile["calibration_status"] == "cad_bemt_estimate"
    assert len(profile["rotors"]) == 4
    physical = profile["physical"]
    assert float(physical["mass_kg"]) > 0
    cg = np.asarray(physical["center_of_mass_m"], dtype=float)
    inertia = np.asarray(physical["inertia_kg_m2"], dtype=float)
    assert cg.shape == (3,) and np.all(np.isfinite(cg))
    assert inertia.size in (3, 9) and np.all(np.isfinite(inertia))
    inertia_matrix = np.diag(inertia) if inertia.size == 3 else inertia.reshape(3, 3)
    assert np.allclose(inertia_matrix, inertia_matrix.T, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(inertia_matrix)) > 0
    assert physical["mass_properties_estimate"]["status"] == "provisional_not_mesh_integrated"
    bemt = profile["propulsion"]["propeller_bemt"]
    assert bemt["model"] == "cad_station_blade_element_momentum"
    assert len(bemt["stations"]) >= 3
    assert int(bemt["integration_sections"]) >= 8


def test_smooth2_bemt_curve_is_finite_and_monotonic_with_inflow_cases():
    profile = smooth2_profile()
    model = BladeElementPropeller(profile["propulsion"]["propeller_bemt"])
    speeds = np.linspace(0.0, float(profile["propulsion"]["max_rotor_speed_rad_s"]), 17)
    for inflow in (-3.0, 0.0, 3.0):
        results = [model.evaluate(float(w), inflow) for w in speeds]
        values = np.asarray(results, dtype=float)
        assert values.shape == (len(speeds), 2)
        assert np.all(np.isfinite(values))
        assert np.all(values >= -1e-9)
        if inflow == 0.0:
            assert np.all(np.diff(values[:, 0]) >= -1e-7), "static BEMT thrust curve must be monotonic"


def test_smooth2_runtime_plant_reports_bemt_and_finite_rigid_body_outputs():
    params = profile_to_parameters(smooth2_profile())
    plant = FastQuadrotor(params, dt=0.02)
    plant.reset(position=(0.0, 0.0, 1.0))
    state, info = plant.step(np.full(4, 0.65, dtype=np.float64))
    assert info["thrust_model_status"] == (
        "runtime_CAD_station_BEMT_with_axial_inflow_and_randomized_effectiveness"
    )
    assert len(info["rotor_shaft_torque_nm"]) == 4
    assert len(info["rotor_thrust_n"]) == 4
    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(info["rotor_shaft_torque_nm"]))
    assert np.all(np.isfinite(info["rotor_thrust_n"]))
    assert np.all(np.asarray(info["rotor_shaft_torque_nm"]) >= 0)
    assert np.all(np.asarray(info["rotor_thrust_n"]) >= 0)
    assert np.isclose(info["mass_kg"], params.mass_kg)


def test_smooth2_wrench_allocator_returns_bounded_finite_motor_commands():
    params = profile_to_parameters(smooth2_profile())
    requests = (
        np.array([params.mass_kg * 9.80665, 0.0, 0.0, 0.0]),
        np.array([params.mass_kg * 9.80665, 0.01, -0.01, 0.001]),
        np.array([1000.0, 10.0, -10.0, 10.0]),
    )
    for wrench in requests:
        commands = wrench_to_motor_commands(wrench, params)
        assert commands.shape == (4,)
        assert np.all(np.isfinite(commands))
        assert np.all(commands >= 0.0) and np.all(commands <= 1.0)
