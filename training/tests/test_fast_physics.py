from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics import AdaptiveDroneEnv, FastQuadrotor, VehicleParameters, load_vehicle_parameters, sample_scenario
AIRFRAME = ROOT / "vehicle_models/smooth2_parametric/base_airframe.json"
CALIBRATION = ROOT / "vehicle_models/smooth2_parametric/physics_calibration.json"
RANDOMIZATION = ROOT / "vehicle_models/smooth2_parametric/domain_randomization.json"


def synthetic_params():
    return VehicleParameters(
        mass_kg=0.6,
        center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(0.004, 0.006, 0.004),
        rotor_positions_m=((0.082, 0.082, 0.0), (0.082, -0.082, 0.0),
                           (-0.082, -0.082, 0.0), (-0.082, 0.082, 0.0)),
        spin_signs=(1.0, -1.0, 1.0, -1.0),
        max_rotor_speed_rad_s=1100.0,
        thrust_coefficient_n_per_rad_s2=2.0e-6,
        torque_to_thrust_m=0.012,
    )


def test_unknown_calibration_is_not_silently_filled():
    with pytest.raises(ValueError, match="Physical calibration is incomplete"):
        load_vehicle_parameters(AIRFRAME, CALIBRATION)


def test_synthetic_calibration_requires_explicit_opt_in():
    params, metadata = load_vehicle_parameters(
        AIRFRAME, CALIBRATION, allow_synthetic_calibration=True
    )
    assert metadata["synthetic_calibration_used"] is True
    assert params.mass_kg == pytest.approx(0.6)


def test_airframe_geometry_is_fixed_and_four_rotors_are_present():
    data = json.loads(AIRFRAME.read_text(encoding="utf-8"))
    assert data["vehicle_id"] == "smooth2-parametric-v1"
    assert len(data["rotors"]) == 4
    assert all(np.allclose(r["axis_body"], [0, 0, 1]) for r in data["rotors"])
    assert data["geometry"]["model_scale"] == pytest.approx(1.414028)
    assert data["uncertain_physical_properties"]["takeoff_mass_kg"] is None


def test_dynamics_hover_force_balances_weight_at_expected_command():
    params = synthetic_params()
    plant = FastQuadrotor(params)
    plant.reset(position=(0, 0, 1))
    # A full-scale command is not a hover command; this test checks finite response.
    state, telemetry = plant.step(np.ones(4, dtype=np.float32) * 0.5)
    assert np.all(np.isfinite(state))
    assert len(telemetry["rotor_thrust_n"]) == 4
    assert telemetry["total_thrust_n"] > 0


def test_domain_randomization_is_reproducible_and_changes_mass_cg_inertia():
    params = synthetic_params()
    config = json.loads(RANDOMIZATION.read_text(encoding="utf-8"))
    a, details_a = sample_scenario(params, np.random.default_rng(123), config)
    b, details_b = sample_scenario(params, np.random.default_rng(123), config)
    assert a == b
    assert details_a == details_b
    assert a.mass_kg != params.mass_kg
    assert a.center_of_mass_m != params.center_of_mass_m
    assert a.inertia_kg_m2 != params.inertia_kg_m2


def test_env_observation_contains_history_but_not_privileged_parameters():
    env = AdaptiveDroneEnv(
        params=synthetic_params(), randomization_path=RANDOMIZATION,
        history_length=4, max_steps=20,
    )
    try:
        obs, info = env.reset(seed=9)
        assert obs.shape == (64,)
        assert info["policy_observation_excludes_privileged_parameters"] is True
        assert "mass_kg" not in info
        next_obs, reward, terminated, truncated, step_info = env.step(np.full(4, 0.5))
        assert next_obs.shape == (64,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "mass_kg" in step_info
    finally:
        env.close()


def test_randomized_env_seed_repeats_scenario():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION)
    try:
        _, info_a = env.reset(seed=77)
        _, info_b = env.reset(seed=77)
        assert info_a["scenario"] == info_b["scenario"]
        assert info_a["wind_m_s"] == info_b["wind_m_s"]
    finally:
        env.close()


def test_action_shape_is_validated():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION)
    try:
        env.reset(seed=1)
        with pytest.raises(ValueError, match="four motor commands"):
            env.step(np.zeros(3))
    finally:
        env.close()
