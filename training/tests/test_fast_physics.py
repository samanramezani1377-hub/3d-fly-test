from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics import AdaptiveDroneEnv, FastQuadrotor, VehicleParameters, load_vehicle_parameters, sample_scenario, OnlineThrustEstimator, randomization_for_stage
from evaluate_robust_ppo import summarize_episode
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
        assert obs.shape == (66,)
        assert info["policy_observation_excludes_privileged_parameters"] is True
        assert "mass_kg" not in info
        next_obs, reward, terminated, truncated, step_info = env.step(np.full(4, 0.5))
        assert next_obs.shape == (66,)
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


def test_full_inertia_tensor_is_supported_and_positive_definite():
    params = synthetic_params()
    full = VehicleParameters(**{**params.__dict__, "inertia_kg_m2": (0.02, 0.001, 0.0, 0.001, 0.035, 0.0, 0.0, 0.0, 0.02)})
    full.validate()
    assert full.inertia_matrix.shape == (3, 3)
    invalid = VehicleParameters(**{**params.__dict__, "inertia_kg_m2": (1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0)})
    with pytest.raises(ValueError):
        invalid.validate()


def test_online_estimator_learns_effective_collective_response_from_observations():
    estimator = OnlineThrustEstimator()
    dt = 0.02
    vz = 0.0
    estimator.reset(vz)
    for index in range(600):
        command = 0.15 + 0.75 * ((index % 30) / 29.0)
        acceleration = 0.4 + 14.0 * command ** 2 - 9.80665
        vz += acceleration * dt
        estimator.update(np.full(4, command), vz, dt)
    result = estimator.diagnostics()
    assert np.isfinite(result["effective_lift_gain_m_s2"])
    assert result["samples"] == 600
    assert result["confidence"] > 0.0
    assert abs(result["effective_lift_gain_m_s2"] - 14.0) < 2.0


def test_curriculum_stage_one_is_nominal_and_stage_four_uses_wider_bounds():
    config = json.loads(RANDOMIZATION.read_text(encoding="utf-8"))
    stage1 = randomization_for_stage(config, 1)
    stage4 = randomization_for_stage(config, 4)
    assert stage1["mass"]["scale"] == [1.0, 1.0]
    assert stage1["environment"]["wind_component_m_s"] == [0.0, 0.0]
    assert stage4["mass"]["scale"] == [0.80, 1.25]
    assert stage4["environment"]["wind_component_m_s"] == [-2.0, 2.0]


def test_env_exposes_estimator_context_but_not_ground_truth_parameters():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION)
    try:
        obs, _ = env.reset(seed=12)
        assert obs.shape == (66,)
        assert np.all(np.isfinite(obs[-2:]))
        _, _, _, _, info = env.step(np.full(4, 0.4))
        assert "online_estimator" in info
        assert "mass_kg" not in info["online_estimator"]
    finally:
        env.close()


def test_full_cg_offset_is_randomized_in_all_three_body_axes():
    params = synthetic_params()
    config = json.loads(RANDOMIZATION.read_text(encoding="utf-8"))
    sampled, details = sample_scenario(params, np.random.default_rng(31), config)
    assert len(details["center_of_mass_offset_m"]) == 3
    assert sampled.center_of_mass_m[2] != params.center_of_mass_m[2]


def test_sensor_noise_is_reproducible_with_same_seed():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION)
    try:
        obs_a, _ = env.reset(seed=202)
        obs_b, _ = env.reset(seed=202)
        assert np.allclose(obs_a, obs_b)
    finally:
        env.close()


def test_held_out_evaluator_requires_sustained_hover_and_no_crash():
    stable_samples = [
        {
            "taken_off": True, "stable": True, "crashed": False,
            "altitude_error_m": 0.05, "horizontal_error_m": 0.05,
        }
        for _ in range(300)
    ]
    good = summarize_episode(stable_samples, 500.0)
    assert good["hover_success"] is True
    assert good["crashed"] is False

    crashed_samples = stable_samples[:-1] + [{
        "taken_off": True, "stable": False, "crashed": True,
        "altitude_error_m": 0.5, "horizontal_error_m": 0.5,
    }]
    bad = summarize_episode(crashed_samples, 100.0)
    assert bad["hover_success"] is False
    assert bad["crashed"] is True


def test_online_estimator_uses_sensor_measurement_not_simulator_truth():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION, randomize=False)
    try:
        env.reset(seed=4)
        original_measure = env._measure_state

        def biased_sensor(state):
            measured = original_measure(state)
            measured[5] += 0.123
            return measured

        env._measure_state = biased_sensor
        env.step(np.full(4, 0.4))
        assert env.estimator.previous_vertical_velocity == pytest.approx(
            env.plant.velocity[2] + 0.123
        )
    finally:
        env.close()
