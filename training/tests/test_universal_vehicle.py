from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics.curriculum import randomization_for_stage
from fast_physics.dynamics import FastQuadrotor
from fast_physics.propeller import BladeElementPropeller
from fast_physics.env import AdaptiveDroneEnv
from train_universal_ppo import build_velocity_evaluation_sequence
from fast_physics.vehicle_family import (
    DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles,
    profile_to_parameters, vehicle_feature_vector,
)


def test_family_has_distinct_provisional_and_cad_estimated_quadrotor_profiles():
    profiles = load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)
    assert len(profiles) >= 3
    assert len({p["vehicle_id"] for p in profiles}) == len(profiles)
    assert profiles[0]["calibration_status"] == "cad_bemt_estimate"
    assert all(p["calibration_status"] == "provisional_estimate" for p in profiles[1:])
    assert len({p["physical"]["mass_kg"] for p in profiles}) > 1
    assert len({p["geometry"]["propeller_diameter_m"] for p in profiles}) > 1


def test_profiles_build_valid_four_rotor_models():
    for profile in load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY):
        params = profile_to_parameters(profile)
        params.validate()
        assert len(params.rotor_positions_m) == 4
        assert params.inertia_matrix.shape == (3, 3)


def test_every_profile_has_nominal_static_hover_equilibrium():
    for profile in load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY):
        params = profile_to_parameters(profile)
        if params.propeller_bemt is not None:
            propeller = BladeElementPropeller(params.propeller_bemt)
            target_per_rotor = params.mass_kg * 9.80665 / 4.0
            lo, hi = 0.0, params.max_rotor_speed_rad_s
            for _ in range(36):
                mid = 0.5 * (lo + hi)
                if propeller.evaluate(mid)[0] < target_per_rotor:
                    lo = mid
                else:
                    hi = mid
            hover_omega = 0.5 * (lo + hi)
        else:
            hover_omega = np.sqrt(params.mass_kg * 9.80665 / (4.0 * params.thrust_coefficient_n_per_rad_s2))
        assert 0.0 < hover_omega < params.max_rotor_speed_rad_s
        command = np.full(4, hover_omega / params.max_rotor_speed_rad_s, dtype=np.float64)
        plant = FastQuadrotor(params)
        plant.reset(position=(0.0, 0.0, 1.0))
        plant.motor_omega[:] = hover_omega
        for _ in range(100):
            state, telemetry = plant.step(command)
        assert abs(float(state[2]) - 1.0) < 1e-5
        assert np.linalg.norm(state[6:12]) < 1e-6
        assert telemetry["total_thrust_n"] == pytest.approx(params.mass_kg * 9.80665, rel=2e-3)


def test_all_curriculum_stages_preserve_minimum_hover_thrust_margin():
    randomization_path = ROOT / "vehicle_models/family/domain_randomization.json"
    config = json.loads(randomization_path.read_text(encoding="utf-8"))
    for profile in load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY):
        params = profile_to_parameters(profile)
        nominal_twr = (
            4.0 * params.thrust_coefficient_n_per_rad_s2 * params.max_rotor_speed_rad_s**2
            / (params.mass_kg * 9.80665)
        )
        for stage_number in range(1, 6):
            stage = randomization_for_stage(config, stage_number)
            worst_mass = max(stage["mass"]["scale"])
            worst_voltage = min(stage["battery"]["voltage_factor"])
            worst_motor = min(stage["motor"]["effectiveness"])
            worst_rotor_speed = min(stage["motor"]["max_rotor_speed_scale"])
            worst_thrust_coefficient = min(stage["thrust"]["coefficient_scale"])
            worst_case_twr = (
                nominal_twr * worst_voltage**2 * worst_motor
                * worst_rotor_speed**2 * worst_thrust_coefficient / worst_mass
            )
            assert worst_case_twr > 1.05, (
                f"{profile['vehicle_id']} stage {stage_number} has insufficient "
                f"worst-case thrust-to-weight margin: {worst_case_twr:.3f}"
            )


def test_descriptor_is_fixed_size_finite_and_vehicle_specific():
    vectors = [vehicle_feature_vector(p) for p in load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)]
    assert all(v.shape == (12,) and np.all(np.isfinite(v)) for v in vectors)
    assert not np.allclose(vectors[0], vectors[1])


def test_family_env_selects_profiles_reproducibly_and_adds_config_to_observation():
    env = VehicleFamilyEnv(vehicle_family_path=DEFAULT_VEHICLE_FAMILY, max_steps=10)
    try:
        obs_a, info_a = env.reset(seed=123)
        obs_b, info_b = env.reset(seed=123)
        assert obs_a.shape == (78,)
        assert info_a["vehicle_id"] == info_b["vehicle_id"]
        assert np.allclose(obs_a, obs_b)
        assert info_a["policy_observation_includes_vehicle_configuration"] is True
        assert len(info_a["vehicle_configuration_features"]) == 12
        assert info_a["policy_observation_excludes_episode_randomization_ground_truth"] is True
    finally:
        env.close()

    # The movement policy adds exactly four inputs after the unchanged 12
    # nominal hardware features: forward, left, world-up and yaw rate.
    velocity_env = VehicleFamilyEnv(
        vehicle_family_path=DEFAULT_VEHICLE_FAMILY, max_steps=1500,
        curriculum_stage=2, velocity_control=True,
    )
    try:
        sequence = build_velocity_evaluation_sequence(episode_index=0, stage=2)
        assert len(sequence) == 30
        assert 100 + sum(item["duration_steps"] for item in sequence) <= 1500
        velocity_obs, _ = velocity_env.reset(seed=123, options={"velocity_command_sequence": sequence})
        assert velocity_obs.shape == (82,)
        assert np.allclose(velocity_obs[-4:], 0.0)
    finally:
        velocity_env.close()

    # At 90 degrees yaw, nose-forward points world-left and nose-left points
    # world-backward in the ENU convention used by the plant.
    forward_world = AdaptiveDroneEnv._body_yaw_velocity_to_world(
        np.array([1.0, 0.0, 0.0]), np.pi / 2
    )
    left_world = AdaptiveDroneEnv._body_yaw_velocity_to_world(
        np.array([0.0, 1.0, 0.0]), np.pi / 2
    )
    assert np.allclose(forward_world, [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(left_world, [-1.0, 0.0, 0.0], atol=1e-6)


def test_environment_can_be_pinned_to_one_vehicle():
    vehicle_id = load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)[-1]["vehicle_id"]
    env = VehicleFamilyEnv(vehicle_family_path=DEFAULT_VEHICLE_FAMILY, vehicle_id=vehicle_id, max_steps=10)
    try:
        obs, info = env.reset(seed=5)
        assert info["vehicle_id"] == vehicle_id
        assert obs.shape == env.observation_space.shape
    finally:
        env.close()


def test_incomplete_user_profile_fails_closed(tmp_path):
    profile = {
        "vehicle_id":"incomplete","calibration_status":"user_supplied",
        "geometry":{"rotor_count":4,"arm_span_diagonal_m":0.2,"propeller_diameter_m":0.1},
        "physical":{"mass_kg":None,"inertia_kg_m2":[0.01,0.01,0.01]},
        "propulsion":{},"rotors":[{"position_m":[0,0,0],"spin_sign":1} for _ in range(4)]
    }
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="positive takeoff mass"):
        load_vehicle_profiles(path)


def test_partial_user_profile_derives_secondary_physics_and_labels_it(tmp_path):
    source = json.loads(DEFAULT_VEHICLE_FAMILY.read_text(encoding="utf-8"))
    profile = source["profiles"][0]
    profile["vehicle_id"] = "user-config-with-known-mass"
    profile["calibration_status"] = "user_supplied"
    profile["physical"]["inertia_kg_m2"] = None
    profile["physical"]["center_of_mass_m"] = None
    profile["propulsion"]["max_rotor_speed_rad_s"] = None
    profile["propulsion"]["thrust_coefficient_n_per_rad_s2"] = None
    profile["propulsion"]["torque_to_thrust_m"] = None
    profile["propulsion"]["motor_time_constant_s"] = None
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    loaded = load_vehicle_profiles(path)[0]
    assert loaded["physical"]["mass_kg"] == profile["physical"]["mass_kg"]
    assert len(loaded["derived_estimates"]) == 6
    assert "inertia_kg_m2:point-mass-arm-estimate" in loaded["derived_estimates"]
    assert "center_of_mass_m:geometric-center-assumption" in loaded["derived_estimates"]
    assert profile_to_parameters(loaded).inertia_matrix.shape == (3, 3)
    features = vehicle_feature_vector(loaded)
    assert features.shape == (12,)
    assert features[-2] > 0.0


def test_non_positive_definite_inertia_is_rejected(tmp_path):
    source = json.loads(DEFAULT_VEHICLE_FAMILY.read_text(encoding="utf-8"))
    profile = source["profiles"][0]
    profile["vehicle_id"] = "bad-inertia"
    profile["physical"]["inertia_kg_m2"] = [1.0, 2.0, 0.0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError):
        load_vehicle_profiles(path)


def test_production_velocity_workflow_trains_and_evaluates_all_four_profiles():
    workflow = (ROOT / ".github" / "workflows" / "universal-ppo-five-stage.yml").read_text(encoding="utf-8")
    expected_ids = {
        "smooth2-a2212-custom-prop",
        "generic-5inch-quad",
        "generic-7inch-long-range",
        "generic-3inch-cinewhoop",
    }
    assert all(vehicle_id in workflow for vehicle_id in expected_ids)
    assert "--vehicle-family vehicle_models/family/vehicle_profiles.json" in workflow
    assert "--vehicle-id" not in workflow
    assert "assert len(profiles) == 4" in workflow
    assert "assert len(results) == 4" in workflow
    assert '"episodes_per_vehicle": 10' in workflow
    assert '"metrics_by_vehicle": metrics_by_vehicle' in workflow
    assert "universal-quad-four-vehicle-five-stage" in workflow
    assert '"physical_flight_eligible": false' in workflow.lower()


def test_universal_evaluation_publishes_battery_diagnostics_without_using_them_for_checkpoint_selection():
    source = (ROOT / "training" / "train_universal_ppo.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "universal-ppo-five-stage.yml").read_text(encoding="utf-8")
    for key in (
        "mean_estimated_battery_current_a",
        "mean_estimated_battery_energy_consumed_wh",
        "mean_estimated_battery_consumed_ah",
        "mean_final_battery_state_of_charge_estimate",
    ):
        assert key in source
        assert key in workflow
    # Electrical diagnostics are report-only and are not added to policy observations.
    score_start = source.index("score = (")
    score_end = source.index("report = {", score_start)
    assert "battery" not in source[score_start:score_end].lower()
