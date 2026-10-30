from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics.vehicle_family import (
    DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles,
    profile_to_parameters, vehicle_feature_vector,
)


def test_family_has_distinct_provisional_quadrotor_profiles():
    profiles = load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)
    assert len(profiles) >= 3
    assert len({p["vehicle_id"] for p in profiles}) == len(profiles)
    assert all(p["calibration_status"] == "provisional_estimate" for p in profiles)
    assert len({p["physical"]["mass_kg"] for p in profiles}) > 1
    assert len({p["geometry"]["propeller_diameter_m"] for p in profiles}) > 1


def test_profiles_build_valid_four_rotor_models():
    for profile in load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY):
        params = profile_to_parameters(profile)
        params.validate()
        assert len(params.rotor_positions_m) == 4
        assert params.inertia_matrix.shape == (3, 3)


def test_descriptor_is_fixed_size_finite_and_vehicle_specific():
    vectors = [vehicle_feature_vector(p) for p in load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)]
    assert all(v.shape == (10,) and np.all(np.isfinite(v)) for v in vectors)
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
    profile["propulsion"]["max_rotor_speed_rad_s"] = None
    profile["propulsion"]["thrust_coefficient_n_per_rad_s2"] = None
    profile["propulsion"]["torque_to_thrust_m"] = None
    profile["propulsion"]["motor_time_constant_s"] = None
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    loaded = load_vehicle_profiles(path)[0]
    assert loaded["physical"]["mass_kg"] == profile["physical"]["mass_kg"]
    assert len(loaded["derived_estimates"]) == 5
    assert "inertia_kg_m2:point-mass-arm-estimate" in loaded["derived_estimates"]
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
