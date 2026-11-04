from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics.propeller import BladeElementPropeller
from fast_physics.ppo_action import _rotor_thrust_limits_and_inverse
from fast_physics.vehicle_family import load_vehicle_profiles, profile_to_parameters


def _smooth2_params():
    profiles = load_vehicle_profiles(ROOT / "vehicle_models/family/vehicle_profiles.json")
    profile = next(p for p in profiles if p["vehicle_id"] == "smooth2-a2212-custom-prop")
    return profile_to_parameters(profile)


def test_fused_bemt_batch_matches_individual_rotor_evaluations():
    params = _smooth2_params()
    model = BladeElementPropeller(params.propeller_bemt)
    omega = np.array([0.0, 180.0, 520.0, 910.0], dtype=np.float64)
    axial = np.array([0.0, -0.35, 0.8, 1.1], dtype=np.float64)
    thrust, torque = model.evaluate_many(omega, axial)
    reference = np.asarray([model.evaluate(w, v) for w, v in zip(omega, axial)])
    np.testing.assert_allclose(thrust, reference[:, 0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(torque, reference[:, 1], rtol=2e-12, atol=2e-12)


def test_cached_bemt_configuration_key_matches_canonical_configuration():
    params = _smooth2_params()
    config_json = json.dumps(params.propeller_bemt, sort_keys=True, separators=(",", ":"))
    direct = _rotor_thrust_limits_and_inverse(params)
    cached_key = _rotor_thrust_limits_and_inverse(params, bemt_config_json=config_json)
    for left, right in zip(direct, cached_key):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif left is None:
            assert right is None
        else:
            assert left == right


def test_bemt_batch_rejects_mismatched_input_shapes():
    params = _smooth2_params()
    model = BladeElementPropeller(params.propeller_bemt)
    try:
        model.evaluate_many([100.0, 200.0], [0.0])
    except ValueError as exc:
        assert "matching 1-D arrays" in str(exc)
    else:
        raise AssertionError("mismatched rotor input arrays must be rejected")
