from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics.propeller import BladeElementPropeller
from fast_physics.ppo_action import _rotor_thrust_limits_and_inverse, wrench_to_motor_commands
from fast_physics.dynamics import _dynamics_accelerations_python, _dynamics_accelerations_jit
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


def test_numba_motor_allocator_matches_original_numpy_path():
    params = _smooth2_params()
    rng = np.random.default_rng(20261104)
    arm = float(np.mean(np.linalg.norm(np.asarray(params.rotor_positions_m)[:, :2], axis=1)))
    weight = params.mass_kg * 9.80665
    for _ in range(32):
        wrench = np.array([
            rng.uniform(0.55, 1.35) * weight,
            rng.uniform(-0.18, 0.18) * weight * arm,
            rng.uniform(-0.18, 0.18) * weight * arm,
            rng.uniform(-0.18, 0.18) * weight * params.torque_to_thrust_m,
        ], dtype=np.float64)
        reference = wrench_to_motor_commands(wrench, params, _use_numba=False)
        optimized = wrench_to_motor_commands(wrench, params, _use_numba=True)
        np.testing.assert_allclose(optimized, reference, rtol=2e-6, atol=2e-6)


def test_compiled_dynamics_acceleration_core_matches_python_equations():
    rng = np.random.default_rng(731)
    for _ in range(40):
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1.0
        air_velocity = rng.normal(size=3)
        force = rng.normal(size=3)
        torque = rng.normal(size=3) * 0.1
        inertia = np.diag(rng.uniform(0.002, 0.02, size=3))
        inverse = np.linalg.inv(inertia)
        angular_velocity = rng.normal(size=3)
        spin = np.array([1.0, -1.0, 1.0, -1.0])
        omega = rng.uniform(0.0, 1000.0, size=4)
        omega_dot = rng.normal(size=4) * 100.0
        args = (
            rotation, air_velocity, force, torque, 0.6, 9.80665,
            np.array([0.08, 0.09, 0.1]), np.array([0.01, 0.02, 0.03]),
            inertia, inverse, angular_velocity, spin, omega, omega_dot, 1.2e-5, 0.002,
        )
        expected = _dynamics_accelerations_python(*args)
        actual = _dynamics_accelerations_jit(*args)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)


def test_numba_motor_allocator_matches_original_numpy_path():
    params = _smooth2_params()
    rng = np.random.default_rng(20261104)
    arm = float(np.mean(np.linalg.norm(np.asarray(params.rotor_positions_m)[:, :2], axis=1)))
    weight = params.mass_kg * 9.80665
    for _ in range(32):
        wrench = np.array([
            rng.uniform(0.55, 1.35) * weight,
            rng.uniform(-0.18, 0.18) * weight * arm,
            rng.uniform(-0.18, 0.18) * weight * arm,
            rng.uniform(-0.18, 0.18) * weight * params.torque_to_thrust_m,
        ], dtype=np.float64)
        reference = wrench_to_motor_commands(wrench, params, _use_numba=False)
        optimized = wrench_to_motor_commands(wrench, params, _use_numba=True)
        np.testing.assert_allclose(optimized, reference, rtol=2e-6, atol=2e-6)


def test_compiled_dynamics_acceleration_core_matches_python_equations():
    rng = np.random.default_rng(731)
    for _ in range(40):
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1.0
        air_velocity = rng.normal(size=3)
        force = rng.normal(size=3)
        torque = rng.normal(size=3) * 0.1
        inertia = np.diag(rng.uniform(0.002, 0.02, size=3))
        inverse = np.linalg.inv(inertia)
        angular_velocity = rng.normal(size=3)
        spin = np.array([1.0, -1.0, 1.0, -1.0])
        omega = rng.uniform(0.0, 1000.0, size=4)
        omega_dot = rng.normal(size=4) * 100.0
        args = (
            rotation, air_velocity, force, torque, 0.6, 9.80665,
            np.array([0.08, 0.09, 0.1]), np.array([0.01, 0.02, 0.03]),
            inertia, inverse, angular_velocity, spin, omega, omega_dot, 1.2e-5, 0.002,
        )
        expected = _dynamics_accelerations_python(*args)
        actual = _dynamics_accelerations_jit(*args)
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_allclose(actual_value, expected_value, rtol=2e-12, atol=2e-12)
