from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_env import NativeSkySimEnv


def test_symmetric_action_mapping_avoids_zero_throttle_bias():
    _, throttle = NativeSkySimEnv.map_action(np.array([0.0, 0.0, 0.0, 0.0]))
    assert np.isclose(throttle, 0.35)
    _, min_throttle = NativeSkySimEnv.map_action(np.array([0.0, 0.0, 0.0, -1.0]))
    _, max_throttle = NativeSkySimEnv.map_action(np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.isclose(min_throttle, 0.20)
    assert np.isclose(max_throttle, 0.50)


def test_native_core_reset_and_step():
    env = NativeSkySimEnv(max_steps=3)
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == (12,)
        assert np.isfinite(obs).all()
        assert info["backend"] == "native-drone-core"
        assert info["flight_phase"] == "grounded"
        assert not info["has_taken_off"]
        assert np.isclose(obs[1], env.ground_height, atol=1e-3)
        next_obs, reward, terminated, truncated, step_info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        assert next_obs.shape == (12,)
        assert np.isfinite(next_obs).all()
        assert np.isfinite(reward)
        assert not truncated
        assert np.isclose(step_info["throttle"], 0.35)
        assert "position_xyz_m" in step_info
        assert "reward_components" in step_info
        assert np.isclose(step_info["reward_total"], reward)
        assert "crash_penalty" in step_info["reward_components"]
        assert "ground_wait_penalty" in step_info["reward_components"]
        assert step_info["flight_phase"] == "grounded"
        assert not step_info["crashed"]
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
        assert truncated
    finally:
        env.close()



def test_reward_prefers_target_altitude_and_position_to_low_hover():
    common = {
        "velocity": np.zeros(3),
        "attitude": np.zeros(3),
        "angular_velocity": np.zeros(3),
        "target": np.array([0.0, 3.0, 0.0]),
        "action": np.zeros(4),
        "throttle": 0.35,
    }
    target_reward, _, target_crashed, _ = NativeSkySimEnv.calculate_reward(
        position=np.array([0.0, 3.0, 0.0]), **common
    )
    low_reward, _, low_crashed, _ = NativeSkySimEnv.calculate_reward(
        position=np.array([0.0, 0.2, 0.0]), has_taken_off=False, **common
    )
    assert not target_crashed
    assert not low_crashed
    assert target_reward > 0
    assert low_reward > 0
    assert target_reward > low_reward


def test_reward_penalizes_linear_and_angular_motion():
    common = {
        "position": np.array([0.0, 3.0, 0.0]),
        "attitude": np.zeros(3),
        "target": np.array([0.0, 3.0, 0.0]),
        "action": np.zeros(4),
        "throttle": 0.35,
    }
    calm_reward, _, _, _ = NativeSkySimEnv.calculate_reward(
        velocity=np.zeros(3), angular_velocity=np.zeros(3), **common
    )
    moving_reward, moving_terms, _, _ = NativeSkySimEnv.calculate_reward(
        velocity=np.array([2.0, 2.0, 0.0]),
        angular_velocity=np.array([1.0, 0.0, 0.0]),
        **common
    )
    assert moving_reward < calm_reward
    assert moving_terms["angular_rate_penalty"] < 0


def test_crash_gets_explicit_terminal_penalty_and_reason():
    reward, terms, crashed, reason = NativeSkySimEnv.calculate_reward(
        position=np.array([0.0, 0.1, 0.0]),
        velocity=np.zeros(3),
        attitude=np.zeros(3),
        angular_velocity=np.zeros(3),
        target=np.array([0.0, 3.0, 0.0]),
        action=np.zeros(4),
        throttle=0.35,
    )
    assert crashed
    assert reason == "ground_contact_after_takeoff"
    assert terms["crash_penalty"] == -100.0
    assert reward < 0.0


def test_ground_wait_penalty_increases_without_ground_crash_before_takeoff():
    common = {
        "position": np.array([0.0, 0.16, 0.0]),
        "velocity": np.zeros(3),
        "attitude": np.zeros(3),
        "angular_velocity": np.zeros(3),
        "target": np.array([0.0, 3.0, 0.0]),
        "action": np.zeros(4),
        "throttle": 0.35,
        "has_taken_off": False,
    }
    early_reward, early_terms, early_crashed, _ = NativeSkySimEnv.calculate_reward(
        ground_wait_steps=1, **common
    )
    late_reward, late_terms, late_crashed, _ = NativeSkySimEnv.calculate_reward(
        ground_wait_steps=100, **common
    )
    assert not early_crashed and not late_crashed
    assert early_terms["ground_wait_penalty"] < 0
    assert late_terms["ground_wait_penalty"] < early_terms["ground_wait_penalty"]
    assert late_reward < early_reward


def test_ground_contact_is_crash_only_after_takeoff():
    common = {
        "position": np.array([0.0, 0.16, 0.0]),
        "velocity": np.zeros(3),
        "attitude": np.zeros(3),
        "angular_velocity": np.zeros(3),
        "target": np.array([0.0, 3.0, 0.0]),
        "action": np.zeros(4),
        "throttle": 0.35,
    }
    _, grounded_terms, grounded_crashed, grounded_reason = NativeSkySimEnv.calculate_reward(
        has_taken_off=False, **common
    )
    _, airborne_terms, airborne_crashed, airborne_reason = NativeSkySimEnv.calculate_reward(
        has_taken_off=True, **common
    )
    assert not grounded_crashed and grounded_reason is None
    assert grounded_terms["crash_penalty"] == 0.0
    assert airborne_crashed
    assert airborne_reason == "ground_contact_after_takeoff"
    assert airborne_terms["crash_penalty"] == -100.0


def test_takeoff_bonus_is_reported_as_a_separate_reward_component():
    reward, terms, crashed, reason = NativeSkySimEnv.calculate_reward(
        position=np.array([0.0, 0.5, 0.0]),
        velocity=np.array([0.0, 1.0, 0.0]),
        attitude=np.zeros(3),
        angular_velocity=np.zeros(3),
        target=np.array([0.0, 3.0, 0.0]),
        action=np.zeros(4),
        throttle=0.35,
        has_taken_off=True,
        takeoff_bonus=2.0,
    )
    assert not crashed and reason is None
    assert terms["takeoff_bonus"] == 2.0
    assert np.isfinite(reward)


def test_reset_after_motor_spin_matches_fresh_core():
    """Reset must clear rotor, ESC, inflow, VRS, battery, and turbulence runtime."""
    used = NativeSkySimEnv(max_steps=100)
    fresh = NativeSkySimEnv(max_steps=100)
    try:
        used.reset(seed=123)
        for _ in range(20):
            used.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        reset_obs, _ = used.reset(seed=456)
        fresh_obs, _ = fresh.reset(seed=456)
        assert np.allclose(reset_obs, fresh_obs, atol=1e-7)
        action = np.array([0.0, 0.0, 0.0, 0.25], dtype=np.float32)
        used_next, _, _, _, used_info = used.step(action)
        fresh_next, _, _, _, fresh_info = fresh.step(action)
        assert np.allclose(used_next, fresh_next, atol=1e-6)
        assert np.allclose(used_info["rotor_thrust_n"], fresh_info["rotor_thrust_n"], atol=1e-8)
        assert np.allclose(used_info["rotor_omega_rads"], fresh_info["rotor_omega_rads"], atol=1e-8)
    finally:
        used.close()
        fresh.close()


def test_telemetry_altitude_is_post_integration_state():
    env = NativeSkySimEnv(max_steps=10)
    try:
        env.reset(seed=1)
        obs, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        assert np.isclose(info["telemetry_altitude_m"], obs[1], atol=1e-7)
    finally:
        env.close()
