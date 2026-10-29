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
        position=np.array([0.0, 0.2, 0.0]), **common
    )
    assert not target_crashed
    assert not low_crashed
    assert target_reward > 0
    assert low_reward < 0
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
    assert reason == "ground_contact"
    assert terms["crash_penalty"] == -100.0
    assert reward < -100.0
