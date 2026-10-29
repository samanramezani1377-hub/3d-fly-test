"""Smoke-test the C API state vector and the policy observation transform."""
from __future__ import annotations

import json
import numpy as np

from native_env import NativeSkySimEnv, PolicyObservationWrapper


def main():
    raw_env = NativeSkySimEnv()
    env = PolicyObservationWrapper(raw_env)
    checks = []
    try:
        obs, info = env.reset(seed=123)
        raw = raw_env._observation()
        expected = raw.copy()
        expected[0:3] = (expected[0:3] - raw_env.target.astype(np.float32)) / 12.0
        expected[3:6] /= 8.0
        expected[6:9] /= np.pi
        expected[9:12] /= 10.0
        assert obs.shape == (12,), f"expected 12 observations, got {obs.shape}"
        assert np.all(np.isfinite(raw)), f"raw C API state contains non-finite values: {raw}"
        assert np.allclose(obs, expected, atol=1e-6), "policy transform does not match documented feature mapping"
        assert np.allclose(raw[0:3], [0.0, raw_env.ground_height, 0.0], atol=1e-4), (
            f"unexpected reset position/order: {raw[0:3]}"
        )
        checks.append("reset vector finite; 12 features; expected reset position and feature scaling verified")

        for i, action in enumerate((
            np.array([0, 0, 0, -1], dtype=np.float32),
            np.array([0, 0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0, 1], dtype=np.float32),
        )):
            obs, reward, terminated, truncated, step_info = env.step(action)
            raw = raw_env._observation()
            assert obs.shape == (12,) and np.all(np.isfinite(obs)), "invalid policy observation after step"
            assert np.allclose(step_info["position_xyz_m"], raw[0:3], atol=1e-5), (
                "info position does not match C API observation position slots"
            )
            assert np.allclose(step_info["velocity_xyz_mps"], raw[3:6], atol=1e-5), (
                "info velocity does not match C API observation velocity slots"
            )
            expected_throttle = float(np.clip(0.35 + 0.15 * action[3], 0.0, 1.0))
            assert abs(step_info["throttle"] - expected_throttle) < 1e-6, (
                f"throttle mapping mismatch: {step_info['throttle']} != {expected_throttle}"
            )
            checks.append(f"action {i}: state/info alignment and throttle mapping verified")
            if terminated or truncated:
                obs, info = env.reset(seed=124 + i)
        print(json.dumps({
            "passed": True,
            "raw_observation_order": [
                "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
                "roll_rad", "pitch_rad", "yaw_rad", "wx_rads", "wy_rads", "wz_rads"
            ],
            "policy_transform": {
                "position": "(position - target) / 12 m",
                "linear_velocity": "velocity / 8 m/s",
                "euler_angles": "angle / pi rad",
                "angular_velocity": "angular_velocity / 10 rad/s"
            },
            "checks": checks,
            "important_limit": "This checks Python-to-C API slot consistency and internal units/transform; it does not independently prove the upstream C++ physical semantics or real-world accuracy."
        }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
