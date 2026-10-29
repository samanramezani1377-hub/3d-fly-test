from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_env import NativeSkySimEnv


def test_native_core_reset_and_step():
    env = NativeSkySimEnv(max_steps=3)
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == (12,)
        assert np.isfinite(obs).all()
        assert info["backend"] == "native-drone-core"
        next_obs, reward, terminated, truncated, step_info = env.step(
            np.array([0.0, 0.0, 0.0, 0.34], dtype=np.float32)
        )
        assert next_obs.shape == (12,)
        assert np.isfinite(next_obs).all()
        assert np.isfinite(reward)
        assert not truncated
        assert "position_xyz_m" in step_info
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(np.array([0, 0, 0, 0.34], dtype=np.float32))
        assert truncated
    finally:
        env.close()
