"""Gymnasium environment over the same native C++ DroneCore sources used for WASM."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NativeSkySimEnv(gym.Env):
    """Hover task backed directly by the project's native DroneCore shared library.

    Action: normalized roll, pitch, yaw-rate, throttle.
    Observation: [x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz] in core SI/radian units.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        library_path=None,
        target=(0.0, 3.0, 0.0),
        max_steps=1500,
        dt=1.0 / 50.0,
        max_tilt_rad=0.785,
        max_yaw_rate_rads=3.14,
    ):
        super().__init__()
        self.library_path = Path(library_path or Path(__file__).with_name("libskysim_core.so")).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(
                f"Native core not found: {self.library_path}. Run "
                "'python training/build_native_core.py' first."
            )
        self.target = np.asarray(target, dtype=np.float64)
        if self.target.shape != (3,):
            raise ValueError("target must contain exactly three coordinates")
        if max_steps <= 0 or dt <= 0:
            raise ValueError("max_steps and dt must be positive")
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_yaw_rate_rads = float(max_yaw_rate_rads)

        self.core = ctypes.CDLL(str(self.library_path))
        ptr = ctypes.c_void_p
        dbl = ctypes.c_double
        self.core.skysim_create.restype = ptr
        self.core.skysim_destroy.argtypes = [ptr]
        self.core.skysim_reset.argtypes = [ptr, dbl, dbl, dbl]
        self.core.skysim_arm.argtypes = [ptr, ctypes.c_int]
        self.core.skysim_set_attitude.argtypes = [ptr, dbl, dbl, dbl, dbl]
        self.core.skysim_set_wind.argtypes = [ptr, dbl, dbl, dbl]
        self._config_path = Path(__file__).resolve().parents[1] / "config" / "drone.json"
        self._config = json.loads(self._config_path.read_text(encoding="utf-8"))
        self.core.skysim_step.argtypes = [ptr, dbl]
        self.core.skysim_get_obs.argtypes = [ptr, ctypes.POINTER(dbl)]
        self.core.skysim_altitude.argtypes = [ptr]
        self.core.skysim_altitude.restype = dbl
        self.handle = self.core.skysim_create()
        if not self.handle:
            raise RuntimeError("skysim_create returned a null pointer")

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self._step_count = 0
        self._closed = False

    def _observation(self):
        out = (ctypes.c_double * 12)()
        self.core.skysim_get_obs(self.handle, out)
        return np.asarray(list(out), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset to a fixed initial state for reproducible comparisons.
        self.core.skysim_reset(self.handle, 0.0, 2.0, 0.0)
        wind = self._config.get("wind_mps", {})
        self.core.skysim_set_wind(self.handle, float(wind.get("x", 0)), float(wind.get("y", 0)), float(wind.get("z", 0)))
        self.core.skysim_arm(self.handle, 1)
        self._step_count = 0
        obs = self._observation()
        return obs, {"target_xyz_m": self.target.tolist(), "backend": "native-drone-core"}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,):
            raise ValueError("action must be [roll, pitch, yaw_rate, throttle]")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.core.skysim_set_attitude(
            self.handle,
            float(action[0] * self.max_tilt_rad),
            float(action[1] * self.max_tilt_rad),
            float(action[2] * self.max_yaw_rate_rads),
            float(action[3]),
        )
        self.core.skysim_step(self.handle, self.dt)
        self._step_count += 1
        obs = self._observation()

        position = obs[0:3].astype(np.float64)
        velocity = obs[3:6].astype(np.float64)
        attitude = obs[6:9].astype(np.float64)
        position_error = position - self.target
        reward = (
            -1.0 * float(np.dot(position_error, position_error))
            -0.08 * float(np.dot(velocity, velocity))
            -0.15 * float(np.dot(attitude, attitude))
            -0.002 * float(np.dot(action[:3], action[:3]))
        )
        crashed = bool(position[1] <= 0.16 or position[1] > 12.0 or np.linalg.norm(position[[0, 2]]) > 12.0)
        terminated = crashed
        truncated = self._step_count >= self.max_steps
        info = {
            "position_xyz_m": position.tolist(),
            "velocity_xyz_mps": velocity.tolist(),
            "target_xyz_m": self.target.tolist(),
            "crashed": crashed,
            "backend": "native-drone-core",
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if not self._closed:
            self.core.skysim_destroy(self.handle)
            self._closed = True
