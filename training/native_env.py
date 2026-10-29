"""Gymnasium environment over the project's native C++ DroneCore build."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NativeSkySimEnv(gym.Env):
    """Hover task backed by the native DroneCore C API.

    Observation: [x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz].
    Action is symmetric [-1, 1] on every dimension so PPO does not clip a
    zero-centered Gaussian throttle distribution to zero. Throttle is mapped
    to 0..0.70 with 0.35 as the initial hover estimate from the control sweep.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        library_path=None,
        target=(0.0, 3.0, 0.0),
        max_steps=1500,
        dt=1.0 / 50.0,
        max_tilt_rad=0.55,
        max_yaw_rate_rads=3.14,
        throttle_center=0.35,
        throttle_span=0.15,
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
        if not 0.0 < throttle_center < 1.0 or not 0.0 < throttle_span <= 1.0:
            raise ValueError("throttle_center must be in (0,1) and throttle_span in (0,1]")
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_yaw_rate_rads = float(max_yaw_rate_rads)
        self.throttle_center = float(throttle_center)
        self.throttle_span = float(throttle_span)

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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self._step_count = 0
        self._closed = False

    @staticmethod
    def map_action(action, throttle_center=0.35, throttle_span=0.15):
        """Map PPO's symmetric action vector into simulator controls."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,):
            raise ValueError("action must be [roll, pitch, yaw_rate, throttle]")
        action = np.clip(action, -1.0, 1.0)
        throttle = float(np.clip(throttle_center + throttle_span * action[3], 0.0, 1.0))
        return action, throttle

    def _observation(self):
        out = (ctypes.c_double * 12)()
        self.core.skysim_get_obs(self.handle, out)
        return np.asarray(list(out), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.core.skysim_reset(self.handle, 0.0, 2.0, 0.0)
        wind = self._config.get("wind_mps", {})
        self.core.skysim_set_wind(
            self.handle, float(wind.get("x", 0)), float(wind.get("y", 0)), float(wind.get("z", 0))
        )
        self.core.skysim_arm(self.handle, 1)
        self._step_count = 0
        return self._observation(), {"target_xyz_m": self.target.tolist(), "backend": "native-drone-core"}

    def step(self, action):
        action, throttle = self.map_action(action, self.throttle_center, self.throttle_span)
        self.core.skysim_set_attitude(
            self.handle,
            float(action[0] * self.max_tilt_rad),
            float(action[1] * self.max_tilt_rad),
            float(action[2] * self.max_yaw_rate_rads),
            throttle,
        )
        self.core.skysim_step(self.handle, self.dt)
        self._step_count += 1
        obs = self._observation()

        position = obs[0:3].astype(np.float64)
        velocity = obs[3:6].astype(np.float64)
        attitude = obs[6:9].astype(np.float64)
        position_error = position - self.target
        horizontal_error_sq = float(position_error[0] ** 2 + position_error[2] ** 2)
        crashed = bool(
            position[1] <= 0.16
            or position[1] > 12.0
            or np.linalg.norm(position[[0, 2]]) > 12.0
        )

        # Bounded shaping prevents squared position error from dominating every
        # timestep and makes sustained proximity to the target more valuable
        # than briefly surviving near the ground. The terminal crash penalty
        # remains large relative to any single step reward.
        altitude_error = abs(float(position_error[1]))
        horizontal_error = float(np.sqrt(horizontal_error_sq))
        altitude_score = float(np.exp(-altitude_error / 0.75))
        horizontal_score = float(np.exp(-horizontal_error / 1.0))
        vertical_speed_penalty = 0.12 * min(float(velocity[1] ** 2), 16.0)
        horizontal_speed_sq = float(velocity[0] ** 2 + velocity[2] ** 2)
        horizontal_speed_penalty = 0.04 * min(horizontal_speed_sq, 25.0)
        attitude_penalty = 0.08 * min(float(np.dot(attitude, attitude)), 9.0)
        action_penalty = 0.01 * float(np.dot(action[:3], action[:3]))
        throttle_penalty = 0.05 * float((throttle - self.throttle_center) ** 2)
        reward = (
            0.05
            + 2.0 * altitude_score
            + 0.5 * horizontal_score
            - vertical_speed_penalty
            - horizontal_speed_penalty
            - attitude_penalty
            - action_penalty
            - throttle_penalty
        )
        if crashed:
            reward -= 100.0

        terminated = crashed
        truncated = self._step_count >= self.max_steps
        crash_reason = None
        if position[1] <= 0.16:
            crash_reason = "ground_contact"
        elif position[1] > 12.0:
            crash_reason = "ceiling"
        elif np.linalg.norm(position[[0, 2]]) > 12.0:
            crash_reason = "horizontal_boundary"
        speed_norm = float(np.linalg.norm(velocity))
        hover_stable = bool(
            altitude_error <= 0.35
            and horizontal_error <= 0.5
            and speed_norm <= 0.5
        )
        info = {
            "position_xyz_m": position.tolist(),
            "velocity_xyz_mps": velocity.tolist(),
            "target_xyz_m": self.target.tolist(),
            "altitude_error_m": float(position_error[1]),
            "horizontal_error_m": horizontal_error,
            "speed_norm_mps": speed_norm,
            "throttle": throttle,
            "crashed": crashed,
            "crash_reason": crash_reason,
            "hover_stable": hover_stable,
            "backend": "native-drone-core",
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if not self._closed:
            self.core.skysim_destroy(self.handle)
            self._closed = True
