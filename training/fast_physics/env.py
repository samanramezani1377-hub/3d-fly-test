"""Gymnasium environment for robust control across randomized smooth2 variants."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dynamics import FastQuadrotor
from .curriculum import randomization_for_stage
from .system_identification import OnlineThrustEstimator
from .parameters import (
    DEFAULT_AIRFRAME,
    DEFAULT_CALIBRATION,
    VehicleParameters,
    load_vehicle_parameters,
    sample_scenario,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANDOMIZATION = ROOT / "vehicle_models" / "smooth2_parametric" / "domain_randomization.json"


class AdaptiveDroneEnv(gym.Env):
    """Direct four-rotor control with hidden episode parameters and observation history.

    The policy sees only state estimates and recent control history. Randomized
    mass, CG, inertia, motor effectiveness and voltage are kept in info for
    diagnostics, not appended to the policy observation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        params: VehicleParameters | None = None,
        airframe_path: str | Path = DEFAULT_AIRFRAME,
        calibration_path: str | Path = DEFAULT_CALIBRATION,
        randomization_path: str | Path = DEFAULT_RANDOMIZATION,
        allow_synthetic_calibration: bool = False,
        dt: float = 0.02,
        max_steps: int = 1500,
        target_position=(0.0, 0.0, 1.0),
        history_length: int = 4,
        randomize: bool = True,
        curriculum_stage: int | None = None,
    ):
        super().__init__()
        if params is None:
            params, self.model_metadata = load_vehicle_parameters(
                airframe_path, calibration_path,
                allow_synthetic_calibration=allow_synthetic_calibration,
            )
        else:
            params.validate()
            self.model_metadata = {
                "vehicle_id": "injected-test-or-caller-parameters",
                "calibration_status": "caller-supplied",
                "synthetic_calibration_used": False,
                "missing_fields": [],
            }
        self.base_params = params
        self.randomization = json.loads(Path(randomization_path).read_text(encoding="utf-8"))
        self.curriculum_stage = curriculum_stage
        self.active_randomization = (
            randomization_for_stage(self.randomization, curriculum_stage)
            if curriculum_stage is not None else self.randomization
        )
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.target = np.asarray(target_position, dtype=np.float64)
        self.history_length = int(history_length)
        self.randomize = bool(randomize)
        if self.dt <= 0 or self.max_steps <= 0 or self.history_length < 1:
            raise ValueError("dt, max_steps and history_length must be positive")
        if self.target.shape != (3,):
            raise ValueError("target_position must be a 3-vector")
        self.action_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        # Each history frame contains 12 state values and 4 previous motor commands.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.history_length * 16 + 2,), dtype=np.float32,
        )
        self.plant = FastQuadrotor(self.base_params, dt=self.dt)
        self.estimator = OnlineThrustEstimator()
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._last_action = np.zeros(4, dtype=np.float32)
        self._steps = 0
        self._taken_off = False
        self._scenario_details: dict = {}

    def _policy_frame(self, state: np.ndarray) -> np.ndarray:
        relative = state[:3].astype(np.float32) - self.target.astype(np.float32)
        return np.concatenate((relative, state[3:12], self._last_action)).astype(np.float32)

    def _observation(self) -> np.ndarray:
        if not self._history:
            state = self.plant.state()
            self._history.extend([self._policy_frame(state)] * self.history_length)
        return np.concatenate((*tuple(self._history), self.estimator.context())).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize:
            episode_params, self._scenario_details = sample_scenario(
                self.base_params, self.np_random, self.active_randomization
            )
        else:
            episode_params = self.base_params
            self._scenario_details = {
                "mass_scale": 1.0,
                "center_of_mass_offset_m": [0.0, 0.0, 0.0],
                "inertia_scale_xyz": [1.0, 1.0, 1.0],
                "motor_effectiveness": list(episode_params.motor_effectiveness),
                "voltage_factor": episode_params.voltage_factor,
                "drag_scale": 1.0,
            }
        self.plant = FastQuadrotor(episode_params, dt=self.dt)
        env_cfg = self.active_randomization.get("environment", {})
        wind_bound = float(env_cfg.get("wind_component_m_s", [-2.0, 2.0])[1])
        wind = self.np_random.uniform(-wind_bound, wind_bound, size=3) if self.randomize else np.zeros(3)
        state = self.plant.reset(wind=wind)
        self.estimator.reset(float(state[5]))
        self._history.clear()
        self._last_action = np.zeros(4, dtype=np.float32)
        frame = self._policy_frame(state)
        self._history.extend([frame.copy() for _ in range(self.history_length)])
        self._steps = 0
        self._taken_off = False
        info = {
            "vehicle_id": self.model_metadata["vehicle_id"],
            "calibration_status": self.model_metadata["calibration_status"],
            "synthetic_calibration_used": self.model_metadata["synthetic_calibration_used"],
            "scenario": dict(self._scenario_details),
            "wind_m_s": wind.tolist(),
            "policy_observation_excludes_privileged_parameters": True,
            "curriculum_stage": self.active_randomization.get("active_curriculum_stage"),
        }
        return self._observation(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError("action must contain four motor commands")
        action = np.clip(action, 0.0, 1.0)
        state, telemetry = self.plant.step(action)
        self._steps += 1
        self._taken_off = self._taken_off or bool(state[2] > 0.18)
        self.estimator.update(action, float(state[5]), self.dt)

        position_error = state[:3].astype(np.float64) - self.target
        horizontal_error = float(np.linalg.norm(position_error[:2]))
        altitude_error = abs(float(position_error[2]))
        attitude = state[6:9].astype(np.float64)
        rates = state[9:12].astype(np.float64)
        speed = float(np.linalg.norm(state[3:6]))
        reward = (
            2.0 * np.exp(-altitude_error / 0.45)
            + 1.0 * np.exp(-horizontal_error / 0.65)
            - 0.10 * speed ** 2
            - 0.22 * float(np.dot(attitude, attitude))
            - 0.025 * float(np.dot(rates, rates))
            - 0.015 * float(np.dot(action - self._last_action, action - self._last_action))
        )
        crashed = bool(
            (self._taken_off and state[2] <= 0.025)
            or state[2] > 10.0
            or horizontal_error > 10.0
            or np.max(np.abs(attitude[:2])) > 1.25
            or not np.all(np.isfinite(state))
        )
        if crashed:
            reward -= 100.0
        terminated = crashed
        truncated = self._steps >= self.max_steps and not terminated

        self._last_action = action.copy()
        self._history.append(self._policy_frame(state))
        obs = self._observation()
        info = {
            **telemetry,
            "crashed": crashed,
            "crash_reason": "state_or_boundary_limit" if crashed else None,
            "has_taken_off": self._taken_off,
            "position_error_m": float(np.linalg.norm(position_error)),
            "altitude_error_m": altitude_error,
            "horizontal_error_m": horizontal_error,
            "scenario": dict(self._scenario_details),
            "online_estimator": self.estimator.diagnostics(),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        return None
