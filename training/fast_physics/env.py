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
        self._last_finite_state = np.zeros(12, dtype=np.float32)
        self._previous_altitude_potential = 0.0
        self._ground_contact = False

    def _measure_state(self, state: np.ndarray) -> np.ndarray:
        """Return the sensor-visible state; never pass simulator truth to the policy estimator."""
        measured = state.astype(np.float64).copy()
        noise = self.active_randomization.get("environment", {}).get("sensor_noise_std", {})
        if self.randomize and isinstance(noise, dict):
            std = np.array(
                [noise.get("position_m", 0.0)] * 3
                + [noise.get("velocity_m_s", 0.0)] * 3
                + [noise.get("attitude_rad", 0.0)] * 3
                + [noise.get("angular_rate_rad_s", 0.0)] * 3,
                dtype=np.float64,
            )
            measured += self.np_random.normal(0.0, std, size=12)
        return measured

    def _policy_frame(self, state: np.ndarray, measured: np.ndarray | None = None) -> np.ndarray:
        """Return dimensionless, bounded-scale state features for PPO."""
        if measured is None:
            measured = self._measure_state(state)
        relative = (measured[:3].astype(np.float32) - self.target.astype(np.float32)) / np.array(
            [2.0, 2.0, 2.0], dtype=np.float32
        )
        velocity = measured[3:6].astype(np.float32) / 5.0
        attitude = measured[6:9].astype(np.float32) / np.pi
        angular_rate = measured[9:12].astype(np.float32) / 10.0
        previous_motor_commands = 2.0 * self._last_action.astype(np.float32) - 1.0
        return np.concatenate(
            (relative, velocity, attitude, angular_rate, previous_motor_commands)
        ).astype(np.float32)

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
        wind_bounds = env_cfg.get("wind_component_m_s", [-2.0, 2.0])
        wind_low, wind_high = float(wind_bounds[0]), float(wind_bounds[1])
        if wind_low > wind_high:
            raise ValueError("environment.wind_component_m_s lower bound must not exceed upper bound")
        wind = self.np_random.uniform(wind_low, wind_high, size=3) if self.randomize else np.zeros(3)
        state = self.plant.reset(wind=wind)
        # Progressive ground-start perturbations: retain the takeoff task while
        # gradually teaching recovery from small initial attitude/position errors.
        stage_info = self.active_randomization.get("active_curriculum_stage") or {}
        stage_number = int(stage_info.get("stage", 1))
        perturbation = {
            1: (0.00, 0.00), 2: (0.02, 0.02), 3: (0.04, 0.04),
            4: (0.07, 0.07), 5: (0.10, 0.10),
        }.get(stage_number, (0.0, 0.0))
        position_bound_m, attitude_bound_rad = perturbation
        if self.randomize and position_bound_m > 0:
            self.plant.position[:2] = self.np_random.uniform(
                -position_bound_m, position_bound_m, size=2
            )
            self.plant.attitude[:2] = self.np_random.uniform(
                -attitude_bound_rad, attitude_bound_rad, size=2
            )
            state = self.plant.state()
        measured = self._measure_state(state)
        self.estimator.reset(float(measured[5]))
        self._history.clear()
        self._last_action = np.zeros(4, dtype=np.float32)
        frame = self._policy_frame(state, measured)
        self._history.extend([frame.copy() for _ in range(self.history_length)])
        self._steps = 0
        self._taken_off = False
        self._last_finite_state = state.astype(np.float32).copy()
        self._previous_altitude_potential = float(np.exp(-abs(float(state[2]) - float(self.target[2])) / 0.45))
        self._ground_contact = False
        info = {
            "vehicle_id": self.model_metadata["vehicle_id"],
            "calibration_status": self.model_metadata["calibration_status"],
            "synthetic_calibration_used": self.model_metadata["synthetic_calibration_used"],
            "scenario": dict(self._scenario_details),
            "wind_m_s": wind.tolist(),
            "policy_observation_excludes_privileged_parameters": True,
            "curriculum_stage": self.active_randomization.get("active_curriculum_stage"),
            "initial_position_xy_m": self.plant.position[:2].astype(float).tolist(),
            "initial_roll_pitch_rad": self.plant.attitude[:2].astype(float).tolist(),
            "initial_state_perturbation": {
                "position_bound_m": position_bound_m,
                "attitude_bound_rad": attitude_bound_rad,
            },
        }
        return self._observation(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError("action must contain four motor commands")
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)
        state, telemetry = self.plant.step(action)
        self._steps += 1

        state_finite = bool(np.all(np.isfinite(state)))
        crash_reasons = []
        if not state_finite:
            crash_reasons.append("non_finite_state")
            # Never pass NaN/Inf into the reward, estimator, observation or PPO buffer.
            safe_state = self._last_finite_state.copy()
        else:
            safe_state = state.astype(np.float32, copy=True)
            self._last_finite_state = safe_state.copy()

        z = float(safe_state[2])
        was_taken_off = self._taken_off
        self._taken_off = self._taken_off or bool(z > 0.18)
        measured = self._measure_state(safe_state)
        ground_contact = bool(state_finite and z <= 0.0)
        if ground_contact:
            self.plant.position[2] = 0.0
            if self.plant.velocity[2] < 0.0:
                self.plant.velocity[2] = 0.0
            safe_state = self.plant.state()
            self._last_finite_state = safe_state.copy()
            z = 0.0
            measured = self._measure_state(safe_state)
        if state_finite:
            self.estimator.update(action, float(measured[5]), self.dt)

        position_error = safe_state[:3].astype(np.float64) - self.target
        horizontal_error = float(np.linalg.norm(position_error[:2]))
        altitude_error = abs(float(position_error[2]))
        attitude = safe_state[6:9].astype(np.float64)
        rates = safe_state[9:12].astype(np.float64)
        speed = float(np.linalg.norm(safe_state[3:6]))
        tilt = float(np.max(np.abs(attitude[:2])))

        # Dense objective: position/altitude tracking, low translational and
        # angular motion, level attitude, and smooth motor commands.
        base_reward = (
            2.0 * np.exp(-altitude_error / 0.45)
            + 1.0 * np.exp(-horizontal_error / 0.65)
            - 0.10 * speed ** 2
            - 0.22 * float(np.dot(attitude[:2], attitude[:2]))
            - 0.025 * float(np.dot(rates, rates))
            - 0.015 * float(np.dot(action - self._last_action, action - self._last_action))
            - 0.01 * float(np.dot(action, action))
        )
        # Potential-based altitude progress shaping. It rewards progress toward
        # target altitude, not merely remaining airborne or accumulating time.
        altitude_potential = float(np.exp(-altitude_error / 0.45))
        reward = base_reward + 0.5 * (
            0.99 * altitude_potential - self._previous_altitude_potential
        )
        self._previous_altitude_potential = altitude_potential
        if self._taken_off and not was_taken_off:
            reward += 1.0
        stable_hover_now = bool(
            self._taken_off
            and altitude_error <= 0.15
            and horizontal_error <= 0.20
            and speed <= 0.35
            and tilt <= 0.25
            and float(np.linalg.norm(rates)) <= 0.5
        )
        if stable_hover_now:
            reward += 0.5

        # Unilateral ground contact prevents the simulated vehicle from passing
        # through z=0. This is not a detailed landing or impact-damage model.
        self._ground_contact = ground_contact

        if ground_contact and self._taken_off:
            crash_reasons.append("ground_impact_after_takeoff")
        elif ground_contact and not self._taken_off:
            # Contact before takeoff is not a crash; the episode may continue.
            pass
        if z > 10.0:
            crash_reasons.append("altitude_limit_exceeded")
        if horizontal_error > 10.0:
            crash_reasons.append("horizontal_boundary_exceeded")
        if tilt > 1.25:
            crash_reasons.append("attitude_limit_exceeded")

        crashed = bool(crash_reasons)
        if crashed:
            reward = -100.0 if "non_finite_state" in crash_reasons else float(reward - 100.0)

        terminated = crashed
        truncated = self._steps >= self.max_steps and not terminated
        takeoff_failed = bool(truncated and not self._taken_off)
        if takeoff_failed:
            reward -= 20.0
        self._last_action = action.copy()
        self._history.append(self._policy_frame(safe_state, measured))
        obs = self._observation()
        if not np.all(np.isfinite(obs)):
            # Last-resort observation sanitation; mark the transition as invalid.
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
            if "non_finite_observation" not in crash_reasons:
                crash_reasons.append("non_finite_observation")
            crashed = terminated = True
            truncated = False
            reward = -100.0

        info = {
            **telemetry,
            "crashed": crashed,
            "crash_reason": crash_reasons[0] if crash_reasons else None,
            "crash_reasons": crash_reasons,
            "has_taken_off": self._taken_off,
            "takeoff_failed": takeoff_failed,
            "ground_contact": ground_contact,
            "altitude_m": float(z),
            "position_xyz_m": safe_state[:3].astype(float).tolist(),
            "position_error_m": float(np.linalg.norm(position_error)),
            "altitude_error_m": altitude_error,
            "horizontal_error_m": horizontal_error,
            "scenario": dict(self._scenario_details),
            "online_estimator": self.estimator.diagnostics(),
            "state_velocity_m_s": safe_state[3:6].astype(float).tolist(),
            "attitude_rad": safe_state[6:9].astype(float).tolist(),
            "angular_velocity_rad_s": safe_state[9:12].astype(float).tolist(),
            "state_finite": state_finite,
            "control_effort_proxy": float(np.mean(np.square(action))),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        return None
