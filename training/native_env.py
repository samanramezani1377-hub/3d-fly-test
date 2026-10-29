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

    Observation: world-Godot [x,y,z,vx,vy,vz] + body-FRD [roll,pitch,yaw,wx,wy,wz].
    Action is symmetric [-1, 1] on every dimension so PPO does not clip a
    zero-centered Gaussian throttle distribution to zero. Throttle is mapped
    to 0.20..0.50 around a configurable throttle bias. The default 0.35 is not assumed to be a calibrated hover throttle.
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
        self.core.skysim_rotor_thrust.argtypes = [ptr, ctypes.c_int]
        self.core.skysim_rotor_thrust.restype = dbl
        self.core.skysim_rotor_omega.argtypes = [ptr, ctypes.c_int]
        self.core.skysim_rotor_omega.restype = dbl
        self.handle = self.core.skysim_create()
        if not self.handle:
            raise RuntimeError("skysim_create returned a null pointer")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self._step_count = 0
        self._closed = False
        self.ground_height = 0.16
        self.ground_contact_height = 0.22
        self.takeoff_altitude = 0.45
        self._has_taken_off = False
        self._ground_wait_steps = 0
        self._flight_phase = "grounded"

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
        # Start at ground level; takeoff is the first learning stage.
        self.core.skysim_reset(self.handle, 0.0, self.ground_height, 0.0)
        wind = self._config.get("wind_mps", {})
        self.core.skysim_set_wind(
            self.handle, float(wind.get("x", 0)), float(wind.get("y", 0)), float(wind.get("z", 0))
        )
        self.core.skysim_arm(self.handle, 1)
        self._step_count = 0
        self._has_taken_off = False
        self._ground_wait_steps = 0
        self._flight_phase = "grounded"
        obs = self._observation()
        self._previous_altitude_error = abs(float(self.target[1] - obs[1]))
        return obs, {
            "target_xyz_m": self.target.tolist(),
            "backend": "native-drone-core",
            "flight_phase": self._flight_phase,
            "has_taken_off": self._has_taken_off,
        }

    @staticmethod
    def calculate_reward(
        position,
        velocity,
        attitude,
        angular_velocity,
        target,
        action,
        throttle,
        throttle_center=0.35,
        has_taken_off=True,
        ground_wait_steps=0,
        takeoff_bonus=0.0,
        altitude_progress_reward=0.0,
    ):
        """Return a shaped reward and transparent signed component breakdown.

        Exponential proximity scores stay non-negative so a distant but safe
        state is not made artificially worse than terminating early in a crash.
        Hovering at the target still earns substantially more than remaining
        grounded, while velocity and attitude penalties discourage instability.
        """
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        attitude = np.asarray(attitude, dtype=np.float64)
        angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        position_error = position - target
        altitude_error = abs(float(position_error[1]))
        horizontal_error = float(np.linalg.norm(position_error[[0, 2]]))
        altitude_score = float(np.exp(-altitude_error / 0.75))
        horizontal_score = float(np.exp(-horizontal_error / 1.0))

        # Keep every non-crashed state reward non-negative at rest. The old
        # centered scores made distant states strongly negative, so PPO could
        # improve discounted return by ending an episode in a crash rather than
        # continuing to control the vehicle. Absolute proximity scores preserve
        # a clear gradient toward the target while making survival valuable.
        altitude_term = 2.0 * altitude_score
        horizontal_term = 0.5 * horizontal_score
        # Penalize runaway vertical/horizontal speed without a hard cap.
        # Previously, 4 m/s and 8 m/s vertical speed received the same penalty.
        abs_vertical_speed = abs(float(velocity[1]))
        vertical_speed_penalty = (
            0.12 * abs_vertical_speed ** 2
            + 0.20 * max(abs_vertical_speed - 2.0, 0.0) ** 2
        )
        horizontal_speed_sq = float(velocity[0] ** 2 + velocity[2] ** 2)
        horizontal_speed = float(np.sqrt(horizontal_speed_sq))
        horizontal_speed_penalty = (
            0.04 * horizontal_speed_sq
            + 0.08 * max(horizontal_speed - 2.0, 0.0) ** 2
        )
        attitude_penalty = 0.08 * min(float(np.dot(attitude, attitude)), 9.0)
        angular_rate_penalty = 0.03 * min(float(np.dot(angular_velocity, angular_velocity)), 25.0)
        action_penalty = 0.01 * float(np.dot(action[:3], action[:3]))
        throttle_penalty = 0.05 * float((float(throttle) - throttle_center) ** 2)

        ground_contact = bool(has_taken_off and position[1] <= 0.22)
        ceiling_contact = bool(position[1] > 12.0)
        horizontal_boundary = bool(np.linalg.norm(position[[0, 2]]) > 12.0)
        crashed = bool(ground_contact or ceiling_contact or horizontal_boundary)
        crash_reason = None
        if ground_contact:
            crash_reason = "ground_contact_after_takeoff"
        elif ceiling_contact:
            crash_reason = "ceiling"
        elif horizontal_boundary:
            crash_reason = "horizontal_boundary"

        crash_penalty = 100.0 if crashed else 0.0
        # A small bounded delay cost discourages idle resets without making
        # the grounded state itself strongly negative (which can reward early
        # terminal crashes under discounted return).
        ground_wait_penalty = min(0.005 * max(int(ground_wait_steps), 0), 0.05) if not has_taken_off else 0.0
        reward = (
            altitude_term
            + horizontal_term
            + float(takeoff_bonus)
            + float(altitude_progress_reward)
            - ground_wait_penalty
            - vertical_speed_penalty
            - horizontal_speed_penalty
            - attitude_penalty
            - angular_rate_penalty
            - action_penalty
            - throttle_penalty
            - crash_penalty
        )
        components = {
            "altitude_term": float(altitude_term),
            "horizontal_term": float(horizontal_term),
            "vertical_speed_penalty": float(-vertical_speed_penalty),
            "horizontal_speed_penalty": float(-horizontal_speed_penalty),
            "attitude_penalty": float(-attitude_penalty),
            "angular_rate_penalty": float(-angular_rate_penalty),
            "action_penalty": float(-action_penalty),
            "throttle_penalty": float(-throttle_penalty),
            "crash_penalty": float(-crash_penalty),
            "ground_wait_penalty": float(-ground_wait_penalty),
            "takeoff_bonus": float(takeoff_bonus),
            "altitude_progress_reward": float(altitude_progress_reward),
            "altitude_error_m": float(altitude_error),
            "horizontal_error_m": float(horizontal_error),
            "altitude_score": float(altitude_score),
            "horizontal_score": float(horizontal_score),
        }
        return float(reward), components, crashed, crash_reason

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
        altitude_error = abs(float(position_error[1]))
        horizontal_error = float(np.sqrt(horizontal_error_sq))
        angular_velocity = obs[9:12].astype(np.float64)

        # Latch takeoff exactly once. A later ground contact is a crash.
        newly_took_off = bool(
            not self._has_taken_off and position[1] >= self.takeoff_altitude
        )
        if newly_took_off:
            self._has_taken_off = True
            self._ground_wait_steps = 0
            self._flight_phase = "airborne"
        elif not self._has_taken_off:
            self._ground_wait_steps += 1
            self._flight_phase = "grounded"
        elif position[1] <= self.ground_contact_height:
            self._flight_phase = "crashed"
        elif (
            altitude_error <= 0.35
            and horizontal_error <= 0.5
            and float(np.linalg.norm(velocity)) <= 0.5
        ):
            self._flight_phase = "hovering"
        else:
            self._flight_phase = "airborne"

        altitude_progress_reward = float(np.clip(
            1.5 * (self._previous_altitude_error - altitude_error), -0.5, 0.5
        ))
        reward, reward_components, crashed, crash_reason = self.calculate_reward(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            target=self.target,
            action=action,
            throttle=throttle,
            throttle_center=self.throttle_center,
            has_taken_off=self._has_taken_off,
            ground_wait_steps=self._ground_wait_steps,
            altitude_progress_reward=altitude_progress_reward,
            takeoff_bonus=(
                2.0
                if (
                    newly_took_off
                    and float(np.linalg.norm(attitude)) <= 0.6
                    and horizontal_error <= 1.0
                    and abs(float(velocity[1])) <= 2.0
                )
                else 0.0
            ),
        )
        self._previous_altitude_error = altitude_error
        if crashed:
            self._flight_phase = "crashed"

        terminated = crashed
        truncated = self._step_count >= self.max_steps
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
            "flight_phase": self._flight_phase,
            "has_taken_off": self._has_taken_off,
            "ground_wait_steps": self._ground_wait_steps,
            "takeoff_altitude_m": self.takeoff_altitude,
            "hover_stable": hover_stable,
            "reward_components": reward_components,
            "reward_total": float(reward),
            "angular_velocity_xyz_rads": angular_velocity.tolist(),
            "backend": "native-drone-core",
            "telemetry_altitude_m": float(self.core.skysim_altitude(self.handle)),
            "rotor_thrust_n": [float(self.core.skysim_rotor_thrust(self.handle, i)) for i in range(4)],
            "rotor_omega_rads": [float(self.core.skysim_rotor_omega(self.handle, i)) for i in range(4)],
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if not self._closed:
            self.core.skysim_destroy(self.handle)
            self._closed = True


class PolicyObservationWrapper(gym.ObservationWrapper):
    """Deterministically scale state features and express position relative to the target.

    The underlying environment and info dictionaries retain physical SI units.
    Only the observation sent to the policy is transformed, so evaluation metrics
    and diagnostics continue to use physical measurements.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32).copy()
        # Position becomes target-relative; normalize heterogeneous SI units.
        obs[0:3] = (obs[0:3] - self.env.target.astype(np.float32)) / 12.0
        obs[3:6] = obs[3:6] / 8.0
        obs[6:9] = obs[6:9] / np.pi
        obs[9:12] = obs[9:12] / 10.0
        if obs.shape != (12,) or not np.all(np.isfinite(obs)):
            raise ValueError(f"Invalid policy observation: shape={obs.shape}, values={obs}")
        return obs
