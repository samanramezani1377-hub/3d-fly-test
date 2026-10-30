"""Privileged-parameter cascaded PID reference controller for hover diagnostics.

This controller intentionally receives the episode's true randomized parameters.
It is an oracle baseline for checking plant controllability and reward/learning
problems, NOT a deployable controller or a fair comparison with observation-only PPO.
"""
from __future__ import annotations

import numpy as np

from .dynamics import rotation_matrix
from .parameters import VehicleParameters


class HoverReferenceController:
    """Cascaded position/attitude controller with a four-rotor control allocator."""

    def __init__(
        self,
        params: VehicleParameters,
        *,
        target_position=(0.0, 0.0, 1.0),
        wind_m_s=(0.0, 0.0, 0.0),
        position_kp=(1.8, 1.8, 4.0),
        position_kd=(3.0, 3.0, 3.2),
        position_ki=(0.65, 0.65, 0.35),
        integral_limit=(2.5, 2.5, 1.5),
        dt_s=0.02,
        attitude_kp=(12.0, 12.0, 4.0),
        attitude_kd=(6.0, 6.0, 3.0),
        max_tilt_rad=0.40,
        max_horizontal_accel_m_s2=3.0,
    ):
        params.validate()
        self.params = params
        self.target = np.asarray(target_position, dtype=np.float64)
        self.wind = np.asarray(wind_m_s, dtype=np.float64)
        self.position_kp = np.asarray(position_kp, dtype=np.float64)
        self.position_kd = np.asarray(position_kd, dtype=np.float64)
        self.position_ki = np.asarray(position_ki, dtype=np.float64)
        self.integral_limit = np.asarray(integral_limit, dtype=np.float64)
        self.dt = float(dt_s)
        self.position_integral = np.zeros(3, dtype=np.float64)
        self.attitude_kp = np.asarray(attitude_kp, dtype=np.float64)
        self.attitude_kd = np.asarray(attitude_kd, dtype=np.float64)
        self.max_tilt = float(max_tilt_rad)
        self.max_horizontal_accel = float(max_horizontal_accel_m_s2)
        if self.target.shape != (3,) or self.wind.shape != (3,) or self.max_tilt <= 0 or self.dt <= 0:
            raise ValueError("target_position must be a 3-vector; tilt and dt must be positive")
        if any(v.shape != (3,) for v in (self.position_kp, self.position_kd, self.position_ki, self.integral_limit)):
            raise ValueError("position controller gains and integral limits must be 3-vectors")
        self.allocation = self._build_allocation_matrix()

    def _build_allocation_matrix(self) -> np.ndarray:
        p = self.params
        rotor_positions = np.asarray(p.rotor_positions_m, dtype=np.float64)
        com = np.asarray(p.center_of_mass_m, dtype=np.float64)
        arms = rotor_positions - com
        # Map individual rotor thrusts to [collective, roll, pitch, yaw] body wrench.
        matrix = np.vstack((
            np.ones(4),
            arms[:, 1],
            -arms[:, 0],
            np.asarray(p.spin_signs, dtype=np.float64) * p.torque_to_thrust_m,
        ))
        if np.linalg.matrix_rank(matrix) < 4:
            raise ValueError("rotor geometry does not provide a full-rank four-axis allocation")
        return matrix

    def desired_wrench(self, state: np.ndarray) -> np.ndarray:
        """Return oracle [collective, roll, pitch, yaw] wrench for a plant state."""
        s = np.asarray(state, dtype=np.float64)
        if s.shape != (12,) or not np.all(np.isfinite(s)):
            raise ValueError("state must be a finite 12-value vector")
        p = self.params
        position, velocity = s[:3], s[3:6]
        attitude, rates = s[6:9], s[9:12]
        position_error = self.target - position
        self.position_integral = np.clip(
            self.position_integral + position_error * self.dt,
            -self.integral_limit, self.integral_limit,
        )
        accel_world = (
            self.position_kp * position_error
            - self.position_kd * velocity
            + self.position_ki * self.position_integral
        )
        # Oracle feed-forward: cancel the plant drag implied by the known wind.
        air_velocity = velocity - self.wind
        air_speed = float(np.linalg.norm(air_velocity))
        drag_compensation = (
            p.linear_drag_kg_s * air_velocity
            + p.quadratic_drag_kg_m * air_speed * air_velocity
        ) / p.mass_kg
        accel_world += drag_compensation
        accel_world[:2] = np.clip(accel_world[:2], -self.max_horizontal_accel, self.max_horizontal_accel)

        yaw = float(attitude[2])
        cy, sy = np.cos(yaw), np.sin(yaw)
        accel_body_x = cy * accel_world[0] + sy * accel_world[1]
        accel_body_y = -sy * accel_world[0] + cy * accel_world[1]
        desired_pitch = np.clip(np.arctan2(accel_body_x, 9.80665), -self.max_tilt, self.max_tilt)
        desired_roll = np.clip(-np.arctan2(accel_body_y, 9.80665), -self.max_tilt, self.max_tilt)
        desired_attitude = np.array([desired_roll, desired_pitch, 0.0])
        attitude_error = desired_attitude - attitude
        attitude_error[2] = (attitude_error[2] + np.pi) % (2.0 * np.pi) - np.pi

        inertia = p.inertia_matrix
        angular_momentum = inertia @ rates
        desired_angular_accel = self.attitude_kp * attitude_error - self.attitude_kd * rates
        torque = (
            inertia @ desired_angular_accel
            + np.cross(rates, angular_momentum)
            + p.angular_damping_nms * rates
        )

        rot = rotation_matrix(attitude)
        vertical_cosine = max(float(rot[2, 2]), 0.35)
        collective = p.mass_kg * (9.80665 + accel_world[2]) / vertical_cosine
        collective = float(np.clip(collective, 0.0, 4.0 * p.thrust_coefficient_n_per_rad_s2
                                   * (p.max_rotor_speed_rad_s * p.voltage_factor) ** 2
                                   * max(p.motor_effectiveness)))
        return np.array([collective, torque[0], torque[1], torque[2]], dtype=np.float64)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Return four normalized motor commands for the legacy oracle benchmark."""
        desired_wrench = self.desired_wrench(state)
        rotor_thrust = np.linalg.solve(self.allocation, desired_wrench)
        max_thrust = self.params.thrust_coefficient_n_per_rad_s2 * (
            self.params.max_rotor_speed_rad_s * self.params.voltage_factor
        ) ** 2 * np.asarray(self.params.motor_effectiveness, dtype=np.float64)
        rotor_thrust = np.clip(rotor_thrust, 0.0, max_thrust)
        omega = np.sqrt(rotor_thrust / (
            self.params.thrust_coefficient_n_per_rad_s2
            * np.asarray(self.params.motor_effectiveness, dtype=np.float64)
        ))
        commands = omega / (self.params.max_rotor_speed_rad_s * self.params.voltage_factor)
        return np.clip(commands, 0.0, 1.0).astype(np.float32)
