"""Privileged velocity and yaw-rate reference controller for PPO teacher guidance.

Tracks nose-relative 3D velocity and world-vertical yaw rate while following the
moving virtual position target. True episode parameters/state are used only by
the oracle action wrapper; randomized simulator ground truth is not added to PPO
observations.
"""
from __future__ import annotations

import numpy as np

from .dynamics import rotation_matrix
from .parameters import VehicleParameters
from .propeller import BladeElementPropeller


class VelocityYawReferenceController:
    """Cascaded velocity/position-to-attitude controller with yaw-rate feedback."""

    def __init__(
        self, params: VehicleParameters, *, wind_m_s=(0.0, 0.0, 0.0),
        dt_s: float = 0.02, velocity_kp=(2.4, 2.4, 2.8),
        position_kp=(1.0, 1.0, 1.4), attitude_kp=(12.0, 12.0, 0.0),
        attitude_kd=(6.0, 6.0, 0.0), yaw_rate_kp=4.0, yaw_rate_kd=0.35,
        max_horizontal_accel_m_s2=3.0, max_vertical_accel_m_s2=3.0,
        max_tilt_rad=0.40, max_yaw_accel_rad_s2=8.0,
    ):
        params.validate()
        self.params = params
        self.wind = np.asarray(wind_m_s, dtype=np.float64)
        if params.propeller_bemt:
            model = BladeElementPropeller(params.propeller_bemt)
            reference_kf = float(params.propeller_bemt.get(
                "reference_thrust_coefficient_n_per_rad_s2",
                params.thrust_coefficient_n_per_rad_s2,
            ))
            scale = params.thrust_coefficient_n_per_rad_s2 / reference_kf
            self.max_collective_thrust_n = float(
                model.evaluate(
                    params.max_rotor_speed_rad_s * params.voltage_factor, 0.0
                )[0] * scale * np.sum(params.motor_effectiveness)
            )
        else:
            self.max_collective_thrust_n = (
                4.0 * params.thrust_coefficient_n_per_rad_s2
                * (params.max_rotor_speed_rad_s * params.voltage_factor) ** 2
                * max(params.motor_effectiveness)
            )
        self.dt = float(dt_s)
        self.velocity_kp = np.asarray(velocity_kp, dtype=np.float64)
        self.position_kp = np.asarray(position_kp, dtype=np.float64)
        self.attitude_kp = np.asarray(attitude_kp, dtype=np.float64)
        self.attitude_kd = np.asarray(attitude_kd, dtype=np.float64)
        self.yaw_rate_kp, self.yaw_rate_kd = float(yaw_rate_kp), float(yaw_rate_kd)
        self.max_horizontal_accel = float(max_horizontal_accel_m_s2)
        self.max_vertical_accel = float(max_vertical_accel_m_s2)
        self.max_tilt, self.max_yaw_accel = float(max_tilt_rad), float(max_yaw_accel_rad_s2)
        if self.wind.shape != (3,) or any(
            x.shape != (3,) for x in
            (self.velocity_kp, self.position_kp, self.attitude_kp, self.attitude_kd)
        ):
            raise ValueError("wind and controller gains must be 3-vectors")
        if self.dt <= 0 or min(
            self.max_horizontal_accel, self.max_vertical_accel,
            self.max_tilt, self.max_yaw_accel
        ) <= 0:
            raise ValueError("controller timestep and acceleration/tilt limits must be positive")

    @staticmethod
    def _body_yaw_velocity_to_world(velocity_body: np.ndarray, yaw: float) -> np.ndarray:
        cy, sy = np.cos(float(yaw)), np.sin(float(yaw))
        forward, left, up = np.asarray(velocity_body, dtype=np.float64)
        return np.array([cy * forward - sy * left, sy * forward + cy * left, up])

    def desired_wrench(
        self, state: np.ndarray, *, target_position: np.ndarray,
        desired_velocity_body: np.ndarray, desired_yaw_rate_rad_s: float,
    ) -> np.ndarray:
        """Return the oracle [collective, roll, pitch, yaw] wrench."""
        s = np.asarray(state, dtype=np.float64)
        target = np.asarray(target_position, dtype=np.float64)
        command_body = np.asarray(desired_velocity_body, dtype=np.float64)
        if (s.shape != (12,) or target.shape != (3,) or command_body.shape != (3,)
                or not np.all(np.isfinite(s)) or not np.all(np.isfinite(target))
                or not np.all(np.isfinite(command_body))
                or not np.isfinite(desired_yaw_rate_rad_s)):
            raise ValueError("teacher state, target and velocity command must be finite")
        p = self.params
        position, velocity = s[:3], s[3:6]
        attitude, rates = s[6:9], s[9:12]
        yaw = float(attitude[2])

        # Horizontal commands rotate with the nose; vertical is world-up.
        commanded_velocity_world = self._body_yaw_velocity_to_world(command_body, yaw)
        position_error = target - position
        desired_velocity_world = commanded_velocity_world + self.position_kp * position_error
        accel_world = self.velocity_kp * (desired_velocity_world - velocity)

        # Privileged feed-forward compensation for the plant's body-axis drag.
        rot = rotation_matrix(attitude)
        air_velocity_body = rot.T @ (velocity - self.wind)
        linear_drag = np.asarray(
            p.linear_drag_body_kg_s
            if p.linear_drag_body_kg_s is not None
            else (p.linear_drag_kg_s,) * 3,
            dtype=np.float64,
        )
        quadratic_drag = np.asarray(
            p.quadratic_drag_body_kg_m
            if p.quadratic_drag_body_kg_m is not None
            else (p.quadratic_drag_kg_m,) * 3,
            dtype=np.float64,
        )
        drag_body = (
            -linear_drag * air_velocity_body
            -quadratic_drag * np.abs(air_velocity_body) * air_velocity_body
        )
        accel_world -= (rot @ drag_body) / p.mass_kg
        accel_world[:2] = np.clip(
            accel_world[:2], -self.max_horizontal_accel, self.max_horizontal_accel
        )
        accel_world[2] = np.clip(
            accel_world[2], -self.max_vertical_accel, self.max_vertical_accel
        )

        cy, sy = np.cos(yaw), np.sin(yaw)
        accel_body_x = cy * accel_world[0] + sy * accel_world[1]
        accel_body_y = -sy * accel_world[0] + cy * accel_world[1]
        desired_pitch = float(np.clip(
            np.arctan2(accel_body_x, 9.80665), -self.max_tilt, self.max_tilt
        ))
        desired_roll = float(np.clip(
            np.arctan2(accel_body_y, 9.80665), -self.max_tilt, self.max_tilt
        ))

        # Roll/pitch stabilization plus independent world-vertical yaw-rate control.
        desired_attitude = np.array([desired_roll, desired_pitch, attitude[2]])
        desired_angular_accel = (
            self.attitude_kp * (desired_attitude - attitude) - self.attitude_kd * rates
        )
        cos_pitch = max(abs(float(np.cos(attitude[1]))), 1e-3)
        cos_roll = max(abs(float(np.cos(attitude[0]))), 1e-3)
        measured_yaw_rate = (
            np.sin(attitude[0]) / cos_pitch * rates[1]
            + np.cos(attitude[0]) / cos_pitch * rates[2]
        )
        # Convert the requested world-vertical Euler yaw rate into the body-r
        # rate required at the current roll/pitch. Controlling r directly as
        # though it were yaw-dot is only correct when the vehicle is level.
        desired_body_r = (
            cos_pitch / cos_roll
            * (
                float(desired_yaw_rate_rad_s)
                - np.sin(attitude[0]) / cos_pitch * rates[1]
            )
        )
        body_r_error = desired_body_r - rates[2]
        desired_angular_accel[2] = np.clip(
            self.yaw_rate_kp * body_r_error - self.yaw_rate_kd * rates[2],
            -self.max_yaw_accel, self.max_yaw_accel,
        )

        inertia = p.inertia_matrix
        angular_momentum = inertia @ rates
        torque = (
            inertia @ desired_angular_accel
            + np.cross(rates, angular_momentum)
            + p.angular_damping_nms * rates
        )
        rot = rotation_matrix(attitude)
        vertical_cosine = max(float(rot[2, 2]), 0.35)
        collective = p.mass_kg * (9.80665 + accel_world[2]) / vertical_cosine
        collective = float(np.clip(
            collective, 0.0,
            self.max_collective_thrust_n,
        ))
        return np.array([collective, torque[0], torque[1], torque[2]], dtype=np.float64)
