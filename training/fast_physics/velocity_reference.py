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
        dt_s: float = 0.02, velocity_kp=(2.8, 2.8, 3.2),
        velocity_ki=(0.35, 0.35, 0.45), velocity_integral_limit=(1.25, 1.25, 1.0),
        position_kp=(1.15, 1.15, 1.55), attitude_kp=(14.0, 14.0, 0.0),
        attitude_kd=(7.0, 7.0, 0.0), yaw_rate_kp=5.0, yaw_rate_kd=0.45,
        max_horizontal_accel_m_s2=4.5, max_vertical_accel_m_s2=3.5,
        max_tilt_rad=0.48, max_yaw_accel_rad_s2=10.0,
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
        self.velocity_ki = np.asarray(velocity_ki, dtype=np.float64)
        self.velocity_integral_limit = np.asarray(velocity_integral_limit, dtype=np.float64)
        self.velocity_integral = np.zeros(3, dtype=np.float64)
        self._previous_command = np.zeros(4, dtype=np.float64)
        self.position_kp = np.asarray(position_kp, dtype=np.float64)
        self.attitude_kp = np.asarray(attitude_kp, dtype=np.float64)
        self.attitude_kd = np.asarray(attitude_kd, dtype=np.float64)
        self.yaw_rate_kp, self.yaw_rate_kd = float(yaw_rate_kp), float(yaw_rate_kd)
        self.max_horizontal_accel = float(max_horizontal_accel_m_s2)
        self.max_vertical_accel = float(max_vertical_accel_m_s2)
        self.max_tilt, self.max_yaw_accel = float(max_tilt_rad), float(max_yaw_accel_rad_s2)
        if self.wind.shape != (3,) or any(
            x.shape != (3,) for x in
            (self.velocity_kp, self.velocity_ki, self.velocity_integral_limit,
             self.position_kp, self.attitude_kp, self.attitude_kd)
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
        velocity_error = desired_velocity_world - velocity

        # Reset accumulated bias when the pilot changes direction/speed sharply;
        # otherwise an old command's integral can fight the new command.
        command = np.concatenate((command_body, [float(desired_yaw_rate_rad_s)]))
        if np.linalg.norm(command - self._previous_command) > 0.35:
            self.velocity_integral *= 0.35
        self._previous_command = command

        # Integrate only when the requested acceleration is not already
        # saturated in the same direction (conditional anti-windup).
        candidate_integral = np.clip(
            self.velocity_integral + velocity_error * self.dt,
            -self.velocity_integral_limit, self.velocity_integral_limit,
        )

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
        drag_accel_world = -(rot @ drag_body) / p.mass_kg
        accel_without_integral = self.velocity_kp * velocity_error + drag_accel_world
        accel_candidate = accel_without_integral + self.velocity_ki * candidate_integral

        # Derive the available tilt/acceleration from this episode's actual
        # thrust-to-weight margin. A controller must not request a tilt that
        # leaves insufficient thrust to support the vehicle.
        thrust_to_weight = self.max_collective_thrust_n / max(p.mass_kg * 9.80665, 1e-9)
        if thrust_to_weight <= 1.0:
            tilt_authority = 0.0
        else:
            tilt_authority = float(np.arccos(np.clip(1.04 / thrust_to_weight, -1.0, 1.0)))
        effective_tilt = min(self.max_tilt, tilt_authority)
        horizontal_limit = min(
            self.max_horizontal_accel,
            9.80665 * np.tan(max(effective_tilt, 0.0)),
        )
        vertical_up_limit = max(0.0, self.max_collective_thrust_n / p.mass_kg - 9.80665)
        vertical_limit = min(self.max_vertical_accel, max(vertical_up_limit, 0.15))

        # Conditional anti-windup independently on horizontal and vertical axes.
        if (np.linalg.norm(accel_candidate[:2]) > horizontal_limit
                and float(np.dot(velocity_error[:2], accel_candidate[:2])) > 0.0):
            candidate_integral[:2] = self.velocity_integral[:2]
        if (abs(float(accel_candidate[2])) > vertical_limit
                and velocity_error[2] * accel_candidate[2] > 0.0):
            candidate_integral[2] = self.velocity_integral[2]
        self.velocity_integral = candidate_integral
        accel_world = accel_without_integral + self.velocity_ki * self.velocity_integral

        horizontal_norm = float(np.linalg.norm(accel_world[:2]))
        if horizontal_norm > horizontal_limit and horizontal_norm > 1e-9:
            accel_world[:2] *= horizontal_limit / horizontal_norm
        accel_world[2] = float(np.clip(accel_world[2], -self.max_vertical_accel, vertical_limit))

        cy, sy = np.cos(yaw), np.sin(yaw)
        accel_body_x = cy * accel_world[0] + sy * accel_world[1]
        accel_body_y = -sy * accel_world[0] + cy * accel_world[1]
        desired_pitch = float(np.clip(
            np.arctan2(accel_body_x, 9.80665), -self.max_tilt, self.max_tilt
        ))
        desired_roll = float(np.clip(
            -np.arctan2(accel_body_y, 9.80665), -self.max_tilt, self.max_tilt
        ))

        # Roll/pitch stabilization plus independent world-vertical yaw-rate control.
        desired_attitude = np.array([desired_roll, desired_pitch, attitude[2]])
        # Geometric SO(3) attitude error avoids treating Euler roll/pitch
        # differences as independent when the vehicle is yawed or tilted.
        desired_rotation = rotation_matrix(desired_attitude)
        current_rotation = rotation_matrix(attitude)
        relative_rotation = current_rotation.T @ desired_rotation
        attitude_error = 0.5 * np.array([
            relative_rotation[2, 1] - relative_rotation[1, 2],
            relative_rotation[0, 2] - relative_rotation[2, 0],
            relative_rotation[1, 0] - relative_rotation[0, 1],
        ], dtype=np.float64)
        desired_angular_accel = (
            self.attitude_kp * attitude_error - self.attitude_kd * rates
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
        # Bound yaw acceleration by the available counter-rotating rotor
        # reaction torque and the episode's actual yaw inertia.
        max_thrust = self.max_collective_thrust_n
        positive = np.asarray(p.spin_signs) > 0
        negative = ~positive
        per_rotor_max = (
            self.max_collective_thrust_n
            * np.asarray(p.motor_effectiveness, dtype=np.float64)
            / max(float(np.sum(p.motor_effectiveness)), 1e-9)
        )
        positive_torque = float(np.sum(per_rotor_max[positive])) * p.torque_to_thrust_m
        negative_torque = float(np.sum(per_rotor_max[negative])) * p.torque_to_thrust_m
        yaw_torque_authority = min(positive_torque, negative_torque)
        yaw_accel_authority = 0.80 * yaw_torque_authority / max(float(p.inertia_matrix[2, 2]), 1e-9)
        effective_yaw_accel = min(self.max_yaw_accel, yaw_accel_authority)
        desired_angular_accel[2] = np.clip(
            self.yaw_rate_kp * body_r_error - self.yaw_rate_kd * rates[2],
            -effective_yaw_accel, effective_yaw_accel,
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
