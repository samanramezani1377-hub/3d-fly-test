"""Vectorized-friendly, deterministic 6-DoF rigid-body quadrotor plant.

This is a fast research model, not CFD. It uses first-order motor lag,
quadratic rotor thrust, reaction torque, rigid-body Euler equations and
quadratic translational drag. All physical coefficients are explicit inputs.
"""
from __future__ import annotations

import numpy as np

from .parameters import VehicleParameters


def rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Body-to-world rotation for Z-up ENU and intrinsic roll/pitch/yaw."""
    roll, pitch, yaw = (float(v) for v in angles)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


class FastQuadrotor:
    """Small deterministic plant with explicit per-episode vehicle parameters."""

    def __init__(self, params: VehicleParameters, dt: float = 0.02, gravity_m_s2: float = 9.80665):
        params.validate()
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.params = params
        self.dt = float(dt)
        self.gravity = float(gravity_m_s2)
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.attitude = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.motor_omega = np.zeros(4, dtype=np.float64)
        self.wind = np.zeros(3, dtype=np.float64)
        self.time = 0.0

    def reset(self, position=(0.0, 0.0, 0.08), wind=(0.0, 0.0, 0.0)) -> np.ndarray:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity[:] = 0.0
        self.attitude[:] = 0.0
        self.angular_velocity[:] = 0.0
        self.motor_omega[:] = 0.0
        self.wind = np.asarray(wind, dtype=np.float64).copy()
        if self.position.shape != (3,) or self.wind.shape != (3,):
            raise ValueError("position and wind must be 3-vectors")
        self.time = 0.0
        return self.state()

    def state(self) -> np.ndarray:
        return np.concatenate((self.position, self.velocity, self.attitude, self.angular_velocity)).astype(np.float32)

    def step(self, motor_commands: np.ndarray) -> tuple[np.ndarray, dict]:
        command = np.asarray(motor_commands, dtype=np.float64)
        if command.shape != (4,):
            raise ValueError("motor_commands must contain four normalized rotor commands")
        command = np.clip(command, 0.0, 1.0)
        p = self.params
        target_omega = command * p.max_rotor_speed_rad_s * p.voltage_factor
        alpha = min(self.dt / p.motor_time_constant_s, 1.0)
        self.motor_omega += alpha * (target_omega - self.motor_omega)

        thrusts = (
            p.thrust_coefficient_n_per_rad_s2
            * self.motor_omega ** 2
            * np.asarray(p.motor_effectiveness, dtype=np.float64)
        )
        rotor_positions = np.asarray(p.rotor_positions_m, dtype=np.float64) - np.asarray(
            p.center_of_mass_m, dtype=np.float64
        )
        forces_body = np.zeros((4, 3), dtype=np.float64)
        forces_body[:, 2] = thrusts
        torques_body = np.cross(rotor_positions, forces_body)
        torques_body[:, 2] += (
            np.asarray(p.spin_signs, dtype=np.float64)
            * p.torque_to_thrust_m
            * thrusts
        )
        total_force_body = forces_body.sum(axis=0)
        total_torque_body = torques_body.sum(axis=0)

        rot = rotation_matrix(self.attitude)
        air_velocity = self.velocity - self.wind
        speed = float(np.linalg.norm(air_velocity))
        drag_world = (
            -p.linear_drag_kg_s * air_velocity
            -p.quadratic_drag_kg_m * speed * air_velocity
        )
        force_world = rot @ total_force_body + drag_world
        force_world[2] -= p.mass_kg * self.gravity
        acceleration = force_world / p.mass_kg
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt

        inertia = p.inertia_matrix
        angular_momentum = inertia @ self.angular_velocity
        angular_acceleration = np.linalg.solve(
            inertia,
            total_torque_body
            - np.cross(self.angular_velocity, angular_momentum)
            - p.angular_damping_nms * self.angular_velocity,
        )
        self.angular_velocity += angular_acceleration * self.dt

        roll, pitch, _ = self.attitude
        pitch_cos = max(abs(float(np.cos(pitch))), 1e-3)
        roll_dot = (
            self.angular_velocity[0]
            + np.sin(roll) * np.tan(pitch) * self.angular_velocity[1]
            + np.cos(roll) * np.tan(pitch) * self.angular_velocity[2]
        )
        pitch_dot = (
            np.cos(roll) * self.angular_velocity[1]
            - np.sin(roll) * self.angular_velocity[2]
        )
        yaw_dot = (
            np.sin(roll) / pitch_cos * self.angular_velocity[1]
            + np.cos(roll) / pitch_cos * self.angular_velocity[2]
        )
        self.attitude += np.array([roll_dot, pitch_dot, yaw_dot]) * self.dt
        self.attitude = (self.attitude + np.pi) % (2.0 * np.pi) - np.pi
        self.time += self.dt

        return self.state(), {
            "rotor_thrust_n": thrusts.astype(float).tolist(),
            "rotor_omega_rad_s": self.motor_omega.astype(float).tolist(),
            "total_thrust_n": float(thrusts.sum()),
            "mass_kg": float(p.mass_kg),
            "center_of_mass_m": list(p.center_of_mass_m),
            "inertia_kg_m2": list(p.inertia_kg_m2),
            "time_s": float(self.time),
        }
