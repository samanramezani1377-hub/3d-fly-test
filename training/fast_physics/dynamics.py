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


def _quaternion_from_euler(angles: np.ndarray) -> np.ndarray:
    """Return a normalized body-to-world quaternion in [w, x, y, z] order."""
    roll, pitch, yaw = (float(v) * 0.5 for v in angles)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def _euler_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


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
        previous_motor_omega = self.motor_omega.copy()
        # Exact discretization of a first-order motor-speed response.
        alpha = -np.expm1(-self.dt / p.motor_time_constant_s)
        self.motor_omega += alpha * (target_omega - self.motor_omega)
        motor_omega_dot = (self.motor_omega - previous_motor_omega) / self.dt

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

        # The attitude vector is the public state contract. Reconstructing q
        # here also keeps direct reset-time attitude perturbations synchronized.
        quaternion = _quaternion_from_euler(self.attitude)
        rot = _rotation_matrix_from_quaternion(quaternion)
        air_velocity_world = self.velocity - self.wind
        air_velocity_body = rot.T @ air_velocity_world
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
        drag_world = rot @ drag_body
        force_world = rot @ total_force_body + drag_world
        force_world[2] -= p.mass_kg * self.gravity
        acceleration = force_world / p.mass_kg
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt

        inertia = p.inertia_matrix
        old_body_rates = self.angular_velocity.copy()
        angular_momentum = inertia @ self.angular_velocity
        spin_signs = np.asarray(p.spin_signs, dtype=np.float64)
        rotor_spin_momentum = p.rotor_inertia_kg_m2 * float(np.dot(spin_signs, self.motor_omega))
        rotor_angular_momentum = np.array([0.0, 0.0, rotor_spin_momentum])
        rotor_angular_acceleration = p.rotor_inertia_kg_m2 * float(np.dot(spin_signs, motor_omega_dot))
        rotor_gyro_torque = (
            -np.cross(self.angular_velocity, rotor_angular_momentum)
            - np.array([0.0, 0.0, rotor_angular_acceleration])
        )
        angular_acceleration = np.linalg.solve(
            inertia,
            total_torque_body
            + rotor_gyro_torque
            - np.cross(self.angular_velocity, angular_momentum)
            - p.angular_damping_nms * self.angular_velocity,
        )
        self.angular_velocity += angular_acceleration * self.dt

        # Integrate attitude on SO(3) with a unit quaternion and the midpoint
        # body rate. This avoids Euler-angle rate singularities in the plant.
        midpoint_rates = 0.5 * (old_body_rates + self.angular_velocity)
        rotation_speed = float(np.linalg.norm(midpoint_rates))
        rotation_angle = rotation_speed * self.dt
        if rotation_angle > 1e-12:
            half = 0.5 * rotation_angle
            delta_q = np.concatenate((
                [np.cos(half)],
                midpoint_rates * (np.sin(half) / rotation_speed),
            ))
            quaternion = _quaternion_multiply(quaternion, delta_q)
            quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        self.attitude = _euler_from_quaternion(quaternion)
        self.attitude = (self.attitude + np.pi) % (2.0 * np.pi) - np.pi
        self.time += self.dt

        return self.state(), {
            "rotor_thrust_n": thrusts.astype(float).tolist(),
            "rotor_omega_rad_s": self.motor_omega.astype(float).tolist(),
            "total_thrust_n": float(thrusts.sum()),
            "body_force_n": total_force_body.astype(float).tolist(),
            "body_torque_nm": total_torque_body.astype(float).tolist(),
            "rotor_gyroscopic_torque_nm": rotor_gyro_torque.astype(float).tolist(),
            "drag_world_n": drag_world.astype(float).tolist(),
            "linear_acceleration_world_m_s2": acceleration.astype(float).tolist(),
            "angular_acceleration_body_rad_s2": angular_acceleration.astype(float).tolist(),
            "attitude_quaternion_wxyz": quaternion.astype(float).tolist(),
            "mass_kg": float(p.mass_kg),
            "center_of_mass_m": list(p.center_of_mass_m),
            "inertia_kg_m2": list(p.inertia_kg_m2),
            "time_s": float(self.time),
        }
