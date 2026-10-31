"""Online, observation-only residual/system identification utilities."""
from __future__ import annotations

import numpy as np


class OnlineThrustEstimator:
    """Recursive least-squares estimate of effective collective lift response.

    It estimates the approximate relationship:
        vertical_acceleration + g ~= bias + gain * mean(rotor_command**2)

    Only commanded motor values and observed vertical velocity are used.
    The estimate is an effective response parameter, not a unique mass or
    motor diagnosis. Battery sag, payload, motor loss and disturbances can
    be observationally confounded.
    """

    def __init__(self, gravity_m_s2: float = 9.80665, forgetting_factor: float = 0.995):
        if gravity_m_s2 <= 0 or not 0.9 < forgetting_factor <= 1.0:
            raise ValueError("invalid estimator gravity or forgetting factor")
        self.gravity = float(gravity_m_s2)
        self.forgetting_factor = float(forgetting_factor)
        self.theta = np.array([0.0, 10.0], dtype=np.float64)
        self.covariance = np.eye(2, dtype=np.float64) * 100.0
        self.previous_vertical_velocity: float | None = None
        self.samples = 0
        self.residual_ema = 0.0

    def reset(self, vertical_velocity_m_s: float = 0.0) -> None:
        self.theta[:] = (0.0, 10.0)
        self.covariance[:] = np.eye(2) * 100.0
        self.previous_vertical_velocity = float(vertical_velocity_m_s)
        self.samples = 0
        self.residual_ema = 0.0

    def update(self, motor_commands, vertical_velocity_m_s: float, dt: float) -> np.ndarray:
        commands = np.asarray(motor_commands, dtype=np.float64)
        if commands.shape != (4,) or dt <= 0:
            raise ValueError("four motor commands and positive dt are required")
        velocity = float(vertical_velocity_m_s)
        if not np.isfinite(velocity):
            return self.context()
        if self.previous_vertical_velocity is None:
            self.previous_vertical_velocity = velocity
            return self.context()

        acceleration = (velocity - self.previous_vertical_velocity) / float(dt)
        self.previous_vertical_velocity = velocity
        collective = float(np.mean(np.clip(commands, 0.0, 1.0) ** 2))
        phi = np.array([1.0, collective], dtype=np.float64)
        target = acceleration + self.gravity

        p_phi = self.covariance @ phi
        denominator = self.forgetting_factor + float(phi @ p_phi)
        gain = p_phi / max(denominator, 1e-12)
        residual = target - float(phi @ self.theta)
        self.theta += gain * residual
        self.covariance = (
            self.covariance - np.outer(gain, phi @ self.covariance)
        ) / self.forgetting_factor
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.samples += 1
        self.residual_ema = 0.95 * self.residual_ema + 0.05 * abs(residual)
        return self.context()

    def context(self) -> np.ndarray:
        gain_estimate = float(np.clip(self.theta[1], 0.0, 100.0))
        uncertainty = float(np.sqrt(max(self.covariance[1, 1], 0.0)))
        confidence = (self.samples / (self.samples + 25.0)) / (1.0 + uncertainty / 10.0)
        return np.array([
            np.clip(gain_estimate / 30.0, 0.0, 3.333),
            np.clip(confidence, 0.0, 1.0),
        ], dtype=np.float32)

    def diagnostics(self) -> dict:
        return {
            "effective_lift_gain_m_s2": float(self.theta[1]),
            "vertical_acceleration_bias_m_s2": float(self.theta[0]),
            "confidence": float(self.context()[1]),
            "residual_ema_m_s2": float(self.residual_ema),
            "samples": int(self.samples),
        }


def estimate_first_order_time_constant(time_s, target_speed_rad_s, measured_speed_rad_s) -> dict:
    """Fit a first-order motor speed lag from command-target and RPM logs.

    Inputs must be synchronized samples in seconds and rad/s. The fit uses
    d(omega)/dt = (omega_target - omega)/tau. It estimates actuator lag only;
    it does not infer thrust, torque, current, or propeller performance.
    """
    time = np.asarray(time_s, dtype=np.float64)
    target = np.asarray(target_speed_rad_s, dtype=np.float64)
    measured = np.asarray(measured_speed_rad_s, dtype=np.float64)
    if time.ndim != 1 or target.shape != time.shape or measured.shape != time.shape or time.size < 5:
        raise ValueError("time, target speed and measured speed must be equal-length 1D arrays with at least five samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(measured)):
        raise ValueError("motor response samples must be finite")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time samples must be strictly increasing")
    drive = target - measured
    derivative = np.gradient(measured, time)
    mask = np.abs(drive) > max(float(np.max(np.abs(drive))) * 1e-3, 1e-6)
    if int(mask.sum()) < 3:
        raise ValueError("motor response log does not contain enough excitation")
    denominator = float(np.dot(drive[mask], drive[mask]))
    rate = float(np.dot(drive[mask], derivative[mask]) / denominator)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("first-order motor response fit is not physically positive")
    predicted = rate * drive[mask]
    residual = derivative[mask] - predicted
    total = derivative[mask] - float(np.mean(derivative[mask]))
    r_squared = 1.0 - float(np.dot(residual, residual)) / max(float(np.dot(total, total)), 1e-12)
    return {
        "motor_time_constant_s": float(1.0 / rate),
        "fit_r_squared": float(r_squared),
        "sample_count": int(mask.sum()),
        "model": "d_omega_dt=(omega_target-omega)/tau",
        "scope": "motor_speed_response_only_no_thrust_or_propeller_inference",
    }


def estimate_body_axis_drag_coefficients(relative_air_velocity_body_m_s, drag_force_body_n) -> dict:
    """Fit per-axis linear/quadratic drag from synchronized body-frame samples.

    drag_force_body_n must be the isolated aerodynamic force after gravity,
    rotor force and other known forces are removed. This function does not
    infer drag from raw acceleration without those force terms.
    """
    velocity = np.asarray(relative_air_velocity_body_m_s, dtype=np.float64)
    force = np.asarray(drag_force_body_n, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 3 or force.shape != velocity.shape or velocity.shape[0] < 8:
        raise ValueError("velocity and drag force must be matching Nx3 arrays with at least eight samples")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(force)):
        raise ValueError("drag identification samples must be finite")
    linear, quadratic, scores = [], [], []
    for axis in range(3):
        speed = velocity[:, axis]
        design = np.column_stack((-speed, -np.abs(speed) * speed))
        if np.linalg.matrix_rank(design) < 2:
            raise ValueError(f"axis {axis}: velocity samples do not excite both linear and quadratic drag terms")
        coefficients, _, _, _ = np.linalg.lstsq(design, force[:, axis], rcond=None)
        if np.any(coefficients < -1e-10):
            raise ValueError(f"axis {axis}: fitted drag coefficients are negative; check force isolation and frame")
        prediction = design @ coefficients
        residual = force[:, axis] - prediction
        centered = force[:, axis] - float(np.mean(force[:, axis]))
        r_squared = 1.0 - float(np.dot(residual, residual)) / max(float(np.dot(centered, centered)), 1e-12)
        linear.append(float(max(coefficients[0], 0.0)))
        quadratic.append(float(max(coefficients[1], 0.0)))
        scores.append(float(r_squared))
    return {
        "linear_drag_body_kg_s": linear,
        "quadratic_drag_body_kg_m": quadratic,
        "fit_r_squared_by_axis": scores,
        "sample_count": int(velocity.shape[0]),
        "coordinate_frame": "BODY_FORWARD_LEFT_UP",
        "scope": "isolated_aerodynamic_drag_only_requires_known_force_subtraction",
    }
