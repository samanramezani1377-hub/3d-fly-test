"""Runtime blade-element/momentum propeller model for Smooth2.

This is a deterministic screening model, not CFD or a thrust-stand calibration.
It evaluates the CAD blade stations at each rotor speed and axial inflow.
"""
from __future__ import annotations

import math
import numpy as np


def _lift_drag(alpha: float, alpha0: float, cl_limit: float, cd0: float, cd_k: float):
    cl = float(np.clip(2.0 * math.pi * (alpha - alpha0), -cl_limit, cl_limit))
    return cl, cd0 + cd_k * cl * cl


class BladeElementPropeller:
    def __init__(self, config: dict):
        self.radius_m = float(config["radius_m"])
        self.hub_radius_m = float(config.get("hub_radius_m", 0.0))
        self.blade_count = int(config["blade_count"])
        self.stations = np.asarray(config["stations"], dtype=np.float64)
        self.rho = float(config.get("air_density_kg_m3", 1.225))
        self.alpha0 = math.radians(float(config.get("zero_lift_angle_deg", -4.15448)))
        self.cl_limit = float(config.get("cl_limit", 1.2))
        self.cd0 = float(config.get("cd0", 0.012))
        self.cd_k = float(config.get("cd_k", 0.01))
        self.n_integral = int(config.get("integration_sections", 48))
        if self.radius_m <= self.hub_radius_m or self.blade_count < 2:
            raise ValueError("invalid propeller radius, hub radius or blade count")
        if self.stations.ndim != 2 or self.stations.shape[1] != 3 or len(self.stations) < 3:
            raise ValueError("propeller stations must be [r/R, chord_m, pitch_deg] rows")
        if np.any(np.diff(self.stations[:, 0]) <= 0) or self.stations[0, 0] <= 0 or self.stations[-1, 0] > 1.000001:
            raise ValueError("propeller radial stations must increase from root to tip")
        if self.rho <= 0 or self.n_integral < 8:
            raise ValueError("invalid air density or integration resolution")

    def evaluate(self, omega_rad_s: float, axial_inflow_m_s: float = 0.0) -> tuple[float, float]:
        """Return positive thrust (N) and shaft torque magnitude (Nm)."""
        omega = max(float(omega_rad_s), 0.0)
        if omega < 1e-6:
            return 0.0, 0.0
        r0 = max(self.hub_radius_m, self.stations[0, 0] * self.radius_m)
        dr = (self.radius_m - r0) / self.n_integral
        thrust = torque = 0.0
        radial = self.stations[:, 0]
        chord = self.stations[:, 1]
        pitch = np.radians(self.stations[:, 2])
        for idx in range(self.n_integral):
            r = r0 + (idx + 0.5) * dr
            x = r / self.radius_m
            c = float(np.interp(x, radial, chord))
            beta = float(np.interp(x, radial, pitch))
            vt = omega * r

            def section(vi):
                va = axial_inflow_m_s + vi
                phi = math.atan2(va, vt)
                v2 = vt * vt + va * va
                cl, cd = _lift_drag(beta - phi, self.alpha0, self.cl_limit, self.cd0, self.cd_k)
                d_t = 0.5 * self.rho * v2 * self.blade_count * c * (cl * math.cos(phi) - cd * math.sin(phi))
                d_q = 0.5 * self.rho * v2 * self.blade_count * c * (cl * math.sin(phi) + cd * math.cos(phi)) * r
                f_arg = self.blade_count * 0.5 * (self.radius_m - r) / max(r * abs(math.sin(phi)), 1e-8)
                loss = max(2.0 / math.pi * math.acos(min(1.0, math.exp(-f_arg))), 0.05)
                momentum = 4.0 * math.pi * self.rho * r * vi * (vi + axial_inflow_m_s) * loss
                return d_t - momentum, d_t, d_q

            lo, hi = 0.0, max(0.6 * omega * self.radius_m, 1.0)
            flo = section(lo)[0]
            if flo <= 0.0:
                _, dt, dq = section(lo)
            else:
                for _ in range(36):
                    mid = 0.5 * (lo + hi)
                    if section(mid)[0] > 0.0:
                        lo = mid
                    else:
                        hi = mid
                _, dt, dq = section(0.5 * (lo + hi))
            thrust += dt * dr
            torque += dq * dr
        return max(float(thrust), 0.0), max(float(torque), 0.0)
