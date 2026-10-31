"""Vectorized runtime blade-element/momentum propeller model for Smooth2.

Uses the pinned CAD blade stations and local axial inflow. This is a fast
screening model, not CFD or thrust-stand calibration.
"""
from __future__ import annotations

import math
import numpy as np


class BladeElementPropeller:
    def __init__(self, config: dict):
        self.radius_m = float(config["radius_m"])
        self.hub_radius_m = float(config.get("hub_radius_m", 0.0))
        self.blade_count = int(config["blade_count"])
        stations = np.asarray(config["stations"], dtype=np.float64)
        self.rho = float(config.get("air_density_kg_m3", 1.225))
        self.alpha0 = math.radians(float(config.get("zero_lift_angle_deg", -4.15448)))
        self.cl_limit = float(config.get("cl_limit", 1.2))
        self.cd0 = float(config.get("cd0", 0.012))
        self.cd_k = float(config.get("cd_k", 0.01))
        self.n_integral = int(config.get("integration_sections", 120))
        if self.radius_m <= self.hub_radius_m or self.blade_count < 2:
            raise ValueError("invalid propeller radius, hub radius or blade count")
        if stations.ndim != 2 or stations.shape[1] != 3 or len(stations) < 3:
            raise ValueError("propeller stations must be [r/R, chord_m, pitch_deg] rows")
        if np.any(np.diff(stations[:, 0]) <= 0) or stations[0, 0] <= 0 or stations[-1, 0] > 1.000001:
            raise ValueError("propeller radial stations must increase from root to tip")
        if self.rho <= 0 or self.n_integral < 8:
            raise ValueError("invalid air density or integration resolution")
        r0 = max(self.hub_radius_m, stations[0, 0] * self.radius_m)
        dr = (self.radius_m - r0) / self.n_integral
        self.r = r0 + (np.arange(self.n_integral, dtype=np.float64) + 0.5) * dr
        self.dr = dr
        radial = stations[:, 0]
        self.chord = np.interp(self.r / self.radius_m, radial, stations[:, 1])
        self.beta = np.radians(np.interp(self.r / self.radius_m, radial, stations[:, 2]))
        tip_factor = 0.5 * self.blade_count * (self.radius_m - self.r) / self.r
        self.tip_factor = tip_factor

    def evaluate(self, omega_rad_s: float, axial_inflow_m_s: float = 0.0) -> tuple[float, float]:
        """Return positive thrust (N) and shaft torque magnitude (Nm)."""
        omega = max(float(omega_rad_s), 0.0)
        if omega < 1e-6:
            return 0.0, 0.0
        vt = omega * self.r
        axial = float(axial_inflow_m_s)

        def section(induced):
            va = axial + induced
            phi = np.arctan2(va, vt)
            vrel2 = vt * vt + va * va
            cl = np.clip(
                2.0 * math.pi * (self.beta - phi - self.alpha0),
                -self.cl_limit, self.cl_limit,
            )
            cd = self.cd0 + self.cd_k * cl * cl
            dynamic = 0.5 * self.rho * vrel2 * self.blade_count * self.chord
            d_thrust = dynamic * (cl * np.cos(phi) - cd * np.sin(phi))
            d_torque = dynamic * (cl * np.sin(phi) + cd * np.cos(phi)) * self.r
            sin_phi = np.maximum(np.abs(np.sin(phi)), 1e-8)
            f_arg = self.tip_factor / sin_phi
            loss = np.maximum(
                2.0 / math.pi * np.arccos(np.minimum(1.0, np.exp(-f_arg))),
                0.05,
            )
            momentum = 4.0 * math.pi * self.rho * self.r * induced * (induced + axial) * loss
            return d_thrust - momentum, d_thrust, d_torque

        lo = np.zeros_like(self.r)
        hi = np.full_like(self.r, max(0.6 * omega * self.radius_m, 1.0))
        residual_lo = section(lo)[0]
        active = residual_lo > 0.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            residual = section(mid)[0]
            above = residual > 0.0
            lo = np.where(active & above, mid, lo)
            hi = np.where(active & ~above, mid, hi)
        induced = np.where(active, 0.5 * (lo + hi), 0.0)
        _, d_thrust, d_torque = section(induced)
        thrust = float(np.sum(d_thrust) * self.dr)
        torque = float(np.sum(d_torque) * self.dr)
        return max(thrust, 0.0), max(torque, 0.0)
