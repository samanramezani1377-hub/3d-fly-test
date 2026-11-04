"""Smooth2 CAD blade-element/momentum propeller model.

The numerical model retains configured radial integration resolution, all 36
induced-inflow bisection iterations, Prandtl tip loss and axial inflow. Numba
accelerates the same equations without reducing physical resolution.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _section_terms(induced, axial, omega, r, chord, beta, tip_factor,
                   rho, blade_count, alpha0, cl_limit, cd0, cd_k):
    va = axial + induced
    vt = omega * r
    phi = math.atan2(va, vt)
    vrel2 = vt * vt + va * va
    cl = min(max(2.0 * math.pi * (beta - phi - alpha0), -cl_limit), cl_limit)
    cd = cd0 + cd_k * cl * cl
    dynamic = 0.5 * rho * vrel2 * blade_count * chord
    d_thrust = dynamic * (cl * math.cos(phi) - cd * math.sin(phi))
    d_torque = dynamic * (cl * math.sin(phi) + cd * math.cos(phi)) * r
    sin_phi = max(abs(math.sin(phi)), 1e-8)
    f_arg = tip_factor / sin_phi
    loss = max(2.0 / math.pi * math.acos(min(1.0, math.exp(-f_arg))), 0.05)
    momentum = 4.0 * math.pi * rho * r * induced * (induced + axial) * loss
    return d_thrust - momentum, d_thrust, d_torque


if njit is not None:
    _section_terms_jit = njit(cache=True, fastmath=False)(_section_terms)

    @njit(cache=True, fastmath=False)
    def _bemt_evaluate_jit(omega, axial, r, chord, beta, tip_factor, dr,
                           rho, blade_count, alpha0, cl_limit, cd0, cd_k):
        n = r.shape[0]
        lo = np.zeros(n, dtype=np.float64)
        hi = np.full(n, max(0.6 * omega * (r[-1] + 0.5 * dr), 1.0), dtype=np.float64)
        active = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            residual, _, _ = _section_terms_jit(
                0.0, axial, omega, r[i], chord[i], beta[i], tip_factor[i],
                rho, blade_count, alpha0, cl_limit, cd0, cd_k
            )
            active[i] = residual > 0.0
        for _ in range(36):
            for i in range(n):
                if active[i]:
                    mid = 0.5 * (lo[i] + hi[i])
                    residual, _, _ = _section_terms_jit(
                        mid, axial, omega, r[i], chord[i], beta[i], tip_factor[i],
                        rho, blade_count, alpha0, cl_limit, cd0, cd_k
                    )
                    if residual > 0.0:
                        lo[i] = mid
                    else:
                        hi[i] = mid
        thrust = 0.0
        torque = 0.0
        for i in range(n):
            induced = 0.5 * (lo[i] + hi[i]) if active[i] else 0.0
            _, d_thrust, d_torque = _section_terms_jit(
                induced, axial, omega, r[i], chord[i], beta[i], tip_factor[i],
                rho, blade_count, alpha0, cl_limit, cd0, cd_k
            )
            thrust += d_thrust * dr
            torque += d_torque * dr
        return max(thrust, 0.0), max(torque, 0.0)


if njit is not None:
    @njit(cache=True, fastmath=False)
    def _bemt_evaluate_many_jit(omegas, axial_inflows, r, chord, beta, tip_factor, dr,
                                rho, blade_count, alpha0, cl_limit, cd0, cd_k):
        """Evaluate rotor batches in one compiled call, preserving each BEMT solve."""
        count = omegas.shape[0]
        thrust = np.zeros(count, dtype=np.float64)
        torque = np.zeros(count, dtype=np.float64)
        for i in range(count):
            if omegas[i] >= 1e-6:
                thrust[i], torque[i] = _bemt_evaluate_jit(
                    omegas[i], axial_inflows[i], r, chord, beta, tip_factor, dr,
                    rho, blade_count, alpha0, cl_limit, cd0, cd_k,
                )
        return thrust, torque


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
        self.dr = (self.radius_m - r0) / self.n_integral
        self.r = r0 + (np.arange(self.n_integral, dtype=np.float64) + 0.5) * self.dr
        radial = stations[:, 0]
        self.chord = np.interp(self.r / self.radius_m, radial, stations[:, 1])
        self.beta = np.radians(np.interp(self.r / self.radius_m, radial, stations[:, 2]))
        self.tip_factor = 0.5 * self.blade_count * (self.radius_m - self.r) / self.r


    def evaluate_many(self, omega_rad_s, axial_inflow_m_s):
        """Evaluate matching rotor arrays with one Python-to-Numba transition."""
        omegas = np.asarray(omega_rad_s, dtype=np.float64)
        axial = np.asarray(axial_inflow_m_s, dtype=np.float64)
        if omegas.ndim != 1 or axial.shape != omegas.shape:
            raise ValueError("rotor speeds and axial inflows must be matching 1-D arrays")
        omegas = np.maximum(omegas, 0.0)
        if njit is not None:
            return _bemt_evaluate_many_jit(
                omegas, axial, self.r, self.chord, self.beta, self.tip_factor, self.dr,
                self.rho, self.blade_count, self.alpha0, self.cl_limit, self.cd0, self.cd_k,
            )
        results = [self.evaluate(w, v) for w, v in zip(omegas, axial)]
        return (np.asarray([x[0] for x in results], dtype=np.float64),
                np.asarray([x[1] for x in results], dtype=np.float64))

    def evaluate(self, omega_rad_s: float, axial_inflow_m_s: float = 0.0) -> tuple[float, float]:
        """Return positive thrust (N) and shaft torque magnitude (Nm)."""
        omega = max(float(omega_rad_s), 0.0)
        if omega < 1e-6:
            return 0.0, 0.0
        axial = float(axial_inflow_m_s)
        if njit is not None:
            return _bemt_evaluate_jit(
                omega, axial, self.r, self.chord, self.beta, self.tip_factor, self.dr,
                self.rho, self.blade_count, self.alpha0, self.cl_limit, self.cd0, self.cd_k,
            )

        vt = omega * self.r

        def section(induced):
            va = axial + induced
            phi = np.arctan2(va, vt)
            vrel2 = vt * vt + va * va
            cl = np.clip(2.0 * math.pi * (self.beta - phi - self.alpha0), -self.cl_limit, self.cl_limit)
            cd = self.cd0 + self.cd_k * cl * cl
            dynamic = 0.5 * self.rho * vrel2 * self.blade_count * self.chord
            d_thrust = dynamic * (cl * np.cos(phi) - cd * np.sin(phi))
            d_torque = dynamic * (cl * np.sin(phi) + cd * np.cos(phi)) * self.r
            sin_phi = np.maximum(np.abs(np.sin(phi)), 1e-8)
            f_arg = self.tip_factor / sin_phi
            loss = np.maximum(2.0 / math.pi * np.arccos(np.minimum(1.0, np.exp(-f_arg))), 0.05)
            momentum = 4.0 * math.pi * self.rho * self.r * induced * (induced + axial) * loss
            return d_thrust - momentum, d_thrust, d_torque

        lo = np.zeros_like(self.r)
        hi = np.full_like(self.r, max(0.6 * omega * self.radius_m, 1.0))
        active = section(lo)[0] > 0.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            residual = section(mid)[0]
            above = residual > 0.0
            lo = np.where(active & above, mid, lo)
            hi = np.where(active & ~above, mid, hi)
        induced = np.where(active, 0.5 * (lo + hi), 0.0)
        _, d_thrust, d_torque = section(induced)
        return max(float(np.sum(d_thrust) * self.dr), 0.0), max(float(np.sum(d_torque) * self.dr), 0.0)
