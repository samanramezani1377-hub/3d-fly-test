"""Screening-level blade-element/momentum (BEMT) tools for the Smooth2 propeller.

This module has no CadQuery dependency so the twist law can be designed, tested and
re-used by ``drone_parametric.py``.  It reproduces the model described in the
"Smooth2 propeller engineering report" (annular momentum balance + 2*pi*alpha lift
slope clipped at +-1.2, CD = 0.012 + 0.01*CL^2) and adds three corrections the report
asked for: (1) the region under the hub is excluded from the integral, (2) the root
twist is no longer capped at an arbitrary 42 deg, (3) optional Prandtl tip/root loss.

All results are *screening estimates*.  They are not calibrated against a thrust
stand and must not be used to claim real thrust or efficiency.  Lengths here are
in metres unless a name ends in ``_mm``.
"""
import math

RHO = 1.225          # kg/m^3, sea level
CL_LIMIT = 1.2       # same clip as the report
CD0, CD_K = 0.012, 0.01


def thin_airfoil_alpha0(camber, camber_pos, n=400):
    """Zero-lift angle (rad) of a NACA 4-digit mean line from thin-airfoil theory."""
    total = 0.0
    for i in range(n):
        theta = math.pi * (i + 0.5) / n
        x = 0.5 * (1.0 - math.cos(theta))
        if x < camber_pos:
            slope = 2.0 * camber / camber_pos**2 * (camber_pos - x)
        else:
            slope = 2.0 * camber / (1.0 - camber_pos) ** 2 * (camber_pos - x)
        total += slope * (math.cos(theta) - 1.0) * (math.pi / n)
    return -total / math.pi


def lift_drag(alpha, alpha0):
    cl = max(-CL_LIMIT, min(CL_LIMIT, 2.0 * math.pi * (alpha - alpha0)))
    return cl, CD0 + CD_K * cl * cl


def _station_solution(r, chord, beta, omega, radius, blades, alpha0, prandtl):
    """Solve the induced velocity of one annulus by bisection on the momentum balance."""
    def residual(vi):
        v_t = omega * r
        phi = math.atan2(vi, v_t)
        vrel2 = v_t * v_t + vi * vi
        cl, cd = lift_drag(beta - phi, alpha0)
        dt_be = 0.5 * RHO * vrel2 * blades * chord * (cl * math.cos(phi) - cd * math.sin(phi))
        f = 1.0
        if prandtl and phi > 1e-6:
            tip = blades / 2.0 * (radius - r) / (r * math.sin(phi))
            f = 2.0 / math.pi * math.acos(min(1.0, math.exp(-tip)))
            f = max(f, 0.05)
        dt_mom = 4.0 * math.pi * RHO * r * vi * vi * f
        return dt_be - dt_mom, dt_be, cl, cd, phi, vrel2
    lo, hi = 0.0, 0.6 * omega * radius
    if residual(lo)[0] <= 0.0:           # blade cannot lift at this angle of attack
        return residual(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if residual(mid)[0] > 0.0:
            lo = mid
        else:
            hi = mid
    return residual(0.5 * (lo + hi))


def analyse(radius, stations, betas_deg, rpm, hub_radius=0.0, blades=2,
            alpha0=0.0, prandtl=False, n_int=120):
    """Integrate thrust/torque from hub radius to tip.

    ``stations``: list of (r/R, chord_m).  ``betas_deg``: geometric pitch angle per station.
    Returns dict with thrust (N), torque (N*m), power (W) and per-station details.
    """
    omega = 2.0 * math.pi * rpm / 60.0
    fr = [s[0] for s in stations]
    ch = [s[1] for s in stations]

    def interp(xs, ys, x):
        if x <= xs[0]:
            return ys[0]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] * (1 - t) + ys[i] * t
        return ys[-1]

    r0 = max(hub_radius, fr[0] * radius)
    dr = (radius - r0) / n_int
    thrust = torque = 0.0
    detail = []
    for i in range(n_int):
        r = r0 + (i + 0.5) * dr
        x = r / radius
        c = interp(fr, ch, x)
        b = math.radians(interp(fr, betas_deg, x))
        _, dt, cl, cd, phi, vrel2 = _station_solution(r, c, b, omega, radius, blades, alpha0, prandtl)
        dq = 0.5 * RHO * vrel2 * blades * c * (cl * math.sin(phi) + cd * math.cos(phi)) * r
        thrust += dt * dr
        torque += dq * dr
        detail.append((x, math.degrees(b - phi), cl))
    return {"thrust_N": thrust, "torque_Nm": torque, "power_W": torque * omega,
            "rpm": rpm, "detail": detail}


def constant_pitch_betas(radius, stations, pitch, cap_deg=None):
    betas = []
    for x, _ in stations:
        b = math.degrees(math.atan(pitch / (2.0 * math.pi * x * radius)))
        betas.append(min(b, cap_deg) if cap_deg else b)
    return betas


def design_twist(radius, stations, rpm, design_cl, alpha0, blades=2, prandtl=True,
                 max_root_deg=None):
    """Twist law holding every station at ``design_cl`` for its own induced velocity.

    For each annulus the momentum balance is solved with the lift coefficient pinned to
    ``design_cl`` (unknowns: induced velocity and pitch angle).  The result is the pitch
    angle distribution that makes the whole blade work at one lift coefficient at ``rpm``.
    """
    omega = 2.0 * math.pi * rpm / 60.0
    out = []
    for x, chord in stations:
        r = x * radius
        v_t = omega * r

        def balance(vi):
            phi = math.atan2(vi, v_t)
            cl = design_cl
            cd = CD0 + CD_K * cl * cl
            vrel2 = v_t * v_t + vi * vi
            dt_be = 0.5 * RHO * vrel2 * blades * chord * (cl * math.cos(phi) - cd * math.sin(phi))
            f = 1.0
            if prandtl and phi > 1e-6:
                tip = blades / 2.0 * (radius - r) / (r * math.sin(phi))
                f = max(2.0 / math.pi * math.acos(min(1.0, math.exp(-tip))), 0.05)
            return dt_be - 4.0 * math.pi * RHO * r * vi * vi * f, phi
        lo, hi = 0.0, 0.6 * omega * radius
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if balance(mid)[0] > 0.0:
                lo = mid
            else:
                hi = mid
        phi = balance(0.5 * (lo + hi))[1]
        alpha = design_cl / (2.0 * math.pi) + alpha0
        beta = math.degrees(phi + alpha)
        out.append(min(beta, max_root_deg) if max_root_deg else beta)
    return out


# Typical A2212/13T 1000KV values.  They are NOT measured on the user's motor and the
# winding resistance in particular varies by manufacturer; the operating-point results
# are therefore assumption-based screening numbers.
A2212_ASSUMED = {"kv": 1000.0, "resistance_ohm": 0.12, "no_load_current_a": 0.5,
                 "voltage_v": 11.1}


def motor_operating_point(radius, stations, betas_deg, hub_radius, blades, alpha0,
                          motor=None, voltage=None, n_int=60):
    """RPM, current and thrust where the motor's torque balance meets the propeller load.

    Brushless DC model: V = I*R + omega/Kv_rad, torque = (I - I0)/Kt.  ``voltage`` lets the
    caller evaluate part-throttle (equivalent average voltage).
    """
    m = dict(A2212_ASSUMED if motor is None else motor)
    v = float(m["voltage_v"] if voltage is None else voltage)
    kt = 60.0 / (2.0 * math.pi * m["kv"])
    lo, hi = 50.0, 2.0 * math.pi * m["kv"] * v / 60.0 * 1.02
    res = None
    for _ in range(40):
        om = 0.5 * (lo + hi)
        rpm = om * 60.0 / (2.0 * math.pi)
        res = analyse(radius, stations, betas_deg, rpm, hub_radius=hub_radius, blades=blades,
                      alpha0=alpha0, prandtl=True, n_int=n_int)
        current = res["torque_Nm"] / kt + m["no_load_current_a"]
        if current * m["resistance_ohm"] + om * kt > v:
            hi = om
        else:
            lo = om
    res["current_a"] = current
    res["electrical_power_w"] = current * v
    return res
