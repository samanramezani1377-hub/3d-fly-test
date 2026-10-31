"""Design the Smooth2 propeller for maximum thrust the existing motor can drive.

Run from the repository root:  python cad/parametric/design_prop.py

Inputs live in ``prop_design`` inside model.params.json (mass, thrust-to-weight target, motor
constants, blade count, chord shape, design lift coefficient, tip clearance).  The script
grows the propeller radius until the motor reaches its current limit (best thrust), checks the thrust-to-weight target, designs the
twist of every station with BEMT at the motor's own full-throttle RPM, and writes back:
blade stations, radius, hub, blade count and guard growth.  Motor positions are kept unless
``allow_move_motors`` is true; otherwise the rotor is limited by the space left for its guard.  Everything else in the params file is untouched.

All aerodynamic numbers are screening estimates; the A2212 constants are typical values,
not measurements of the user's motor.
"""
import json
import math
from pathlib import Path

import prop_bemt as pb

PARAMS_PATH = Path(__file__).resolve().with_name("model.params.json")
G = 9.80665


def planform(x, prop_design, root_x):
    """Chord / radius at r/R = x: fuller root, peak near 0.3R, linear taper to the tip."""
    peak = float(prop_design["chord_peak_fraction"])
    cmax = float(prop_design["max_chord_over_radius"])
    tip = float(prop_design["tip_chord_ratio"])
    root = float(prop_design["root_chord_ratio"])
    if x <= peak:
        t = root + (1.0 - root) * (x - root_x) / max(peak - root_x, 1e-6)
    else:
        t = 1.0 - (1.0 - tip) * (x - peak) / (1.0 - peak)
    return cmax * t


def build(params, radius_mm):
    dp = params["prop_design"]
    scale = float(params["model_scale"])
    radius_m = radius_mm / 1000.0
    radius_local = radius_mm / scale
    hub_local = float(dp["hub_radius_mm"]) / scale
    root_x = round(hub_local / radius_local * float(dp["root_start_fraction_of_hub"]), 4)
    xs = [root_x] + [float(v) for v in dp["radial_fractions"]]
    stations_m = [(x, planform(x, dp, root_x) * radius_m) for x in xs]
    alpha0 = pb.thin_airfoil_alpha0(float(params["prop_airfoil_camber"]),
                                    float(params["prop_airfoil_camber_position"]))
    blades = int(dp["blade_count"])
    motor = dict(pb.A2212_ASSUMED, **dp.get("motor", {}))
    hub_m = hub_local * scale / 1000.0
    rpm = 0.9 * 60.0 * motor["kv"] * motor["voltage_v"] / 60.0
    for _ in range(4):
        betas = pb.design_twist(radius_m, stations_m, rpm, float(dp["design_cl"]), alpha0,
                                blades=blades, prandtl=False)
        full = pb.motor_operating_point(radius_m, stations_m, betas, hub_m, blades, alpha0, motor)
        rpm = full["rpm"]
    return xs, stations_m, betas, full, (radius_m, hub_m, blades, alpha0, motor)


def hover_point(ctx, stations_m, betas, thrust_target):
    radius_m, hub_m, blades, alpha0, motor = ctx
    lo, hi = 0.2 * motor["voltage_v"], motor["voltage_v"]
    for _ in range(30):
        v = 0.5 * (lo + hi)
        res = pb.motor_operating_point(radius_m, stations_m, betas, hub_m, blades, alpha0,
                                       motor, voltage=v, n_int=40)
        if res["thrust_N"] > thrust_target:
            hi = v
        else:
            lo = v
    res["equivalent_voltage_v"] = v
    return res


def body_distance(params, x, z):
    import drone_parametric as d
    pts = d.canopy_base_points(params)
    best = 1e9
    for i, (ax, az) in enumerate(pts):
        bx, bz = pts[(i + 1) % len(pts)]
        vx, vz = bx - ax, bz - az
        t = max(0.0, min(1.0, ((x - ax) * vx + (z - az) * vz) / (vx * vx + vz * vz)))
        best = min(best, math.hypot(x - ax - t * vx, z - az - t * vz))
    return best


def main():
    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    dp = params["prop_design"]
    scale = float(params["model_scale"])
    weight_per_rotor = float(dp["vehicle_mass_kg"]) * G / 4.0
    target = float(dp["target_thrust_to_weight"]) * weight_per_rotor
    # Best thrust: grow the rotor until the assumed motor reaches its current limit (or the
    # maximum radius), then keep the last radius that stays within the limit.
    max_radius_mm = float(dp["max_radius_mm"])
    allow_move = bool(dp.get("allow_move_motors", False))
    if not allow_move:
        # Keep the user's motor positions: the rotor may only grow until its guard would touch
        # the body or the neighbouring guards.
        gap = float(dp["guard_to_body_gap_mm"]) / scale
        spread = max(params["guard_profile_outer"]) - min(params["guard_profile_inner"])
        m_now = float(params["motor_x"])
        outer_allowed = min(body_distance(params, m_now, m_now) - gap, m_now - gap)
        space_mm = (outer_allowed - spread - float(dp["tip_clearance_mm"]) / scale) * scale
        max_radius_mm = min(max_radius_mm, math.floor(space_mm * 2.0) / 2.0)
        print("space-limited radius mm:", round(space_mm, 2))
    radius_mm = min(float(dp["min_radius_mm"]), max_radius_mm)
    best = None
    while radius_mm <= max_radius_mm + 1e-9:
        result = build(params, radius_mm)
        if result[3]["current_a"] > float(dp["max_motor_current_a"]) and best is not None:
            break
        best = (radius_mm, result)
        radius_mm += 0.5
    radius_mm, (xs, st, betas, full, ctx) = best
    if full["thrust_N"] < weight_per_rotor:
        raise SystemExit("Even full throttle cannot hover within the radius/current limits")
    if full["thrust_N"] < target:
        print("WARNING: thrust-to-weight target not reached; best achievable in the available "
              "space is TWR %.2f" % (4.0 * full["thrust_N"] / (float(dp["vehicle_mass_kg"]) * G)))
    radius_local = radius_mm / scale
    hover = hover_point(ctx, st, betas, weight_per_rotor)
    # chord in local units: metres -> mm -> / scale
    rows = [[round(x, 4), round(c * 1000.0 / scale, 3), round(b, 2)]
            for x, (_, c), b in zip(xs, st, betas)]
    b75 = None
    for (x0, _, b0), (x1, _, b1) in zip(rows, rows[1:]):
        if x0 <= 0.75 <= x1:
            b75 = b0 + (b1 - b0) * (0.75 - x0) / (x1 - x0)
    params["prop_radius"] = round(radius_local, 3)
    params["prop_hub_radius"] = round(float(dp["hub_radius_mm"]) / scale, 3)
    params["prop_hub_height"] = round(float(dp["hub_height_mm"]) / scale, 3)
    params["prop_blade_count"] = int(dp["blade_count"])
    params["prop_blade_stations"] = rows
    params["prop_pitch_mm"] = round(2 * math.pi * 0.75 * radius_mm * math.tan(math.radians(b75)), 3)
    # Guard: inner wall keeps the requested tip clearance; motors move out so the larger guard
    # keeps the original gap to the body.
    inner_min = min(params["guard_profile_inner"])
    params["guard_grow"] = round(radius_local + float(dp["tip_clearance_mm"]) / scale - inner_min, 3)
    m = float(params["motor_x"])
    if allow_move:
        outer_max = max(params["guard_profile_outer"]) + params["guard_grow"]
        gap = float(dp["guard_to_body_gap_mm"]) / scale
        m = 40.0
        while body_distance(params, m, m) < outer_max + gap:
            m += 0.5
        params["motor_x"] = params["motor_z"] = m
    PARAMS_PATH.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("radius_mm", radius_mm, "diameter_mm", 2 * radius_mm, "blades", dp["blade_count"])
    print("target thrust N", round(target, 3), "full-throttle", round(full["thrust_N"], 3),
          "rpm", round(full["rpm"]), "A", round(full["current_a"], 1), "W", round(full["electrical_power_w"], 1))
    print("hover rpm", round(hover["rpm"]), "A", round(hover["current_a"], 2), "thrust", round(hover["thrust_N"], 3))
    print("guard_grow", params["guard_grow"], "motor_x/z", m)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
