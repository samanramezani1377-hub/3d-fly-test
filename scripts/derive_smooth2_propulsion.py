"""Reproducible geometry extraction and explicitly approximate static BEMT envelope."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAD = ROOT / "cad/source/smooth2_model_params.json"
PROFILE = ROOT / "vehicle_models/family/vehicle_profiles.json"
OUT = ROOT / "vehicle_models/family/smooth2_cad_propulsion_estimate.json"
CAD_BLOB_SHA = "cc77601fb0e4837ca1f7d9c40024c2bc942462"
RHO = 1.225
MU = 1.81e-5
CL_MAX = 1.2
CD0 = 0.012
CD2 = 0.01


def _static_bemt(rpm, stations, radius_local_m, scale, pitch_m):
    """Simple annular momentum + blade-element estimate; not a validated low-Re polar."""
    omega = rpm * 2.0 * math.pi / 60.0
    radii = [float(row[0]) * radius_local_m for row in stations]
    chords = [float(row[1]) * scale / 1000.0 for row in stations]
    thrust = torque = 0.0
    reynolds = []
    for i in range(len(radii) - 1):
        r_local = 0.5 * (radii[i] + radii[i + 1])
        r = r_local * scale
        dr = (radii[i + 1] - radii[i]) * scale
        chord = 0.5 * (chords[i] + chords[i + 1])
        # Match the CAD generator: pitch is converted back to local model units.
        beta = min(math.atan((pitch_m / scale) / (2.0 * math.pi * r_local)),
                   math.radians(42.0))

        def section(axial_velocity):
            tangential_velocity = omega * r
            relative_speed = math.hypot(tangential_velocity, axial_velocity)
            inflow_angle = math.atan2(axial_velocity, tangential_velocity)
            cl = max(-CL_MAX, min(CL_MAX, 2.0 * math.pi * (beta - inflow_angle)))
            cd = CD0 + CD2 * cl * cl
            q = 0.5 * RHO * relative_speed * relative_speed
            d_t = q * 2 * chord * (
                cl * math.cos(inflow_angle) - cd * math.sin(inflow_angle)
            )
            d_q = q * 2 * chord * r * (
                cd * math.cos(inflow_angle) + cl * math.sin(inflow_angle)
            )
            # Annular actuator-disk momentum balance, per unit radial span.
            momentum_t = 4.0 * math.pi * RHO * r * axial_velocity**2
            return d_t - momentum_t, d_t, d_q, relative_speed

        lo, hi = 0.0, 80.0
        for _ in range(72):
            mid = 0.5 * (lo + hi)
            if section(mid)[0] > 0:
                lo = mid
            else:
                hi = mid
        _, d_t, d_q, relative_speed = section(0.5 * (lo + hi))
        thrust += d_t * dr
        torque += d_q * dr
        reynolds.append(RHO * relative_speed * chord / MU)

    diameter = 2.0 * radius_local_m * scale
    rev_per_sec = rpm / 60.0
    ct = thrust / (RHO * rev_per_sec**2 * diameter**4)
    cq = torque / (RHO * rev_per_sec**2 * diameter**5)
    return {
        "rpm": rpm,
        "omega_rad_s": omega,
        "thrust_N_per_rotor": thrust,
        "torque_Nm_per_rotor": torque,
        "CT": ct,
        "CQ": cq,
        "thrust_coefficient_N_per_rad_s2": thrust / omega**2,
        "torque_coefficient_Nm_per_rad_s2": torque / omega**2,
        "Reynolds_min": min(reynolds),
        "Reynolds_max": max(reynolds),
    }


def derive():
    params = json.loads(CAD.read_text())
    family = json.loads(PROFILE.read_text())
    vehicle = next(x for x in family["profiles"]
                   if x["vehicle_id"] == "smooth2-a2212-custom-prop")
    propulsion = vehicle["propulsion"]
    scale = float(params["model_scale"])
    radius_local_m = float(params["prop_radius"]) / 1000.0
    radius_m = radius_local_m * scale
    diameter_m = 2.0 * radius_m
    pitch_m = float(params["prop_pitch_mm"]) / 1000.0
    no_load_rpm = float(propulsion["kv_rpm_per_volt"]) * float(
        propulsion["nominal_battery_voltage_v"])
    max_profile_rpm = float(propulsion["max_rotor_speed_rad_s"]) * 60.0 / (2.0 * math.pi)
    rpm_cap = min(no_load_rpm, max_profile_rpm)
    rpm_cases = [0.50, 0.65, 0.80, 0.95]
    envelope = [_static_bemt(rpm_cap * fraction, params["prop_blade_stations"],
                             radius_local_m, scale, pitch_m)
                for fraction in rpm_cases]
    max_case = envelope[-1]
    mass = float(vehicle["physical"]["mass_kg"])
    rotor_count = int(vehicle["geometry"]["rotor_count"])
    required_hover_per_rotor = mass * 9.80665 / rotor_count
    chord_stations = [
        {"radius_fraction": float(row[0]), "chord_mm_final": float(row[1]) * scale}
        for row in params["prop_blade_stations"]
    ]
    return {
        "schema_version": 2,
        "vehicle_id": vehicle["vehicle_id"],
        "calibration_status": "analytical_estimate_not_measured",
        "source": {
            "repository": "samanramezani1377-hub/3D",
            "branch": "kilo",
            "file": "cad/parametric/model.params.json",
            "blob_sha": CAD_BLOB_SHA,
            "source_repository_modified": False
        },
        "propeller": {
            "blades": 2,
            "radius_m": radius_m,
            "diameter_m": diameter_m,
            "pitch_mm": pitch_m * 1000.0,
            "pitch_diameter_ratio": pitch_m / diameter_m,
            "airfoil": {
                "designation_style": "NACA 4410 parameterization",
                "camber_ratio": float(params["prop_airfoil_camber"]),
                "maximum_camber_position_chord": float(params["prop_airfoil_camber_position"]),
                "thickness_ratio": float(params["prop_airfoil_thickness_ratio"]),
                "radial_stations": chord_stations,
                "twist_law": "atan(P/(2*pi*r)), capped at 42 degrees, matching CAD builder"
            }
        },
        "motor_reference": {
            "name": propulsion["motor_name"],
            "KV_rpm_per_volt": propulsion["kv_rpm_per_volt"],
            "battery_voltage_V_nominal": propulsion["nominal_battery_voltage_v"],
            "battery_cells": propulsion["battery_cells"],
            "capacity_Ah": propulsion["battery_capacity_ah"],
            "ideal_no_load_rpm": no_load_rpm,
            "profile_max_omega_rad_s": propulsion["max_rotor_speed_rad_s"],
            "analysis_rpm_cap": rpm_cap,
            "rpm_cap_basis": "minimum of ideal KV*nominal-voltage no-load RPM and configured simulation rotor-speed cap",
            "loaded_operating_rpm_known": False
        },
        "BEMT_model": {
            "method": "annular actuator-disk momentum balance coupled to blade-element lift/drag",
            "air_density_kg_m3": RHO,
            "dynamic_viscosity_Pa_s": MU,
            "lift_model": "2*pi*alpha per radian, clipped to +/-1.2",
            "drag_model": "CD = 0.012 + 0.01*CL^2",
            "rpm_sweep_fraction_of_cap": rpm_cases,
            "rpm_cases": envelope,
            "maximum_case_four_rotor_thrust_N": rotor_count * max_case["thrust_N_per_rotor"]
        },
        "hover_sanity_check": {
            "vehicle_mass_kg_from_profile": mass,
            "rotor_count": rotor_count,
            "required_hover_thrust_N_per_rotor": required_hover_per_rotor,
            "estimated_max_case_thrust_N_per_rotor": max_case["thrust_N_per_rotor"],
            "estimated_max_case_to_hover_requirement_ratio": max_case["thrust_N_per_rotor"] / required_hover_per_rotor,
            "not_a_certified_flight_performance_result": True
        },
        "limitations": [
            "The CAD calls the section NACA 4410-style, but supplies geometry only; it does not contain measured airfoil polars.",
            "The calculated Reynolds range is very low; thin-airfoil lift slope and the assumed drag polar are not reliable substitutes for measured low-Reynolds-number data.",
            "No Prandtl tip/root-loss correction, induced-swirl solution, blade/guard interference, ground effect, or forward-flight model.",
            "The 50/65/80/95 percent RPM values are analysis points, not predicted loaded motor operating points.",
            "Motor winding resistance, current limits, ESC, propeller load curve, torque-speed curve, and motor temperature are unknown.",
            "CAD motor dimensions are simplified envelopes; no measured motor mass or inertia is available.",
            "No measured airframe mass, thrust-stand data, or flight validation is available."
        ],
        "interpretation": "This is a screening estimate only. A low predicted thrust-to-hover ratio is a reason to validate the motor/propeller pairing, not proof that the CAD propeller is defective.",
        "active_vehicle_profile_modified": False,
        "physical_flight_eligible": False
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = derive()
    if args.check:
        assert 0.074 < report["propeller"]["diameter_m"] < 0.076
        assert len(report["propeller"]["airfoil"]["radial_stations"]) == 11
        cases = report["BEMT_model"]["rpm_cases"]
        assert len(cases) == 4 and all(c["thrust_N_per_rotor"] > 0 for c in cases)
        assert all(a["thrust_N_per_rotor"] < b["thrust_N_per_rotor"]
                   for a, b in zip(cases, cases[1:]))
        assert report["motor_reference"]["loaded_operating_rpm_known"] is False
        assert report["active_vehicle_profile_modified"] is False
        assert report["physical_flight_eligible"] is False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
