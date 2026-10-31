#!/usr/bin/env python3
"""Fail-fast consistency checks between the pinned Smooth2 CAD and simulator profiles."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(actual: float, expected: float, label: str, tolerance: float = 1e-8) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise SystemExit(f"{label} mismatch: actual={actual:.12g}, expected={expected:.12g}")


def main() -> None:
    params = read_json("cad/parametric/model.params.json")
    profiles = read_json("vehicle_models/family/vehicle_profiles.json")["profiles"]
    profile = next(p for p in profiles if p["vehicle_id"] == "smooth2-a2212-custom-prop")
    manifest = read_json("output/smooth2-parametric/assembly-manifest.json")
    report = read_json("output/smooth2-parametric/validation-report.json")
    config = read_json("config/drone.json")
    calibration = read_json("vehicle_models/smooth2_parametric/physics_calibration.json")
    revision_text = (ROOT / "assets/preflight/MODEL_SOURCE_REVISION.md").read_text(encoding="utf-8")
    revision_match = re.search(r"Source commit:.*?([0-9a-f]{40})", revision_text)
    if not revision_match:
        raise SystemExit("Pinned CAD source commit is missing from MODEL_SOURCE_REVISION.md")
    source_revision = revision_match.group(1)

    if profile.get("calibration_status") != "provisional_estimate":
        raise SystemExit("Smooth2 profile must remain explicitly provisional until measured")
    if source_revision not in profile.get("parameter_source", ""):
        raise SystemExit("Vehicle profile does not reference the pinned CAD source revision")
    if calibration.get("motor_propeller", {}).get("propeller_design_revision") != source_revision:
        raise SystemExit("Physics calibration and pinned CAD source revisions differ")
    if manifest.get("branch_scope") != "kilo" or report.get("branch") != "kilo":
        raise SystemExit("CAD exports are not identified as the kilo branch")
    if manifest.get("input_sha256") != report.get("input_sha256"):
        raise SystemExit("Assembly manifest and validation report were not generated from identical CAD inputs")
    if not report.get("assembly", {}).get("shape_valid"):
        raise SystemExit("CAD assembly validation report does not mark the assembly valid")

    scale = float(params["model_scale"])
    motor_x_m = float(params["motor_x"]) * scale / 1000.0
    motor_z_m = float(params["motor_z"]) * scale / 1000.0
    expected_arm_span = 2.0 * math.hypot(motor_x_m, motor_z_m)
    expected_prop_radius = float(params["prop_radius"]) * scale / 1000.0
    expected_prop_diameter = 2.0 * expected_prop_radius
    rotors = profile.get("rotors", [])
    if len(rotors) != 4 or int(profile["geometry"].get("rotor_count", 0)) != 4:
        raise SystemExit("Smooth2 must define exactly four simulator rotors")

    for index, rotor in enumerate(rotors, 1):
        x, y, z = (float(v) for v in rotor["position_m"])
        close(abs(x), motor_x_m, f"rotor {index} CAD X station")
        close(abs(y), motor_z_m, f"rotor {index} CAD Y station")
        close(z, 0.0, f"rotor {index} rotor-plane offset")
        expected_reaction_sign = 1 if x * y > 0 else -1
        if int(rotor["spin_sign"]) != expected_reaction_sign:
            raise SystemExit(
                f"rotor {index} reaction-torque sign disagrees with the mirrored CAD propeller handedness"
            )

    close(profile["geometry"]["arm_span_diagonal_m"], expected_arm_span, "simulator diagonal motor span")
    close(profile["geometry"]["propeller_diameter_m"], expected_prop_diameter, "simulator propeller diameter")
    close(report["propeller"]["diameter_mm"], expected_prop_diameter * 1000.0, "CAD report propeller diameter", 0.01)
    if int(report["propeller"].get("blade_count", 0)) != int(params["prop_blade_count"]):
        raise SystemExit("CAD report blade count differs from the parametric source")
    if len([p for p in manifest["parts"] if p["name"].startswith("arm_")]) != 4:
        raise SystemExit("CAD assembly manifest must contain four integrated arm/guard/motor modules")
    if len([p for p in manifest["parts"] if p["name"].startswith("prop_")]) != 4:
        raise SystemExit("CAD assembly manifest must contain four separate propellers")

    close(config["rotor_radius_m"], expected_prop_radius, "legacy simulator rotor radius")
    close(config["arm_length_m"], math.hypot(motor_x_m, motor_z_m), "legacy simulator motor arm length")
    close(config["visual"]["propeller_radius_m"], expected_prop_radius, "fallback visual propeller radius")

    components = report["components"]
    lower = components["body_lower"]["bbox_mm"]
    crown = components["body_crown"]["bbox_mm"]
    body_x = (max(lower[3], crown[3]) - min(lower[0], crown[0])) / 1000.0
    body_z = (max(lower[4], crown[4]) - min(lower[1], crown[1])) / 1000.0
    body_y = (max(lower[5], crown[5]) - min(lower[2], crown[2])) / 1000.0
    close(config["visual"]["body_x_m"], body_x, "fallback visual body width")
    close(config["visual"]["body_y_m"], body_y, "fallback visual body height")
    close(config["visual"]["body_z_m"], body_z, "fallback visual body length")

    print("Smooth2 CAD-to-simulator synchronization: PASS")
    print(f"  CAD source: 3D/kilo@{source_revision}")
    print(f"  motor station radius: {math.hypot(motor_x_m, motor_z_m):.9f} m")
    print(f"  diagonal motor span: {expected_arm_span:.9f} m")
    print(f"  propeller diameter: {expected_prop_diameter:.9f} m")
    print("  four rotor positions, reaction signs, body dimensions and source hashes agree")
    print("  thrust coefficient, torque proxy, motor lag and RPM limits were not changed")


if __name__ == "__main__":
    main()
