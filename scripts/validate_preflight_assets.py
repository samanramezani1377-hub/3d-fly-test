#!/usr/bin/env python3
"""Structural checks for the authoritative kilo CAD package (stdlib only).

This verifies file integrity and source consistency. It does not replace CAD-kernel
BREP checks, physical mass-property calibration, thrust-stand tests, or flight tests.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "output" / "smooth2-parametric"
CAD = ROOT / "cad" / "parametric"
EXPECTED_COMPONENTS = 26


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def read_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(errors, f"{display_path(path)}: invalid or unreadable JSON: {exc}")
        return None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_step(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        fail(errors, f"Missing STEP file: {display_path(path)}")
        return
    if path.stat().st_size < 100:
        fail(errors, f"STEP file is unexpectedly small: {display_path(path)}")
        return
    with path.open("rb") as stream:
        head = stream.read(4096)
        stream.seek(max(0, path.stat().st_size - 4096))
        tail = stream.read()
    if b"ISO-10303-21;" not in head:
        fail(errors, f"Missing ISO-10303-21 header: {display_path(path)}")
    if b"END-ISO-10303-21;" not in tail:
        fail(errors, f"Missing END-ISO-10303-21 trailer: {display_path(path)}")


def check_stl(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        fail(errors, f"Missing STL file: {display_path(path)}")
        return
    data = path.read_bytes()
    if len(data) < 84:
        fail(errors, f"STL file is unexpectedly small: {display_path(path)}")
        return
    # Accept standard binary STL or a valid-looking ASCII STL.
    if len(data) == 84 + 50 * int.from_bytes(data[80:84], "little"):
        return
    stripped = data.lstrip()
    if stripped.startswith(b"solid") and b"endsolid" in data[-256:]:
        return
    fail(errors, f"STL size/header is inconsistent: {display_path(path)}")


def main() -> int:
    errors: list[str] = []
    params_path = ASSETS / "drone_parametric.params.json"
    canonical_params_path = CAD / "model.params.json"
    manifest_path = ASSETS / "assembly-manifest.json"
    report_path = ASSETS / "validation-report.json"

    params = read_json(params_path, errors)
    canonical_params = read_json(canonical_params_path, errors)
    manifest = read_json(manifest_path, errors)
    report = read_json(report_path, errors)

    if isinstance(params, dict):
        for key in ("motor_x", "motor_z", "prop_pitch_mm", "prop_airfoil_camber",
                    "prop_airfoil_thickness_ratio", "model_scale"):
            if key not in params:
                fail(errors, f"Missing required design parameter: {key}")
    else:
        fail(errors, "Parameter snapshot must be a JSON object")

    if isinstance(params, dict) and isinstance(canonical_params, dict) and params != canonical_params:
        fail(errors, "Generated parameter snapshot differs from cad/parametric/model.params.json")

    if not (CAD / "drone_parametric.py").is_file():
        fail(errors, "Missing canonical parametric CAD builder")
    if not (CAD / "body_sections.json").is_file():
        fail(errors, "Missing measured body section input")

    parts = manifest.get("parts") if isinstance(manifest, dict) else None
    if not isinstance(parts, list):
        fail(errors, "Assembly manifest must contain a parts array")
        parts = []
    if len(parts) != EXPECTED_COMPONENTS:
        fail(errors, f"Expected {EXPECTED_COMPONENTS} manifest components, found {len(parts)}")

    if isinstance(manifest, dict) and manifest.get("branch_scope") != "kilo":
        fail(errors, f"Expected source branch_scope 'kilo', found {manifest.get('branch_scope')!r}")
    if isinstance(report, dict) and report.get("branch") != "kilo":
        fail(errors, f"Expected validation report branch 'kilo', found {report.get('branch')!r}")

    seen_names: set[str] = set()
    for part in parts:
        if not isinstance(part, dict):
            fail(errors, "Manifest part entry is not an object")
            continue
        name = part.get("name")
        if not isinstance(name, str) or name in seen_names:
            fail(errors, f"Missing or duplicate component name: {name!r}")
        else:
            seen_names.add(name)
        step_name, stl_name = part.get("step"), part.get("stl")
        if not isinstance(step_name, str):
            fail(errors, f"Invalid STEP path for component {name!r}")
        else:
            check_step(ASSETS / step_name, errors)
        if not isinstance(stl_name, str):
            fail(errors, f"Invalid STL path for component {name!r}")
        else:
            check_stl(ASSETS / stl_name, errors)

    assembly_step = ASSETS / "drone_parametric.step"
    assembly_stl = ASSETS / "drone_parametric.stl"
    assembled_stl = ASSETS / "drone_parametric_assembled.stl"
    exploded_stl = ASSETS / "drone_parametric_exploded.stl"
    check_step(assembly_step, errors)
    for path in (assembly_stl, assembled_stl, exploded_stl):
        check_stl(path, errors)

    if assembly_stl.is_file() and assembled_stl.is_file():
        if assembly_stl.read_bytes() != assembled_stl.read_bytes():
            fail(errors, "drone_parametric.stl must be byte-identical to drone_parametric_assembled.stl")

    if isinstance(report, dict):
        assembly = report.get("assembly", {})
        if assembly.get("shape_valid") is not True:
            fail(errors, "Source validation report does not mark assembly shape_valid=true")
        if assembly.get("solid_count") != EXPECTED_COMPONENTS:
            fail(errors, f"Validation report expected {EXPECTED_COMPONENTS} solids, found {assembly.get('solid_count')}")

    # Confirm that the committed validation report and assembly manifest describe
    # the exact CAD inputs checked into this repository.
    expected_hashes = {}
    if isinstance(manifest, dict):
        expected_hashes.update(manifest.get("input_sha256", {}))
    if isinstance(report, dict):
        report_hashes = report.get("input_sha256", {})
        if expected_hashes and report_hashes != expected_hashes:
            fail(errors, "Manifest and validation report input SHA-256 maps differ")
        expected_hashes.update(report_hashes)
    source_files = {
        "builder": CAD / "drone_parametric.py",
        "parameters": canonical_params_path,
        "body_sections": CAD / "body_sections.json",
    }
    for key, path in source_files.items():
        expected = expected_hashes.get(key)
        if expected and path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                fail(errors, f"Source SHA-256 mismatch for {display_path(path)}")

    if errors:
        print("PREFLIGHT ASSET VALIDATION: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PREFLIGHT ASSET VALIDATION: PASS")
    print(f"- Source branch: {manifest.get('branch_scope')}")
    print(f"- Manifest components: {len(parts)}")
    print(f"- Individual STEP and STL files checked: {len(parts)} each")
    print("- Full assembly STEP and three assembly STL exports checked")
    print("- Canonical parameters, generated snapshot, manifest, report and SHA-256 fingerprints agree")
    print("NOTE: Structural checks only; no physical mass-property, thrust, CFD or flight validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
