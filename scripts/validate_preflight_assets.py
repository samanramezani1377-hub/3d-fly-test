#!/usr/bin/env python3
"""Structural checks for imported preflight CAD assets (stdlib only).

This does not replace CAD-kernel BREP validation or physical testing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "preflight" / "smooth2-parametric"
CHUNKS = ASSETS / "assembly_parts"
EXPECTED_CHUNKS = 6
EXPECTED_COMPONENTS = 26
EXPECTED_CHUNK_BYTES = 5_970_685


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def read_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # report all missing/broken inputs together
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


def main() -> int:
    errors: list[str] = []
    params_path = ASSETS / "drone_parametric.params.json"
    manifest_path = ASSETS / "assembly-manifest.json"
    report_path = ASSETS / "validation-report.json"
    params = read_json(params_path, errors)
    manifest = read_json(manifest_path, errors)
    report = read_json(report_path, errors)

    if not isinstance(params, dict):
        fail(errors, "Parameter snapshot must be a JSON object")
    else:
        for key in ("motor_x", "motor_z", "prop_pitch_mm", "prop_airfoil_camber",
                    "prop_airfoil_thickness_ratio", "model_scale"):
            if key not in params:
                fail(errors, f"Missing required design parameter: {key}")

    parts = manifest.get("parts") if isinstance(manifest, dict) else None
    if not isinstance(parts, list):
        fail(errors, "Assembly manifest must contain a parts array")
        parts = []
    if len(parts) != EXPECTED_COMPONENTS:
        fail(errors, f"Expected {EXPECTED_COMPONENTS} manifest components, found {len(parts)}")

    seen_names: set[str] = set()
    for part in parts:
        if not isinstance(part, dict):
            fail(errors, "Manifest part entry is not an object")
            continue
        name = part.get("name")
        step_name = part.get("step")
        if not isinstance(name, str) or name in seen_names:
            fail(errors, f"Missing or duplicate component name: {name!r}")
        else:
            seen_names.add(name)
        if not isinstance(step_name, str):
            fail(errors, f"Invalid STEP path for component {name!r}")
            continue
        check_step(ASSETS / step_name, errors)

    chunk_paths = sorted(CHUNKS.glob("drone_parametric.step.part*"))
    if len(chunk_paths) != EXPECTED_CHUNKS:
        fail(errors, f"Expected {EXPECTED_CHUNKS} assembly chunks, found {len(chunk_paths)}")
    chunk_size = sum(p.stat().st_size for p in chunk_paths)
    if chunk_size != EXPECTED_CHUNK_BYTES:
        fail(errors, f"Assembly chunk bytes mismatch: expected {EXPECTED_CHUNK_BYTES}, found {chunk_size}")

    if len(chunk_paths) == EXPECTED_CHUNKS:
        rebuilt = ASSETS / "drone_parametric.step"
        try:
            with rebuilt.open("wb") as output:
                for chunk in chunk_paths:
                    output.write(chunk.read_bytes())
            check_step(rebuilt, errors)
        except OSError as exc:
            fail(errors, f"Could not reconstruct assembly STEP: {exc}")
        finally:
            if rebuilt.exists():
                rebuilt.unlink()

    if isinstance(report, dict):
        reported_count = report.get("assembly", {}).get("solid_count")
        if reported_count != EXPECTED_COMPONENTS:
            fail(errors, f"Validation report expected {EXPECTED_COMPONENTS} solids, found {reported_count}")

    # STL files are referenced in the manifest but are not currently part of
    # the imported STEP-only asset set. Report this as a warning, not a failure.
    stl_refs = [p.get("stl") for p in parts if isinstance(p, dict) and p.get("stl")]
    missing_stl = [name for name in stl_refs if not (ASSETS / name).is_file()]
    if missing_stl:
        print(f"WARNING: {len(missing_stl)} manifest STL files are not present in this import.")

    if errors:
        print("PREFLIGHT ASSET VALIDATION: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PREFLIGHT ASSET VALIDATION: PASS")
    print(f"- Manifest components: {len(parts)}")
    print(f"- Individual STEP files checked: {len(parts)}")
    print(f"- Assembly chunks: {len(chunk_paths)} ({chunk_size:,} bytes)")
    print("- Full assembly STEP reconstructed and envelope-checked")
    print("- JSON parameter, manifest and report files parsed")
    print("NOTE: Structural checks only; no BREP, CFD, mass-property or flight validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
