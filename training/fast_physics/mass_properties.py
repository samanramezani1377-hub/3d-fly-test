"""Mass-property aggregation from measured component masses and CAD-local inertias.

All input coordinates and inertia tensors use the declared common frame
(default CAD_EXPORT_Z_UP). No density or component mass is invented.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CSV_FIELDS = (
    "component_id", "mass_kg", "cg_x_m", "cg_y_m", "cg_z_m",
    "ixx_kg_m2", "iyy_kg_m2", "izz_kg_m2",
    "ixy_kg_m2", "ixz_kg_m2", "iyz_kg_m2",
)


def aggregate_mass_properties(components: list[dict], coordinate_frame: str = "CAD_EXPORT_Z_UP") -> dict:
    if not components:
        raise ValueError("at least one measured component is required")
    masses, positions, local_inertias = [], [], []
    names = set()
    for item in components:
        name = str(item.get("component_id", "")).strip()
        if not name or name in names:
            raise ValueError("component_id values must be non-empty and unique")
        names.add(name)
        mass = float(item["mass_kg"])
        position = np.asarray(item["center_m"], dtype=np.float64)
        inertia = np.asarray(item.get("inertia_kg_m2", np.zeros((3, 3))), dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{name}: center_m must be a finite 3-vector")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"{name}: mass_kg must be finite and positive")
        if inertia.shape == (3,):
            inertia = np.diag(inertia)
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ValueError(f"{name}: inertia_kg_m2 must be a finite diagonal 3-vector or 3x3 tensor")
        if not np.allclose(inertia, inertia.T, atol=1e-10):
            raise ValueError(f"{name}: component inertia must be symmetric")
        if float(np.min(np.linalg.eigvalsh(inertia))) < -1e-10:
            raise ValueError(f"{name}: component inertia must be positive semidefinite")
        masses.append(mass)
        positions.append(position)
        local_inertias.append(inertia)

    masses_array = np.asarray(masses, dtype=np.float64)
    positions_array = np.asarray(positions, dtype=np.float64)
    total_mass = float(masses_array.sum())
    center = np.sum(positions_array * masses_array[:, None], axis=0) / total_mass
    total_inertia = np.zeros((3, 3), dtype=np.float64)
    identity = np.eye(3)
    for mass, position, local in zip(masses_array, positions_array, local_inertias):
        offset = position - center
        total_inertia += local + mass * (
            float(np.dot(offset, offset)) * identity - np.outer(offset, offset)
        )
    total_inertia = 0.5 * (total_inertia + total_inertia.T)
    eigenvalues = np.linalg.eigvalsh(total_inertia)
    if float(np.min(eigenvalues)) < -1e-9:
        raise ValueError("aggregated inertia is not positive semidefinite")
    return {
        "status": "calculated_from_user_supplied_component_measurements",
        "coordinate_frame": coordinate_frame,
        "component_count": len(components),
        "component_ids": sorted(names),
        "takeoff_mass_kg": total_mass,
        "center_of_mass_m": center.tolist(),
        "inertia_tensor_kg_m2": total_inertia.tolist(),
        "principal_moments_kg_m2": eigenvalues.tolist(),
        "method": "sum_component_centroidal_inertias_plus_parallel_axis_theorem",
        "warning": "Output is only as accurate as the supplied component masses, CG coordinates and local inertias. Empty template values must not be treated as zero-mass aircraft parts.",
    }


def read_component_csv(path: str | Path) -> list[dict]:
    components = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CSV_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if not row.get("component_id", "").strip():
                continue
            required = ("mass_kg", "cg_x_m", "cg_y_m", "cg_z_m")
            if any(row.get(key, "").strip() == "" for key in required):
                raise ValueError(f"line {line_number}: mass and CG coordinates are required")
            values = [float(row[key]) if row.get(key, "").strip() else 0.0 for key in CSV_FIELDS[5:]]
            ixx, iyy, izz, ixy, ixz, iyz = values
            components.append({
                "component_id": row["component_id"].strip(),
                "mass_kg": float(row["mass_kg"]),
                "center_m": [float(row["cg_x_m"]), float(row["cg_y_m"]), float(row["cg_z_m"])],
                "inertia_kg_m2": [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
            })
    return components


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate measured component masses into aircraft mass, CG and inertia.")
    parser.add_argument("input_csv")
    parser.add_argument("output_json")
    parser.add_argument("--frame", default="CAD_EXPORT_Z_UP")
    args = parser.parse_args()
    result = aggregate_mass_properties(read_component_csv(args.input_csv), args.frame)
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
