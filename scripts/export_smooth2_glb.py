#!/usr/bin/env python3
"""Build a named, browser-ready GLB from the committed Smooth2 CAD STL parts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
CAD_DIR = ROOT / "output" / "smooth2-parametric"

PART_COLORS = {
    "body_lower": [37, 55, 63, 255],
    "body_crown": [54, 78, 86, 255],
}


def load_part(name: str) -> trimesh.Trimesh:
    path = CAD_DIR / (name + ".stl")
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty CAD component STL: {path.relative_to(ROOT)}")
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty or len(mesh.faces) == 0:
        raise SystemExit(f"Could not load CAD component as a mesh: {name}")
    if not mesh.is_watertight:
        raise SystemExit(f"CAD component is not watertight: {name}")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads((CAD_DIR / "assembly-manifest.json").read_text(encoding="utf-8"))
    parts = manifest.get("parts", [])
    names = [part["name"] for part in parts]
    if len(names) != 26 or len(set(names)) != len(names):
        raise SystemExit(f"Expected 26 unique CAD parts; found {len(names)}")

    params = json.loads((ROOT / "cad" / "parametric" / "model.params.json").read_text(encoding="utf-8"))
    scale = float(params["model_scale"])
    rotor_plane_z_mm = (float(params["prop_y"]) + float(params["prop_thickness"]) * 0.5) * scale
    prop_bounds = []
    loaded = {}
    for name in names:
        mesh = load_part(name)
        loaded[name] = mesh
        if name.startswith("prop_"):
            prop_bounds.append(mesh.bounds[:, 2])
    if len(prop_bounds) != 4:
        raise SystemExit("The current CAD assembly must contain four separate propeller meshes")
    for index, bounds in enumerate(prop_bounds, 1):
        if not float(bounds[0]) - 0.5 <= rotor_plane_z_mm <= float(bounds[1]) + 0.5:
            raise SystemExit(f"CAD propeller {index} does not cross the parametric rotor plane")

    # CAD export is X/Y/Z-up. The Flight Lab scene uses X/right, Y/up, Z/depth.
    # This is a proper +90-degree X rotation: (x, y, z) -> (x, z, -y).
    cad_to_three = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    scene = trimesh.Scene()
    motor_x_mm = float(params["motor_x"]) * scale
    motor_z_mm = float(params["motor_z"]) * scale
    prop_stations = {
        "prop_1": (motor_x_mm, motor_z_mm),
        "prop_2": (motor_x_mm, -motor_z_mm),
        "prop_3": (-motor_x_mm, motor_z_mm),
        "prop_4": (-motor_x_mm, -motor_z_mm),
    }
    for name in names:
        mesh = loaded[name].copy()
        vertices = (np.asarray(mesh.vertices, dtype=np.float64) @ cad_to_three.T) * 0.001
        vertices[:, 1] -= rotor_plane_z_mm * 0.001
        if name in prop_stations:
            station_x_mm, station_z_mm = prop_stations[name]
            cad_pivot = np.array([station_x_mm, -station_z_mm, rotor_plane_z_mm])
            if np.any(cad_pivot < mesh.bounds[0] - 0.5) or np.any(cad_pivot > mesh.bounds[1] + 0.5):
                raise SystemExit(f"{name} CAD hub axis does not match its parametric motor station")
            centre = np.array([station_x_mm * 0.001, 0.0, station_z_mm * 0.001])
        else:
            centre = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        mesh.vertices = vertices - centre
        color = PART_COLORS.get(
            name,
            [72, 91, 101, 255] if name.startswith("arm_")
            else [119, 153, 164, 255] if name.startswith("prop_")
            else [105, 113, 119, 255],
        )
        mesh.visual.face_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.faces), 1))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = centre
        scene.add_geometry(mesh, geom_name=name, node_name=name, transform=transform)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    glb_bytes = scene.export(file_type="glb")
    if not isinstance(glb_bytes, (bytes, bytearray)) or len(glb_bytes) < 1024:
        raise SystemExit("Trimesh did not produce a valid non-empty GLB payload")
    args.output.write_bytes(glb_bytes)

    check = trimesh.load(args.output, force="scene")
    exported_nodes = set(check.graph.nodes_geometry)
    missing = sorted(set("prop_" + str(i) for i in range(1, 5)) - exported_nodes)
    if missing:
        raise SystemExit("GLB export lost named animated propeller nodes: " + ", ".join(missing))
    if len(exported_nodes) != len(names):
        raise SystemExit(f"GLB node count mismatch: expected {len(names)}, got {len(exported_nodes)}")
    print(f"Generated {args.output} ({args.output.stat().st_size} bytes)")
    print(f"  {len(exported_nodes)} named components; parametric rotor plane at {rotor_plane_z_mm:.3f} mm")
    print("  units: metres; up axis: +Y; front: +Z; propellers remain separate and animatable")


if __name__ == "__main__":
    main()
