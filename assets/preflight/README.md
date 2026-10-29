# Preflight engineering assets

This directory contains the imported CAD model and the engineering data contract for preflight analysis. It is separate from PPO training. Unverified physical parameters must never be treated as measured facts.

## Current model

See [smooth2-parametric](./smooth2-parametric/README.md), the [assembly manifest](./smooth2-parametric/assembly-manifest.json), [design parameters](./smooth2-parametric/drone_parametric.params.json), [stage-1 measurements](./smooth2-parametric/stage1-measurements.json), and [geometry validation report](./smooth2-parametric/validation-report.json).

The repository `3d-fly-test` is the working location. No workflow here should write to the source 3D repository.

## Recorded geometry values and discrepancies

| Item | Recorded value | Qualification |
|---|---:|---|
| Rotors | 4 | Assembly manifest |
| Blades per propeller | 2 | Validation report |
| Airfoil concept | NACA 4410 | Parametric design documentation |
| Propeller diameter | 74.802 mm | Validation report; conflicts with older 63.5 mm README value |
| Geometric pitch | 53.086 mm / 2.09 in | Design value, not measured aerodynamic pitch |
| Airfoil camber / thickness | 0.04 / 0.10 | Parametric snapshot |
| Assembly bounding box | 271.890 × 271.856 × 49.938 mm | Validation report; conflicts with older ~247 mm documentation |

Do not silently choose between conflicting dimensions. Re-measure exported STEP geometry and document the CAD-kernel version, units and method.

## Asset inventory

The imported directory contains STEP files for the lower body, crown, four arm/motor/guard modules, four separate propellers, eight screw placeholders and eight nut placeholders. The full assembly is split into six chunks and rebuilt with `smooth2-parametric/rebuild_assembly.py`. STL files are referenced by the manifest but are not assumed to be present in this import.

## Engineering gates

1. Structural asset integrity.
2. CAD-kernel BREP validation, transforms, dimensions and interference checks.
3. Verified mass, center of gravity and inertia (CAD volume alone is not mass).
4. Exact motor/ESC/battery data and motor-propeller characterization.
5. Rotor and installed-airframe aerodynamic model correlated to bench data.
6. 6-DoF dynamics and baseline hover-controller validation.
7. PPO integration only after the plant and baseline controller pass.

Structural file checks do not establish aerodynamic performance or flight readiness. See [analysis plan](./ANALYSIS_PLAN.md) and [data dictionary](./DATA_DICTIONARY.md).