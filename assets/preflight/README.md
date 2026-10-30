# Preflight engineering assets

This directory contains engineering plans and the data contract for preflight analysis. The authoritative CAD package is stored in the same layout as its source repository under `output/smooth2-parametric/`; it is deliberately separate from PPO training. Unverified physical parameters must never be treated as measured facts.

## Authoritative model source

- Source repository: [3D](https://github.com/samanramezani1377-hub/3D)
- Source branch: `kilo`
- Immutable source commit: `d591b81bd671e04abe7d4bdeb0865fd3c3bf381b`
- Parametric builder: [cad/parametric/drone_parametric.py](../../cad/parametric/drone_parametric.py)
- Canonical parameters: [cad/parametric/model.params.json](../../cad/parametric/model.params.json)
- Measured body sections: [cad/parametric/body_sections.json](../../cad/parametric/body_sections.json)
- Pinned CAD builder dependencies: [cad/parametric/requirements.txt](../../cad/parametric/requirements.txt)
- Generated package: [output/smooth2-parametric](../../output/smooth2-parametric/README.md)
- Assembly manifest: [assembly-manifest.json](../../output/smooth2-parametric/assembly-manifest.json)
- Validation report: [validation-report.json](../../output/smooth2-parametric/validation-report.json)
- Provenance record: [MODEL_SOURCE_REVISION.md](./MODEL_SOURCE_REVISION.md)

The CAD builder, canonical parameters, body sections, STEP/STL exports, assembly manifest, parameter snapshot and validation report are synchronized from that one source revision. Do not combine model files from other branches or older exports.

## Recorded design and validation values

| Item | Value | Qualification |
|---|---:|---|
| Assembly solids | 26 | Source validation report |
| Rotors | 4 | Assembly manifest |
| Blades per propeller | 2 | Parametric design |
| Airfoil parameters | 4% camber, 40% camber position, 10% thickness | Parametric design; NACA 4410-derived |
| Propeller radius | 26.45 mm design radius | Before model scale |
| Propeller pitch | 53.086 mm | Geometric design pitch, not measured aerodynamic pitch |
| Model scale | 1.414028 | Canonical CAD parameter |
| Assembly bounding box | 271.890 × 271.856 × 49.938 mm | Source CAD validation report |

Older imported notes referring to six split STEP chunks, missing STL exports, a 63.5 mm propeller diameter, or an approximately 247 mm footprint describe superseded model packages. The current kilo package contains a complete assembly STEP, individual component STEP/STL files, and assembled/exploded STL exports.

## Engineering gates

1. Structural asset integrity and source-hash consistency.
2. CAD-kernel BREP validation, transforms, dimensions and interference checks.
3. Verified mass, centre of gravity and inertia (CAD volume alone is not mass).
4. Exact motor/ESC/battery data and motor-propeller characterization on the actual custom propeller.
5. Rotor and installed-airframe aerodynamic model correlated to bench data.
6. 6-DoF dynamics and baseline hover-controller validation.
7. PPO integration only after the plant and baseline controller pass.

Structural file checks do not establish aerodynamic performance or flight readiness. See [analysis plan](./ANALYSIS_PLAN.md) and [data dictionary](./DATA_DICTIONARY.md).
