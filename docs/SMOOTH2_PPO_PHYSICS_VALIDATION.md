# Smooth2 PPO / physics integration — engineering notes

Status: audit and implementation notes for PR #30. This document is not an authorization to start PPO training or change the five-stage workflow.

## Scope and protected files

- Target branch: `fix/smooth2-only-ppo-curriculum` (PR #30).
- The five-stage workflow `.github/workflows/universal-ppo-five-stage.yml` is protected from edits under the current instruction. Do not modify it unless the owner first receives a complete change proposal and explicitly approves that exact change.
- Do not modify the source CAD repository (`3D`, branch `kilo`). Smooth2 geometry and propeller inputs are consumed as source data only.
- PR #23 is out of scope.
- Do not start PPO training, dispatch the five-stage workflow, or run expensive model optimization during this audit.

## Intended training contract (documented, not executed)

Five sequential stages, each with:
- Phase A: 400,000 transitions, teacher blend weight 0.80.
- Phase B: 400,000 transitions, teacher disabled and blend weight 0.00.
- Phase A best evaluated checkpoint initializes Phase B in the same stage.
- Phase B best evaluated checkpoint initializes Phase A of the next stage.
- Total new training budget: 4,000,000 transitions.
- The selected legacy 1M checkpoint is a policy warm start only. It is not evidence of calibration or prior training under the current Smooth2 BEMT plant.
- Periodic checkpoint selection and independent final evaluation must be teacher-free.
- Only `smooth2-a2212-custom-prop` is eligible for training and evaluation.
- Evaluation command set includes 26 signed 3D translation directions, four pure in-place yaw commands (left/right at two magnitudes), combined translation/yaw, and neutral linear/angular braking.

## Physics model and provenance

The Smooth2 profile is marked `cad_bemt_estimate`. It uses the CAD-station blade-element/momentum propeller model with Prandtl tip-loss treatment and axial inflow in the runtime plant. The propeller model returns rotor thrust and shaft torque; runtime dynamics applies the thrust force, reaction torque, body rigid-body dynamics, motor lag, and optional rotor gyroscopic coupling.

The current CAD-derived estimates are provisional:
- Mass baseline: approximately 0.60 kg.
- CG and inertia tensor: derived from component-lumped geometric estimates and parallel-axis calculations.
- Stated uncertainty in the calibration record: mass ±25%, CG ±15 mm, inertia ±35%.
- Aerodynamic drag is an anisotropic body-axis engineering estimate, not CFD or wind-tunnel calibration.
- BEMT polars and motor/propeller performance are not thrust-stand measurements. Guard interference, Reynolds effects, manufacturing tolerances, battery sag and motor variation remain uncertain.

These values must never be described as measured hardware properties or used to claim physical-flight readiness. Physical flight remains disabled until traceable mass/CG/inertia measurements and propulsion tests replace the estimates.

## Audit findings on the current PR branch

### Confirmed by source inspection

1. `train_universal_ppo.py` accepts repeatable `--vehicle-id` filters and writes a selected-profile JSON for the training environment. The PR workflow passes the Smooth2 ID for the two phase commands.
2. `VehicleFamilyEnv` supports a pinned vehicle ID; the training command's generated selected-profile file and the callback's selected profile list constrain periodic evaluation to the selected set.
3. The preflight script selects exactly the Smooth2 profile and requests three randomized teacher-only episodes. Its mission sequence contains 26 signed translation commands plus four pure yaw commands; it also measures the neutral braking tail.
4. Runtime dynamics exposes `thrust_model_status` with the exact BEMT status string used by the current implementation:
   `runtime_CAD_station_BEMT_with_axial_inflow_and_randomized_effectiveness`.
5. The BEMT model's `evaluate(omega, axial_inflow)` API returns thrust and positive shaft-torque magnitude.
6. Parameter validation checks finite positive mass, symmetric positive-definite inertia, four rotor positions, spin signs and positive propulsion parameters.

### Open engineering blockers / limitations

1. **Yaw allocation is not yet BEMT-consistent.** The plant uses BEMT shaft torque, but `ppo_action.py` still builds its yaw row from the nominal `torque_to_thrust_m` proxy. This means the wrench allocator's yaw prediction is not based on the actual BEMT torque-vs-RPM curve. Do not describe the actuator allocator as fully BEMT-aligned until this is corrected and validated.
2. **Static allocator lookup is an approximation.** The thrust-to-RPM inverse is generated from zero-axial-inflow BEMT. Runtime plant evaluation includes axial inflow. This is an acceptable documented static feed-forward approximation only if saturation and tracking tests demonstrate adequate robustness; it is not an exact inverse of the in-flight plant.
3. **Preflight is not a complete physics certification.** The current preflight primarily checks teacher-guided mission capability. It does not, by itself, prove allocator fidelity, BEMT torque tracking, mass-property correctness, or PPO policy competence. The report's three episodes are a small deterministic randomized sample.
4. **Mass properties remain estimates.** Positive-definite tensor checks prove numerical validity, not physical accuracy.
5. **CI wiring is a separate authorization boundary.** The protected five-stage workflow currently has `workflow_dispatch` as its trigger. The repository README describes other push/PR validation workflows, but new Smooth2-specific checks must be placed in an existing non-training CI workflow (or an already existing validation job) to run automatically. This PR must not edit the protected five-stage workflow without explicit approval.

## Required validation matrix (non-training CI only)

These checks are intended for the repository's existing automatic, non-training CI. They must not invoke PPO learning, download training checkpoints, or dispatch the five-stage workflow.

- Static: Python compile checks for changed physics/controller modules.
- Data contract: parse the Smooth2 profile, verify exactly four rotors, CAD frame conversion, BEMT stations and integration resolution, positive mass, CG shape, symmetric positive-definite inertia, and declared uncertainty/provenance.
- Propeller: BEMT thrust and shaft torque finite/non-negative at zero, intermediate and maximum RPM; static curve monotonicity; positive/negative axial inflow cases; no NaN/Inf.
- Dynamics: assert runtime BEMT status, four per-rotor thrust/shaft-torque outputs, finite rigid-body state, and correct mass/inertia use.
- Allocator: collective/roll/pitch/yaw wrench round-trip and saturation tests; after yaw allocator correction, compare requested and realized yaw torque against BEMT shaft torque.
- Teacher: Smooth2-only, teacher-only, three deterministic randomized episodes; coverage and braking thresholds are reported separately from PPO results.
- Scope: assert no non-Smooth2 profile enters training selection, periodic evaluation, or final evaluation.
- Curriculum contract: parse/inspect the training plan and verify 5 × (400K + 400K), weights 0.80/0.00, checkpoint handoff, and teacher-free periodic/final evaluation without launching training.

All of the above belong in the existing automatic CI path that runs physics/vehicle/CAD validation. Do not add a new workflow, and do not put unrelated checks into the PPO training job.

## Approval required before touching the five-stage workflow

Before changing `.github/workflows/universal-ppo-five-stage.yml`, provide the owner with:
1. The exact proposed diff.
2. Which existing automatic non-training CI job will execute the new checks.
3. Confirmation that no training command, checkpoint download, or training artifact publication is part of that CI job.
4. Expected runtime and dependency impact.

Wait for explicit approval of that proposal. Until then, leave the five-stage workflow byte-for-byte unchanged.

## Current execution state

- PPO training: not run.
- Five-stage workflow: not dispatched by this audit.
- Merge: not performed.
- Physical flight: not approved.
