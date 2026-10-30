# Universal hardware-conditioned quadrotor family

## Purpose

Train one PPO policy across multiple quadrotor configurations. At episode reset, the environment selects a named vehicle profile and exposes a normalized descriptor of its **declared nominal hardware configuration** to the policy. The policy then sees the same state/action history and online estimator context used by the fast-physics environment.

Episode-specific disturbances and randomized physical parameters remain hidden. This preserves the distinction between (a) what the configuration file says about the nominal aircraft and (b) the actual episode realization the controller must handle robustly.

## Profile contract

Profiles live in `vehicle_models/family/vehicle_profiles.json`. Each profile contains:

- geometry: rotor count, arm span and propeller diameter;
- physical: complete takeoff mass, CG, inertia and drag placeholders;
- four rotor positions and spin directions;
- propulsion: KV, maximum rotor speed, thrust coefficient, torque/thrust ratio, motor time constant, battery voltage and capacity;
- calibration status and provenance.

`provisional_estimate`, `user_supplied`, and `measured` are distinct statuses. A user profile must supply all fields needed by the current simplified dynamics. Missing required fields fail closed. A profile marked measured is only as trustworthy as its attached measurement record; the software does not independently certify measurements.

## Shared policy observation

The current policy receives the existing environment observation plus twelve dimensionless nominal hardware/provenance features:

1. nominal mass;
2. mean rotor arm radius;
3. propeller diameter;
4. configured maximum rotor angular speed;
5. configured thrust coefficient;
6. torque/thrust ratio;
7. motor time constant;
8. nominal battery voltage;
9. battery capacity;
10. motor KV;
11. fraction of the profile values that were derived estimates;
12. calibration provenance confidence (a software label, not a measurement-quality certification).

The policy does **not** receive episode-randomized true mass, CG offset, inertia scale, per-motor effectiveness, battery sag or wind. The existing online effective-lift estimator remains observation-derived and approximate; it is not a full system-identification solution.

The resulting observation has 78 values with the default four history frames: 66 environment values plus 12 hardware/provenance features. Existing 66-value smooth2-only checkpoints are not observation-compatible.

## Training and evaluation

- `training/train_universal_ppo.py`: one shared PPO model; random vehicle selection each episode; periodic evaluation writes `best/best_model.zip` and the terminal policy separately as `final_model.zip`.
- `training/evaluate_universal_ppo.py`: deterministic checkpoint evaluation, separately grouped by vehicle profile.
- `training/run_universal_curriculum.py`: four sequential stages, loading the preceding stage's best evaluated checkpoint before continuing. The final checkpoint is retained separately for reproducibility.
- `training/tests/test_universal_vehicle.py`: profile validation, observation dimensions, deterministic selection, pinning and fail-closed tests.

Example commands are in `training/README.md`.

The test suite also checks nominal static-hover force balance for every example profile and verifies that the configured worst-case mass, voltage and motor-effectiveness bounds in each curriculum stage retain at least a 1.05 thrust-to-weight ratio. These are internal consistency checks on provisional coefficients, not proof that the real aircraft can hover.

## Calibration workflow for a new aircraft

1. Copy `vehicle_config_template.json` and assign a unique vehicle ID.
2. Measure total takeoff mass with battery and intended payload.
3. Measure CG in the coordinate frame used by the CAD/physics model.
4. Obtain a verified inertia tensor from a validated CAD mass model or physical measurement.
5. Record rotor locations, numbering and spin directions.
6. Identify exact motor, ESC, battery and propeller; collect loaded RPM, voltage, current, thrust and torque over a guarded test matrix.
7. Measure motor/ESC response to command steps and thermal/current limits.
8. Record provenance, measurement uncertainty and test conditions; set calibration status honestly.
9. Run profile validation, train/continue the shared policy, and evaluate on held-out seeds and the new profile.
10. Perform separate bench, restrained and controlled flight validation with independent failsafes.

## Limitations and safety boundary

This is a software training architecture, not a universal flight controller. The example profiles are synthetic engineering estimates. They are useful for exercising multi-domain training code, not for predicting real flight. The simplified fast-physics model does not yet include a validated aerodynamic model for the custom CAD propeller, full motor electrical dynamics, ESC behavior, battery equivalent-circuit behavior, sensor/estimator stack or flight-controller timing. Training success on these profiles is not evidence of sim-to-real transfer. No checkpoint produced by this path is eligible for physical flight without aircraft-specific calibration, independent validation and a separate safety review.
