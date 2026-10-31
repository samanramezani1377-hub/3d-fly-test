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
- `training/run_universal_curriculum.py`: five sequential stages, loading the preceding stage's balanced best evaluated checkpoint before continuing. The final checkpoint is retained separately for reproducibility.
- `training/tests/test_universal_vehicle.py`: profile validation, observation dimensions, deterministic selection, pinning and fail-closed tests.

Example commands are in `training/README.md`.

The test suite also checks nominal static-hover force balance for every example profile and verifies that the configured worst-case mass, voltage and motor-effectiveness bounds in each curriculum stage retain at least a 1.05 thrust-to-weight ratio. During training, periodic family evaluation selects `best/best_model.zip` lexicographically by: (1) maximum minimum per-profile five-second sustained-hover success, (2) minimum worst-profile crash rate, (3) maximum minimum per-profile takeoff rate, and (4) mean return as a final tie-breaker. The last policy snapshot is saved independently as `final_model.zip`. The sustained-hover definition requires takeoff, altitude error ≤0.15 m, horizontal error ≤0.20 m, speed ≤0.35 m/s, absolute roll/pitch ≤0.25 rad, angular-rate norm ≤0.5 rad/s, and five continuous seconds without a crash. These are simulation checks on provisional coefficients, not proof that the real aircraft can hover.

## Reference hover controller and plant feasibility gate

- `training/fast_physics/hover_reference.py` implements a cascaded position/attitude PID controller with bounded integral disturbance rejection, true-wind drag feed-forward and a four-rotor wrench allocator.
- `training/evaluate_hover_reference.py` runs sustained-hover episodes independently for every profile, in nominal or Stage 4 randomized conditions.
- `training/tests/test_hover_reference.py` checks sustained nominal hover for all four profiles.
- CI runs sustained-hover benchmarks for all four profiles. Five episodes per profile achieved 100% success with zero crashes in nominal conditions and again under Stage 4 randomization, including independent wind components up to ±2 m/s. The separate fixed-wind regression test also passes.

**This is a privileged-parameter oracle benchmark:** it reads the exact 12-value simulator state and uses the true per-episode mass, inertia, motor effectiveness, voltage factor, drag coefficients and wind from the simulator. It is intentionally an upper-bound diagnostic to determine whether the simplified plant can be controlled; it is not an observation-only controller, cannot be compared directly with PPO, and is not deployable. If this oracle cannot hold hover, investigate physics, actuator allocation and control conventions before spending compute on PPO. If it succeeds while PPO fails, focus next on PPO reward design, exploration, observation scaling and curriculum.

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


## Reward, crash handling and release criteria (current)

The universal fast-physics environment uses dense altitude and horizontal position tracking, penalties for translational speed, roll/pitch tilt, angular rates, abrupt motor-command changes, and a small squared normalized-command effort proxy. It intentionally does not penalize yaw angle without a defined yaw target. Altitude progress uses potential-based shaping; crossing the takeoff threshold receives a one-time bonus, stable-hover steps receive a shaping bonus, and an episode that reaches its time limit without takeoff receives a takeoff-failure penalty.

The command-effort term is only a normalized control-effort proxy. The plant does not yet contain validated motor current, ESC losses, battery internal resistance or electrical power telemetry, so this term must not be described as measured energy consumption.

Non-finite states are detected before reward/estimator/observation use. The environment substitutes the last finite state for the returned observation, emits a finite terminal penalty and records the failure reason. The ground plane is enforced as a unilateral contact constraint: the simulated vehicle cannot continue below z=0; contact before takeoff is allowed, while contact after takeoff is a crash.

Evaluation reports separately track takeoff success/failure, sustained-hover success, crash rate, maximum altitude, stable-hover duration and merely surviving to the time limit. Surviving to the time limit is not mission completion. The release gate requires minimum per-profile sustained-hover success, maximum per-profile crash rate, and minimum per-profile takeoff success. Simulation acceptance is not physical-flight approval.


## PPO control and learning pipeline v4

The universal PPO path now emits an unbounded four-dimensional Gaussian sample
which is smoothly squashed with tanh and interpreted as collective thrust,
roll torque, pitch torque and yaw torque. A geometry-derived full-rank wrench
allocation matrix maps the requested body wrench into per-rotor thrust and
then into normalized motor commands. The allocator uses the nominal declared
vehicle profile only; randomized episode ground truth remains hidden.

The fast-physics state/action history is dimensionlessly scaled before entering
the policy. Ground takeoff remains required in all five curriculum stages;
small initial horizontal-position and roll/pitch perturbations grow from stage
2 to stage 5. The reward retains altitude progress, takeoff transition and
sustained-hover shaping, with an additional vertical-speed damping term.

The PPO policy/value networks use separate 128-unit layers, reduced initial
Gaussian standard deviation, a 0.15 clipping range, target-KL guard and
gradient-norm cap. These are training design choices to improve conditioning,
not empirically proven optimal hyperparameters.

The existing workflow now stores these incompatible action/observation
semantics under `training/models/universal-quad-v4/`. Do not load v2 or v3
checkpoints into v4. No training run or new test suite is triggered by these
changes. The four included vehicle profiles are provisional estimates; static
mixer rank and nominal thrust-envelope calculations do not validate the real
aircraft or establish sim-to-real transfer.


## PPO learning improvements pipeline v5

Pipeline v5 adds staged teacher guidance to the body-wrench policy. The
privileged hover-reference controller contributes requested body wrench with
weights 0.80 / 0.60 / 0.40 / 0.20 / 0.00 across curriculum stages 1–5.
Its true simulator state, randomized physical parameters and wind are used
only inside the teacher controller; none are added to the PPO observation.
Stage 5 and the final held-out evaluation run with teacher weight zero.

The same action wrapper is used for PPO training, periodic balanced checkpoint
evaluation and standalone evaluation. Mixer desaturation preserves collective
thrust where feasible and scales requested torques down to respect per-rotor
limits. The environment adds vertical-speed damping and reports motor
saturation fraction.

The existing family sampler now updates per-profile sampling weights from
balanced evaluation results, increasing exposure to profiles with weak
takeoff/hover performance or high crash rate. Updates are smoothed, bounded,
and normalized. Checkpoint selection is consistent across in-training and
final held-out selection: weakest-profile sustained-hover success, worst
profile crash rate, weakest takeoff success, stable-hover duration, lower
motor saturation, then episode return.

No new workflow or test suite was created. The existing five-stage workflow
stores v5 checkpoints separately at `training/models/universal-quad-v5/`.
The oracle teacher is a simulation training aid, not a flight controller.
Successful PPO simulation metrics do not establish real-flight safety.


## Five-stage nose-relative velocity and yaw-rate curriculum

The production workflow is `.github/workflows/universal-ppo-five-stage.yml`. It
uses one shared, hardware-conditioned PPO policy across all four profiles in
`vehicle_models/family/vehicle_profiles.json`:

- `smooth2-a2212-custom-prop`
- `generic-5inch-quad`
- `generic-7inch-long-range`
- `generic-3inch-cinewhoop`

No profile is pinned during training. The sampler can adapt exposure using
per-profile evaluation weaknesses. Domain-randomization ranges are proportional
to each profile's own nominal parameters; the same absolute mass, motor, or
thrust constants are not imposed across unlike vehicles.

Each of five curriculum stages chains a 400K-step teacher-assisted phase into a
400K-step teacher-free phase. The teacher is an oracle available only inside
the training action wrapper. It tracks nose-forward/nose-left horizontal
velocity, world-up vertical velocity, and an independent yaw-rate command.
Positive yaw rate turns the nose left; zero translational command with nonzero
yaw-rate command produces an in-place horizontal turn. Stage-specific yaw-rate
limits increase from 20 to 90 degrees per second.

The final held-out evaluation is independent of periodic checkpoint selection:
10 episodes are evaluated separately for each of the four profiles. The
workflow fails if any profile is missing, duplicated, or has fewer than 10
episodes, and stores metrics by vehicle rather than averaging away a weak
profile.

All four profile records currently carry `provisional_estimate` calibration
status. Their simulation results are software research evidence only, not
measured performance, flight clearance, or evidence of sim-to-real transfer.
The Smooth2 propeller geometry and thrust calibration remain deliberately
separate from this training workflow and must be updated only after the
redesign and subsequent motor/propeller characterization are available.
