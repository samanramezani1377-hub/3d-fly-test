# Robust and Adaptive Flight Learning — Fast Physics Architecture

**Status:** Experimental software implementation; physical calibration incomplete  
**Initial vehicle:** `smooth2-parametric-v1`  
**Primary objective:** train one policy across a family of uncertain physical configurations without exposing privileged simulator parameters to the policy.

## 1. Executive summary

The project is moving from a single-configuration hover policy toward a robust adaptive flight-control research platform. Version 1 deliberately keeps the imported smooth2-parametric airframe geometry fixed and varies uncertain physical properties around a documented reference model.

The selected engineering trade-off is **fast physics + balanced accuracy/performance**:

- a deterministic 6-degree-of-freedom rigid-body plant for many training episodes;
- first-order rotor actuator dynamics and a configurable thrust/torque model;
- episode-level domain randomization for mass, centre of mass, inertia, motor effectiveness, battery voltage factor, drag and wind;
- a short state/action history window so a feed-forward policy can infer changes from observed response;
- a separate calibration record that refuses to silently invent missing physical data;
- a new PPO entry point that does not replace or silently alter the existing SkySim/NativeSkySimEnv pipeline.

This is not CFD, not a validated model of the real aircraft, and not a real-flight clearance.

## 2. Fixed baseline airframe

The only v1 airframe is:

- ID: `smooth2-parametric-v1`
- CAD source: `cad/parametric/drone_parametric.py`
- authoritative generated engineering assets: `output/smooth2-parametric/`
- parameter snapshot: `output/smooth2-parametric/drone_parametric.params.json`
- assembly description: `output/smooth2-parametric/assembly-manifest.json`
- normalized model definition: `vehicle_models/smooth2_parametric/base_airframe.json`

The geometry is held constant in v1:

- central body and canopy geometry;
- four arm/motor/guard assemblies;
- four propellers and their design geometry;
- motor mounting positions and rotor axes;
- camera and guard geometry.

The CAD parameter snapshot records `motor_x = 58 mm`, `motor_z = 58 mm`, `model_scale = 1.414028`, and `prop_radius = 26.45 mm`. The normalized rotor centres therefore use approximately 82.014 mm offsets in the horizontal plane. The CAD export is Z-up. The CAD export frame has X=lateral, Y=longitudinal with the nose at negative Y, and Z=up. The rigid-body plant uses `BODY_FORWARD_LEFT_UP` (X=forward, Y=left, Z=up), so the CAD-to-body transform is `(x_body, y_body, z_body) = (-Y_cad, X_cad, Z_cad)`. Rotor locations, centre of mass and inertia are transformed together; the transform is right-handed and is applied once by the profile/airframe loaders. World yaw zero points along world +X, and positive yaw turns toward world +Y. Regression tests protect the convention.

### Geometry discrepancies

The authoritative CAD package is pinned to the `3D` repository's `kilo` branch commit `d591b81bd671e04abe7d4bdeb0865fd3c3bf381b`. Its committed validation report records a 271.890 × 271.856 × 49.938 mm assembly bounding box and the 74.802 mm scaled propeller diameter. Earlier 247 mm / 63.5 mm figures belong to superseded imports. These are CAD-derived dimensions, not certified manufacturing measurements; repeat BREP and physical print-fit checks before fabrication.

The alternating rotor spin signs in the normalized model are a conventional quadrotor assumption. Confirm motor numbering and actual CW/CCW propeller directions before using yaw torque results as aircraft-specific.

## 3. Separate geometry from physical calibration

Geometry and physics are intentionally separate records.

### Fixed geometry

`base_airframe.json` stores the source, frame, design dimensions, rotor count, rotor positions, rotor axes and assumed spin signs.

### Unknown or calibrated physical properties

`physics_calibration.json` is the only source for aircraft-specific mass and propulsion values. Unknown values are represented as JSON `null`, not zero and not copied from legacy simulator defaults.

Required measured/manufacturer-backed values include:

- complete takeoff mass (airframe + motors + ESCs + propellers + battery + wiring + payload);
- centre of mass in the declared body frame;
- full symmetric positive-definite 3×3 inertia tensor;
- exact motor model and version;
- ESC limits and motor time constant / response data;
- exact battery cell count, loaded voltage and internal resistance;
- per-rotor thrust and reaction torque versus RPM, voltage and preferably inflow;
- actual rotor direction and motor numbering.

The current loader supports diagonal inertia and a full 3×3 inertia tensor. The baseline record deliberately leaves the real inertia unknown.

### Synthetic values

The source contains a clearly labelled synthetic set to let software tests and algorithm prototypes execute. It is never silently selected. The caller must pass `allow_synthetic_calibration=True`, or the PPO CLI flag `--software-only-synthetic-calibration`.

Synthetic results are for software plumbing, numerical stability and algorithm experiments only. They are not evidence that this aircraft can hover, not aircraft-specific performance estimates, and not suitable for real hardware.

## 4. Fast plant equations and assumptions

The plant is implemented in `training/fast_physics/dynamics.py`.

### Rotor actuator

Each normalized command is clipped to [0, 1]. The first-order motor response is discretized exactly over each simulation step:

[
omega_{k+1} = omega_k + (1-e^{-\Delta t/\tau_m})(omega_{cmd}-omega_k)
]

The command speed includes the configured voltage factor and maximum rotor speed.

### Rotor thrust and reaction torque

The current fast approximation is:

[
T_i = k_f omega_i^2 eta_i
]

[
Q_i = s_i k_Q T_i
]

where (k_f), (k_Q), the body reaction-torque sign (s_i), and motor effectiveness (eta_i) are explicit model parameters. Rotor angular momentum uses the opposite sign to the body reaction torque. This is a compact fitted/parametric rotor model, not a BEMT solver. It must be fitted to the exact motor + propeller + voltage test data before physical interpretation.

The propeller geometry is preserved as metadata for later BEMT integration; geometry alone does not determine a reliable thrust curve.

### Rigid-body translation

The four rotor forces are summed in the forward-left-up body frame and rotated to the world frame. Wind-relative velocity is transformed back into the body frame. Optional per-axis drag coefficients represent different forward, lateral and vertical resistance; when they are unavailable, the legacy scalar coefficients are used isotropically:

[
F_d^b = -C_1 v_{air}^b - C_2 (|v_{air}^b| \odot v_{air}^b),\quad F_d^w=R F_d^b
]

[
dot v = (R F_{rotor}+F_d^w)/m + [0,0,-g]
]

### Rigid-body rotation

Rotor moment is computed from each rotor position relative to the episode centre of mass:

[
	au = sum_i (r_i-r_{CG})	imes F_i + 	au_{reaction}
]

Angular motion uses the rigid-body equation, including optional rotor angular-momentum coupling when rotor polar inertia is calibrated:

[
I\dot{\omega} = \tau - \omega\times(I\omega+H_r) - \dot H_r - D_\omega\omega
]

Attitude is integrated with a normalized body-to-world quaternion using the midpoint body rate; Euler angles are derived only for the existing policy/state interface. This removes the Euler-rate singularity from the plant integration. The public policy state still uses Euler angles, so near-inverted flight and extreme manoeuvres need additional observation/controller work.

### Fidelity limits

Not modelled as high-fidelity physics in v1:

- blade-element/momentum-theory inflow solution and advance-ratio-dependent thrust/torque;
- blade Reynolds/Mach effects and airfoil polar lookup;
- rotor-rotor interference and rotor/guard/body flow interaction;
- ground effect and vortex-ring-state dynamics;
- blade flexibility, structural resonance and propeller imbalance;
- ESC commutation, motor heating, current draw, battery sag and detailed electrical dynamics;
- detailed battery state-of-charge and thermal model;
- full sensor/estimator dynamics, delays and correlated noise;
- collision geometry from STEP solids.

The plant is therefore a stronger 6-DoF control-learning approximation with explicit uncertainty, not a complete high-fidelity aerodynamic simulator. In particular, the newly supported body-axis drag and rotor gyroscopic terms remain disabled/unparameterized until measured values are supplied. Correct coordinate handling and sign tests prove internal consistency, not aircraft-specific accuracy.

## 5. Domain randomization and family of scenarios

`domain_randomization.json` defines initial software experiment bounds. A new physical parameter set is sampled at reset; the airframe geometry does not change.

Randomized values:

| Parameter | Initial range | Reason |
|---|---:|---|
| Total mass multiplier | 0.80–1.25 | added/removed payload and battery variation |
| CG offset, all body axes | ±15 mm | payload placement and assembly variation |
| Inertia multiplier per axis | 0.70–1.40 | distribution of mass changes |
| Per-motor effectiveness | 0.90–1.10 | motor/propeller mismatch |
| Voltage factor | 0.82–1.00 | loaded battery / sag proxy |
| Drag multiplier | 0.70–1.40 | aerodynamic uncertainty |
| Wind component | ±2 m/s | disturbance robustness |

These are starter bounds, not certified aircraft limits. They must be constrained by measured mass limits, structural strength, motor thermal/current limits, battery discharge capability and a safe operating envelope. Randomization must never create physically impossible or unsafe test cases without labelling them as deliberate stress tests.

The configuration also describes a four-stage curriculum:

1. baseline;
2. mass and CG variation;
3. actuator and battery variation;
4. combined robustness.

The environment can select an individual curriculum stage. `training/run_robust_curriculum.py` chains stages 1–4 and warm-starts each stage from the previous model. It is a simple sequential curriculum runner; it does not yet implement automatic performance-based range expansion.

## 6. What the policy can and cannot see

The environment returns a 4-frame history. Each frame contains:

- position relative to target;
- world velocity;
- roll, pitch and yaw;
- body angular rates;
- the preceding four rotor commands.

The observation contains 64 history values plus two online-estimator context values (effective lift-response estimate and confidence), for 66 values at a history length of four.

The policy does **not** receive mass, centre of mass, inertia, motor effectiveness, voltage factor or wind as observation features. Those values are retained in the environment info/telemetry for experiment analysis only.

This makes the control task partially observable. A feed-forward MLP can use the short history as a crude adaptation/context signal. It is not equivalent to a recurrent policy and does not guarantee identification of unobservable parameters. Future options include RecurrentPPO/LSTM, a learned context encoder, and a model-based online estimator.

### Identifiability caveat

Different causes can produce similar observed motion. For example, higher mass, reduced motor effectiveness, battery sag and a vertical wind disturbance can all change altitude response. The estimator must represent uncertainty and avoid claiming a unique physical diagnosis when the available observations cannot distinguish causes.

## 7. Software architecture

```text
vehicle_models/smooth2_parametric/
  base_airframe.json          fixed CAD-derived geometry
  physics_calibration.json    measured values or explicit unknowns
  domain_randomization.json   bounded scenario distribution

training/fast_physics/
  parameters.py               validation, loading, domain randomization
  dynamics.py                 fast 6-DoF plant
  env.py                      Gymnasium environment + observation history

training/train_robust_ppo.py  new PPO entry point
training/tests/test_fast_physics.py
docs/ROBUST_ADAPTIVE_FLIGHT_ARCHITECTURE.md
```

The existing SkySim native/WASM path and `training/native_env.py` remain intact. The new fast plant is an additive experimental backend. Do not silently swap it into the existing five-stage production training workflow. First compare state/action conventions, motor allocation, thrust mapping, reset behaviour, and repeatable trajectories. A separate adapter can be implemented after these comparisons.

## 8. Training entry point

Install the existing training requirements:

```bash
python -m pip install -r training/requirements.txt
```

A software-only smoke experiment may be started explicitly with synthetic values:

```bash
python training/train_robust_ppo.py \
  --timesteps 100000 \
  --n-envs 4 \
  --run-name robust-ppo-smooth2-software-only \
  --software-only-synthetic-calibration
```

Without the explicit flag, startup is blocked while required aircraft calibration fields remain unknown. This is intentional.

Each run stores a model and metadata. Metadata identifies the plant, policy history, randomization state and whether synthetic calibration was used. A run trained with synthetic calibration is always marked `physical_flight_eligible: false`.

The PPO entry point is experimental and does not inherit the existing native SkySim five-stage checkpoint chain. Do not mix its checkpoints with the native SkySim policy.

## 9. Validation protocol

### Unit and deterministic checks

- unknown calibration blocks default loading;
- synthetic calibration requires explicit opt-in;
- four CAD rotor definitions and fixed geometry are present;
- plant state and telemetry remain finite;
- randomized scenarios are reproducible from a seed;
- mass, CG, inertia, motor and voltage parameters vary within configured bounds;
- policy observation has the expected shape and excludes privileged parameters;
- invalid action shapes fail fast.

### Plant validation required before aircraft-specific use

- remeasure STEP geometry and confirm units, transforms, rotor centres and propeller dimensions;
- weigh each component and complete takeoff configuration;
- measure CG and calculate full inertia tensor;
- confirm rotor spin directions and motor numbering;
- characterize motor/ESC/propeller static thrust, torque, RPM, voltage, current and temperature;
- fit the rotor model on one subset of test points and validate on held-out points;
- compare hover throttle and response against bench data;
- validate motor lag, battery sag and drag independently;
- test coordinate transforms with known vectors;
- compare the fast plant against the native SkySim backend only after matching conventions and calibration.

### Robustness evaluation

Keep three non-overlapping scenario sets:

- Train: randomized scenarios used for gradient updates;
- Validation: different seeds and held-out combinations for model selection;
- Unseen test: configurations and disturbance combinations never used during training or tuning.

The dedicated `training/evaluate_robust_ppo.py` script evaluates checkpoints on held-out episode seeds and writes per-episode plus aggregate reports. The curriculum runner invokes it after every stage. These are held-out random seeds from the declared distribution, not proof of extrapolation outside the configured ranges.

Report at minimum:

- takeoff and sustained-hover success rates;
- crash/termination rate;
- position and altitude error;
- attitude and angular-rate error;
- recovery time after disturbances;
- motor saturation duration;
- energy estimate and voltage margin (once the electrical model supports them);
- adaptation delay after a parameter change;
- worst-case and lower-percentile performance, not only mean reward;
- out-of-distribution detection and safe-abort rate.

Never select a checkpoint on mean reward alone. A policy that improves reward while increasing crash probability or motor saturation must be rejected.

## 10. Online adaptation roadmap

The current version includes a bounded recursive-least-squares estimator of effective collective lift response. It uses recent motor commands and observed vertical velocity only; its two context outputs are included in the policy observation. It does not uniquely identify mass, motor degradation, battery sag or wind. It does not update PPO weights online during flight.

Recommended staged additions:

1. **Residual monitor:** compare predicted response with measured state change.
2. **Parameter estimator:** estimate bounded effective mass/thrust/response parameters with confidence intervals.
3. **Context encoder:** summarize recent input/output history for the policy.
4. **Adaptive controller interface:** condition the policy on the context estimate, never on inaccessible ground-truth simulator parameters.
5. **Safety supervisor:** enforce attitude, altitude, voltage, motor and uncertainty bounds independently of PPO.
6. **Continual-learning harness:** replay prior vehicle configurations and reject updates that cause catastrophic forgetting.

Online adaptation should initially update a small context state, not freely rewrite all PPO weights during operation. This is easier to test and safer to constrain.

## 11. Safety and operating boundary

This repository is for simulation and software research. No learned policy is cleared for real aircraft operation.

Before any hardware flight, a separate engineering process is required: guarded thrust-stand tests, propeller balancing and structural inspection, motor/ESC thermal and current limits, battery safety, independent failsafe, containment, staged tethered tests and qualified supervision. Printed propellers must not be spun at high RPM without appropriate material, balance and containment validation.

If the model is outside the validated parameter envelope, the correct response is to report uncertainty and inhibit/abort the test—not to claim guaranteed adaptation.

## 12. Current implementation status

| Capability | Status |
|---|---|
| Fixed smooth2 CAD baseline record | Implemented |
| Geometry/physics data separation | Implemented |
| Unknown physical values remain null | Implemented |
| Fast 6-DoF rigid-body plant | Implemented (simplified) |
| Motor first-order lag + quadratic thrust | Implemented (requires calibration) |
| Full 3×3 inertia support | Implemented in parameter model |
| Episode randomization of mass/CG/inertia/actuators/battery/drag | Implemented |
| Wind disturbance randomization | Implemented |
| Hidden privileged parameters | Implemented |
| Four-frame state/action history + sensor noise | Implemented |
| Online effective-lift response estimator | Implemented (approximate; not unique physical identification) |
| Four-stage chained PPO curriculum | Implemented (fixed stage sequence) |
| Held-out randomized scenario evaluator | Implemented (seed-separated, same configured bounds) |
| New PPO training entry point | Implemented; calibration-gated by default |
| Synthetic software-only mode | Implemented; explicit opt-in |
| Automated unit tests | Added; CI result must be checked after push |
| Performance-driven automatic curriculum expansion | Not yet implemented |
| Full multi-parameter physical estimator with uncertainty bounds | Not yet implemented |
| Recurrent policy/context encoder | Not yet implemented |
| Native SkySim adapter/parity validation | Not yet implemented |
| Motor/propeller calibration from measurements | Blocked on exact components and test data |
| Physical flight validation | Not performed; not authorized by this software |

## 13. Required next real-world inputs

To turn the software scaffold into an aircraft-specific predictive model, collect:

1. complete takeoff weight with battery and all payload;
2. CG coordinates in the CAD_EXPORT_Z_UP frame;
3. component masses/materials or a verified full inertia tensor;
4. exact motor make/model and KV;
5. ESC model and current/thermal limits;
6. battery chemistry, cell count, capacity, internal resistance and loaded voltage;
7. propeller test data for this exact propeller/motor/voltage combination;
8. actual rotor numbering and spin directions.

Until these inputs are available, the model can be tested as software but cannot produce defensible aircraft-specific flight predictions.


## 14. A2212/13T 1000KV motor reference profile

A sourced generic reference profile is stored at `vehicle_models/components/a2212_1000kv_motor.json` and referenced from `physics_calibration.json`. The sources agree on the 1000KV class, 2S–3S LiPo use, approximately 0.5A no-load current at 10V, approximately 90mΩ catalog resistance, 3.17mm shaft, and roughly 28mm-class motor envelope. Several vendor sheets identify a 14-pole motor and recommend an ESC of at least 18–20A, with 30A commonly suggested as additional margin.

The catalog sources are not fully consistent. Published mass spans approximately 47–61g depending on vendor and included accessories; maximum current is listed as 12A/60s by some vendors and 13A/60s by another; maximum power is listed as 150W in common A2212/13T sheets, while a separate A2212-series 1000KV/1047/3S performance row reports 173W and 15.6A. These are preserved as vendor-specific records, not averaged into a fictitious universal rating. Do not treat the 15.6A/173W row as a safe operating limit.

HiModel publishes static test points at 11.1V: a GWS 9x4.7 prop at 7.5A typical / 11.3A peak and 650g typical / 730g maximum thrust; and an APC 10x4.7 prop at 10A typical / 12A peak and 790g typical / 915g maximum thrust. A different E-S Motor data sheet publishes an 1047 propeller point at 15.6A, 885g and 173W. These test points use different propellers and vendor test setups. They are useful reference evidence, but they are not a single continuous thrust curve and do not describe the custom smooth2 CAD propeller.

**Important model boundary:** KV is a no-load speed constant, not a loaded RPM guarantee. The model therefore records KV but leaves `max_rotor_speed_rad_s`, `thrust_coefficient_n_per_rad_s2`, `torque_to_thrust_m`, and the motor time constant unset. The previously stored 0.045s motor lag was an unvalidated default; this profile does not promote it to a measured value. The fast-physics loader must continue to fail closed for physical calibration until mass/CG/inertia and exact motor/ESC/battery/propeller response are measured.

Before using the motor data for aircraft-specific simulation, identify the exact vendor/lot and run a guarded thrust-stand characterization of the actual motor, ESC, battery and propeller. Record voltage, current, RPM, thrust, torque and temperature over multiple steady operating points, plus step response for actuator lag. Never infer thrust or reaction torque coefficients by fitting across the mixed vendor points above.


## 15. Universal hardware-conditioned policy and vehicle family

The new universal path is additive and intentionally separate from the earlier smooth2-only PPO entry point.

### Vehicle profile library

`vehicle_models/family/vehicle_profiles.json` contains four starting profiles: the current smooth2/A2212 reference, a generic 5-inch quad, a generic 7-inch long-range quad, and a generic 3-inch cinewhoop. All four are labelled `provisional_estimate`. The generic profiles are not tied to specific verified bills of materials. Their masses, inertias, motor constants, propeller coefficients and battery values are plausible software starting points only.

A custom vehicle is represented by `vehicle_models/family/vehicle_config_template.json`. Required inputs include:
- complete takeoff mass, including battery and payload;
- centre of mass in the simulator body frame;
- positive-definite diagonal or full 3x3 inertia tensor;
- four motor locations and verified spin signs;
- propeller diameter and motor/propeller performance parameters;
- nominal battery voltage and capacity;
- evidence/source notes and calibration status.

The loader accepts either one profile object or a family object with a profiles array. It fails closed for missing required fields, non-four-rotor geometry, invalid inertia, or invalid dimensions. The template intentionally contains nulls and is not loadable until completed.

### Policy hardware conditioning

`training/fast_physics/vehicle_family.py` wraps the existing Gymnasium plant. At each episode reset it selects a vehicle profile (or pins one when requested), builds a vehicle-specific set of physical parameters and appends twelve normalized nominal configuration features to the observation:
1. nominal mass;
2. rotor arm radius;
3. propeller diameter;
4. maximum rotor angular speed;
5. thrust coefficient;
6. torque-to-thrust ratio;
7. motor time constant;
8. nominal battery voltage;
9. battery capacity;
10. motor KV;
11. fraction of secondary physical parameters initialized from estimates;
12. profile provenance confidence (provisional, user-supplied or measured).

The policy receives the declared configuration. It does not receive the simulator's randomized episode ground truth. The existing history and online effective-lift estimator remain in place. This allows the same policy architecture to learn how nominal configuration changes its control response, while the history/estimator helps it adapt to deviations from nominal values.

The current observation length for four history frames changes from 66 to 76. A universal checkpoint is not interchangeable with the earlier 66-feature smooth2-only checkpoint.

### Universal training, curriculum and evaluation

New entry points:
- `training/train_universal_ppo.py`: PPO training over the vehicle family;
- `training/run_universal_curriculum.py`: four chained training stages, each initialized from the prior stage's checkpoint;
- `training/evaluate_universal_ppo.py`: held-out evaluation reported separately for each vehicle profile.

The family sampler selects a vehicle per episode. Episode randomization continues to perturb mass, CG, inertia, motor effectiveness, battery voltage, drag, wind and sensor noise. Training metadata records the profile identities and the simulation-only boundary.

### Deployment/configuration workflow target

For a new aircraft, the intended software workflow is:
1. Create a profile from the template and enter exact known specifications with units and source records.
2. Validate geometry, frame convention, rotor numbering, mass, CG and inertia; reject inconsistent inputs.
3. Load the universal checkpoint and condition it on the declared vehicle features.
4. Run vehicle-specific simulation and held-out stress scenarios before any hardware test.
5. Perform guarded system identification using measured RPM, voltage, current, thrust and torque; estimate effective response and uncertainty rather than assuming KV uniquely determines loaded motor performance.
6. Keep the online estimator bounded and confidence-aware. If the aircraft lies outside the validated profile envelope or the estimator confidence is low, inhibit control/abort rather than extrapolating silently.
7. Only after independent engineering validation, proceed through controlled bench and tethered test gates with independent failsafe and qualified supervision.

This is a target workflow, not a completed autopilot deployment stack. The current estimator estimates effective collective lift response only; it does not uniquely recover mass, motor degradation, battery sag, wind or all inertia terms. The universal policy is not authorized for real flight.

## 16. Current universal-path status

| Capability | Status |
|---|---|
| Four provisional quadrotor profiles | Implemented |
| User vehicle configuration template | Implemented |
| Strict profile and inertia validation | Implemented |
| Per-episode vehicle profile selection | Implemented |
| Nominal hardware features in policy observation | Implemented |
| Hidden episode-level parameter randomization | Implemented |
| Universal PPO training entry point | Implemented; not yet trained |
| Four-stage checkpoint handoff | Implemented; not yet executed |
| Per-profile held-out evaluation script | Implemented; not yet run |
| Automatic identification of full mass/CG/inertia/rotor curves | Not implemented |
| Real-time motor/ESC/battery safety supervisor | Not implemented |
| Native SkySim universal-policy adapter | Not implemented |
| Real-hardware validation or flight clearance | Not performed |
