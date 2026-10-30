# PPO training using the shared SkySim physics core

This folder provides a native C++ build of SkySim DroneCore, a Gymnasium adapter, PPO training/evaluation scripts, tests, and a Google Colab notebook.

## Important fixes in the current training version

- The shared upstream patch resets rotor RPM/commands, blade-element inflow caches, VRS state, battery, and turbulence RNG at each episode reset, preventing physics state from leaking between episodes.
- Policy attitude Euler angles and body angular rates use FRD body coordinates, matching the flight controller feedback and roll/pitch/yaw-rate action convention. Position and linear velocity remain world-Godot coordinates.
- Fixed-throttle diagnostics run each throttle point in a fresh simulator instance and report per-rotor thrust/RPM. Telemetry altitude is sampled after integration to match the returned state.

- PPO actions now use a symmetric [-1, 1] action space. The old throttle space [0, 1] interacted badly with PPO's zero-centered Gaussian actions: negative samples were clipped to zero, and the trained policy collapsed toward zero throttle.
- The throttle action is mapped to a narrower simulator range (0.20–0.50) around a hover-centered starting estimate (0.35), reducing early exploration into extreme motor commands. This estimate comes from the repository's diagnostic sweep and is a simulation-specific starting point, not a measured physical flight value.
- Episodes now reset on the ground and train through explicit `grounded`, `airborne`, `hovering`, and `crashed` phases. Remaining grounded receives an increasingly negative (capped) waiting penalty; crossing the takeoff threshold latches takeoff and can receive a one-time bonus only when tilt, horizontal displacement, and vertical speed are within limits. Ground contact is expected before takeoff but is a terminal crash after takeoff. The separate `reward_components` breakdown reports waiting, takeoff, and crash terms. The target remains stable hover at the configured target position; this is not a landing task.
- Training defaults to 50,000 total environment transitions, saves checkpoints and evaluates periodically, and stores the best evaluated model separately from the final model. Use `--n-envs` to collect experience in parallel for one shared PPO policy; this does not create separate models.

## Backend architecture

The browser simulator compiles pinned upstream C++ sources to WebAssembly. build_native_core.py checks out the same pinned upstream revision used by .github/workflows/pages.yml, applies this repository's configuration and battery compatibility patch, then compiles those sources as a native shared library. native_env.py exposes that C API through Gymnasium; PPO trains against that physics source rather than a separate Godot server.

**Important:** shared source code does not prove bit-for-bit native/WASM parity. CI and deterministic trajectory comparisons are still needed. This is a simplified simulation, not sim-to-real validation.

## Quick start

Requires Python 3.10+ and a C++ compiler with C++20 support (for example, g++).

    python -m pip install -r training/requirements.txt
    python training/build_native_core.py
    python training/train_ppo.py --timesteps 50000 --seed 42 --run-name ppo-hover-50000 --save-freq 5000 --eval-freq 5000
    python training/evaluate.py --model training/runs/ppo-hover-50000/final_model.zip --episodes 10

## Universal five-stage CPU PPO training

The existing workflow is .github/workflows/ppo-cpu-two-stage.yml. Its event modes are intentionally separate:

- Pull requests run only the existing physics/family/hover regression tests. They do not start PPO training.
- A manual workflow_dispatch run executes one sequential five-stage universal PPO curriculum.

The manual run uses one shared policy, 16 parallel simulator environments, and exactly 1,000,000 aggregate PPO transitions in each of five stages (5,000,000 stage transitions total). The PPO rollout is 500 steps × 16 environments = 8,000 transitions, so the stage budget and resumable checkpoints align to complete rollouts. Each stage warm-starts from the previous stage's best evaluated checkpoint. A previous-stage PPO timestep counter is reset for the new stage budget; restarting an interrupted stage resumes from that same stage's latest checkpoint and consumes only its remaining budget.

To keep evaluation from becoming the main workload:

- The latest resumable training checkpoint is saved at the first complete PPO rollout after each 20,000-transition interval (24,000 transitions with the current 16-environment, 500-step rollout).
- Checkpoint-selection evaluation runs every 100,000 transitions, with 2 episodes per vehicle profile.
- Independent held-out evaluation runs only once, after all five training stages, comparing the final and best stage-5 checkpoints (20 episodes per vehicle profile per candidate).
- No separate training workflow, benchmark workflow, or extra test suite is launched by this pipeline.

The workflow restores committed universal-quad-v2 checkpoints before a manual run and saves updated checkpoints, reports and the release manifest under training/models/universal-quad-v2/. Resume with the same settings and profile file; the runner rejects a manifest with mismatched settings rather than silently mixing runs.

Manual run: open the existing workflow in GitHub Actions and select Run workflow on main. Pull-request checks are validation only; they are not the training run.

The five-stage chain remains a simulation research experiment. The supplied vehicle profiles are provisional estimates, and a passing held-out simulation gate is not physical flight certification.

## Google Colab

Open the [experimental 40-million-transition Colab notebook](https://colab.research.google.com/github/samanramezani1377-hub/3d-fly-test/blob/main/notebooks/colab_train_ppo.ipynb). This is a separate research experiment (one shared policy, 40 parallel environments), not the production five-stage CPU workflow. It may exceed Colab resources and is not a recommended substitute for the chained pipeline. Download its ZIP before closing the session.

## Run artifacts

Each run is stored under training/runs/<run-name>/ with periodic PPO checkpoints, final_model.zip, a separately saved best/best_model.zip when evaluation runs, Monitor CSV files, TensorBoard logs, evaluation reports with crash causes and hover-stability metrics, and metadata.json. Large model weights are not committed automatically.

## Interpreting results

50,000 steps is a larger training run, not a guarantee of success. Inspect crash rate, time spent near the target altitude, vertical velocity, and evaluation returns before deciding whether the policy has learned to hover. If it does not, use the evaluation artifacts to tune the reward, action mapping, or physics rather than blindly increasing timesteps.

## Physical model boundary

config/drone.json still contains placeholder values until the actual aircraft is measured. The native build applies the current configuration but does not infer mass, inertia, motor curves, or aerodynamics from a GLB. Do not treat these PPO results as specific to the final physical quadcopter or as safe for real flight.


## Aircraft-specific preflight model

The imported `smooth2-parametric` CAD model and engineering metadata live under `assets/preflight/`. Read the [preflight asset index](../assets/preflight/README.md), [analysis plan](../assets/preflight/ANALYSIS_PLAN.md), and [data dictionary](../assets/preflight/DATA_DICTIONARY.md) before using aircraft-specific parameters.

The current SkySim/PPO configuration is not automatically derived from this CAD. Do not copy design dimensions into mass, inertia, rotor thrust or motor parameters without validating the STEP geometry, material/density, exact motor/ESC/battery and rotor performance data. The preflight gates must pass before treating the model as the training plant.


## Experimental robust/adaptive PPO on the smooth2 fast-physics plant

The new `training/fast_physics/` package provides a separate simplified 6-DoF plant and Gymnasium environment. It keeps the imported smooth2 geometry fixed and randomizes uncertain physical properties per episode. The policy receives four frames of state/action history; mass, CG, inertia, motor effectiveness and voltage factor are not exposed as policy observations.

**The aircraft calibration is intentionally incomplete.** By default, the environment refuses to load while measured mass/CG/inertia and motor/propeller coefficients are missing. For software-only development, synthetic values can be enabled explicitly:

```bash
python training/train_robust_ppo.py --timesteps 100000 --n-envs 4 --software-only-synthetic-calibration
```

Do not use a synthetic run as evidence of real hover capability or physical flight safety. This new entry point is additive and is not part of the existing SkySim five-stage checkpoint chain. Full details and current limitations are documented in [ROBUST_ADAPTIVE_FLIGHT_ARCHITECTURE.md](../docs/ROBUST_ADAPTIVE_FLIGHT_ARCHITECTURE.md).


### Staged robust/adaptive PPO experiment

The new fast-physics environment supports curriculum stages 1–4. Each stage narrows or expands mass, CG, inertia, actuator, battery, drag and wind ranges. The sequential runner warm-starts each stage from the previous stage's final model:

```bash
python training/run_robust_curriculum.py \
  --timesteps-per-stage 100000 \
  --n-envs 4 \
  --run-prefix robust-smooth2-software-only \
  --software-only-synthetic-calibration
```

The policy input contains a four-frame state/action history, observation noise, and two context values from an online recursive-least-squares estimator of effective collective lift response. The estimator is not a unique mass/motor/battery diagnosis; it provides an approximate adaptation cue and confidence value. Privileged simulator parameters remain outside the policy observation.

The curriculum runner and PPO script refuse to use unknown physical calibration unless the explicit software-only synthetic flag is supplied. The existing native SkySim five-stage workflow is unchanged.


Each curriculum stage is followed by a held-out evaluation (20 episodes by default) using a separate seed range. Reports are written to `held-out-evaluation.json` in the stage run directory. The evaluator reports hover/takeoff success, crash rate, stable-hover fraction and position errors. These are held-out random seeds from the configured distribution, not out-of-range generalization proof.


## Universal hardware-conditioned quadrotor policy (new research path)

The separate universal PPO path trains one policy across several four-rotor vehicle profiles and supplies the policy with a normalized nominal hardware descriptor. It does not replace the existing native SkySim production pipeline or the earlier smooth2-only fast-physics experiment.

Included provisional profiles live in `vehicle_models/family/vehicle_profiles.json`:
- `smooth2-a2212-custom-prop`: current CAD rotor geometry with provisional software-only mass and propulsion coefficients;
- `generic-5inch-quad`: generic 5-inch class estimate;
- `generic-7inch-long-range`: generic 7-inch class estimate;
- `generic-3inch-cinewhoop`: generic ducted 3-inch class estimate.

Every profile is marked `provisional_estimate`. The numerical mass, inertia, thrust coefficient, torque ratio, RPM and motor lag are training hypotheses, not measurements or verified safe limits. Do not interpret a profile name as proof that a particular commercial aircraft has those specifications.

### Train a shared hardware-conditioned policy

```bash
python -m pip install -r training/requirements.txt
python training/train_universal_ppo.py --timesteps 200000 --n-envs 4 --run-name universal-quad-baseline
```

The trainer periodically evaluates every vehicle profile and selects a balanced checkpoint by weakest-profile success, worst-profile crash rate, then mean return at `best/best_model.zip`; the terminal training snapshot is saved separately as `final_model.zip`. The environment samples a vehicle profile at episode reset. The policy observation contains the existing state/action history and online effective-lift estimator context plus twelve normalized nominal hardware/provenance features (mass, arm radius, propeller diameter, motor speed/thrust/torque response, battery voltage/capacity and KV). Episode-randomized ground-truth mass, CG, inertia and actuator effectiveness remain hidden. This deliberately distinguishes declared setup values from uncertain episode physics.

### Five-stage universal curriculum

```bash
python training/run_universal_curriculum.py --timesteps-per-stage 200000 --n-envs 4
```

Each stage starts from the previous stage's best evaluated checkpoint (`best/best_model.zip`), not the last training snapshot. Both best and final checkpoints, metadata, held-out per-profile evaluation and a final curriculum manifest are retained. Stage ranges are inherited from `vehicle_models/family/domain_randomization.json`; the actuator, voltage and mass bounds are constrained so the four provisional profiles retain a nominal worst-case thrust-to-weight ratio above 1.05 in the simplified model. These remain provisional software ranges, not structural or electrical operating limits.

### Validate plant hover feasibility before PPO

Run the privileged reference benchmark before spending compute on policy training:

```bash
python training/evaluate_hover_reference.py --episodes 10 --max-steps 1500 --settle-seconds 5
python training/evaluate_hover_reference.py --episodes 10 --max-steps 1500 --settle-seconds 5 --randomize --start-at-target --wind-max 2
# The second command checks Stage 4 randomized hover-hold with wind feed-forward; it does not test takeoff.
```

The reference controller uses the simulator's true episode parameters. It is an oracle upper-bound diagnostic, not an observation-only policy or a flight controller. It isolates whether the simplified plant and motor allocation can sustain hover. CI checks nominal sustained hover on all four profiles. If the oracle succeeds but PPO fails, debug reward design, action exploration, observation normalization and training curriculum before increasing training duration.

### Evaluate per airframe

```bash
python training/evaluate_universal_ppo.py --model training/runs/universal-quad-baseline/best/best_model.zip --episodes 20
```

To evaluate only one profile, repeat `--vehicle-id <id>`. To add a new airframe, copy `vehicle_models/family/vehicle_config_template.json`, fill the known required fields (mass, frame geometry, rotor locations, motor KV and nominal battery voltage) and use the resulting JSON as `--vehicle-family`. A single profile JSON or a family JSON containing a `profiles` array is accepted. The loader rejects missing mass, geometry, rotor coordinates, motor KV or nominal battery voltage. Unknown inertia, loaded RPM, thrust coefficient, torque ratio and motor lag may receive explicit provisional estimates, recorded in the profile metadata and exposed to the policy as uncertainty/provenance features.

For a real vehicle, enter complete takeoff mass including battery and payload, CG, rotor coordinates and verified spin directions, inertia, motor/ESC/battery details, and a measured or manufacturer-backed thrust/torque response. A KV value or propeller diameter alone is not enough to infer loaded RPM or thrust. If some values remain estimates, keep the profile labelled provisional and include uncertainty in the randomization ranges.

**Important boundary:** this is still a four-rotor research simulator with simplified quadratic rotor thrust, first-order motor lag and rigid-body 6-DoF dynamics. The universal policy is simulation-only. It is not cleared for hardware operation. Before any real flight, independently validate the exact airframe, motor/ESC/propeller/battery combination, actuator allocation, sensor conventions, failsafes and a safe test procedure.


## Universal hardware-conditioned quadrotor policy (simulation research)

The experimental multi-airframe path is separate from the existing smooth2-only trainer:

- `vehicle_models/family/vehicle_profiles.json` contains provisional example profiles for the current smooth2 CAD/A2212 reference, generic 5-inch freestyle, generic 7-inch long-range, and generic 3-inch cinewhoop classes.
- `vehicle_models/family/vehicle_config_template.json` is the starting template for adding a measured aircraft. Replace every required null with traceable data; the loader intentionally rejects incomplete profiles.
- `training/fast_physics/vehicle_family.py` selects a vehicle at reset and appends twelve normalized nominal hardware/provenance descriptors to the observation. Episode-randomized mass, CG, inertia, motor effectiveness and voltage remain hidden from the policy.
- `training/train_universal_ppo.py` trains one shared PPO model over the selected family.
- `training/evaluate_universal_ppo.py` reports results separately for each vehicle.
- `training/run_universal_curriculum.py` runs five sequential curriculum stages, handing each stage's best evaluated checkpoint to the next stage and evaluating after every stage.

Example:

    python -m pip install -r training/requirements.txt
    python training/train_universal_ppo.py --timesteps 200000 --n-envs 4
    python training/evaluate_universal_ppo.py --model training/runs/universal-quad-ppo/best/best_model.zip --episodes 30
    python training/run_universal_curriculum.py --timesteps-per-stage 200000 --n-envs 4

These sample profiles are deliberately marked `provisional_estimate`: they are software-development starting points, not measured or validated flight models. The current smooth2 custom propeller's thrust and torque coefficients are not established by the A2212 catalog data. Do not use any resulting checkpoint to control a physical aircraft. A real aircraft profile needs complete takeoff mass, CG, inertia, exact rotor geometry/directions, motor/ESC/battery details, and thrust/torque/RPM/current/voltage and motor-step-response measurements for the exact propulsion combination. Bench validation and staged safety testing remain separate requirements.

This first implementation conditions the policy on declared nominal hardware specifications. It does not yet infer every unknown parameter online, guarantee zero-shot transfer to unseen aircraft, or provide an automatic physical calibration procedure. New aircraft must be represented in the profile library and independently evaluated before being considered supported.


## Final-tested universal PPO releases and resumable checkpoints

The existing five-stage workflow now writes an optimizer-inclusive `latest_model.zip` checkpoint and `training_state.json` at each existing periodic evaluation point. Re-running the same manual workflow resumes the first incomplete stage from its latest checkpoint and skips stages already marked complete in the curriculum manifest. Resume with the same vehicle-family file and training settings; changing these between runs invalidates a strict apples-to-apples continuation.

At the end of stage 5, both the terminal `final_model.zip` and balanced training-selected `best/best_model.zip` are evaluated on the same independent held-out seeds (20 episodes per vehicle by default). The release `best_model.zip` is the candidate that wins this final held-out comparison. The terminal training snapshot is always preserved separately as `final_model.zip`.

The default release gate requires at least 50% hover success for every vehicle profile and no more than 25% crash rate for any profile. If the gate fails, the workflow reports failure but retains and publishes the models and evaluation reports for diagnosis; a failed model is not silently labelled accepted.

On manual runs, the existing workflow commits stage checkpoints, model weights, evaluation reports, metadata and manifests under `training/models/universal-quad/` in this repository. It also uploads a 90-day Actions backup. Portable release files are:
- `training/models/universal-quad/release/best_model.zip`
- `training/models/universal-quad/release/final_model.zip`
- `training/models/universal-quad/release/release-manifest.json`

These artifacts are simulation research outputs only. They are not flight-certified and must not be loaded onto a physical aircraft.


## Current universal PPO reward and evaluation semantics

The current universal fast-physics path (train_universal_ppo.py, evaluate_universal_ppo.py, and run_universal_curriculum.py) uses these aligned rules:

- Reward combines altitude and horizontal target tracking with penalties for translational speed, roll/pitch tilt, angular rates, motor-command changes and a small normalized squared-command effort proxy. Yaw is not penalized without a defined yaw target.
- Altitude progress is potential-shaped; takeoff gets a one-time transition bonus; stable-hover steps get a shaping bonus; ending the time limit without ever taking off receives a takeoff-failure penalty.
- The command-effort term is not measured electrical energy. The current plant has no validated motor-current, ESC-loss or battery electrical model.
- A stable-hover sample requires takeoff, altitude error ≤0.15 m, horizontal error ≤0.20 m, speed ≤0.35 m/s, absolute roll/pitch ≤0.25 rad, and angular-rate norm ≤0.5 rad/s. Success requires these conditions continuously for 5 seconds without a crash.
- Ground contact is constrained at z=0. Contact before takeoff is permitted; contact after takeoff is a crash. Non-finite state is detected before reward and observation use; the last finite state is used to keep PPO observations finite and the episode receives a terminal penalty.
- Reports distinguish takeoff success, takeoff failure, crash, sustained-hover success and survival to the time limit. Survival alone is not mission completion.
- Checkpoint selection prioritizes weakest-profile sustained-hover success, worst-profile crash rate, weakest-profile takeoff rate, then mean return. Final held-out selection uses the same safety priorities and stable-hover duration as a tie-breaker.
- The final release gate checks minimum sustained-hover success, maximum crash rate and minimum takeoff success on every profile. Passing this simulation gate never certifies physical flight.

No additional workflow or test suite was introduced for these changes. The existing PR workflow remains the only automated validation path; no training or evaluation run is started by editing these files.


### Universal PPO action normalization (pipeline v3)

The universal family policy uses a symmetric four-dimensional action space
[-1, 1]. Each component is a normalized motor delta around the nominal hover
command computed from the declared vehicle profile. A zero action therefore
requests nominal hover thrust; -1 maps to zero motor command and +1 maps to
full command. The mapping uses nominal profile parameters only, not hidden
episode-randomized mass, voltage, motor effectiveness, or other ground truth.
The underlying physics environment and Flight Lab continue to use physical
motor commands in [0, 1].

Pipeline v3 uses a separate checkpoint directory/name. Checkpoints trained
with the old [0, 1] PPO action space are intentionally not resumed because
their policy distribution and action semantics are incompatible. No training
run is started by this code change.
