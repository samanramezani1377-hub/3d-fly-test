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

## Production five-stage CPU training

The production workflow is `.github/workflows/ppo-cpu-two-stage.yml` and runs `training/run_five_stage.py` on CPU. Its current configuration is:

- Five chained stages, each requesting 100,000 aggregate environment transitions.
- Stage 1 trains two independent candidates (seeds 42 and 40) concurrently; each candidate uses 4 environments.
- Stages 2–5 each use 4 environments and continue from the previous stage's selected checkpoint.
- Stage 1 therefore runs 8 simulator environments concurrently in total (two training processes × four environments); each later stage runs four.
- Checkpoints and periodic evaluation are every 20,000 aggregate transitions. Periodic evaluation uses 5 episodes.
- Candidate screening uses 2 episodes; up to 3 finalists receive 10 evaluation episodes each, split evenly across two independent evaluation seeds.
- The standalone 1/4/8-environment benchmarks are separate workflows/scripts and are not invoked by production five-stage training.

Run locally after building the native core:

    python training/run_five_stage.py

The GitHub Actions workflow installs CPU-only PyTorch and runs the five-stage chain. The full run can take several hours. The manifest records the selected seed/checkpoint at each stage and whether the final model meets the simulation-only eligibility gate.

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

The environment samples a vehicle profile at episode reset. The policy observation contains the existing state/action history and online effective-lift estimator context plus twelve normalized nominal hardware features (mass, arm radius, propeller diameter, motor speed/thrust/torque response, battery voltage/capacity and KV). Episode-randomized ground-truth mass, CG, inertia and actuator effectiveness remain hidden. This deliberately distinguishes declared setup values from uncertain episode physics.

### Four-stage universal curriculum

```bash
python training/run_universal_curriculum.py --timesteps-per-stage 200000 --n-envs 4
```

Each stage starts from the previous stage's checkpoint. It writes a per-stage checkpoint, metadata, held-out per-profile evaluation and a final curriculum manifest. Stage ranges are inherited from `vehicle_models/smooth2_parametric/domain_randomization.json`; they are provisional software ranges and are not structural or electrical operating limits.

### Evaluate per airframe

```bash
python training/evaluate_universal_ppo.py --model training/runs/universal-quad-baseline/final_model.zip --episodes 20
```

To evaluate only one profile, repeat `--vehicle-id <id>`. To add a new airframe, copy `vehicle_models/family/vehicle_config_template.json`, fill the known required fields (mass, frame geometry, rotor locations, motor KV and nominal battery voltage) and use the resulting JSON as `--vehicle-family`. A single profile JSON or a family JSON containing a `profiles` array is accepted. The loader rejects missing mass, geometry, rotor coordinates, motor KV or nominal battery voltage. Unknown inertia, loaded RPM, thrust coefficient, torque ratio and motor lag may receive explicit provisional estimates, recorded in the profile metadata and exposed to the policy as uncertainty/provenance features.

For a real vehicle, enter complete takeoff mass including battery and payload, CG, rotor coordinates and verified spin directions, inertia, motor/ESC/battery details, and a measured or manufacturer-backed thrust/torque response. A KV value or propeller diameter alone is not enough to infer loaded RPM or thrust. If some values remain estimates, keep the profile labelled provisional and include uncertainty in the randomization ranges.

**Important boundary:** this is still a four-rotor research simulator with simplified quadratic rotor thrust, first-order motor lag and rigid-body 6-DoF dynamics. The universal policy is simulation-only. It is not cleared for hardware operation. Before any real flight, independently validate the exact airframe, motor/ESC/propeller/battery combination, actuator allocation, sensor conventions, failsafes and a safe test procedure.
