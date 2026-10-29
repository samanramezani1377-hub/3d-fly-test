# PPO training using the shared SkySim physics core

This folder provides a native C++ build of SkySim DroneCore, a Gymnasium adapter, PPO training/evaluation scripts, tests, and a Google Colab notebook.

## Important fixes in the current training version

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

For a single PPO policy trained from 40 parallel environments for 40 million aggregate transitions:

    python training/train_ppo.py --timesteps 40000000 --n-envs 40 --n-steps 1000 --batch-size 4000 --seed 42 --run-name ppo-hover-parallel-40m --save-freq 400000 --eval-freq 400000 --eval-episodes 10

The rollout size is `n-envs * n-steps` (40,000 samples per update for this configuration). Callback intervals are specified in aggregate environment transitions. Forty subprocesses can exceed available CPU/RAM in some Colab sessions; reduce `--n-envs` if needed, and note that 40 million is the aggregate transition target, not 40 million per worker.
    python training/evaluate.py --model training/runs/ppo-hover-50000/final_model.zip --episodes 10

## Google Colab

Open the [parallel 40-million-transition Colab notebook](https://colab.research.google.com/github/samanramezani1377-hub/3d-fly-test/blob/feat/parallel-single-ppo-40m/notebooks/colab_train_ppo.ipynb). It installs dependencies, builds the native core, trains one shared PPO policy using 40 parallel environments, evaluates the final and best evaluated models, and downloads a ZIP containing models, checkpoints, logs, and reports. Colab sessions can terminate, and 40 workers may exceed available resources, so download the ZIP before closing and reduce the worker count if needed.

## Run artifacts

Each run is stored under training/runs/<run-name>/ with periodic PPO checkpoints, final_model.zip, a separately saved best/best_model.zip when evaluation runs, Monitor CSV files, TensorBoard logs, evaluation reports with crash causes and hover-stability metrics, and metadata.json. Large model weights are not committed automatically.

## Interpreting results

50,000 steps is a larger training run, not a guarantee of success. Inspect crash rate, time spent near the target altitude, vertical velocity, and evaluation returns before deciding whether the policy has learned to hover. If it does not, use the evaluation artifacts to tune the reward, action mapping, or physics rather than blindly increasing timesteps.

## Physical model boundary

config/drone.json still contains placeholder values until the actual aircraft is measured. The native build applies the current configuration but does not infer mass, inertia, motor curves, or aerodynamics from a GLB. Do not treat these PPO results as specific to the final physical quadcopter or as safe for real flight.
