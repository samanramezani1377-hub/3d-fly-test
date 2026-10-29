# PPO training using the shared SkySim physics core

This folder provides a native C++ build of SkySim DroneCore, a Gymnasium adapter, PPO training/evaluation scripts, tests, and a Google Colab notebook.

## Important fixes in the current training version

- PPO actions now use a symmetric [-1, 1] action space. The old throttle space [0, 1] interacted badly with PPO's zero-centered Gaussian actions: negative samples were clipped to zero, and the trained policy collapsed toward zero throttle.
- The throttle action is mapped to a narrower simulator range (0.20–0.50) around a hover-centered starting estimate (0.35), reducing early exploration into extreme motor commands. This estimate comes from the repository's diagnostic sweep and is a simulation-specific starting point, not a measured physical flight value.
- Reward shaping uses bounded altitude/horizontal proximity scores, capped velocity and attitude penalties, and an explicit crash penalty. Evaluation reports stable-hover fraction and success rate, not only episode return.
- Training defaults to 50,000 timesteps, saves checkpoints every 5,000 steps, and evaluates every 5,000 steps across 10 episodes. The best evaluated model and the final model are saved separately.

## Backend architecture

The browser simulator compiles pinned upstream C++ sources to WebAssembly. build_native_core.py checks out the same pinned upstream revision used by .github/workflows/pages.yml, applies this repository's configuration and battery compatibility patch, then compiles those sources as a native shared library. native_env.py exposes that C API through Gymnasium; PPO trains against that physics source rather than a separate Godot server.

**Important:** shared source code does not prove bit-for-bit native/WASM parity. CI and deterministic trajectory comparisons are still needed. This is a simplified simulation, not sim-to-real validation.

## Quick start

Requires Python 3.10+ and a C++ compiler with C++20 support (for example, g++).

    python -m pip install -r training/requirements.txt
    python training/build_native_core.py
    python training/train_ppo.py --timesteps 50000 --seed 42 --run-name ppo-hover-50000 --save-freq 5000 --eval-freq 5000
    python training/evaluate.py --model training/runs/ppo-hover-50000/final_model.zip --episodes 10

## Google Colab

Open [the 50,000-step Colab notebook](https://colab.research.google.com/github/samanramezani1377-hub/3d-fly-test/blob/main/notebooks/colab_train_ppo.ipynb). It installs dependencies, builds the native core, trains for 50,000 timesteps, evaluates the final model and best evaluated model, and downloads a ZIP containing models, checkpoints, logs, and reports. Colab sessions can terminate, so download the ZIP before closing.

## Run artifacts

Each run is stored under training/runs/<run-name>/ with periodic PPO checkpoints, final_model.zip, a separately saved best/best_model.zip when evaluation runs, Monitor CSV files, TensorBoard logs, evaluation reports with crash causes and hover-stability metrics, and metadata.json. Large model weights are not committed automatically.

## Interpreting results

50,000 steps is a larger training run, not a guarantee of success. Inspect crash rate, time spent near the target altitude, vertical velocity, and evaluation returns before deciding whether the policy has learned to hover. If it does not, use the evaluation artifacts to tune the reward, action mapping, or physics rather than blindly increasing timesteps.

## Physical model boundary

config/drone.json still contains placeholder values until the actual aircraft is measured. The native build applies the current configuration but does not infer mass, inertia, motor curves, or aerodynamics from a GLB. Do not treat these PPO results as specific to the final physical quadcopter or as safe for real flight.
