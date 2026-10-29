# PPO training using the shared SkySim physics core

This folder provides a native C++ build of SkySim DroneCore, a Gymnasium adapter, PPO training/evaluation scripts, tests, and a Google Colab notebook.

## Backend architecture

The browser simulator compiles these pinned upstream source files to WebAssembly: web/core/drone_core.cpp and web/core/capi.cpp.

build_native_core.py checks out the same pinned upstream revision used by .github/workflows/pages.yml, runs scripts/prepare_site.py to apply this repository's drone configuration and battery compatibility patch, then compiles those same C++ sources as a native shared library. native_env.py exposes that C API through Gymnasium; PPO therefore trains against the same physics source, rather than a separate Godot server.

**Important:** same source code does not by itself prove bit-for-bit native/WASM parity. CI compiles the native core and runs smoke tests; a future regression test should compare deterministic native and browser-WASM trajectories. This is still a simplified simulation and not sim-to-real validation.

## Quick start

Requires Python 3.10+ and a C++ compiler with C++20 support (for example, g++).

    python -m pip install -r training/requirements.txt
    python training/build_native_core.py
    python training/train_ppo.py --timesteps 5000 --seed 42 --run-name smoke-hover
    python training/evaluate.py --model training/runs/smoke-hover/final_model.zip --episodes 10

After a successful smoke run, increase timesteps as desired. PPO's MLP policy does not require a GPU.

## Google Colab

Open notebooks/colab_train_ppo.ipynb in Colab. The notebook clones this branch, installs Python dependencies, builds the native physics core, runs a short training job, evaluates the checkpoint, and downloads a ZIP of the results. Colab sessions can terminate, so download results before closing.

## Run artifacts

Each run is stored under training/runs/<run-name>/ with periodic PPO checkpoints, final_model.zip, monitor CSV, TensorBoard logs, and metadata.json. Large model weights are not committed automatically.

## Physical model boundary

config/drone.json remains unchanged and contains placeholder values until the actual aircraft is ready. The native build applies the current configuration but does not infer mass, inertia, motor curves, or aerodynamics from a GLB. Do not treat current PPO results as specific to the final physical quadcopter or as safe for real flight.

Once the real model and measurements exist, update the config from credible measurements/component data, validate the physics, compare native/WASM trajectories, and only then run aircraft-specific training.