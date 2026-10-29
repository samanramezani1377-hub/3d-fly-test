# Reinforcement-learning training (experimental)

This folder adds a repeatable PPO training entry point and a Google Colab notebook.

## Important backend distinction

The browser deployment in this repository runs SkySim's C++ physics core compiled to WebAssembly. The upstream Python Gymnasium client instead talks over WebSocket to the **Godot headless agent server**. These are separate execution paths; this scaffold does **not** claim their physics or dynamics are identical.

Until a native Gymnasium adapter is connected to the same C++ physics core used by the browser, treat any policy trained here as an **infrastructure prototype**, not as a controller validated for the browser build or the eventual physical quadcopter. Do not use it to fly hardware.

Also, `config/drone.json` still contains placeholder values. The training setup deliberately does not edit or infer the real aircraft's mass, inertia, rotor geometry, battery, or motor parameters. Final aircraft-specific training waits until the real physical model and measured parameters are available.

## Layout

- `train_ppo.py`: PPO trainer using the upstream `skysim` Gymnasium package.
- `evaluate.py`: evaluates a saved PPO checkpoint over fixed episode seeds.
- `requirements.txt`: Python dependencies.
- `tests/test_training_config.py`: offline checks for safe defaults and argument parsing.
- `../notebooks/colab_train_ppo.ipynb`: Colab notebook to clone the pinned upstream source, install dependencies, and run training.

## Local setup

Use the pinned upstream revision from `.github/workflows/pages.yml`:

```bash
git clone https://github.com/vishwagw/Sky-Sim-drone-simulator-3.0.git upstream
cd upstream
git checkout a31864f999418531775980b879979ce4c57e791d
python -m pip install -e "python/[rl]"
cd ..
python -m pip install -r training/requirements.txt
```

Build/install the upstream simulator's Godot extension using its own instructions, then start its headless agent server:

```bash
godot --headless demo/agent_server.tscn -- --port 5557
```

In another terminal, from this repository root:

```bash
python training/train_ppo.py --port 5557 --timesteps 200000 --run-name prototype-hover
python training/evaluate.py --port 5557 --model training/runs/prototype-hover/final_model.zip --episodes 10
```

The simulator server must remain running while training/evaluation is active. The environment is the upstream Godot Gymnasium backend, not the browser WebAssembly instance.

## Outputs and reproducibility

Each run is saved under `training/runs/<run-name>/` with:
- PPO checkpoint(s) and final model
- training monitor CSV
- JSON metadata containing seeds, timesteps, target, backend identity, pinned upstream revision, and an explicit non-equivalence warning

Do not commit large model weights to the source tree by default. Download the run directory from Colab or upload it as a GitHub Actions/Colab artifact when needed.

## Colab notes

Colab sessions can stop and GPU availability is not guaranteed. This PPO setup uses an MLP policy; a GPU is optional and often not the bottleneck. Save checkpoints periodically and copy results out of the ephemeral runtime. The notebook does not modify the deployed site or `config/drone.json`.

## Next engineering milestone

Implement and test a native binding around the exact C++ `DroneCore` used for the browser WASM build, then implement a Gymnasium adapter over that binding. Once that is verified against shared deterministic test cases, switch training to that backend. This is required before claiming browser-physics equivalence.
