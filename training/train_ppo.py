"""PPO training using a native build of the same C++ DroneCore sources as browser WASM."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

UPSTREAM_REVISION = "a31864f999418531775980b879979ce4c57e791d"
BACKEND_WARNING = (
    "Native build shares pinned DroneCore C++ sources with browser WASM; numerical parity "
    "still requires regression tests and this is not sim-to-real validation."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=3.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--run-name", default="prototype-hover")
    parser.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    parser.add_argument("--save-freq", type=int, default=25_000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.timesteps <= 0 or args.max_steps <= 0 or args.save_freq <= 0:
        raise SystemExit("--timesteps, --max-steps, and --save-freq must be positive")
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_name):
        raise SystemExit("--run-name may contain only letters, numbers, '-' and '_'")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.env_checker import check_env
        from native_env import NativeSkySimEnv
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install training/requirements.txt. Build the native core "
            "first with python training/build_native_core.py."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target = (args.target_x, args.target_y, args.target_z)

    raw_env = NativeSkySimEnv(
        target=target,
        max_steps=args.max_steps,
    )
    check_env(raw_env, warn=True)
    try:
        raw_env.reset(seed=args.seed)
    except TypeError:
        # Compatibility with older Gymnasium wrappers; seed still recorded.
        pass
    env = Monitor(raw_env, filename=str(run_dir / "monitor.csv"))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "algorithm": "PPO",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "max_steps_per_episode": args.max_steps,
        "target_xyz_m": list(target),
        "backend": "native-drone-core",
        "browser_core_same_cpp_sources": True,
        "upstream_revision_expected": UPSTREAM_REVISION,
        "backend_warning": "Uses the same pinned DroneCore C++ sources and project config patch as the browser WASM build, compiled natively. Native-vs-WASM numerical parity still requires regression testing.",
        "drone_config_used": True,
        "physical_aircraft_model_ready": False,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    callback = CheckpointCallback(
        save_freq=max(1, args.save_freq),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_hover",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tensorboard"),
        device="auto",
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(run_dir / "final_model"))
    finally:
        env.close()

    print(f"Saved model: {run_dir / 'final_model.zip'}")
    print(f"Saved metadata: {run_dir / 'metadata.json'}")
    print("NOTE: native and WASM builds share C++ sources; verify numerical parity before relying on results.")


if __name__ == "__main__":
    main()
