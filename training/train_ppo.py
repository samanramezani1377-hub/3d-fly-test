"""PPO training with checkpoints and periodic evaluation for the SkySim hover task."""
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
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=3.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--run-name", default="ppo-hover-50000")
    parser.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.timesteps, args.max_steps, args.save_freq, args.eval_freq, args.eval_episodes) <= 0:
        raise SystemExit("timesteps, max-steps, save-freq, eval-freq, and eval-episodes must be positive")
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_name):
        raise SystemExit("--run-name may contain only letters, numbers, '-' and '_'")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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

    raw_env = NativeSkySimEnv(target=target, max_steps=args.max_steps)
    check_env(raw_env, warn=True)
    raw_env.reset(seed=args.seed)
    env = Monitor(raw_env, filename=str(run_dir / "monitor.csv"))

    eval_raw_env = NativeSkySimEnv(target=target, max_steps=args.max_steps)
    eval_env = Monitor(eval_raw_env, filename=str(run_dir / "evaluation-monitor.csv"))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "algorithm": "PPO",
        "seed": args.seed,
        "timesteps_requested": args.timesteps,
        "max_steps_per_episode": args.max_steps,
        "target_xyz_m": list(target),
        "checkpoint_frequency": args.save_freq,
        "evaluation_frequency": args.eval_freq,
        "evaluation_episodes": args.eval_episodes,
        "action_space": "symmetric [-1, 1]; throttle restricted to 0.20..0.50 around hover estimate (0.35)",
        "reward": "bounded exponential altitude/horizontal proximity scores, capped speed/attitude penalties, explicit crash penalty -100",
        "backend": "native-drone-core",
        "browser_core_same_cpp_sources": True,
        "upstream_revision_expected": UPSTREAM_REVISION,
        "backend_warning": BACKEND_WARNING,
        "drone_config_used": True,
        "physical_aircraft_model_ready": False,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_hover",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evaluation"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        warn=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tensorboard"),
        device="auto",
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback, eval_callback])
        model.save(str(run_dir / "final_model"))
    finally:
        env.close()
        eval_env.close()

    print(f"Saved final model: {run_dir / 'final_model.zip'}")
    best_model = run_dir / "best" / "best_model.zip"
    if best_model.exists():
        print(f"Saved best evaluated model: {best_model}")
    print(f"Saved metadata: {run_dir / 'metadata.json'}")
    print("Reminder: this is simulation-only; validate native/WASM parity and the physical model before any real flight.")


if __name__ == "__main__":
    main()
