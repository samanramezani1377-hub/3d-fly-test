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
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total environment transitions across all workers")
    parser.add_argument("--n-envs", type=int, default=1, help="Parallel environments feeding one shared PPO policy")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout steps per environment before each PPO update")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size; must divide n-envs * n-steps")
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Compute device for PPO; simulator physics remains native C++")
    parser.add_argument("--load-model", type=Path, default=None, help="Load an existing PPO checkpoint and continue training instead of creating a fresh policy")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.timesteps, args.n_envs, args.n_steps, args.batch_size, args.max_steps, args.save_freq, args.eval_freq, args.eval_episodes) <= 0:
        raise SystemExit("timesteps, n-envs, n-steps, batch-size, max-steps, save-freq, eval-freq, and eval-episodes must be positive")
    rollout_size = args.n_envs * args.n_steps
    if rollout_size < 2 or args.batch_size > rollout_size or rollout_size % args.batch_size != 0:
        raise SystemExit("n-envs * n-steps must be >= 2 and evenly divisible by batch-size")
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_name):
        raise SystemExit("--run-name may contain only letters, numbers, '-' and '_'")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.env_checker import check_env
        from native_env import NativeSkySimEnv, PolicyObservationWrapper
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

    # Check the Gymnasium contract in the parent process, then close this instance
    # before workers start. Each worker creates its own native C-API handle.
    checker_env = PolicyObservationWrapper(NativeSkySimEnv(target=target, max_steps=args.max_steps))
    try:
        check_env(checker_env, warn=True)
    finally:
        checker_env.close()

    def make_env(rank):
        def _init():
            worker_env = PolicyObservationWrapper(NativeSkySimEnv(target=target, max_steps=args.max_steps))
            return Monitor(worker_env, filename=str(run_dir / f"monitor-{rank}.csv"))
        return _init

    if args.n_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        # Each worker owns an independent simulator instance, but all experiences
        # update the same PPO model. "spawn" avoids inheriting native-library state.
        env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)], start_method="spawn")
    env.seed(args.seed)

    eval_raw_env = PolicyObservationWrapper(NativeSkySimEnv(target=target, max_steps=args.max_steps))
    eval_env = Monitor(eval_raw_env, filename=str(run_dir / "evaluation-monitor.csv"))
    # SB3 callback frequencies count VecEnv calls, each of which advances n_envs steps.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_hover",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evaluation"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        warn=True,
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "algorithm": "PPO",
        "seed": args.seed,
        "timesteps_requested": args.timesteps,
        "parallel_environments": args.n_envs,
        "shared_policy": True,
        "rollout_size_per_update": rollout_size,
        "ppo_n_steps_per_environment": args.n_steps,
        "ppo_batch_size": args.batch_size,
        "expected_ppo_updates_approx": args.timesteps // rollout_size,
        "max_steps_per_episode": args.max_steps,
        "target_xyz_m": list(target),
        "checkpoint_frequency_total_transitions": args.save_freq,
        "evaluation_frequency_total_transitions": args.eval_freq,
        "evaluation_episodes": args.eval_episodes,
        "training_device": args.device,
        "action_space": "symmetric [-1, 1]; throttle mapped to 0.20..0.50 around estimate (0.35)",
        "policy_observation": "target-relative position / 12 m; linear velocity / 8 m/s; Euler angles / pi rad; angular velocity / 10 rad/s; physical env info remains in SI units",
        "policy_network": {"actor_layers": [128, 128], "critic_layers": [128, 128], "activation": "tanh"},
        "reward": "bounded exponential altitude/horizontal proximity scores, uncapped runaway-speed penalties, capped attitude/rate penalties, explicit crash penalty -100",
        "backend": "native-drone-core",
        "browser_core_same_cpp_sources": True,
        "upstream_revision_expected": UPSTREAM_REVISION,
        "backend_warning": BACKEND_WARNING,
        "drone_config_used": True,
        "physical_aircraft_model_ready": False,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.load_model is not None:
        if not args.load_model.is_file():
            env.close()
            eval_env.close()
            raise SystemExit(f"--load-model file not found: {args.load_model}")
        print(f"Continuing PPO training from checkpoint: {args.load_model}")
        model = PPO.load(
            str(args.load_model), env=env, device=args.device,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        # Keep the loaded policy and optimizer state, while applying the requested rollout settings.
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.n_epochs = 10
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=args.device,
            policy_kwargs={
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "activation_fn": __import__("torch").nn.Tanh,
            },
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
