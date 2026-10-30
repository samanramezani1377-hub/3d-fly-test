"""Train PPO on randomized smooth2 variants using the fast research plant.

Physical calibration is required by default. The synthetic flag is for
software-only experimentation and is stamped into run metadata.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fast_physics import AdaptiveDroneEnv, load_vehicle_parameters


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="robust-ppo-smooth2")
    parser.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--curriculum-stage", type=int, choices=(1, 2, 3, 4), default=None)
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument(
        "--software-only-synthetic-calibration", action="store_true",
        help="Explicitly allow unmeasured synthetic motor/mass values. Never use results as physical predictions.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.timesteps <= 0 or args.n_envs <= 0:
        raise SystemExit("--timesteps and --n-envs must be positive")
    load_vehicle_parameters(allow_synthetic_calibration=args.software_only_synthetic_calibration)
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt to run PPO") from exc

    run_dir = (args.output_dir / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    def make_env(rank):
        def _init():
            env = AdaptiveDroneEnv(
                allow_synthetic_calibration=args.software_only_synthetic_calibration,
                history_length=args.history_length,
                randomize=not args.no_randomization,
                curriculum_stage=args.curriculum_stage,
            )
            env.reset(seed=args.seed + rank)
            return env
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    try:
        if args.load_model is not None:
            model = PPO.load(str(args.load_model), env=env, device="auto")
            model.tensorboard_log = str(run_dir / "tensorboard")
        else:
            model = PPO(
                "MlpPolicy", env, learning_rate=args.learning_rate,
                n_steps=1024, batch_size=256, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.005, verbose=1,
                seed=args.seed, device="auto", tensorboard_log=str(run_dir / "tensorboard"),
            )
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(str(run_dir / "final_model"))
        metadata = {
            "run_name": args.run_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "timesteps": args.timesteps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "history_length": args.history_length,
            "curriculum_stage": args.curriculum_stage,
            "initial_model": str(args.load_model) if args.load_model else None,
            "domain_randomization": not args.no_randomization,
            "software_only_synthetic_calibration": args.software_only_synthetic_calibration,
            "policy_observation": "four-frame state/action history; no privileged mass/CG/inertia/motor parameters",
            "vehicle_model": "smooth2-parametric-v1",
            "physics_backend": "training.fast_physics.FastQuadrotor",
            "stable_baselines3_version": stable_baselines3.__version__,
            "simulation_only": True,
            "physical_flight_eligible": False,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved robust PPO model and metadata to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
