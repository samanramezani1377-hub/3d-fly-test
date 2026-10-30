"""Train a single PPO policy across hardware-conditioned quadrotor profiles."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fast_physics.vehicle_family import DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="universal-quad-ppo")
    p.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--history-length", type=int, default=4)
    p.add_argument("--curriculum-stage", type=int, choices=(1,2,3,4), default=None)
    p.add_argument("--load-model", type=Path, default=None)
    p.add_argument("--vehicle-family", type=Path, default=DEFAULT_VEHICLE_FAMILY)
    p.add_argument("--vehicle-id", action="append", default=None, help="Optional profile allow-list; repeat IDs.")
    p.add_argument("--no-randomization", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.timesteps <= 0 or args.n_envs <= 0:
        raise SystemExit("--timesteps and --n-envs must be positive")
    profiles = load_vehicle_profiles(args.vehicle_family)
    if args.vehicle_id:
        selected = set(args.vehicle_id)
        profiles = [p for p in profiles if p["vehicle_id"] in selected]
        missing = selected - {p["vehicle_id"] for p in profiles}
        if missing:
            raise SystemExit(f"unknown vehicle IDs: {sorted(missing)}")
    if not profiles:
        raise SystemExit("at least one vehicle profile must be selected")
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt to run PPO") from exc

    run_dir = args.output_dir.resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    family_path = args.vehicle_family
    if args.vehicle_id:
        family_path = run_dir / "selected_vehicle_profiles.json"
        family_path.write_text(json.dumps({"schema_version":1,"profiles":profiles}, indent=2), encoding="utf-8")

    def make_env(rank):
        def _init():
            return VehicleFamilyEnv(
                vehicle_family_path=family_path,
                randomization_path=Path(__file__).resolve().parents[1] / "vehicle_models/smooth2_parametric/domain_randomization.json",
                history_length=args.history_length, randomize=not args.no_randomization,
                curriculum_stage=args.curriculum_stage,
            )
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    try:
        if args.load_model:
            model = PPO.load(str(args.load_model), env=env, device="auto")
            model.tensorboard_log = str(run_dir / "tensorboard")
        else:
            model = PPO(
                "MlpPolicy", env, learning_rate=args.learning_rate, n_steps=1024,
                batch_size=256, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.005, verbose=1, seed=args.seed, device="auto",
                tensorboard_log=str(run_dir / "tensorboard"),
            )
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(str(run_dir / "final_model"))
        metadata = {
            "run_name": args.run_name, "created_at": datetime.now(timezone.utc).isoformat(),
            "timesteps": args.timesteps, "n_envs": args.n_envs, "seed": args.seed,
            "history_length": args.history_length, "curriculum_stage": args.curriculum_stage,
            "initial_model": str(args.load_model) if args.load_model else None,
            "domain_randomization": not args.no_randomization,
            "vehicle_family_file": str(args.vehicle_family),
            "vehicle_profiles": [{"vehicle_id":p["vehicle_id"],"calibration_status":p["calibration_status"],
                                  "parameter_source":p.get("parameter_source","unspecified")} for p in profiles],
            "policy_observation": "state/action history + online lift estimate/confidence + 12 normalized nominal hardware features",
            "policy_is_hardware_conditioned": True,
            "episode_randomization_ground_truth_is_hidden": True,
            "physics_backend": "training.fast_physics.FastQuadrotor",
            "stable_baselines3_version": stable_baselines3.__version__,
            "simulation_only": True, "physical_flight_eligible": False,
            "warning": "Provisional profiles and this policy are not approved for physical flight."
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved universal PPO model and metadata to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
