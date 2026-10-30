"""Train a single PPO policy across hardware-conditioned quadrotor profiles."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from fast_physics.vehicle_family import DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="universal-quad-ppo")
    p.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-frequency", type=int, default=20000, help="Balanced profile evaluation interval in aggregate environment transitions.")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--history-length", type=int, default=4)
    p.add_argument("--curriculum-stage", type=int, choices=(1,2,3,4,5), default=None)
    p.add_argument("--load-model", type=Path, default=None)
    p.add_argument("--vehicle-family", type=Path, default=DEFAULT_VEHICLE_FAMILY)
    p.add_argument("--randomization-config", type=Path, default=Path(__file__).resolve().parents[1] / "vehicle_models/family/domain_randomization.json")
    p.add_argument("--vehicle-id", action="append", default=None, help="Optional profile allow-list; repeat IDs.")
    p.add_argument("--no-randomization", action="store_true")
    return p.parse_args(argv)



class BalancedProfileCallback(BaseCallback):
    """Choose the checkpoint that best protects the weakest vehicle profile."""
    def __init__(self, profiles, family_path, randomization_path, history_length,
                 stage, episodes, frequency, best_dir, reports_dir, seed):
        super().__init__(verbose=1)
        self.profiles, self.family_path = profiles, family_path
        self.randomization_path, self.history_length = randomization_path, history_length
        self.stage, self.episodes = stage, max(1, int(episodes))
        self.frequency, self.best_dir, self.reports_dir = max(1, frequency), Path(best_dir), Path(reports_dir)
        self.seed, self.best_score, self.best_report = seed, (-1.0, -1.0, float("-inf")), None
        self.latest_dir = self.best_dir.parent
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        selection_path = self.best_dir / "selection.json"
        if selection_path.is_file():
            saved = json.loads(selection_path.read_text(encoding="utf-8"))
            self.best_score = tuple(saved.get("score", self.best_score))
            self.best_report = saved.get("report")

    def _on_step(self):
        if self.n_calls % self.frequency:
            return True
        profile_reports = []
        for pi, profile in enumerate(self.profiles):
            e = VehicleFamilyEnv(vehicle_family_path=self.family_path,
                randomization_path=self.randomization_path, history_length=self.history_length,
                randomize=True, curriculum_stage=self.stage, vehicle_id=profile["vehicle_id"])
            episodes = []
            try:
                for ei in range(self.episodes):
                    seed = self.seed + self.n_calls * 100 + pi * 1000 + ei
                    obs, _ = e.reset(seed=seed)
                    total, crashed = 0.0, False
                    while True:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = e.step(action)
                        total += float(reward)
                        crashed = crashed or bool(info.get("crashed", False))
                        if terminated or truncated:
                            break
                    success = bool(info.get("has_taken_off", False) and not crashed
                        and info.get("altitude_error_m", 999) < 0.35
                        and info.get("horizontal_error_m", 999) < 0.5)
                    episodes.append({"seed":seed,"success":success,"crashed":crashed,
                        "steps":e.inner._steps,"return":total,
                        "altitude_error_m":info.get("altitude_error_m"),
                        "horizontal_error_m":info.get("horizontal_error_m")})
            finally:
                e.close()
            profile_reports.append({"vehicle_id":profile["vehicle_id"],
                "success_rate":sum(x["success"] for x in episodes)/len(episodes),
                "crash_rate":sum(x["crashed"] for x in episodes)/len(episodes),
                "mean_return":sum(x["return"] for x in episodes)/len(episodes),
                "episodes":episodes})
        score=(min(x["success_rate"] for x in profile_reports),
               -max(x["crash_rate"] for x in profile_reports),
               sum(x["mean_return"] for x in profile_reports)/len(profile_reports))
        report={"timesteps":int(self.num_timesteps),"selection_score":list(score),
            "selection_rule":["maximize minimum per-profile takeoff-hover success",
                              "minimize worst-profile crash rate","maximize mean return"],
            "profiles":profile_reports}
        self.reports_dir.mkdir(parents=True,exist_ok=True)
        report_path=self.reports_dir/f"profile-evaluation-{self.num_timesteps}.json"
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        self.model.save(str(self.latest_dir / "latest_model"))
        (self.latest_dir / "training_state.json").write_text(json.dumps({
            "num_timesteps": int(self.num_timesteps), "stage": self.stage,
            "seed": self.seed, "latest_checkpoint": "latest_model.zip"
        }, indent=2), encoding="utf-8")
        if score > self.best_score:
            self.best_score,self.best_report=score,report_path.name
            self.best_dir.mkdir(parents=True,exist_ok=True)
            self.model.save(str(self.best_dir/"best_model"))
            (self.best_dir/"selection.json").write_text(json.dumps({
                "score":list(score),"report":self.best_report,"timesteps":int(self.num_timesteps),
                "selection_rule":report["selection_rule"]},indent=2),encoding="utf-8")
            print(f"Balanced best checkpoint: steps={self.num_timesteps}, score={score}")
        return True


def main(argv=None):
    args = parse_args(argv)
    if args.timesteps <= 0 or args.n_envs <= 0 or args.eval_frequency <= 0 or args.eval_episodes <= 0:
        raise SystemExit("--timesteps, --n-envs, --eval-frequency and --eval-episodes must be positive")
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
                randomization_path=args.randomization_config,
                history_length=args.history_length, randomize=not args.no_randomization,
                curriculum_stage=args.curriculum_stage,
            )
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    aggregate_eval_frequency = min(args.eval_frequency, max(args.timesteps // 2, 1))
    eval_callback = BalancedProfileCallback(profiles, family_path, args.randomization_config,
        args.history_length, args.curriculum_stage, args.eval_episodes,
        max(aggregate_eval_frequency // args.n_envs, 1), run_dir / "best",
        run_dir / "profile-evaluations", args.seed + 500000)
    try:
        if args.load_model:
            model = PPO.load(str(args.load_model), env=env, device="cpu")
            model.tensorboard_log = str(run_dir / "tensorboard")
        else:
            model = PPO(
                "MlpPolicy", env, learning_rate=args.learning_rate, n_steps=1024,
                batch_size=256, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.005, verbose=1, seed=args.seed, device="cpu",
                tensorboard_log=str(run_dir / "tensorboard"),
            )
        # Each curriculum stage gets its own additional training budget. PPO.load() preserves
        # the cumulative timestep counter, so subtracting it from this stage's budget can
        # skip learning and all evaluation callbacks in stages 2-5.
        remaining = args.timesteps
        if remaining:
            model.learn(total_timesteps=remaining, callback=eval_callback,
                        progress_bar=False, reset_num_timesteps=not bool(args.load_model))
        model.save(str(run_dir / "final_model"))
        model.save(str(run_dir / "latest_model"))
        (run_dir / "training_state.json").write_text(json.dumps({
            "num_timesteps": int(model.num_timesteps), "stage": args.curriculum_stage,
            "completed": True, "latest_checkpoint": "latest_model.zip"
        }, indent=2), encoding="utf-8")
        best_checkpoint = run_dir / "best" / "best_model.zip"
        if not best_checkpoint.is_file():
            raise RuntimeError("evaluation did not produce best/best_model.zip; refusing to continue curriculum without an evaluated checkpoint")
        metadata = {
            "run_name": args.run_name, "created_at": datetime.now(timezone.utc).isoformat(),
            "timesteps": args.timesteps, "n_envs": args.n_envs, "seed": args.seed,
            "evaluation_frequency_aggregate_steps": aggregate_eval_frequency,
            "evaluation_episodes": args.eval_episodes,
            "best_checkpoint": str(best_checkpoint),
            "balanced_selection_score": list(eval_callback.best_score),
            "balanced_selection_report": eval_callback.best_report,
            "checkpoint_selection": "max_min_profile_success_then_min_worst_crash_then_mean_return",
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
