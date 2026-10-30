"""Train a single PPO policy across hardware-conditioned quadrotor profiles."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from fast_physics.vehicle_family import DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles
from fast_physics.ppo_action import BodyWrenchActionWrapper, policy_action_to_motor_commands


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="universal-quad-ppo")
    p.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-frequency", type=int, default=100000, help="Balanced profile evaluation interval in aggregate environment transitions.")
    p.add_argument("--checkpoint-frequency", type=int, default=20000, help="Latest resumable checkpoint interval in aggregate environment transitions.")
    p.add_argument("--resume", action="store_true", help="Resume this same stage from --load-model and its training_state.json.")
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
                 stage, episodes, frequency, checkpoint_frequency, best_dir, reports_dir, seed):
        super().__init__(verbose=1)
        self.profiles, self.family_path = profiles, family_path
        self.randomization_path, self.history_length = randomization_path, history_length
        self.stage, self.episodes = stage, max(1, int(episodes))
        self.frequency = max(1, frequency)
        self.checkpoint_frequency = max(1, checkpoint_frequency)
        self.best_dir, self.reports_dir = Path(best_dir), Path(reports_dir)
        self.seed, self.best_score, self.best_report = seed, (-1.0, -1.0, -1.0, float("-inf")), None
        self.latest_dir = self.best_dir.parent
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.last_eval_timesteps = 0
        self.last_checkpoint_timesteps = None
        selection_path = self.best_dir / "selection.json"
        if selection_path.is_file():
            saved = json.loads(selection_path.read_text(encoding="utf-8"))
            saved_score = tuple(saved.get("score", ()))
            # Old checkpoints used a different three-field score and are not
            # comparable to the new sustained-hover/takeoff score.
            if len(saved_score) == 4:
                self.best_score = saved_score
                self.best_report = saved.get("report")

    def _save_latest(self, completed=False):
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.latest_dir / "latest_model"))
        (self.latest_dir / "training_state.json").write_text(json.dumps({
            "num_timesteps": int(self.num_timesteps), "stage": self.stage,
            "seed": self.seed, "completed": bool(completed),
            "last_eval_timesteps": int(self.last_eval_timesteps),
            "latest_checkpoint": "latest_model.zip"
        }, indent=2), encoding="utf-8")

    def _on_step(self) -> bool:
        # BaseCallback requires this method; returning True keeps PPO training running.
        return True

    def _on_training_start(self):
        if self.last_checkpoint_timesteps is None:
            self.last_checkpoint_timesteps = int(self.num_timesteps)

    def _on_rollout_end(self):
        # Save only at rollout boundaries so a resume never counts discarded,
        # partially collected PPO experience as completed training.
        if int(self.num_timesteps) - self.last_checkpoint_timesteps >= self.checkpoint_frequency:
            self._save_latest(completed=False)
            self.last_checkpoint_timesteps = int(self.num_timesteps)
        if int(self.num_timesteps) - self.last_eval_timesteps < self.frequency:
            return
        self.last_eval_timesteps = int(self.num_timesteps)
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
                    total, crashed, took_off = 0.0, False, False
                    stable_steps = max_stable_steps = 0
                    while True:
                        policy_action, _ = self.model.predict(obs, deterministic=True)
                        action = policy_action_to_motor_commands(policy_action, e.inner.base_params)
                        obs, reward, terminated, truncated, info = e.step(action)
                        total += float(reward)
                        crashed = crashed or bool(info.get("crashed", False))
                        took_off = took_off or bool(info.get("has_taken_off", False))
                        velocity = info.get("state_velocity_m_s", [float("inf")] * 3)
                        attitude = info.get("attitude_rad", [float("inf")] * 3)
                        angular_rate = info.get("angular_velocity_rad_s", [float("inf")] * 3)
                        try:
                            speed_sq = sum(float(v) ** 2 for v in velocity)
                            tilt = max(abs(float(attitude[0])), abs(float(attitude[1])))
                            angular_rate_sq = sum(float(v) ** 2 for v in angular_rate)
                            stable = bool(info.get("has_taken_off", False)
                                and not info.get("crashed", False)
                                and info.get("state_finite", True)
                                and float(info.get("altitude_error_m", 999)) <= 0.15
                                and float(info.get("horizontal_error_m", 999)) <= 0.20
                                and speed_sq <= 0.35 ** 2 and tilt <= 0.25
                                and angular_rate_sq <= 0.5 ** 2)
                        except (TypeError, ValueError, IndexError):
                            stable = False
                        if stable:
                            stable_steps += 1
                            max_stable_steps = max(max_stable_steps, stable_steps)
                        else:
                            stable_steps = 0
                        if terminated or truncated:
                            break
                    stable_seconds = max_stable_steps * e.inner.dt
                    success = bool(not crashed and stable_seconds >= 5.0)
                    episodes.append({"seed": seed, "success": success,
                        "takeoff_success": took_off, "takeoff_failed": bool(not took_off and truncated),
                        "crashed": crashed, "steps": e.inner._steps,
                        "stable_hover_seconds_max": stable_seconds, "return": total,
                        "altitude_error_m": info.get("altitude_error_m"),
                        "horizontal_error_m": info.get("horizontal_error_m"),
                        "crash_reasons": info.get("crash_reasons", [])})
            finally:
                e.close()
            count = len(episodes)
            profile_reports.append({"vehicle_id": profile["vehicle_id"],
                "success_rate": sum(x["success"] for x in episodes) / count,
                "takeoff_success_rate": sum(x["takeoff_success"] for x in episodes) / count,
                "takeoff_failure_rate": sum(x["takeoff_failed"] for x in episodes) / count,
                "crash_rate": sum(x["crashed"] for x in episodes) / count,
                "mean_stable_hover_seconds": sum(x["stable_hover_seconds_max"] for x in episodes) / count,
                "mean_return": sum(x["return"] for x in episodes) / count,
                "episodes": episodes})
        score = (min(x["success_rate"] for x in profile_reports),
            -max(x["crash_rate"] for x in profile_reports),
            min(x["takeoff_success_rate"] for x in profile_reports),
            sum(x["mean_return"] for x in profile_reports) / len(profile_reports))
        report = {"timesteps": int(self.num_timesteps), "selection_score": list(score),
            "selection_rule": ["maximize minimum per-profile five-second sustained-hover success",
                "minimize worst-profile crash rate", "maximize minimum per-profile takeoff rate",
                "maximize mean return as final tie-breaker"], "profiles": profile_reports}
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.reports_dir / f"profile-evaluation-{self.num_timesteps}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._save_latest(completed=False)
        self.last_checkpoint_timesteps = int(self.num_timesteps)
        if score > self.best_score:
            self.best_score, self.best_report = score, report_path.name
            self.best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_dir / "best_model"))
            (self.best_dir / "selection.json").write_text(json.dumps({
                "score": list(score), "report": self.best_report,
                "timesteps": int(self.num_timesteps),
                "selection_rule": report["selection_rule"]}, indent=2), encoding="utf-8")
            print(f"Balanced best checkpoint: steps={self.num_timesteps}, score={score}")
        return True


def main(argv=None):
    args = parse_args(argv)
    if args.timesteps <= 0 or args.n_envs <= 0 or args.eval_frequency <= 0 or args.checkpoint_frequency <= 0 or args.eval_episodes <= 0:
        raise SystemExit("--timesteps, --n-envs, --eval-frequency, --checkpoint-frequency and --eval-episodes must be positive")
    if args.resume and not args.load_model:
        raise SystemExit("--resume requires --load-model pointing to this stage latest_model.zip")
    rollout_transitions = args.n_envs * 500
    if args.timesteps % rollout_transitions:
        raise SystemExit(
            f"--timesteps must be divisible by n_envs * 500 ({rollout_transitions}) "
            "so each stage and same-stage resume finish on a complete PPO rollout."
        )
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
            base_env = VehicleFamilyEnv(
                vehicle_family_path=family_path,
                randomization_path=args.randomization_config,
                history_length=args.history_length, randomize=not args.no_randomization,
                curriculum_stage=args.curriculum_stage,
            )
            return BodyWrenchActionWrapper(base_env)
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    aggregate_eval_frequency = args.eval_frequency
    eval_callback = BalancedProfileCallback(profiles, family_path, args.randomization_config,
        args.history_length, args.curriculum_stage, args.eval_episodes,
        aggregate_eval_frequency, args.checkpoint_frequency, run_dir / "best",
        run_dir / "profile-evaluations", args.seed + 500000)
    try:
        if args.load_model:
            model = PPO.load(str(args.load_model), env=env, device="cpu")
            model.tensorboard_log = str(run_dir / "tensorboard")
            if model.n_steps != 500 or model.batch_size != 250:
                raise RuntimeError("Checkpoint PPO rollout settings do not match the v4 training pipeline.")
        else:
            model = PPO(
                "MlpPolicy", env, learning_rate=args.learning_rate, n_steps=500,
                batch_size=250, gamma=0.99, gae_lambda=0.95, clip_range=0.15,
                ent_coef=0.008, vf_coef=0.5, max_grad_norm=0.5,
                target_kl=0.03, normalize_advantage=True,
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "log_std_init": -1.0},
                verbose=1, seed=args.seed, device="cpu",
                tensorboard_log=str(run_dir / "tensorboard"),
            )
        if args.resume:
            state_path = run_dir / "training_state.json"
            if not state_path.is_file():
                raise RuntimeError("--resume requested but this stage has no training_state.json")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            eval_callback.last_eval_timesteps = int(state.get("last_eval_timesteps", 0))
            if state.get("completed") or state.get("stage") != args.curriculum_stage:
                raise RuntimeError("Refusing to resume a completed or mismatched curriculum stage")
            if int(state.get("num_timesteps", -1)) != int(model.num_timesteps):
                raise RuntimeError("Checkpoint timestep count does not match training_state.json")
            remaining = max(args.timesteps - int(model.num_timesteps), 0)
            reset_num_timesteps = False
        else:
            # Previous-stage checkpoint is a warm start, not this stage's budget.
            remaining = args.timesteps
            reset_num_timesteps = True
        if remaining:
            model.learn(total_timesteps=remaining, callback=eval_callback,
                        progress_bar=False, reset_num_timesteps=reset_num_timesteps)
        model.save(str(run_dir / "final_model"))
        eval_callback._save_latest(completed=True)
        (run_dir / "training_state.json").write_text(json.dumps({
            "num_timesteps": int(model.num_timesteps), "stage": args.curriculum_stage,
            "seed": args.seed, "completed": True, "latest_checkpoint": "latest_model.zip"
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
            "checkpoint_selection": "max_min_profile_sustained_hover_then_min_worst_crash_then_max_min_takeoff_then_mean_return",
            "history_length": args.history_length, "curriculum_stage": args.curriculum_stage,
            "initial_model": str(args.load_model) if args.load_model else None,
            "domain_randomization": not args.no_randomization,
            "vehicle_family_file": str(args.vehicle_family),
            "vehicle_profiles": [{"vehicle_id":p["vehicle_id"],"calibration_status":p["calibration_status"],
                                  "parameter_source":p.get("parameter_source","unspecified")} for p in profiles],
            "policy_observation": "dimensionless-scaled state/action history + online lift estimate/confidence + 12 normalized nominal hardware features",
            "policy_action_semantics": "unbounded_gaussian_tanh_squashed_collective_roll_pitch_yaw_body_wrench",
            "motor_allocator": "geometry-derived nominal-profile wrench allocation; randomized episode parameters hidden",
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
