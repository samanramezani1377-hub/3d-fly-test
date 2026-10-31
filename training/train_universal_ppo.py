"""Train a single PPO policy across hardware-conditioned quadrotor profiles."""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from fast_physics.vehicle_family import DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles
from fast_physics.ppo_action import BodyWrenchActionWrapper


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
    p.add_argument("--no-teacher", action="store_true", help="Disable privileged reference guidance while retaining the selected physics curriculum.")
    p.add_argument("--teacher-weight", type=float, default=None, help="Override the curriculum reference-teacher blend weight (0.0 to 1.0).")
    p.add_argument("--velocity-control", action="store_true", help="Train nose-relative 3D velocity plus independent yaw-rate tracking.")
    return p.parse_args(argv)



def build_velocity_evaluation_sequence(episode_index: int, stage: int, limits_m_s=(1.5, 1.5, 0.75)):
    """Exercise 26 translational combinations plus yaw-only turns and combined turns."""
    limits = np.asarray(limits_m_s, dtype=np.float64)
    if limits.shape != (3,) or np.any(limits <= 0):
        raise ValueError("velocity evaluation limits must be three positive values")
    stage_bands = {1: (0.25, 0.02), 2: (0.35, 0.03), 3: (0.45, 0.035), 4: (0.50, 0.04), 5: (0.55, 0.05)}
    base_scale, episode_increment = stage_bands.get(int(stage), stage_bands[5])
    speed_scale = base_scale + episode_increment * (int(episode_index) % 10)
    yaw_limits = {1: np.deg2rad(20), 2: np.deg2rad(35), 3: np.deg2rad(50), 4: np.deg2rad(70), 5: np.deg2rad(90)}
    yaw_limit = float(yaw_limits.get(int(stage), yaw_limits[5]))
    direction_pairs = []
    for axis in range(3):
        vector = np.zeros(3, dtype=np.float64)
        vector[axis] = speed_scale * limits[axis]
        direction_pairs.append(vector)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for sign_second in (-1.0, 1.0):
            vector = np.zeros(3, dtype=np.float64)
            vector[first] = speed_scale * limits[first] / np.sqrt(2.0)
            vector[second] = sign_second * speed_scale * limits[second] / np.sqrt(2.0)
            direction_pairs.append(vector)
    for sign_y in (-1.0, 1.0):
        for sign_z in (-1.0, 1.0):
            direction_pairs.append(speed_scale * limits * np.array([1.0, sign_y, sign_z]) / np.sqrt(3.0))
    offset = (int(episode_index) * 5) % len(direction_pairs)
    direction_pairs = direction_pairs[offset:] + direction_pairs[:offset]
    commands = []
    for index, vector in enumerate(direction_pairs):
        yaw_rate = yaw_limit if index % 2 == 0 else -yaw_limit
        commands.append({"velocity_m_s": vector.tolist(), "yaw_rate_rad_s": yaw_rate, "duration_steps": 35})
        commands.append({"velocity_m_s": (-vector).tolist(), "yaw_rate_rad_s": -yaw_rate, "duration_steps": 35})
    # Pure in-place turns, both directions and lower-rate precision checks.
    for yaw_rate in (yaw_limit, -yaw_limit, 0.5 * yaw_limit, -0.5 * yaw_limit):
        commands.append({"velocity_m_s": [0.0, 0.0, 0.0], "yaw_rate_rad_s": yaw_rate, "duration_steps": 35})
    return commands

class BalancedProfileCallback(BaseCallback):
    """Choose the checkpoint that best protects the weakest vehicle profile."""
    def __init__(self, profiles, family_path, randomization_path, history_length,
                 stage, episodes, frequency, checkpoint_frequency, best_dir, reports_dir, seed, disable_teacher=False, teacher_weight=None, velocity_control=False):
        super().__init__(verbose=1)
        self.profiles, self.family_path = profiles, family_path
        self.randomization_path, self.history_length = randomization_path, history_length
        self.stage, self.episodes = stage, max(1, int(episodes))
        self.disable_teacher = bool(disable_teacher)
        self.teacher_weight = teacher_weight
        self.velocity_control = bool(velocity_control)
        self.frequency = max(1, frequency)
        self.checkpoint_frequency = max(1, checkpoint_frequency)
        self.best_dir, self.reports_dir = Path(best_dir), Path(reports_dir)
        self.seed, self.best_score, self.best_report = seed, (-1.0, -1.0, -1.0, -1.0, float("-inf"), float("-inf")), None
        self.latest_dir = self.best_dir.parent
        self.sampling_weights_path = self.latest_dir / "profile-sampling-weights.json"
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.last_eval_timesteps = 0
        self.last_checkpoint_timesteps = None
        self._speed_started_at = None
        self._speed_start_timesteps = 0
        self._speed_last_at = None
        self._speed_last_timesteps = 0
        selection_path = self.best_dir / "selection.json"
        if selection_path.is_file():
            saved = json.loads(selection_path.read_text(encoding="utf-8"))
            saved_score = tuple(saved.get("score", ()))
            # Older selection rules are not comparable to the v5 six-field rule.
            if len(saved_score) == 6:
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
        now = time.perf_counter()
        self._speed_started_at = now
        self._speed_start_timesteps = int(self.num_timesteps)
        self._speed_last_at = now
        self._speed_last_timesteps = int(self.num_timesteps)

    def _on_rollout_end(self):
        # Emit a machine-searchable throughput record on every PPO rollout.
        # Rates are aggregate environment transitions/sec across all workers.
        now = time.perf_counter()
        steps_now = int(self.num_timesteps)
        if self._speed_last_at is not None and self._speed_started_at is not None:
            interval_seconds = max(now - self._speed_last_at, 1e-9)
            interval_steps = max(steps_now - self._speed_last_timesteps, 0)
            total_seconds = max(now - self._speed_started_at, 1e-9)
            total_steps = max(steps_now - self._speed_start_timesteps, 0)
            print(
                "[PPO_SPEED] "
                f"timesteps={steps_now} "
                f"rollout_steps={interval_steps} "
                f"rollout_seconds={interval_seconds:.2f} "
                f"rollout_steps_per_sec={interval_steps / interval_seconds:.2f} "
                f"average_steps_per_sec={total_steps / total_seconds:.2f}",
                flush=True,
            )
        self._speed_last_at = now
        self._speed_last_timesteps = steps_now
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
            # Checkpoint selection must measure the PPO policy alone. The teacher
            # may guide training, but is always disabled in every periodic evaluation.
            e = BodyWrenchActionWrapper(VehicleFamilyEnv(vehicle_family_path=self.family_path,
                randomization_path=self.randomization_path, history_length=self.history_length,
                randomize=True, curriculum_stage=self.stage, vehicle_id=profile["vehicle_id"],
                velocity_control=self.velocity_control),
                disable_teacher=True, reference_weight_override=0.0)
            plant_env = e.env
            episodes = []
            try:
                for ei in range(self.episodes):
                    seed = self.seed + self.n_calls * 100 + pi * 1000 + ei
                    eval_options = ({
                        "velocity_command_sequence": build_velocity_evaluation_sequence(ei, self.stage),
                        "target_position": [0.0, 0.0, 2.0 + 0.1 * (ei % 5)],
                    } if self.velocity_control else None)
                    obs, _ = e.reset(seed=seed, options=eval_options)
                    total, crashed, took_off = 0.0, False, False
                    stable_steps = max_stable_steps = 0
                    saturation_samples = []
                    zero_command_samples = []
                    velocity_tracking_errors = []
                    velocity_command_errors = {}
                    yaw_rate_errors = []
                    yaw_command_errors = {}
                    neutral_started_step = None
                    neutral_stop_time = None
                    neutral_stop_stable_steps = 0
                    while True:
                        policy_action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = e.step(policy_action)
                        total += float(reward)
                        crashed = crashed or bool(info.get("crashed", False))
                        took_off = took_off or bool(info.get("has_taken_off", False))
                        saturation_samples.append(float(info.get("motor_saturation_fraction", 0.0)))
                        zero_command_samples.append(float(info.get("motor_zero_command_fraction", 0.0)))
                        if self.velocity_control:
                            velocity_tracking_errors.append(float(info.get("velocity_tracking_error_m_s", float("inf"))))
                            command_index = int(info.get("velocity_command_index", -1))
                            if command_index >= 0:
                                if command_index < 26:
                                    velocity_command_errors.setdefault(command_index, []).append(
                                        float(info.get("velocity_tracking_error_m_s", float("inf")))
                                    )
                                yaw_error = abs(float(info.get("yaw_rate_tracking_error_rad_s", float("inf"))))
                                yaw_rate_errors.append(yaw_error)
                                yaw_command_errors.setdefault(command_index, []).append(yaw_error)
                            if info.get("velocity_command_sequence_complete"):
                                if neutral_started_step is None:
                                    neutral_started_step = int(plant_env.inner._steps)
                                if neutral_stop_time is None:
                                    current_speed = float(np.linalg.norm(info.get("state_velocity_m_s", [999, 999, 999])))
                                    current_yaw_rate = abs(float((info.get("angular_velocity_rad_s") or [999, 999, 999])[2]))
                                    neutral_stop_stable_steps = neutral_stop_stable_steps + 1 if current_speed <= 0.25 and current_yaw_rate <= 0.15 else 0
                                    if neutral_stop_stable_steps >= 25:
                                        neutral_stop_time = (
                                            int(plant_env.inner._steps) - neutral_started_step - 24
                                        ) * float(plant_env.inner.dt)
                        velocity = info.get("state_velocity_m_s", [float("inf")] * 3)
                        attitude = info.get("attitude_rad", [float("inf")] * 3)
                        angular_rate = info.get("angular_velocity_rad_s", [float("inf")] * 3)
                        try:
                            speed_sq = sum(float(v) ** 2 for v in velocity)
                            tilt = max(abs(float(attitude[0])), abs(float(attitude[1])))
                            angular_rate_sq = sum(float(v) ** 2 for v in angular_rate)
                            if self.velocity_control:
                                stable = bool(info.get("has_taken_off", False)
                                    and not info.get("crashed", False)
                                    and info.get("state_finite", True)
                                    and float(info.get("velocity_tracking_error_m_s", 999)) <= 0.35
                                    and float(info.get("altitude_error_m", 999)) <= 0.25
                                    and float(info.get("horizontal_error_m", 999)) <= 0.50
                                    and tilt <= 0.25 and angular_rate_sq <= 0.5 ** 2)
                            else:
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
                    stable_seconds = max_stable_steps * plant_env.inner.dt
                    mean_tracking_error = (sum(velocity_tracking_errors) / max(len(velocity_tracking_errors), 1)) if self.velocity_control else 0.0
                    command_means = [sum(values) / len(values) for values in velocity_command_errors.values()]
                    max_command_error = max(command_means, default=float("inf")) if self.velocity_control else 0.0
                    command_coverage = len(velocity_command_errors) if self.velocity_control else 0
                    mean_yaw_rate_error = sum(yaw_rate_errors) / max(len(yaw_rate_errors), 1) if self.velocity_control else 0.0
                    max_yaw_command_error = max((sum(values) / len(values) for values in yaw_command_errors.values()), default=float("inf")) if self.velocity_control else 0.0
                    yaw_command_coverage = len(yaw_command_errors) if self.velocity_control else 0
                    success = bool(not crashed and took_off and (
                        mean_tracking_error <= 0.50 and max_command_error <= 0.85
                        and command_coverage == 26
                        and mean_yaw_rate_error <= 0.30 and max_yaw_command_error <= 0.50
                        and yaw_command_coverage == 30
                        and neutral_stop_time is not None and neutral_stop_time <= 3.5
                        if self.velocity_control else stable_seconds >= 5.0))
                    episodes.append({"seed": seed, "success": success,
                        "takeoff_success": took_off, "takeoff_failed": bool(not took_off and truncated),
                        "crashed": crashed, "steps": plant_env.inner._steps,
                        "stable_hover_seconds_max": stable_seconds,
                        "mean_velocity_tracking_error_m_s": mean_tracking_error,
                        "max_direction_velocity_tracking_error_m_s": max_command_error,
                        "velocity_direction_coverage": command_coverage,
                        "mean_yaw_rate_tracking_error_rad_s": mean_yaw_rate_error,
                        "max_yaw_command_tracking_error_rad_s": max_yaw_command_error,
                        "yaw_command_coverage": yaw_command_coverage,
                        "neutral_braking_time_seconds": neutral_stop_time,
                        "return": total,
                        "mean_motor_saturation_fraction": sum(saturation_samples) / max(len(saturation_samples), 1),
                        "mean_motor_zero_command_fraction": sum(zero_command_samples) / max(len(zero_command_samples), 1),
                        "altitude_error_m": info.get("altitude_error_m"),
                        "horizontal_error_m": info.get("horizontal_error_m"),
                        "crash_reasons": info.get("crash_reasons", [])})
            finally:
                e.close()
            count = len(episodes)
            profile_reports.append({"vehicle_id": profile["vehicle_id"],
                "reference_teacher_weight": e.reference_weight,
                "success_rate": sum(x["success"] for x in episodes) / count,
                "takeoff_success_rate": sum(x["takeoff_success"] for x in episodes) / count,
                "takeoff_failure_rate": sum(x["takeoff_failed"] for x in episodes) / count,
                "crash_rate": sum(x["crashed"] for x in episodes) / count,
                "mean_stable_hover_seconds": sum(x["stable_hover_seconds_max"] for x in episodes) / count,
                "mean_velocity_tracking_error_m_s": sum(x["mean_velocity_tracking_error_m_s"] for x in episodes) / count,
                "mean_max_direction_tracking_error_m_s": sum(x["max_direction_velocity_tracking_error_m_s"] for x in episodes) / count,
                "mean_velocity_direction_coverage": sum(x["velocity_direction_coverage"] for x in episodes) / count,
                "mean_yaw_rate_tracking_error_rad_s": sum(x["mean_yaw_rate_tracking_error_rad_s"] for x in episodes) / count,
                "mean_max_yaw_command_tracking_error_rad_s": sum(x["max_yaw_command_tracking_error_rad_s"] for x in episodes) / count,
                "mean_yaw_command_coverage": sum(x["yaw_command_coverage"] for x in episodes) / count,
                "mean_neutral_braking_time_seconds": (
                    sum(x["neutral_braking_time_seconds"] for x in episodes if x["neutral_braking_time_seconds"] is not None)
                    / sum(x["neutral_braking_time_seconds"] is not None for x in episodes)
                    if any(x["neutral_braking_time_seconds"] is not None for x in episodes) else None
                ),
                "mean_motor_saturation_fraction": sum(x["mean_motor_saturation_fraction"] for x in episodes) / count,
                "mean_motor_zero_command_fraction": sum(x["mean_motor_zero_command_fraction"] for x in episodes) / count,
                "mean_return": sum(x["return"] for x in episodes) / count,
                "episodes": episodes})
        score = (min(x["success_rate"] for x in profile_reports),
            -max(x["crash_rate"] for x in profile_reports),
            min(x["takeoff_success_rate"] for x in profile_reports),
            -sum(x["mean_velocity_tracking_error_m_s"] for x in profile_reports) / len(profile_reports) if self.velocity_control else sum(x["mean_stable_hover_seconds"] for x in profile_reports) / len(profile_reports),
            -sum(x["mean_motor_saturation_fraction"] for x in profile_reports) / len(profile_reports),
            sum(x["mean_return"] for x in profile_reports) / len(profile_reports))
        report = {"timesteps": int(self.num_timesteps), "selection_score": list(score),
            "selection_rule": ["maximize minimum per-profile mission success",
                "minimize worst-profile crash rate", "maximize minimum per-profile takeoff rate",
                ("minimize mean velocity tracking error" if self.velocity_control else "maximize mean stable-hover duration"),
                "minimize motor saturation", "maximize mean return as final tie-breaker"], "profiles": profile_reports}
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.reports_dir / f"profile-evaluation-{self.num_timesteps}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Increase sampling probability for profiles with weaker mission results.
        weaknesses = {
            item["vehicle_id"]: (
                1.0 - item["success_rate"]
                + 0.5 * item["crash_rate"]
                + 0.25 * (1.0 - item["takeoff_success_rate"])
            )
            for item in profile_reports
        }
        mean_weakness = sum(weaknesses.values()) / len(weaknesses)
        previous = {}
        if self.sampling_weights_path.is_file():
            previous = json.loads(self.sampling_weights_path.read_text(encoding="utf-8"))
        proposed = {
            vid: float(np.clip(1.0 + 1.5 * (weakness - mean_weakness), 0.5, 2.5))
            for vid, weakness in weaknesses.items()
        }
        proposed_mean = sum(proposed.values()) / len(proposed)
        proposed = {vid: value / proposed_mean for vid, value in proposed.items()}
        smoothed = {
            vid: 0.65 * float(previous.get(vid, 1.0)) + 0.35 * proposed[vid]
            for vid in proposed
        }
        weight_tmp = self.sampling_weights_path.with_suffix(".json.tmp")
        weight_tmp.write_text(json.dumps(smoothed, indent=2), encoding="utf-8")
        weight_tmp.replace(self.sampling_weights_path)

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


def build_ppo(PPO, env, args, run_dir):
    return PPO(
        "MlpPolicy", env, learning_rate=args.learning_rate, n_steps=500,
        batch_size=250, gamma=0.99, gae_lambda=0.95, clip_range=0.15,
        ent_coef=0.008, vf_coef=0.5, max_grad_norm=0.5,
        target_kl=0.03, normalize_advantage=True,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "log_std_init": -1.0},
        verbose=1, seed=args.seed, device="cpu",
        tensorboard_log=str(run_dir / "tensorboard"),
    )


def load_ppo_checkpoint(PPO, checkpoint, env, args, run_dir):
    # Load without env first so an older hover checkpoint can be inspected safely.
    source = PPO.load(str(checkpoint), device="cpu")
    source_shape = tuple(source.observation_space.shape)
    target_shape = tuple(env.observation_space.shape)
    if source_shape == target_shape:
        model = PPO.load(str(checkpoint), env=env, device="cpu")
        model.tensorboard_log = str(run_dir / "tensorboard")
        return model
    if args.resume:
        raise RuntimeError("A same-stage resume checkpoint must already use the current observation shape.")
    if not args.velocity_control or len(source_shape) != 1 or len(target_shape) != 1:
        raise RuntimeError(f"Checkpoint observation shape {source_shape} is incompatible with environment {target_shape}.")
    if target_shape[0] != source_shape[0] + 4:
        raise RuntimeError(
            f"Only the expected four velocity/yaw command inputs may be added; "
            f"checkpoint={source_shape}, environment={target_shape}."
        )

    # Warm-start the old hover policy. Four new command columns (3D velocity + yaw rate)
    # start at zero, preserving the old policy response at initialization.
    model = build_ppo(PPO, env, args, run_dir)
    old_state = source.policy.state_dict()
    new_state = model.policy.state_dict()
    expanded = []
    for key, old_tensor in old_state.items():
        if key not in new_state:
            raise RuntimeError(f"Cannot migrate checkpoint: missing policy tensor {key}.")
        new_tensor = new_state[key]
        if tuple(old_tensor.shape) == tuple(new_tensor.shape):
            new_state[key] = old_tensor
        elif (key.endswith(".weight") and old_tensor.ndim == 2 and new_tensor.ndim == 2
              and old_tensor.shape[0] == new_tensor.shape[0]
              and new_tensor.shape[1] == old_tensor.shape[1] + 4):
            expanded_tensor = new_tensor.clone()
            expanded_tensor.zero_()
            expanded_tensor[:, :old_tensor.shape[1]] = old_tensor
            new_state[key] = expanded_tensor
            expanded.append(key)
        else:
            raise RuntimeError(
                f"Cannot safely migrate policy tensor {key}: "
                f"{tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)}."
            )
    if len(expanded) != 2:
        raise RuntimeError(f"Expected to expand actor and critic input layers, expanded: {expanded}")
    model.policy.load_state_dict(new_state)
    print(
        "[CHECKPOINT_MIGRATION] preserved all compatible hover-policy weights; "
        f"zero-initialized command inputs in {expanded}; optimizer state reinitialized.",
        flush=True,
    )
    return model


def main(argv=None):
    args = parse_args(argv)
    if args.teacher_weight is not None and not 0.0 <= args.teacher_weight <= 1.0:
        raise SystemExit("--teacher-weight must be between 0.0 and 1.0")
    if args.no_teacher and args.teacher_weight not in (None, 0.0):
        raise SystemExit("--no-teacher cannot be combined with a nonzero --teacher-weight")
    if args.velocity_control and not args.no_teacher:
        raise SystemExit("--velocity-control requires --no-teacher; hover guidance cannot track movement commands")
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
    sampling_weights_path = run_dir / "profile-sampling-weights.json"
    if not sampling_weights_path.is_file():
        sampling_weights_path.write_text(json.dumps({
            profile["vehicle_id"]: 1.0 for profile in profiles
        }, indent=2), encoding="utf-8")
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
                sampling_weights_path=sampling_weights_path,
                velocity_control=args.velocity_control,
            )
            return BodyWrenchActionWrapper(base_env, disable_teacher=args.no_teacher,
                reference_weight_override=args.teacher_weight)
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    aggregate_eval_frequency = args.eval_frequency
    eval_callback = BalancedProfileCallback(profiles, family_path, args.randomization_config,
        args.history_length, args.curriculum_stage, args.eval_episodes,
        aggregate_eval_frequency, args.checkpoint_frequency, run_dir / "best",
        run_dir / "profile-evaluations", args.seed + 500000, disable_teacher=args.no_teacher,
        teacher_weight=args.teacher_weight, velocity_control=args.velocity_control)
    try:
        if args.load_model:
            model = load_ppo_checkpoint(PPO, args.load_model, env, args, run_dir)
            if model.n_steps != 500 or model.batch_size != 250:
                raise RuntimeError("Checkpoint PPO rollout settings do not match the v5 training pipeline.")
        else:
            model = build_ppo(PPO, env, args, run_dir)
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
            "checkpoint_selection": ("min_profile_velocity_mission_success_then_min_worst_crash_then_min_profile_takeoff_then_min_velocity_tracking_error_then_min_motor_saturation_then_mean_return" if args.velocity_control else "min_profile_hover_then_min_worst_crash_then_min_profile_takeoff_then_stable_duration_then_min_upper_motor_saturation_then_mean_episode_return"),
            "history_length": args.history_length, "curriculum_stage": args.curriculum_stage,
            "initial_model": str(args.load_model) if args.load_model else None,
            "domain_randomization": not args.no_randomization,
            "vehicle_family_file": str(args.vehicle_family),
            "vehicle_profiles": [{"vehicle_id":p["vehicle_id"],"calibration_status":p["calibration_status"],
                                  "parameter_source":p.get("parameter_source","unspecified")} for p in profiles],
            "policy_observation": ("dimensionless-scaled state/action history + online lift estimate/confidence + normalized body-relative forward/left, world-up velocity and yaw-rate command + 12 nominal hardware features" if args.velocity_control else "dimensionless-scaled state/action history + online lift estimate/confidence + 12 normalized nominal hardware features"),
            "velocity_control_enabled": args.velocity_control,
            "velocity_command_frame": "yaw-aligned nose-forward/nose-left plus world-up; forward/left rotate with yaw, vertical remains world-up",
            "yaw_rate_command": "independent rotation about world vertical; positive is counter-clockwise viewed from above (nose turns left), negative turns right",
            "yaw_rate_limits_deg_s_by_stage": {"1":20,"2":35,"3":50,"4":70,"5":90} if args.velocity_control else None,
            "velocity_evaluation_protocol": "every periodic evaluation episode covers 26 translational command directions with simultaneous signed yaw-rate commands, four pure in-place yaw commands, and a neutral linear/angular braking tail; 35 steps per command", 
            "velocity_command_limits_m_s": [1.5, 1.5, 0.75] if args.velocity_control else None,
            "velocity_command_curriculum": {"stage_1": "single-axis low speed including yaw-only", "stage_2": "single-axis medium speed including yaw", "stage_3": "two-axis translation/yaw combinations", "stage_4": "two/three-axis translation/yaw combinations", "stage_5": "continuous random 3D velocity plus yaw rate"} if args.velocity_control else None,
            "policy_action_semantics": "unbounded_gaussian_tanh_squashed_collective_roll_pitch_yaw_body_wrench",
            "motor_allocator": "geometry-derived nominal-profile wrench allocation; randomized episode parameters hidden",
            "policy_is_hardware_conditioned": True,
            "profile_sampling": "adaptive weakness-weighted sampling; smoothed updates from balanced profile evaluation",
            "reference_teacher_schedule": ({"stage_1": 0.0, "stage_2": 0.0, "stage_3": 0.0, "stage_4": 0.0, "stage_5": 0.0} if args.velocity_control else {"stage_1": 0.80, "stage_2": 0.60, "stage_3": 0.40, "stage_4": 0.20, "stage_5": 0.0}),
            "reference_teacher_disabled_for_this_run": args.no_teacher,
            "reference_teacher_disabled_for_all_periodic_evaluations": True,
            "reference_teacher_weight_override": args.teacher_weight,
            "evaluation_reference_teacher_weight": 0.0,
            "evaluation_metrics": (["takeoff_success", "all_26_direction_combinations_covered", "yaw_only_and_combined_yaw_command_coverage", "mean_and_worst_velocity_and_yaw_rate_tracking_error", "neutral_linear_and_angular_braking_time_seconds", "path_and_altitude_tracking", "crash_rate", "motor_saturation_fraction", "mean_return"] if args.velocity_control else ["takeoff_success", "five_second_sustained_hover", "crash_rate", "stable_hover_duration", "motor_saturation_fraction", "mean_return"]),
            "sb3_diagnostics": ["train/approx_kl", "train/explained_variance", "train/entropy_loss", "train/value_loss", "train/policy_gradient_loss"],
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
