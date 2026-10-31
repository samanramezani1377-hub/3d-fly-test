"""Fast teacher-only capability gate run before the universal PPO curriculum.

The policy contribution is exactly zero: BodyWrenchActionWrapper blends 100%
privileged teacher wrench into the plant. This checks whether the reference
controller can execute the same velocity/yaw mission used by PPO evaluation
across every vehicle profile and deterministic randomized Stage-5 scenarios.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fast_physics.ppo_action import BodyWrenchActionWrapper
from fast_physics.vehicle_family import load_vehicle_profiles, VehicleFamilyEnv
from train_universal_ppo import build_velocity_evaluation_sequence


EXPECTED_PROFILE_IDS = {
    "smooth2-a2212-custom-prop",
    "generic-5inch-quad",
    "generic-7inch-long-range",
    "generic-3inch-cinewhoop",
}
EPISODES_PER_PROFILE = 3
STAGE = 5
MAX_STEPS = 1500


def run_episode(profile_id: str, episode_index: int, seed: int,
                family_path: Path, randomization_path: Path) -> dict:
    base = VehicleFamilyEnv(
        vehicle_family_path=family_path,
        randomization_path=randomization_path,
        history_length=4,
        randomize=True,
        curriculum_stage=STAGE,
        vehicle_id=profile_id,
        velocity_control=True,
        dt=0.02,
        max_steps=MAX_STEPS,
    )
    # Override to 1.0: PPO action has zero contribution, teacher has 100%.
    env = BodyWrenchActionWrapper(base, reference_weight_override=1.0)
    try:
        sequence = build_velocity_evaluation_sequence(episode_index, STAGE)
        obs, reset_info = env.reset(
            seed=seed,
            options={
                "velocity_command_sequence": sequence,
                "target_position": [0.0, 0.0, 2.0 + 0.1 * (episode_index % 3)],
            },
        )
        assert env.reference_weight == 1.0
        assert reset_info.get("reference_teacher_uses_privileged_simulator_state") is True

        linear_errors: dict[int, list[float]] = {}
        yaw_errors: dict[int, list[float]] = {}
        crashed = False
        took_off = False
        neutral_started = None
        neutral_stable_steps = 0
        neutral_braking_seconds = None
        steps = 0
        last_info = {}

        while steps < MAX_STEPS:
            # Any finite PPO action is ignored by the 100%-teacher blend.
            obs, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            steps += 1
            last_info = info
            crashed = crashed or bool(info.get("crashed", False))
            took_off = took_off or bool(info.get("has_taken_off", False))

            command_index = int(info.get("velocity_command_index", -1))
            if command_index >= 0:
                velocity_error = float(info.get("velocity_tracking_error_m_s", float("inf")))
                yaw_error = abs(float(info.get("yaw_rate_tracking_error_rad_s", float("inf"))))
                if command_index < 26:
                    linear_errors.setdefault(command_index, []).append(velocity_error)
                yaw_errors.setdefault(command_index, []).append(yaw_error)

            if info.get("velocity_command_sequence_complete"):
                if neutral_started is None:
                    neutral_started = steps
                velocity = np.asarray(info.get("state_velocity_m_s", [999, 999, 999]), dtype=float)
                yaw_rate = abs(float(info.get("measured_yaw_rate_rad_s", 999.0)))
                if np.linalg.norm(velocity) <= 0.25 and yaw_rate <= 0.15:
                    neutral_stable_steps += 1
                else:
                    neutral_stable_steps = 0
                if neutral_stable_steps >= 25 and neutral_braking_seconds is None:
                    neutral_braking_seconds = (steps - neutral_started - 24) * 0.02

            if crashed or terminated or truncated:
                break

        linear_means = [float(np.mean(v)) for v in linear_errors.values() if v]
        yaw_means = [float(np.mean(v)) for v in yaw_errors.values() if v]
        mean_velocity_error = float(np.mean([x for values in linear_errors.values() for x in values])) if linear_errors else float("inf")
        mean_yaw_error = float(np.mean([x for values in yaw_errors.values() for x in values])) if yaw_errors else float("inf")
        max_direction_error = max(linear_means, default=float("inf"))
        max_yaw_command_error = max(yaw_means, default=float("inf"))

        checks = {
            "no_crash": not crashed,
            "successful_takeoff": took_off,
            "all_26_translation_commands": len(linear_errors) == 26,
            "all_30_yaw_commands": len(yaw_errors) == 30,
            "mean_velocity_error_le_0_50_m_s": mean_velocity_error <= 0.50,
            "worst_direction_error_le_0_85_m_s": max_direction_error <= 0.85,
            "mean_yaw_error_le_0_30_rad_s": mean_yaw_error <= 0.30,
            "worst_yaw_command_error_le_0_50_rad_s": max_yaw_command_error <= 0.50,
            "neutral_braking_le_3_5_s": neutral_braking_seconds is not None and neutral_braking_seconds <= 3.5,
        }
        return {
            "vehicle_id": profile_id,
            "episode_index": episode_index,
            "seed": seed,
            "teacher_weight": 1.0,
            "randomization": reset_info.get("scenario", {}),
            "wind_m_s": reset_info.get("wind_m_s"),
            "steps": steps,
            "crashed": crashed,
            "took_off": took_off,
            "translation_command_coverage": len(linear_errors),
            "yaw_command_coverage": len(yaw_errors),
            "mean_velocity_tracking_error_m_s": mean_velocity_error,
            "worst_direction_mean_error_m_s": max_direction_error,
            "mean_yaw_rate_error_rad_s": mean_yaw_error,
            "worst_yaw_command_mean_error_rad_s": max_yaw_command_error,
            "neutral_braking_seconds": neutral_braking_seconds,
            "checks": checks,
            "passed": all(checks.values()),
            "crash_reasons": last_info.get("crash_reasons", []),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle-family", type=Path, default=ROOT / "vehicle_models/family/vehicle_profiles.json")
    parser.add_argument("--randomization-config", type=Path, default=ROOT / "vehicle_models/family/domain_randomization.json")
    parser.add_argument("--output", type=Path, default=ROOT / "training/runs/universal-teacher-preflight/report.json")
    args = parser.parse_args()

    profiles = load_vehicle_profiles(args.vehicle_family)
    actual_ids = {p["vehicle_id"] for p in profiles}
    if actual_ids != EXPECTED_PROFILE_IDS or len(profiles) != 4:
        raise RuntimeError(f"Expected exactly the four supported profiles; got {sorted(actual_ids)}")

    episodes = []
    for profile_index, profile in enumerate(sorted(profiles, key=lambda p: p["vehicle_id"])):
        for episode_index in range(EPISODES_PER_PROFILE):
            seed = 810000 + profile_index * 100 + episode_index
            try:
                result = run_episode(
                    profile["vehicle_id"], episode_index, seed,
                    args.vehicle_family, args.randomization_config,
                )
            except Exception as exc:
                result = {
                    "vehicle_id": profile["vehicle_id"],
                    "episode_index": episode_index,
                    "seed": seed,
                    "teacher_weight": 1.0,
                    "passed": False,
                    "checks": {"episode_completed_without_exception": False},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            episodes.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    by_profile = {}
    for profile_id in sorted(EXPECTED_PROFILE_IDS):
        selected = [item for item in episodes if item["vehicle_id"] == profile_id]
        by_profile[profile_id] = {
            "episodes": len(selected),
            "passed": sum(item["passed"] for item in selected),
            "failed": sum(not item["passed"] for item in selected),
        }

    report = {
        "test": "teacher_only_velocity_yaw_capability_preflight",
        "stage": STAGE,
        "teacher_weight": 1.0,
        "policy_contribution": 0.0,
        "randomization_enabled": True,
        "episodes_per_profile": EPISODES_PER_PROFILE,
        "total_episodes": len(episodes),
        "profile_summary": by_profile,
        "passed": len(episodes) == 12 and all(item["passed"] for item in episodes),
        "episodes": episodes,
        "limitations": (
            "Finite deterministic randomized simulation sample only; it cannot prove "
            "success under every physically possible condition or validate real hardware."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("test", "total_episodes", "profile_summary", "passed", "limitations")}, indent=2))
    if not report["passed"]:
        print("TEACHER PREFLIGHT FAILED: PPO training is blocked.", file=sys.stderr)
        return 1
    print("TEACHER PREFLIGHT PASSED: proceeding to PPO training.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
