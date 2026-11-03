"""Fast teacher-only capability gate run before the universal PPO curriculum.

The policy contribution is exactly zero: BodyWrenchActionWrapper blends 100%
privileged teacher wrench into the plant. This checks whether the reference
controller can execute the same velocity/yaw mission used by PPO evaluation
for Smooth2 only under deterministic randomized Stage-5 scenarios.
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


EXPECTED_PROFILE_IDS = {"smooth2-a2212-custom-prop"}
EPISODES_PER_PROFILE = 3
STAGE = 5
MAX_STEPS = 2000  # Includes 1.5 s yaw commands and a long independent neutral tail.


def run_episode(profile_id: str, episode_index: int, seed: int,
                family_path: Path, randomization_path: Path,
                neutral_tail_mode: str = "braking") -> dict:
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
                "neutral_tail_mode": neutral_tail_mode,
            },
        )
        assert env.reference_weight == 1.0
        assert reset_info.get("reference_teacher_uses_privileged_simulator_state") is True

        # Report the physical thrust margin separately from controller quality.
        # A controller cannot hover if the sampled plant cannot produce weight.
        params = base.inner.plant.params
        teacher = env._velocity_reference
        max_collective_thrust_n = float(teacher.max_collective_thrust_n)
        thrust_to_weight_ratio = max_collective_thrust_n / (params.mass_kg * 9.80665)
        physically_hover_capable = thrust_to_weight_ratio >= 1.04

        linear_errors: dict[int, list[float]] = {}
        yaw_errors: dict[int, list[float]] = {}
        linear_steady_errors: dict[int, list[float]] = {}
        yaw_steady_errors: dict[int, list[float]] = {}
        crashed = False
        took_off = False
        neutral_started = None
        neutral_stable_steps = 0
        neutral_braking_seconds = None
        motor_saturation_samples: list[float] = []
        motor_zero_samples: list[float] = []
        rotor_speed_fraction_samples: list[np.ndarray] = []
        neutral_tail_speed_samples: list[float] = []
        neutral_tail_yaw_samples: list[float] = []
        neutral_tail_position_error_samples: list[float] = []
        allocator_samples: list[dict] = []
        teacher_controller_samples: list[dict] = []
        actual_body_angular_acceleration_samples: list[np.ndarray] = []
        actual_linear_acceleration_samples: list[np.ndarray] = []
        actual_tilt_samples: list[float] = []
        measured_yaw_acceleration_samples: list[float] = []
        previous_measured_yaw_rate = None
        neutral_speed_at_1s = None
        neutral_speed_at_2s = None
        neutral_speed_at_3_5s = None
        steps = 0
        last_info = {}

        while steps < MAX_STEPS:
            # Any finite PPO action is ignored by the 100%-teacher blend.
            obs, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            steps += 1
            last_info = info
            if isinstance(info.get("wrench_allocation"), dict):
                allocator_samples.append(info["wrench_allocation"])
            if isinstance(info.get("teacher_controller"), dict):
                teacher_controller_samples.append(info["teacher_controller"])
            actual_angular_acceleration = np.asarray(
                info.get("angular_acceleration_body_rad_s2", []), dtype=float
            )
            if actual_angular_acceleration.shape == (3,) and np.all(np.isfinite(actual_angular_acceleration)):
                actual_body_angular_acceleration_samples.append(actual_angular_acceleration)
            actual_linear_acceleration = np.asarray(info.get("actual_linear_acceleration_world_m_s2", []), dtype=float)
            if actual_linear_acceleration.shape == (3,) and np.all(np.isfinite(actual_linear_acceleration)):
                actual_linear_acceleration_samples.append(actual_linear_acceleration)
            current_tilt = float(info.get("actual_tilt_rad", np.nan))
            if np.isfinite(current_tilt):
                actual_tilt_samples.append(current_tilt)
            current_measured_yaw_rate = float(info.get("measured_yaw_rate_rad_s", 0.0))
            if previous_measured_yaw_rate is not None:
                measured_yaw_acceleration_samples.append(
                    (current_measured_yaw_rate - previous_measured_yaw_rate) / 0.02
                )
            previous_measured_yaw_rate = current_measured_yaw_rate
            motor_saturation_samples.append(float(info.get("motor_saturation_fraction", 0.0)))
            motor_zero_samples.append(float(info.get("motor_zero_command_fraction", 0.0)))
            rotor_omega = np.asarray(info.get("rotor_omega_rad_s", []), dtype=float)
            max_rotor_omega = float(params.max_rotor_speed_rad_s * params.voltage_factor)
            if rotor_omega.shape == (4,) and max_rotor_omega > 0.0:
                rotor_speed_fraction_samples.append(np.clip(rotor_omega / max_rotor_omega, 0.0, 1.5))
            crashed = crashed or bool(info.get("crashed", False))
            took_off = took_off or bool(info.get("has_taken_off", False))

            command_index = int(info.get("velocity_command_index", -1))
            if command_index >= 0:
                velocity_error = float(info.get("velocity_tracking_error_m_s", float("inf")))
                yaw_error = abs(float(info.get("yaw_rate_tracking_error_rad_s", float("inf"))))
                if command_index < 26:
                    values = linear_errors.setdefault(command_index, [])
                    values.append(velocity_error)
                    if len(values) > 20:
                        linear_steady_errors.setdefault(command_index, []).append(velocity_error)
                else:
                    values = yaw_errors.setdefault(command_index, [])
                    values.append(yaw_error)
                    if len(values) > 20:
                        yaw_steady_errors.setdefault(command_index, []).append(yaw_error)

            if info.get("velocity_command_sequence_complete"):
                if neutral_started is None:
                    neutral_started = steps
                velocity = np.asarray(info.get("state_velocity_m_s", [999, 999, 999]), dtype=float)
                yaw_rate = abs(float(info.get("measured_yaw_rate_rad_s", 999.0)))
                neutral_speed = float(np.linalg.norm(velocity))
                neutral_position_error = float(info.get("position_error_m", 999.0))
                neutral_elapsed = (steps - neutral_started) * 0.02
                neutral_tail_speed_samples.append(neutral_speed)
                neutral_tail_yaw_samples.append(yaw_rate)
                neutral_tail_position_error_samples.append(neutral_position_error)
                if neutral_elapsed >= 1.0 and neutral_speed_at_1s is None:
                    neutral_speed_at_1s = neutral_speed
                if neutral_elapsed >= 2.0 and neutral_speed_at_2s is None:
                    neutral_speed_at_2s = neutral_speed
                if neutral_elapsed >= 3.5 and neutral_speed_at_3_5s is None:
                    neutral_speed_at_3_5s = neutral_speed
                position_is_settled = neutral_tail_mode == "braking" or neutral_position_error <= 0.25
                if neutral_speed <= 0.25 and yaw_rate <= 0.15 and position_is_settled:
                    neutral_stable_steps += 1
                else:
                    neutral_stable_steps = 0
                if neutral_stable_steps >= 25 and neutral_braking_seconds is None:
                    neutral_braking_seconds = (steps - neutral_started - 24) * 0.02

            if crashed or terminated or truncated:
                break

        linear_means = [float(np.mean(v)) for v in linear_errors.values() if v]
        yaw_means = [float(np.mean(v)) for v in yaw_errors.values() if v]
        transient_mean_velocity_error = float(np.mean([x for values in linear_errors.values() for x in values])) if linear_errors else float("inf")
        transient_mean_yaw_error = float(np.mean([x for values in yaw_errors.values() for x in values])) if yaw_errors else float("inf")
        mean_velocity_error = float(np.mean([x for values in linear_steady_errors.values() for x in values])) if linear_steady_errors else float("inf")
        mean_yaw_error = float(np.mean([x for values in yaw_steady_errors.values() for x in values])) if yaw_steady_errors else float("inf")
        max_direction_error = max((float(np.mean(v)) for v in linear_steady_errors.values() if v), default=float("inf"))
        max_yaw_command_error = max((float(np.mean(v)) for v in yaw_steady_errors.values() if v), default=float("inf"))

        saturation_array = np.asarray(motor_saturation_samples, dtype=float)
        zero_array = np.asarray(motor_zero_samples, dtype=float)
        rotor_speed_array = np.asarray(rotor_speed_fraction_samples, dtype=float)
        neutral_tail_final_speed = neutral_tail_speed_samples[-1] if neutral_tail_speed_samples else None
        neutral_tail_final_yaw_rate = neutral_tail_yaw_samples[-1] if neutral_tail_yaw_samples else None
        motor_authority_report = {
            "samples": int(len(saturation_array)),
            "mean_motor_command_saturation_fraction": float(np.mean(saturation_array)) if saturation_array.size else None,
            "peak_motor_command_saturation_fraction": float(np.max(saturation_array)) if saturation_array.size else None,
            "fraction_of_steps_with_any_motor_at_limit": float(np.mean(np.any(rotor_speed_array >= 0.995, axis=1))) if rotor_speed_array.size else None,
            "mean_rotor_speed_fraction_of_voltage_limited_max": float(np.mean(rotor_speed_array)) if rotor_speed_array.size else None,
            "peak_rotor_speed_fraction_of_voltage_limited_max": float(np.max(rotor_speed_array)) if rotor_speed_array.size else None,
            "per_rotor_peak_speed_fraction": np.max(rotor_speed_array, axis=0).astype(float).tolist() if rotor_speed_array.size else None,
            "mean_zero_motor_command_fraction": float(np.mean(zero_array)) if zero_array.size else None,
            "interpretation": "diagnostic_only_not_a_pass_fail_gate; rotor speed is normalized by max_rotor_speed_rad_s times episode voltage_factor",
        }
        allocation_report = {
            "samples": len(allocator_samples),
            "mean_torque_scale": float(np.mean([x["torque_scale"] for x in allocator_samples])) if allocator_samples else None,
            "minimum_torque_scale": float(np.min([x["torque_scale"] for x in allocator_samples])) if allocator_samples else None,
            "fraction_steps_with_torque_desaturation": float(np.mean([x["torque_scale"] < 0.999 for x in allocator_samples])) if allocator_samples else None,
            "mean_collective_scale": float(np.mean([x["collective_scale"] for x in allocator_samples])) if allocator_samples else None,
            "maximum_requested_torque_nm": np.max(np.abs(np.asarray([x["requested_torque_nm"] for x in allocator_samples], dtype=float)), axis=0).tolist() if allocator_samples else None,
            "maximum_allocated_torque_nm": np.max(np.abs(np.asarray([x["allocated_torque_nm"] for x in allocator_samples], dtype=float)), axis=0).tolist() if allocator_samples else None,
            "maximum_motor_command": float(np.max([x["max_motor_command"] for x in allocator_samples])) if allocator_samples else None,
        }
        controller_report = {
            "samples": len(teacher_controller_samples),
            "fraction_steps_horizontal_acceleration_clipped": float(np.mean([x["horizontal_acceleration_clip_fraction"] > 1e-6 for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "peak_horizontal_acceleration_clip_fraction": float(np.max([x["horizontal_acceleration_clip_fraction"] for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "mean_requested_horizontal_acceleration_m_s2": float(np.mean([np.linalg.norm(x["acceleration_requested_world_m_s2"][:2]) for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "mean_after_software_limit_horizontal_acceleration_m_s2": float(np.mean([np.linalg.norm(x["acceleration_after_software_limits_world_m_s2"][:2]) for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "mean_actual_physical_horizontal_acceleration_m_s2": float(np.mean([np.linalg.norm(x[:2]) for x in actual_linear_acceleration_samples])) if actual_linear_acceleration_samples else None,
            "peak_actual_physical_horizontal_acceleration_m_s2": float(np.max([np.linalg.norm(x[:2]) for x in actual_linear_acceleration_samples])) if actual_linear_acceleration_samples else None,
            "mean_actual_physical_vertical_acceleration_m_s2": float(np.mean([x[2] for x in actual_linear_acceleration_samples])) if actual_linear_acceleration_samples else None,
            "mean_actual_tilt_rad": float(np.mean(actual_tilt_samples)) if actual_tilt_samples else None,
            "peak_actual_tilt_rad": float(np.max(actual_tilt_samples)) if actual_tilt_samples else None,
            "fraction_steps_vertical_acceleration_clipped": float(np.mean([bool(x["vertical_acceleration_clipped"]) for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "peak_yaw_acceleration_authority_rad_s2": float(np.max([x["yaw_acceleration_authority_rad_s2"] for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "minimum_yaw_acceleration_authority_rad_s2": float(np.min([x["yaw_acceleration_authority_rad_s2"] for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "mean_abs_requested_yaw_torque_nm": float(np.mean([abs(x["requested_body_torque_nm"][2]) for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "mean_abs_requested_body_r_acceleration_rad_s2": float(np.mean([abs(x["requested_body_angular_acceleration_rad_s2"][2]) for x in teacher_controller_samples])) if teacher_controller_samples else None,
            "mean_abs_actual_body_r_acceleration_rad_s2": float(np.mean([abs(x[2]) for x in actual_body_angular_acceleration_samples])) if actual_body_angular_acceleration_samples else None,
            "mean_abs_measured_yaw_acceleration_rad_s2": float(np.mean(np.abs(measured_yaw_acceleration_samples))) if measured_yaw_acceleration_samples else None,
        }
        checks = {
            "physically_hover_capable_with_4pct_margin": physically_hover_capable,
            "no_crash": not crashed,
            "successful_takeoff": took_off,
            "all_26_translation_commands": len(linear_errors) == 26,
            "all_4_yaw_only_commands": len(yaw_errors) == 4,
            "mean_velocity_error_le_0_50_m_s": mean_velocity_error <= 0.50,
            "worst_direction_error_le_0_85_m_s": max_direction_error <= 0.85,
            "mean_yaw_error_le_0_30_rad_s": mean_yaw_error <= 0.30,
            "worst_yaw_command_error_le_0_50_rad_s": max_yaw_command_error <= 0.50,
            "neutral_braking_le_3_5_s": (neutral_braking_seconds is not None and neutral_braking_seconds <= 3.5) if neutral_tail_mode == "braking" else True,
            "position_return_settled_le_8_s": (neutral_braking_seconds is not None and neutral_braking_seconds <= 8.0) if neutral_tail_mode == "position_return" else True,
        }
        return {
            "vehicle_id": profile_id,
            "episode_index": episode_index,
            "seed": seed,
            "teacher_weight": 1.0,
            "neutral_tail_mode": neutral_tail_mode,
            "randomization": reset_info.get("scenario", {}),
            "wind_m_s": reset_info.get("wind_m_s"),
            "max_collective_thrust_n": max_collective_thrust_n,
            "thrust_to_weight_ratio": thrust_to_weight_ratio,
            "physically_hover_capable": physically_hover_capable,
            "steps": steps,
            "crashed": crashed,
            "took_off": took_off,
            "translation_command_coverage": len(linear_errors),
            "yaw_command_coverage": len(yaw_errors),
            "mean_velocity_tracking_error_m_s": mean_velocity_error,
            "transient_mean_velocity_tracking_error_m_s": transient_mean_velocity_error,
            "steady_state_mean_velocity_tracking_error_m_s": mean_velocity_error,
            "worst_direction_mean_error_m_s": max_direction_error,
            "mean_yaw_rate_error_rad_s": mean_yaw_error,
            "transient_mean_yaw_rate_error_rad_s": transient_mean_yaw_error,
            "steady_state_mean_yaw_rate_error_rad_s": mean_yaw_error,
            "worst_yaw_command_mean_error_rad_s": max_yaw_command_error,
            "neutral_braking_seconds": neutral_braking_seconds,
            "neutral_tail_steps_observed": len(neutral_tail_speed_samples),
            "neutral_tail_final_speed_m_s": neutral_tail_final_speed,
            "neutral_tail_speed_at_1s_m_s": neutral_speed_at_1s,
            "neutral_tail_speed_at_2s_m_s": neutral_speed_at_2s,
            "neutral_tail_speed_at_3_5s_m_s": neutral_speed_at_3_5s,
            "neutral_tail_minimum_speed_m_s": float(np.min(neutral_tail_speed_samples)) if neutral_tail_speed_samples else None,
            "neutral_tail_final_yaw_rate_rad_s": neutral_tail_final_yaw_rate,
            "neutral_tail_final_position_error_m": neutral_tail_position_error_samples[-1] if neutral_tail_position_error_samples else None,
            "neutral_tail_max_position_error_m": float(np.max(neutral_tail_position_error_samples)) if neutral_tail_position_error_samples else None,
            "motor_authority_report": motor_authority_report,
            "wrench_allocation_report": allocation_report,
            "teacher_controller_report": controller_report,
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
    parser.add_argument("--vehicle-id", default="smooth2-a2212-custom-prop")
    parser.add_argument("--output", type=Path, default=ROOT / "training/runs/universal-teacher-preflight/report.json")
    args = parser.parse_args()

    profiles = [p for p in load_vehicle_profiles(args.vehicle_family) if p["vehicle_id"] == args.vehicle_id]
    actual_ids = {p["vehicle_id"] for p in profiles}
    if args.vehicle_id != "smooth2-a2212-custom-prop" or actual_ids != EXPECTED_PROFILE_IDS or len(profiles) != 1:
        raise RuntimeError(f"Preflight is locked to exactly Smooth2; got {sorted(actual_ids)}")

    episodes = []
    for profile_index, profile in enumerate(sorted(profiles, key=lambda p: p["vehicle_id"])):
        for episode_index in range(EPISODES_PER_PROFILE):
            seed = 810000 + profile_index * 100 + episode_index
            for neutral_tail_mode in ("braking", "position_return"):
                try:
                    result = run_episode(
                        profile["vehicle_id"], episode_index, seed,
                        args.vehicle_family, args.randomization_config,
                        neutral_tail_mode=neutral_tail_mode,
                    )
                except Exception as exc:
                    result = {
                        "vehicle_id": profile["vehicle_id"],
                        "episode_index": episode_index,
                        "seed": seed,
                        "teacher_weight": 1.0,
                        "neutral_tail_mode": neutral_tail_mode,
                        "passed": False,
                        "checks": {"episode_completed_without_exception": False},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                episodes.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    by_profile = {}
    for profile_id in sorted(EXPECTED_PROFILE_IDS):
        selected = [item for item in episodes if item["vehicle_id"] == profile_id]
        by_profile[profile_id] = {}
        for mode in ("braking", "position_return"):
            selected = [item for item in episodes if item["vehicle_id"] == profile_id and item.get("neutral_tail_mode") == mode]
            by_profile[profile_id][mode] = {
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
        "vehicle_scope": ["smooth2-a2212-custom-prop"],
        "profile_summary": by_profile,
        "passed": len(episodes) == 6 and all(item["passed"] for item in episodes),
        "episodes": episodes,
        "limitations": (
            "Finite deterministic randomized simulation sample only; it cannot prove "
            "success under every physically possible condition or validate real hardware. "
            "Each randomized seed is evaluated twice: pure braking resets the virtual "
            "position target to the current position, while position_return "
            "retains the original target. Reports distinguish transient, "
            "steady-state, software-limited and actual physical acceleration."
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
