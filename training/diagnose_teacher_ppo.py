"""Compare PPO-only, 80/20 PPO-teacher blend, and teacher-only control.

Each mode is evaluated from the same deterministic reset seeds and command
sequence. The JSON contains per-step policy/teacher/blended wrench requests,
allocator saturation, state, tracking and crash diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_universal_ppo import build_velocity_evaluation_sequence
from fast_physics.vehicle_family import VehicleFamilyEnv, load_vehicle_profiles
from fast_physics.ppo_action import BodyWrenchActionWrapper


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def run_mode(model, *, weight, seed, family_path, randomization_path, output_stage=5):
    base = VehicleFamilyEnv(
        vehicle_family_path=family_path,
        randomization_path=randomization_path,
        history_length=4,
        randomize=True,
        curriculum_stage=output_stage,
        vehicle_id="smooth2-a2212-custom-prop",
        velocity_control=True,
        diagnostics_enabled=True,
    )
    env = BodyWrenchActionWrapper(
        base,
        disable_teacher=(weight == 0.0),
        reference_weight_override=weight,
        diagnostics_enabled=True,
    )
    options = {
        "velocity_command_sequence": build_velocity_evaluation_sequence(0, output_stage),
        "target_position": [0.0, 0.0, 2.0],
    }
    trace = []
    total_return = 0.0
    velocity_errors, yaw_errors = [], []
    collective_deltas, torque_deltas = [], []
    torque_desaturation, collective_desaturation, motor_saturation = [], [], []
    crashed = took_off = False
    reset_info = {}
    last_info = {}
    try:
        obs, reset_info = env.reset(seed=seed, options=options)
        initial_observation_sha256 = hashlib.sha256(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            total_return += float(reward)
            crashed = crashed or bool(info.get("crashed", False))
            took_off = took_off or bool(info.get("has_taken_off", False))
            ve = _finite(info.get("velocity_tracking_error_m_s"))
            ye = _finite(info.get("yaw_rate_tracking_error_rad_s"))
            if ve is not None and int(info.get("velocity_command_index", -1)) >= 0:
                velocity_errors.append(ve)
            if ye is not None and int(info.get("velocity_command_index", -1)) >= 0:
                yaw_errors.append(abs(ye))
            action_diag = info.get("action_diagnostics", {})
            allocation = info.get("wrench_allocation", {})
            teacher_diag = info.get("teacher_controller", {})
            delta = action_diag.get("teacher_minus_policy_wrench_n_nm")
            if delta is not None:
                collective_deltas.append(abs(float(delta[0])))
                torque_deltas.append(float(np.linalg.norm(np.asarray(delta[1:4], dtype=float))))
            if allocation:
                torque_desaturation.append(float(allocation.get("torque_desaturation_fraction", 0.0)))
                collective_desaturation.append(float(allocation.get("collective_desaturation_fraction", 0.0)))
            motor_saturation.append(float(info.get("motor_saturation_fraction", 0.0)))
            trace.append({
                "step": int(base.inner._steps),
                "command_index": int(info.get("velocity_command_index", -1)),
                "command_velocity_m_s": info.get("desired_velocity_body_m_s"),
                "command_yaw_rate_rad_s": info.get("desired_yaw_rate_rad_s"),
                "position_m": info.get("position_xyz_m"),
                "velocity_m_s": info.get("state_velocity_m_s"),
                "attitude_rad": info.get("attitude_rad"),
                "angular_velocity_rad_s": info.get("angular_velocity_rad_s"),
                "altitude_error_m": info.get("altitude_error_m"),
                "horizontal_error_m": info.get("horizontal_error_m"),
                "velocity_tracking_error_m_s": ve,
                "yaw_rate_tracking_error_rad_s": ye,
                "policy_action": np.asarray(action, dtype=float).tolist(),
                "action_diagnostics": action_diag,
                "teacher_controller": teacher_diag,
                "wrench_allocation": allocation,
                "crashed": bool(info.get("crashed", False)),
                "crash_reasons": info.get("crash_reasons", []),
                "reward": float(reward),
            })
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "mode": {0.0: "ppo_only", 0.8: "ppo_teacher_80_20", 1.0: "teacher_only"}[weight],
        "teacher_weight": weight,
        "seed": seed,
        "initial_observation_sha256": initial_observation_sha256,
        "initial_randomized_scenario": reset_info.get("scenario"),
        "steps": len(trace),
        "return": total_return,
        "takeoff_success": took_off,
        "crashed": crashed,
        "crash_reasons": last_info.get("crash_reasons", []),
        "mean_velocity_tracking_error_m_s": float(np.mean(velocity_errors)) if velocity_errors else None,
        "mean_yaw_rate_tracking_error_rad_s": float(np.mean(yaw_errors)) if yaw_errors else None,
        "mean_abs_teacher_policy_collective_delta_n": float(np.mean(collective_deltas)) if collective_deltas else None,
        "mean_teacher_policy_torque_delta_nm": float(np.mean(torque_deltas)) if torque_deltas else None,
        "mean_torque_desaturation_fraction": float(np.mean(torque_desaturation)) if torque_desaturation else None,
        "mean_collective_desaturation_fraction": float(np.mean(collective_desaturation)) if collective_desaturation else None,
        "mean_motor_saturation_fraction": float(np.mean(motor_saturation)) if motor_saturation else None,
        "velocity_command_coverage": len({x["command_index"] for x in trace if x["command_index"] >= 0 and x["command_index"] < 26}),
        "yaw_command_coverage": len({x["command_index"] for x in trace if x["command_index"] >= 0}),
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vehicle-family", type=Path, required=True)
    parser.add_argument("--randomization-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[730001, 730002, 730003])
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    profiles = load_vehicle_profiles(args.vehicle_family)
    if [p["vehicle_id"] for p in profiles if p["vehicle_id"] == "smooth2-a2212-custom-prop"] != ["smooth2-a2212-custom-prop"]:
        raise RuntimeError("Smooth2 A2212 profile is missing or duplicated")
    model = PPO.load(str(args.model), device="cpu")
    results = []
    for seed in args.seeds:
        for weight in (0.0, 0.8, 1.0):
            results.append(run_mode(
                model, weight=weight, seed=seed,
                family_path=args.vehicle_family,
                randomization_path=args.randomization_config,
            ))
    summary = {}
    for mode in ("ppo_only", "ppo_teacher_80_20", "teacher_only"):
        rows = [r for r in results if r["mode"] == mode]
        summary[mode] = {
            "episodes": len(rows),
            "takeoff_rate": sum(r["takeoff_success"] for r in rows) / len(rows),
            "crash_rate": sum(r["crashed"] for r in rows) / len(rows),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "mean_velocity_tracking_error_m_s": (
                float(np.mean([r["mean_velocity_tracking_error_m_s"] for r in rows if r["mean_velocity_tracking_error_m_s"] is not None]))
                if any(r["mean_velocity_tracking_error_m_s"] is not None for r in rows) else None
            ),
            "mean_yaw_rate_tracking_error_rad_s": (
                float(np.mean([r["mean_yaw_rate_tracking_error_rad_s"] for r in rows if r["mean_yaw_rate_tracking_error_rad_s"] is not None]))
                if any(r["mean_yaw_rate_tracking_error_rad_s"] is not None for r in rows) else None
            ),
            "mean_abs_teacher_policy_collective_delta_n": float(np.mean([r["mean_abs_teacher_policy_collective_delta_n"] for r in rows if r["mean_abs_teacher_policy_collective_delta_n"] is not None])) if any(r["mean_abs_teacher_policy_collective_delta_n"] is not None for r in rows) else None,
            "mean_teacher_policy_torque_delta_nm": float(np.mean([r["mean_teacher_policy_torque_delta_nm"] for r in rows if r["mean_teacher_policy_torque_delta_nm"] is not None])) if any(r["mean_teacher_policy_torque_delta_nm"] is not None for r in rows) else None,
            "mean_torque_desaturation_fraction": float(np.mean([r["mean_torque_desaturation_fraction"] for r in rows if r["mean_torque_desaturation_fraction"] is not None])) if any(r["mean_torque_desaturation_fraction"] is not None for r in rows) else None,
            "mean_collective_desaturation_fraction": float(np.mean([r["mean_collective_desaturation_fraction"] for r in rows if r["mean_collective_desaturation_fraction"] is not None])) if any(r["mean_collective_desaturation_fraction"] is not None for r in rows) else None,
            "mean_motor_saturation_fraction": float(np.mean([r["mean_motor_saturation_fraction"] for r in rows if r["mean_motor_saturation_fraction"] is not None])) if any(r["mean_motor_saturation_fraction"] is not None for r in rows) else None,
            "initial_observation_hashes": sorted({r["initial_observation_sha256"] for r in rows}),
        }
    payload = {
        "diagnostic_type": "same_seed_teacher_ppo_action_and_trajectory_comparison",
        "vehicle_id": "smooth2-a2212-custom-prop",
        "stage": 5,
        "model": str(args.model),
        "model_timesteps": int(model.num_timesteps),
        "randomization_enabled": True,
        "same_seed_across_modes": True,
        "identical_initial_observation_expected_per_seed": True,
        "initial_observation_hashes_are_reported_per_episode": True,
        "modes": summary,
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
