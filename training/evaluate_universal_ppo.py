"""Evaluate takeoff, sustained hover, crashes and mission completion per vehicle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fast_physics.vehicle_family import DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles

STABLE_ALTITUDE_ERROR_M = 0.15
STABLE_HORIZONTAL_ERROR_M = 0.20
STABLE_SPEED_M_S = 0.35
STABLE_TILT_RAD = 0.25
STABLE_ANGULAR_RATE_RAD_S = 0.5
STABLE_HOVER_SECONDS = 5.0


def stable_hover(info):
    velocity = info.get("state_velocity_m_s", [float("inf")] * 3)
    attitude = info.get("attitude_rad", [float("inf")] * 3)
    angular_rate = info.get("angular_velocity_rad_s", [float("inf")] * 3)
    try:
        speed_sq = sum(float(v) ** 2 for v in velocity)
        tilt = max(abs(float(attitude[0])), abs(float(attitude[1])))
        angular_rate_sq = sum(float(v) ** 2 for v in angular_rate)
        return bool(
            info.get("has_taken_off", False)
            and not info.get("crashed", False)
            and float(info.get("altitude_error_m", float("inf"))) <= STABLE_ALTITUDE_ERROR_M
            and float(info.get("horizontal_error_m", float("inf"))) <= STABLE_HORIZONTAL_ERROR_M
            and speed_sq <= STABLE_SPEED_M_S ** 2
            and tilt <= STABLE_TILT_RAD
            and angular_rate_sq <= STABLE_ANGULAR_RATE_RAD_S ** 2
            and info.get("state_finite", True)
        )
    except (TypeError, ValueError, IndexError):
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--vehicle-family", type=Path, default=DEFAULT_VEHICLE_FAMILY)
    p.add_argument("--vehicle-id", action="append", default=None)
    p.add_argument("--randomization-config", type=Path, default=Path(__file__).resolve().parents[1] / "vehicle_models/family/domain_randomization.json")
    p.add_argument("--curriculum-stage", type=int, choices=(1, 2, 3, 4, 5), default=5)
    p.add_argument("--held-out", action="store_true", help="Use independent evaluation bounds without applying a training stage.")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=90000)
    p.add_argument("--stable-hover-seconds", type=float, default=STABLE_HOVER_SECONDS)
    p.add_argument("--output", type=Path, default=Path("training/runs/universal-evaluation.json"))
    args = p.parse_args(argv)
    if args.episodes <= 0 or args.stable_hover_seconds <= 0:
        raise SystemExit("--episodes and --stable-hover-seconds must be positive")
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt to evaluate PPO") from exc

    profiles = load_vehicle_profiles(args.vehicle_family)
    wanted = set(args.vehicle_id or [profile["vehicle_id"] for profile in profiles])
    selected = [profile for profile in profiles if profile["vehicle_id"] in wanted]
    if wanted - {profile["vehicle_id"] for profile in selected}:
        raise SystemExit(f"unknown vehicle IDs: {sorted(wanted - {profile['vehicle_id'] for profile in selected})}")
    model = PPO.load(str(args.model), device="auto")
    reports, episode_index = [], 0

    for profile in selected:
        env = VehicleFamilyEnv(
            vehicle_family_path=args.vehicle_family,
            randomization_path=args.randomization_config,
            curriculum_stage=None if args.held_out else args.curriculum_stage,
            vehicle_id=profile["vehicle_id"], randomize=True,
        )
        episodes = []
        required_stable_steps = max(1, int(round(args.stable_hover_seconds / env.inner.dt)))
        try:
            for _ in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode_index)
                episode_seed = args.seed + episode_index
                episode_index += 1
                rewards, crashed = [], False
                takeoff_seen = False
                max_altitude_m = 0.08
                max_stable_steps = stable_steps = 0
                crash_reasons = set()
                terminated = truncated = False
                info = {}
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(float(reward))
                    crashed = crashed or bool(info.get("crashed", False))
                    takeoff_seen = takeoff_seen or bool(info.get("has_taken_off", False))
                    max_altitude_m = max(max_altitude_m, float(info.get("altitude_m", 0.08)))
                    crash_reasons.update(info.get("crash_reasons", []))
                    if stable_hover(info):
                        stable_steps += 1
                        max_stable_steps = max(max_stable_steps, stable_steps)
                    else:
                        stable_steps = 0
                    if terminated or truncated:
                        break
                hover_success = bool(not crashed and max_stable_steps >= required_stable_steps)
                episodes.append({
                    "seed": episode_seed,
                    "vehicle_id": profile["vehicle_id"],
                    "hover_success": hover_success,
                    "stable_hover_seconds_max": max_stable_steps * env.inner.dt,
                    "crashed": crashed,
                    "takeoff_success": takeoff_seen,
                    "takeoff_failed": bool(not takeoff_seen and truncated),
                    "steps": len(rewards),
                    "max_altitude_m": max_altitude_m,
                    "final_altitude_m": info.get("altitude_m"),
                    "crash_reasons": sorted(crash_reasons),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "survived_to_time_limit": bool(truncated and not crashed),
                    "mission_completed": hover_success,
                    "mean_reward": sum(rewards) / max(len(rewards), 1),
                    "final_altitude_error_m": info.get("altitude_error_m"),
                    "final_horizontal_error_m": info.get("horizontal_error_m"),
                    "online_estimator": info.get("online_estimator"),
                })
        finally:
            env.close()

        count = len(episodes)
        reports.append({
            "vehicle_id": profile["vehicle_id"],
            "calibration_status": profile["calibration_status"],
            "episodes": episodes,
            "hover_success_rate": sum(e["hover_success"] for e in episodes) / count,
            "crash_rate": sum(e["crashed"] for e in episodes) / count,
            "takeoff_success_rate": sum(e["takeoff_success"] for e in episodes) / count,
            "takeoff_failure_rate": sum(e["takeoff_failed"] for e in episodes) / count,
            "mean_max_altitude_m": sum(e["max_altitude_m"] for e in episodes) / count,
            "mean_stable_hover_seconds": sum(e["stable_hover_seconds_max"] for e in episodes) / count,
            "mission_completion_rate": sum(e["mission_completed"] for e in episodes) / count,
            "survived_to_time_limit_rate": sum(e["survived_to_time_limit"] for e in episodes) / count,
            # Kept as an explicit compatibility alias; it now means mission completion,
            # not merely surviving until the time limit.
            "completion_rate": sum(e["mission_completed"] for e in episodes) / count,
        })

    result = {
        "model": str(args.model),
        "vehicle_family": str(args.vehicle_family),
        "episodes_per_vehicle": args.episodes,
        "seed_start": args.seed,
        "stable_hover_definition": {
            "continuous_seconds_required": args.stable_hover_seconds,
            "altitude_error_m_max": STABLE_ALTITUDE_ERROR_M,
            "horizontal_error_m_max": STABLE_HORIZONTAL_ERROR_M,
            "speed_m_s_max": STABLE_SPEED_M_S,
            "absolute_roll_pitch_rad_max": STABLE_TILT_RAD,
            "angular_rate_rad_s_max": STABLE_ANGULAR_RATE_RAD_S,
            "requires_takeoff": True,
            "crash_allowed": False,
        },
        "profiles": reports,
        "evaluation_distribution": "independent_held_out" if args.held_out else f"curriculum_stage_{args.curriculum_stage}",
        "simulation_only": True,
        "physical_flight_eligible": False,
        "warning": "Simulation metrics are not real-world flight safety certification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
