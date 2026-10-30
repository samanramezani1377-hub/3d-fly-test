"""Run an oracle hover-reference benchmark on each provisional vehicle profile.

The controller uses the true randomized episode parameters. This is deliberately
an upper-bound diagnostic to isolate physics/control feasibility, not a policy
that can be deployed with unknown hardware parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fast_physics.hover_reference import HoverReferenceController
from fast_physics.vehicle_family import DEFAULT_RANDOMIZATION, DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles


def run_episode(env, seed: int, settle_seconds: float, success_altitude_error: float,
                success_horizontal_error: float, success_tilt_rad: float, start_at_target: bool = False):
    obs, reset_info = env.reset(seed=seed)
    del obs
    plant = env.inner.plant
    if start_at_target:
        plant.position[:] = env.inner.target
        plant.velocity[:] = 0.0
        plant.attitude[:] = 0.0
        plant.angular_velocity[:] = 0.0
        plant.motor_omega[:] = np.sqrt(
            plant.params.mass_kg * 9.80665
            / (4.0 * plant.params.thrust_coefficient_n_per_rad_s2
               * np.asarray(plant.params.motor_effectiveness, dtype=np.float64))
        )
    controller = HoverReferenceController(plant.params, target_position=env.inner.target, wind_m_s=plant.wind)
    required_steps = int(round(settle_seconds / env.inner.dt))
    stable_steps = 0
    crashed = False
    max_tilt = 0.0
    rewards = []
    final_info = {}
    while True:
        action = controller.act(plant.state())
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        final_info = info
        max_tilt = max(max_tilt, float(np.max(np.abs(plant.attitude[:2]))))
        stable = (
            abs(float(plant.position[2] - env.inner.target[2])) <= success_altitude_error
            and float(np.linalg.norm(plant.position[:2] - env.inner.target[:2])) <= success_horizontal_error
            and max_tilt <= success_tilt_rad
            and float(np.linalg.norm(plant.velocity)) <= 0.35
        )
        stable_steps = stable_steps + 1 if stable else 0
        crashed = crashed or bool(info.get("crashed", False))
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "vehicle_id": reset_info["vehicle_id"],
        "success": bool(not crashed and stable_steps >= required_steps),
        "crashed": crashed,
        "steps": len(rewards),
        "stable_steps_at_end": stable_steps,
        "max_abs_tilt_rad": max_tilt,
        "final_position_m": plant.position.astype(float).tolist(),
        "final_velocity_m_s": plant.velocity.astype(float).tolist(),
        "final_attitude_rad": plant.attitude.astype(float).tolist(),
        "final_altitude_error_m": float(final_info.get("altitude_error_m", 999.0)),
        "final_horizontal_error_m": float(final_info.get("horizontal_error_m", 999.0)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle-family", type=Path, default=DEFAULT_VEHICLE_FAMILY)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=71000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--start-at-target", action="store_true", help="Initialize at target altitude with nominal hover rotor speed; tests hover hold, not takeoff.")
    parser.add_argument("--wind-max", type=float, default=None, help="Override symmetric Stage 4 wind-component bound in m/s.")
    parser.add_argument("--output", type=Path, default=Path("training/runs/hover-reference-report.json"))
    args = parser.parse_args(argv)
    if min(args.episodes, args.max_steps) <= 0 or args.settle_seconds <= 0:
        raise SystemExit("episodes, max-steps and settle-seconds must be positive")

    profiles = load_vehicle_profiles(args.vehicle_family)
    randomization_path = DEFAULT_RANDOMIZATION
    if args.wind_max is not None:
        if args.wind_max < 0:
            raise SystemExit("--wind-max cannot be negative")
        config = json.loads(DEFAULT_RANDOMIZATION.read_text(encoding="utf-8"))
        for stage in config.get("curriculum", []):
            if int(stage["stage"]) == 4:
                stage["wind_m_s"] = [-args.wind_max, args.wind_max]
        randomization_path = args.output.with_name(args.output.stem + "-randomization.json")
        randomization_path.parent.mkdir(parents=True, exist_ok=True)
        randomization_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    reports = []
    for profile_index, profile in enumerate(profiles):
        env = VehicleFamilyEnv(
            vehicle_family_path=args.vehicle_family,
            vehicle_id=profile["vehicle_id"],
            max_steps=args.max_steps,
            randomization_path=randomization_path,
            randomize=args.randomize,
            curriculum_stage=4 if args.randomize else 1,
        )
        try:
            episodes = [
                run_episode(env, args.seed + profile_index * 10000 + episode,
                            args.settle_seconds, 0.15, 0.20, 0.25, args.start_at_target)
                for episode in range(args.episodes)
            ]
        finally:
            env.close()
        reports.append({
            "vehicle_id": profile["vehicle_id"],
            "calibration_status": profile["calibration_status"],
            "parameter_access": "privileged_episode_ground_truth_oracle",
            "success_rate": sum(e["success"] for e in episodes) / len(episodes),
            "crash_rate": sum(e["crashed"] for e in episodes) / len(episodes),
            "episodes": episodes,
        })
    result = {
        "controller": "cascaded_pid_with_wrench_allocation",
        "controller_class": "privileged_parameter_oracle_reference_only",
        "randomized": args.randomize,
        "start_at_target": args.start_at_target,
        "stage4_wind_component_max_m_s": args.wind_max if args.wind_max is not None else (2.0 if args.randomize else 0.0),
        "episodes_per_vehicle": args.episodes,
        "settle_seconds": args.settle_seconds,
        "success_criteria": {
            "continuous_stable_seconds": args.settle_seconds,
            "altitude_error_m_max": 0.15,
            "horizontal_error_m_max": 0.20,
            "max_tilt_rad": 0.25,
            "speed_m_s_max": 0.35,
            "crash_allowed": False,
        },
        "profiles": reports,
        "simulation_only": True,
        "physical_flight_eligible": False,
        "warning": "Oracle controller uses true episode parameters; this is not deployable and not a real-flight safety certification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
