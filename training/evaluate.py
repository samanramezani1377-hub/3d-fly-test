"""Evaluate a saved PPO model against the native DroneCore Gymnasium environment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def summarize_flight_metrics(step_infos: list[dict]) -> dict:
    """Summarize only samples recorded after the environment latched takeoff."""
    flight = [info for info in step_infos if bool(info.get("has_taken_off", False))]
    flight_steps = len(flight)
    if not flight:
        return {
            "flight_steps": 0,
            "hover_stable_steps": 0,
            "hover_stable_fraction": 0.0,
            "mean_absolute_altitude_error_m": None,
            "mean_horizontal_error_m": None,
            "mean_absolute_vertical_speed_mps": None,
        }

    return {
        "flight_steps": flight_steps,
        "hover_stable_steps": sum(bool(info.get("hover_stable", False)) for info in flight),
        "hover_stable_fraction": (
            sum(bool(info.get("hover_stable", False)) for info in flight) / flight_steps
        ),
        "mean_absolute_altitude_error_m": float(np.mean([
            abs(float(info["altitude_error_m"])) for info in flight
        ])),
        "mean_horizontal_error_m": float(np.mean([
            float(info["horizontal_error_m"]) for info in flight
        ])),
        "mean_absolute_vertical_speed_mps": float(np.mean([
            abs(float(info["velocity_xyz_mps"][1])) for info in flight
        ])),
    }


def is_hover_success(
    flight_metrics: dict,
    *,
    crashed: bool,
    takeoff_seen: bool,
    min_hover_stable_steps: int,
    stable_fraction_threshold: float = 0.8,
) -> bool:
    """Require takeoff, survival, a minimum stable duration, and a stable flight fraction."""
    return bool(
        takeoff_seen
        and not crashed
        and flight_metrics["flight_steps"] > 0
        and flight_metrics["hover_stable_steps"] >= min_hover_stable_steps
        and flight_metrics["hover_stable_fraction"] >= stable_fraction_threshold
    )


def _mean_available(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=3.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--min-hover-stable-seconds", type=float, default=5.0,
        help="Minimum accumulated stable-hover time required for an episode to count as a hover success",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.episodes <= 0 or args.max_steps <= 0 or args.min_hover_stable_seconds <= 0:
        raise SystemExit("--episodes, --max-steps, and --min-hover-stable-seconds must be positive")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    try:
        from stable_baselines3 import PPO
        from native_env import NativeSkySimEnv, PolicyObservationWrapper
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt and build the native core.") from exc

    target = (args.target_x, args.target_y, args.target_z)
    env = PolicyObservationWrapper(NativeSkySimEnv(target=target, max_steps=args.max_steps))
    simulation_dt = float(env.unwrapped.dt)
    min_hover_stable_steps = max(1, math.ceil(args.min_hover_stable_seconds / simulation_dt))
    model = PPO.load(str(args.model), device="auto")
    episode_results = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            steps = 0
            takeoff_seen = False
            step_infos = []
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                step_infos.append(info)
                takeoff_seen = takeoff_seen or bool(info.get("has_taken_off", False))

            final_position = info.get("position_xyz_m")
            final_velocity = info.get("velocity_xyz_mps")
            final_altitude_error = (
                float(final_position[1] - args.target_y) if final_position is not None else None
            )
            flight_metrics = summarize_flight_metrics(step_infos)
            crashed = bool(info.get("crashed", False))
            episode_results.append({
                "episode": episode,
                "seed": args.seed + episode,
                "return": total_reward,
                "steps": steps,
                "flight_steps": flight_metrics["flight_steps"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "crashed": crashed,
                "takeoff_success": bool(takeoff_seen),
                "crash_reason": info.get("crash_reason"),
                "hover_stable_steps": flight_metrics["hover_stable_steps"],
                "hover_stable_fraction": flight_metrics["hover_stable_fraction"],
                "min_hover_stable_steps_required": min_hover_stable_steps,
                "episode_hover_success": is_hover_success(
                    flight_metrics,
                    crashed=crashed,
                    takeoff_seen=takeoff_seen,
                    min_hover_stable_steps=min_hover_stable_steps,
                ),
                "mean_absolute_altitude_error_m": flight_metrics["mean_absolute_altitude_error_m"],
                "mean_horizontal_error_m": flight_metrics["mean_horizontal_error_m"],
                "mean_absolute_vertical_speed_mps": flight_metrics["mean_absolute_vertical_speed_mps"],
                "final_position_xyz_m": final_position,
                "final_velocity_xyz_mps": final_velocity,
                "final_altitude_error_m": final_altitude_error,
            })
    finally:
        env.close()

    returns = [row["return"] for row in episode_results]
    altitude_errors = [
        abs(row["final_altitude_error_m"])
        for row in episode_results if row["final_altitude_error_m"] is not None
    ]
    vertical_speeds = [
        abs(row["final_velocity_xyz_mps"][1])
        for row in episode_results if row["final_velocity_xyz_mps"] is not None
    ]
    report = {
        "backend": "native-drone-core",
        "browser_core_same_cpp_sources": True,
        "backend_warning": (
            "Native and WASM builds use the same pinned C++ sources and config patch, "
            "but numerical parity still requires regression tests; this is not sim-to-real validation."
        ),
        "model": str(args.model),
        "episodes": args.episodes,
        "target_xyz_m": list(target),
        "simulation_dt_seconds": simulation_dt,
        "min_hover_stable_seconds": args.min_hover_stable_seconds,
        "min_hover_stable_steps_required": min_hover_stable_steps,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "crash_rate": float(np.mean([row["crashed"] for row in episode_results])),
        "takeoff_success_rate": float(np.mean([row["takeoff_success"] for row in episode_results])),
        "hover_success_rate": float(np.mean([row["episode_hover_success"] for row in episode_results])),
        "mean_hover_stable_fraction": float(np.mean([row["hover_stable_fraction"] for row in episode_results])),
        "mean_steps": float(np.mean([row["steps"] for row in episode_results])),
        "mean_flight_steps": float(np.mean([row["flight_steps"] for row in episode_results])),
        "flight_episodes": sum(row["flight_steps"] > 0 for row in episode_results),
        "mean_absolute_altitude_error_during_flight_m": _mean_available(
            episode_results, "mean_absolute_altitude_error_m"
        ),
        "mean_horizontal_error_during_flight_m": _mean_available(
            episode_results, "mean_horizontal_error_m"
        ),
        "mean_absolute_vertical_speed_during_flight_mps": _mean_available(
            episode_results, "mean_absolute_vertical_speed_mps"
        ),
        "mean_absolute_final_altitude_error_m": float(np.mean(altitude_errors)) if altitude_errors else None,
        "mean_absolute_final_vertical_speed_mps": float(np.mean(vertical_speeds)) if vertical_speeds else None,
        "episode_results": episode_results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
