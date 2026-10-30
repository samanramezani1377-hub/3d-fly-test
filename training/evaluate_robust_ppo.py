"""Evaluate a robust PPO checkpoint on held-out randomized smooth2 scenarios."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_physics import AdaptiveDroneEnv, load_vehicle_parameters


def summarize_episode(samples: list[dict], total_reward: float) -> dict:
    airborne = [item for item in samples if item["taken_off"]]
    stable = [item for item in airborne if item["stable"]]
    crashed = any(item["crashed"] for item in samples)
    took_off = bool(airborne)
    stable_fraction = len(stable) / len(airborne) if airborne else 0.0
    hover_success = bool(
        took_off and not crashed and len(stable) >= 250 and stable_fraction >= 0.80
    )
    return {
        "return": float(total_reward),
        "takeoff_success": took_off,
        "crashed": crashed,
        "hover_success": hover_success,
        "airborne_steps": len(airborne),
        "stable_hover_steps": len(stable),
        "stable_hover_fraction": float(stable_fraction),
        "mean_altitude_error_m": float(np.mean([x["altitude_error_m"] for x in airborne])) if airborne else None,
        "mean_horizontal_error_m": float(np.mean([x["horizontal_error_m"] for x in airborne])) if airborne else None,
    }


def evaluate(model, env, episodes: int, seed: int) -> dict:
    results = []
    for episode in range(episodes):
        obs, reset_info = env.reset(seed=seed + episode)
        samples = []
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            velocity = np.asarray(info.get("state_velocity_m_s", [np.inf] * 3))
            attitude = np.asarray(info.get("attitude_rad", [np.inf] * 3))
            rates = np.asarray(info.get("angular_velocity_rad_s", [np.inf] * 3))
            stable = bool(
                info["altitude_error_m"] <= 0.20
                and info["horizontal_error_m"] <= 0.25
                and np.linalg.norm(velocity) <= 0.25
                and np.max(np.abs(attitude[:2])) <= 0.20
                and np.linalg.norm(rates) <= 0.30
            )
            samples.append({
                "taken_off": bool(info["has_taken_off"]),
                "stable": stable,
                "crashed": bool(info["crashed"]),
                "altitude_error_m": float(info["altitude_error_m"]),
                "horizontal_error_m": float(info["horizontal_error_m"]),
            })
        results.append({
            "episode": episode,
            "seed": seed + episode,
            "initial_scenario": reset_info.get("scenario"),
            **summarize_episode(samples, total_reward),
        })

    return {
        "episodes": episodes,
        "seed_start": seed,
        "hover_success_rate": float(np.mean([x["hover_success"] for x in results])),
        "takeoff_success_rate": float(np.mean([x["takeoff_success"] for x in results])),
        "crash_rate": float(np.mean([x["crashed"] for x in results])),
        "mean_return": float(np.mean([x["return"] for x in results])),
        "mean_stable_hover_fraction": float(np.mean([x["stable_hover_fraction"] for x in results])),
        "worst_stable_hover_fraction": float(np.min([x["stable_hover_fraction"] for x in results])),
        "episodes_detail": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--software-only-synthetic-calibration", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    load_vehicle_parameters(allow_synthetic_calibration=args.software_only_synthetic_calibration)
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt to evaluate PPO") from exc

    model = PPO.load(str(args.model), device="auto")
    env = AdaptiveDroneEnv(
        allow_synthetic_calibration=args.software_only_synthetic_calibration,
        randomize=True,
    )
    try:
        report = evaluate(model, env, args.episodes, args.seed)
    finally:
        env.close()
    report.update({
        "model": str(args.model),
        "vehicle_model": "smooth2-parametric-v1",
        "physics_backend": "fast-6dof-research-plant",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_calibration": args.software_only_synthetic_calibration,
        "simulation_only": True,
        "physical_flight_eligible": False,
        "evaluation_split": "held-out episode seeds; same declared randomization bounds",
    })
    output = args.output or args.model.parent / "held-out-evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Hover success: {report['hover_success_rate']:.3f}; crash rate: {report['crash_rate']:.3f}")
    print(f"Saved evaluation report: {output}")


if __name__ == "__main__":
    main()
