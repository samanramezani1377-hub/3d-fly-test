"""Evaluate one hardware-conditioned checkpoint separately on each vehicle profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fast_physics.vehicle_family import DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--vehicle-family", type=Path, default=DEFAULT_VEHICLE_FAMILY)
    p.add_argument("--vehicle-id", action="append", default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=90000)
    p.add_argument("--output", type=Path, default=Path("training/runs/universal-evaluation.json"))
    args = p.parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt to evaluate PPO") from exc
    profiles = load_vehicle_profiles(args.vehicle_family)
    wanted = set(args.vehicle_id or [p["vehicle_id"] for p in profiles])
    selected = [p for p in profiles if p["vehicle_id"] in wanted]
    if wanted - {p["vehicle_id"] for p in selected}:
        raise SystemExit(f"unknown vehicle IDs: {sorted(wanted - {p['vehicle_id'] for p in selected})}")
    model = PPO.load(str(args.model), device="auto")
    reports, episode_index = [], 0
    for profile in selected:
        env = VehicleFamilyEnv(vehicle_family_path=args.vehicle_family, vehicle_id=profile["vehicle_id"], randomize=True)
        episodes = []
        try:
            for _ in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode_index)
                episode_seed = args.seed + episode_index
                episode_index += 1
                rewards, crashed = [], False
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(float(reward))
                    crashed = crashed or bool(info.get("crashed", False))
                    if terminated or truncated:
                        break
                episodes.append({
                    "seed": episode_seed, "vehicle_id": profile["vehicle_id"],
                    "hover_success": bool(info.get("has_taken_off", False) and not crashed
                        and info.get("altitude_error_m", 999) < 0.35
                        and info.get("horizontal_error_m", 999) < 0.5),
                    "crashed": crashed, "steps": len(rewards),
                    "mean_reward": sum(rewards) / max(len(rewards), 1),
                    "final_altitude_error_m": info.get("altitude_error_m"),
                    "final_horizontal_error_m": info.get("horizontal_error_m"),
                    "online_estimator": info.get("online_estimator"),
                })
        finally:
            env.close()
        reports.append({
            "vehicle_id": profile["vehicle_id"], "calibration_status": profile["calibration_status"],
            "episodes": episodes,
            "hover_success_rate": sum(e["hover_success"] for e in episodes) / len(episodes),
            "crash_rate": sum(e["crashed"] for e in episodes) / len(episodes),
        })
    result = {
        "model": str(args.model), "vehicle_family": str(args.vehicle_family),
        "episodes_per_vehicle": args.episodes, "seed_start": args.seed, "profiles": reports,
        "simulation_only": True, "physical_flight_eligible": False,
        "warning": "Held-out simulation evaluation is not real-world flight safety certification."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
