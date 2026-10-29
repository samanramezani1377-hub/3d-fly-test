"""Evaluate a saved PPO model against the native DroneCore Gymnasium environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=3.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("--episodes and --max-steps must be positive")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    try:
        from stable_baselines3 import PPO
        from native_env import NativeSkySimEnv
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt and build the native core.") from exc

    target = (args.target_x, args.target_y, args.target_z)
    env = NativeSkySimEnv(target=target, max_steps=args.max_steps)
    model = PPO.load(str(args.model), device="auto")
    episode_results = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            final_position = info.get("position_xyz_m")
            final_velocity = info.get("velocity_xyz_mps")
            final_altitude_error = (
                float(final_position[1] - args.target_y) if final_position is not None else None
            )
            episode_results.append({
                "episode": episode,
                "seed": args.seed + episode,
                "return": total_reward,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "crashed": bool(info.get("crashed", False)),
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
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "crash_rate": float(np.mean([row["crashed"] for row in episode_results])),
        "mean_steps": float(np.mean([row["steps"] for row in episode_results])),
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
