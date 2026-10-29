"""Evaluate an experimental PPO model against the upstream SkySim server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    try:
        from stable_baselines3 import PPO
        from skysim import SkySimEnv, HoverTask
    except ImportError as exc:
        raise SystemExit("Install training requirements and upstream skysim[rl].") from exc

    env = SkySimEnv(
        port=args.port,
        task=HoverTask(target=(0.0, 3.0, 0.0)),
        max_steps=1500,
    )
    model = PPO.load(str(args.model), device="auto")
    episode_results = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            episode_results.append({
                "episode": episode,
                "seed": args.seed + episode,
                "return": total_reward,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
    finally:
        env.close()

    returns = [row["return"] for row in episode_results]
    report = {
        "backend": "upstream-skysim-godot-headless",
        "backend_warning": (
            "Not verified equivalent to browser WebAssembly physics; not validated "
            "for the physical aircraft."
        ),
        "model": str(args.model),
        "episodes": args.episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "episode_results": episode_results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
