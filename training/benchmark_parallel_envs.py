"""Benchmark PPO throughput with 1, 4, and 8 parallel simulator environments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=12_000)
    parser.add_argument("--output", type=Path, default=Path("training/runs/parallel-env-benchmark.json"))
    args = parser.parse_args()
    results = []
    for n_envs in (1, 4, 8):
        name = f"parallel-benchmark-envs-{n_envs}"
        command = [
            sys.executable, "training/train_ppo.py",
            "--timesteps", str(args.timesteps), "--seed", "2026",
            "--run-name", name, "--output-dir", "training/runs",
            "--device", "cpu", "--n-envs", str(n_envs),
            "--n-steps", "1024", "--batch-size", "64",
            "--save-freq", str(args.timesteps), "--eval-freq", str(args.timesteps),
            "--eval-episodes", "2",
        ]
        start = time.perf_counter()
        subprocess.run(command, cwd=ROOT, check=True)
        elapsed = time.perf_counter() - start
        results.append({
            "n_envs": n_envs,
            "timesteps": args.timesteps,
            "wall_seconds": round(elapsed, 3),
            "transitions_per_second": round(args.timesteps / elapsed, 3),
            "run_name": name,
        })
    results.sort(key=lambda row: row["transitions_per_second"], reverse=True)
    payload = {
        "benchmark": "PPO CPU throughput; same seed and transition budget",
        "results_ranked_fastest_first": results,
        "recommended_n_envs": results[0]["n_envs"],
        "note": "The production five-stage pipeline is configured for 4 environments per training process; this standalone benchmark is diagnostic and never changes that setting.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
