"""Compare PPO training speed and learned hover quality for 1, 4, and 8 envs.

Each configuration is trained from scratch with the same seeds, transition budget,
PPO rollout size, and hyperparameters. Final policies are evaluated on identical
episode seeds. This experiment is standalone and never changes the production
five-stage training configuration.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (1, 4, 8)
ROLLOUT_SIZE = 1024


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def aggregate(reports: list[dict]) -> dict:
    keys = (
        "mean_return",
        "crash_rate",
        "takeoff_success_rate",
        "hover_success_rate",
        "mean_hover_stable_fraction",
        "mean_absolute_altitude_error_during_flight_m",
        "mean_horizontal_error_during_flight_m",
        "mean_absolute_vertical_speed_during_flight_mps",
        "mean_absolute_final_altitude_error_m",
    )
    result = {"seed_runs": len(reports), "evaluation_episodes_per_seed": reports[0]["episodes"]}
    for key in keys:
        values = [float(r[key]) for r in reports if r.get(key) is not None and math.isfinite(float(r[key]))]
        result[key] = mean(values) if values else None
    return result


def quality_gate(candidate: dict, baseline: dict) -> dict:
    # Tolerances are explicit practical non-inferiority margins, not statistical proof.
    checks = {
        "hover_success_within_5pp": candidate["hover_success_rate"] >= baseline["hover_success_rate"] - 0.05,
        "crash_rate_within_5pp": candidate["crash_rate"] <= baseline["crash_rate"] + 0.05,
        "takeoff_success_within_5pp": candidate["takeoff_success_rate"] >= baseline["takeoff_success_rate"] - 0.05,
        "altitude_error_within_5pct_plus_5cm": candidate["mean_absolute_altitude_error_during_flight_m"] <= baseline["mean_absolute_altitude_error_during_flight_m"] * 1.05 + 0.05,
        "vertical_speed_within_5pct_plus_5cm_s": candidate["mean_absolute_vertical_speed_during_flight_mps"] <= baseline["mean_absolute_vertical_speed_during_flight_mps"] * 1.05 + 0.05,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--output", type=Path, default=Path("training/runs/parallel-env-quality-experiment.json"))
    args = parser.parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    if args.timesteps < ROLLOUT_SIZE or args.eval_episodes <= 0 or not seeds:
        raise SystemExit("timesteps must be >= 1024, eval-episodes positive, and at least one seed required")

    results = {}
    for n_envs in CONFIGS:
        per_seed = []
        n_steps = ROLLOUT_SIZE // n_envs
        for seed in seeds:
            run_name = f"parallel-quality-envs{n_envs}-seed{seed}"
            run_dir = Path("training/runs/parallel-env-quality") / run_name
            train_command = [
                sys.executable, "training/train_ppo.py",
                "--timesteps", str(args.timesteps),
                "--seed", str(seed),
                "--run-name", run_name,
                "--output-dir", str(run_dir.parent),
                "--device", "cpu",
                "--n-envs", str(n_envs),
                "--n-steps", str(n_steps),
                "--batch-size", "64",
                "--save-freq", str(args.timesteps),
                "--eval-freq", str(args.timesteps),
                "--eval-episodes", "5",
            ]
            started = time.perf_counter()
            run(train_command)
            training_seconds = time.perf_counter() - started
            model_path = run_dir / "final_model.zip"
            report_path = run_dir / "matched-seed-evaluation.json"
            evaluation_seed = 9000
            run([
                sys.executable, "training/evaluate.py",
                "--model", str(model_path),
                "--episodes", str(args.eval_episodes),
                "--seed", str(evaluation_seed),
                "--output", str(report_path),
            ])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            per_seed.append({
                "training_seed": seed,
                "evaluation_seed_start": evaluation_seed,
                "n_envs": n_envs,
                "n_steps_per_env": n_steps,
                "rollout_size_per_update": n_envs * n_steps,
                "requested_timesteps": args.timesteps,
                "training_wall_seconds": round(training_seconds, 3),
                "transitions_per_second": round(args.timesteps / training_seconds, 3),
                "model": str(model_path),
                "evaluation_report": str(report_path),
                "evaluation": report,
            })
        results[str(n_envs)] = {
            "n_envs": n_envs,
            "runs": per_seed,
            "mean_training_wall_seconds": mean([r["training_wall_seconds"] for r in per_seed]),
            "mean_transitions_per_second": mean([r["transitions_per_second"] for r in per_seed]),
            "quality": aggregate([r["evaluation"] for r in per_seed]),
        }

    baseline = results["1"]
    comparisons = {}
    for n_envs in ("4", "8"):
        quality = quality_gate(results[n_envs]["quality"], baseline["quality"])
        faster = results[n_envs]["mean_transitions_per_second"] > baseline["mean_transitions_per_second"]
        comparisons[n_envs] = {
            "quality_vs_1_env": quality,
            "faster_than_1_env": faster,
            "eligible_for_next_run_consideration": bool(quality["passed"] and faster),
        }

    payload = {
        "experiment": "standalone PPO parallel-environment speed and quality comparison",
        "production_training_modified": False,
        "configs": list(CONFIGS),
        "seeds": seeds,
        "requested_timesteps_per_seed_and_config": args.timesteps,
        "evaluation_episodes_per_seed": args.eval_episodes,
        "evaluation_seed_range": [9000, 9000 + args.eval_episodes - 1],
        "matched_rollout_size_per_update": ROLLOUT_SIZE,
        "same_training_and_evaluation_seed_sets_across_configs": True,
        "quality_gate_note": "Practical non-inferiority thresholds only; two training seeds are preliminary evidence, not statistical proof. Inspect per-seed reports before changing production settings.",
        "results": results,
        "comparisons_to_1_env": comparisons,
        "recommendation": "Consider 4/8 envs for a future run only when faster_than_1_env and all quality checks pass; no automatic setting change is made.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n===== PARALLEL ENVIRONMENT SPEED + QUALITY REPORT =====")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
