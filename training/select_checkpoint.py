"""Choose PPO checkpoints with a cheap screening pass and a robust finalist pass."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _metric(report: dict, key: str, default: float) -> float:
    value = report.get(key)
    return default if value is None else float(value)


def rank(report):
    return (
        _metric(report, "hover_success_rate", 0.0),
        -_metric(report, "crash_rate", 1.0),
        _metric(report, "takeoff_success_rate", 0.0),
        -_metric(report, "mean_absolute_final_altitude_error_m", float("inf")),
        _metric(report, "mean_hover_stable_fraction", 0.0),
        -_metric(report, "mean_absolute_vertical_speed_during_flight_mps", float("inf")),
        -_metric(report, "mean_absolute_final_altitude_error_m", float("inf")),
        _metric(report, "mean_return", float("-inf")),
    )


def evaluate(model_path: Path, report_path: Path, episodes: int, seed: int) -> dict:
    subprocess.run([
        sys.executable, "training/evaluate.py",
        "--model", str(model_path),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", str(report_path),
    ], check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10,
                        help="Full evaluation episodes per finalist")
    parser.add_argument("--screen-episodes", type=int, default=2,
                        help="Cheap screening episodes per checkpoint")
    parser.add_argument("--finalist-count", type=int, default=3,
                        help="Number of screened checkpoints to re-evaluate fully")
    parser.add_argument("--seed", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.episodes < 2 or args.screen_episodes <= 0 or args.finalist_count <= 0:
        parser.error("--episodes must be at least 2; --screen-episodes and --finalist-count must be positive")
    if args.screen_episodes > args.episodes:
        parser.error("--screen-episodes must not exceed --episodes")

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    candidates = set(run_dir.glob("checkpoints/*.zip"))
    candidates.update((run_dir / "best" / "best_model.zip", run_dir / "final_model.zip"))
    candidates = sorted(p.resolve() for p in candidates if p.is_file())
    if not candidates:
        raise SystemExit(f"No PPO checkpoints found under {run_dir}")

    report_dir = run_dir / "checkpoint-evaluations"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Screen every checkpoint cheaply, then spend the larger evaluation budget only
    # on the strongest candidates. Final ranking always uses the full episode count.
    screened = []
    for index, model_path in enumerate(candidates):
        report_path = report_dir / f"screen-candidate-{index:03d}.json"
        report = evaluate(model_path, report_path, args.screen_episodes, args.seed)
        screened.append({
            "model": str(model_path),
            "screen_evaluation": str(report_path),
            "screen_episodes": args.screen_episodes,
            "screen_rank": list(rank(report)),
            "screen_metrics": report,
        })

    finalists = sorted(
        screened,
        key=lambda item: tuple(item["screen_rank"]),
        reverse=True,
    )[:min(args.finalist_count, len(screened))]

    evaluated = []
    for index, candidate in enumerate(finalists):
        model_path = Path(candidate["model"])
        reports = []
        seed_reports = []
        episode_budgets = (args.episodes // 2, args.episodes - args.episodes // 2)
        evaluation_seeds = (args.seed, args.seed + 10_000)
        for seed_offset, episode_budget in zip((0, 10_000), episode_budgets):
            seed_report_path = report_dir / f"finalist-{index:03d}-seed-{args.seed + seed_offset}.json"
            reports.append(evaluate(
                model_path, seed_report_path, episode_budget, args.seed + seed_offset
            ))
            seed_reports.append(str(seed_report_path))

        keys = [
            "hover_success_rate", "takeoff_success_rate", "crash_rate",
            "mean_hover_stable_fraction", "mean_absolute_vertical_speed_during_flight_mps",
            "mean_absolute_final_altitude_error_m", "mean_return",
        ]
        aggregate = {}
        for key in keys:
            usable = [report for report in reports if report.get(key) is not None]
            if not usable:
                aggregate[key] = None
                continue
            weights = [
                int(report.get("flight_episodes", 0))
                if key == "mean_absolute_vertical_speed_during_flight_mps"
                else int(report["episodes"])
                for report in usable
            ]
            aggregate[key] = sum(
                float(report[key]) * weight for report, weight in zip(usable, weights)
            ) / sum(weights)
        evaluated.append({
            "model": str(model_path),
            "evaluation": seed_reports,
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_episode_budgets": list(episode_budgets),
            **aggregate,
        })

    winner = max(evaluated, key=rank)
    destination = run_dir / "best" / "best_model.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(winner["model"]).resolve()
    if source != destination.resolve():
        shutil.copy2(source, destination)
    selection = {
        "selection_metric": [
            "hover_success_rate descending",
            "crash_rate ascending",
            "takeoff_success_rate descending",
            "mean_absolute_final_altitude_error_m ascending",
            "mean_hover_stable_fraction descending",
            "mean_absolute_vertical_speed_during_flight_mps ascending",
            "mean_return descending (tie-breaker only)",
        ],
        "screening": {
            "episodes_per_candidate": args.screen_episodes,
            "candidate_count": len(screened),
            "finalist_count": len(finalists),
            "full_evaluation_episodes_per_finalist": args.episodes,
            "evaluation_seeds": [args.seed, args.seed + 10_000],
            "candidates": screened,
        },
        "candidate_count": len(candidates),
        "finalists_evaluated": len(evaluated),
        "candidates": evaluated,
        "selected": winner,
        "selected_checkpoint_path": str(destination),
        "deployment_eligible": winner["crash_rate"] == 0.0 and winner["hover_success_rate"] == 1.0,
        "simulation_only": True,
    }
    (run_dir / "checkpoint-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "candidate_count": len(candidates),
        "finalists_evaluated": len(evaluated),
        "selected": winner,
        "destination": str(destination),
        "deployment_eligible": selection["deployment_eligible"],
    }, indent=2))


if __name__ == "__main__":
    main()
