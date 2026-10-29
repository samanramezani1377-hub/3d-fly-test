"""Choose PPO checkpoints by hover safety/stability, not episode return alone."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def rank(report):
    return (
        float(report["hover_success_rate"]),
        float(report.get("takeoff_success_rate", 0.0)),
        -float(report["crash_rate"]),
        -float(report["mean_absolute_final_altitude_error_m"]),
        float(report["mean_hover_stable_fraction"]),
        -float(report["mean_absolute_vertical_speed_during_flight_mps"]),
        -float(report["mean_absolute_final_altitude_error_m"]),
        float(report["mean_return"]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
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
    evaluated = []
    for index, model_path in enumerate(candidates):
        report_path = report_dir / f"candidate-{index:03d}.json"
        subprocess.run([
            sys.executable, "training/evaluate.py",
            "--model", str(model_path),
            "--episodes", str(args.episodes),
            "--seed", str(args.seed),
            "--output", str(report_path),
        ], check=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evaluated.append({
            "model": str(model_path),
            "evaluation": str(report_path),
            "hover_success_rate": float(report["hover_success_rate"]),
            "takeoff_success_rate": float(report.get("takeoff_success_rate", 0.0)),
            "crash_rate": float(report["crash_rate"]),
            "mean_hover_stable_fraction": float(report["mean_hover_stable_fraction"]),
            "mean_absolute_vertical_speed_during_flight_mps": float(report["mean_absolute_vertical_speed_during_flight_mps"]),
            "mean_absolute_final_altitude_error_m": float(report["mean_absolute_final_altitude_error_m"]),
            "mean_return": float(report["mean_return"]),
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
            "takeoff_success_rate descending",
            "crash_rate ascending",
            "mean_absolute_final_altitude_error_m ascending",
            "mean_hover_stable_fraction descending",
            "mean_absolute_vertical_speed_during_flight_mps ascending",
            "mean_absolute_final_altitude_error_m ascending",
            "mean_return descending (tie-breaker only)",
        ],
        "candidate_count": len(evaluated),
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
        "candidate_count": len(evaluated),
        "selected": winner,
        "destination": str(destination),
        "deployment_eligible": selection["deployment_eligible"],
    }, indent=2))


if __name__ == "__main__":
    main()
