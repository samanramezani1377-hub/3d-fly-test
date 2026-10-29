"""Run five chained PPO training stages, retaining the best evaluated checkpoint."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs"
STEPS = 100_000
# Periodic training diagnostics are cheaper; checkpoint finalists and each stage's
# selected model still receive a larger evaluation budget.
EVAL_EPISODES = 5
SELECTION_EPISODES = 10
SCREENING_EPISODES = 2
FINALIST_COUNT = 3
PARALLEL_ENVS = 4
CHECKPOINT_FREQ = 20_000
EVAL_FREQ = 20_000


def run(args, log_path: Path | None = None) -> None:
    print("+", " ".join(map(str, args)), flush=True)
    if log_path:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(args, cwd=ROOT, check=True)


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict, key: str, default: float) -> float:
    value = report.get(key)
    return default if value is None else float(value)


def rank(report: dict) -> tuple:
    # Missing flight metrics are worst-case, not a reason to crash selection.
    # First maximize sustained hover, then safety, then takeoff and precision.
    return (
        _metric(report, "hover_success_rate", 0.0),
        -_metric(report, "crash_rate", 1.0),
        _metric(report, "takeoff_success_rate", 0.0),
        -_metric(report, "mean_absolute_final_altitude_error_m", float("inf")),
        _metric(report, "mean_hover_stable_fraction", 0.0),
        -_metric(report, "mean_absolute_vertical_speed_during_flight_mps", float("inf")),
        _metric(report, "mean_return", float("-inf")),
    )


def evaluate(model: Path, output: Path, episodes: int = 20, seed: int = 2000) -> dict:
    run([
        sys.executable, "training/evaluate.py", "--model", str(model),
        "--episodes", str(episodes), "--seed", str(seed), "--output", str(output),
    ])
    return read_report(output)


def train(run_name: str, seed: int, initial_model: Path | None = None, n_envs: int = PARALLEL_ENVS) -> Path:
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if initial_model is not None:
        # Include the incoming best checkpoint in candidate selection. This guarantees
        # a stage cannot silently replace it with a worse checkpoint.
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(initial_model, checkpoint_dir / "previous-best.zip")
    args = [
        sys.executable, "training/train_ppo.py",
        "--timesteps", str(STEPS), "--seed", str(seed),
        "--run-name", run_name, "--output-dir", "training/runs",
        "--device", "cpu", "--n-envs", str(n_envs), "--n-steps", "1024",
        "--batch-size", "64", "--save-freq", str(CHECKPOINT_FREQ),
        "--eval-freq", str(EVAL_FREQ), "--eval-episodes", str(EVAL_EPISODES),
    ]
    if initial_model is not None:
        args.extend(["--load-model", str(initial_model)])
    run(args, run_dir / "training.log")
    run([
        sys.executable, "training/select_checkpoint.py",
        "--run-dir", str(run_dir), "--episodes", str(SELECTION_EPISODES),
        "--screen-episodes", str(SCREENING_EPISODES),
        "--finalist-count", str(FINALIST_COUNT), "--seed", "3000",
    ], run_dir / "checkpoint-selection.log")
    selected = run_dir / "best" / "best_model.zip"
    if not selected.is_file():
        raise RuntimeError(f"Stage {run_name} produced no selected checkpoint")
    return selected


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    stage_reports: list[dict] = []

    # Stage 1 explores two independent seeds; choose the better evaluated checkpoint.
    def train_and_evaluate_seed(seed: int) -> dict:
        name = f"ppo-stage1-seed{seed}"
        model = train(name, seed, n_envs=4)
        report_path = RUNS / name / "matched-seed-evaluation.json"
        report = evaluate(model, report_path)
        return {"stage": 1, "seed": seed, "model": str(model),
                "evaluation": str(report_path), **report}

    # Two seeds run concurrently; each uses 4 environments (8 workers total).
    with ThreadPoolExecutor(max_workers=2) as pool:
        stage1_candidates = list(pool.map(train_and_evaluate_seed, (42, 40)))
    winner = max(stage1_candidates, key=rank)
    current = Path(winner["model"]).resolve()
    stage_reports.append({
        "stage": 1, "selected_seed": winner["seed"],
        "selected_model": str(current),
        "selection_metric": [
            "hover_success_rate descending", "crash_rate ascending",
            "takeoff_success_rate descending", "mean_absolute_final_altitude_error_m ascending",
            "mean_hover_stable_fraction descending",
            "mean_absolute_vertical_speed_during_flight_mps ascending",
            "mean_return descending (tie-breaker)",
        ],
        "candidates": stage1_candidates,
        "selected_evaluation": {k: v for k, v in winner.items() if k not in ("model", "evaluation")},
    })
    shutil.copy2(current, RUNS / "stage1-selected-model.zip")

    # Stages 2-5 each start from the previous stage's selected best checkpoint.
    for stage in range(2, 6):
        name = f"ppo-stage{stage}-continued"
        selected = train(name, 43 + stage, current)
        report_path = RUNS / name / "matched-seed-evaluation.json"
        report = evaluate(selected, report_path)
        current = selected.resolve()
        stage_reports.append({
            "stage": stage, "initial_model": str(stage_reports[-1]["selected_model"]),
            "selected_model": str(current), "evaluation": str(report_path),
            "selected_evaluation": report,
            "selected_metrics_rank": list(rank(report)),
        })
        shutil.copy2(current, RUNS / f"stage{stage}-selected-model.zip")

    final_report = read_report(RUNS / "ppo-stage5-continued" / "matched-seed-evaluation.json")
    deployment_eligible = (
        float(final_report.get("crash_rate", 1.0)) == 0.0
        and float(final_report.get("hover_success_rate", 0.0)) == 1.0
    )
    manifest = {
        "training_stages": 5,
        "environment_steps_per_training_run": STEPS,
        "parallel_environments_default": PARALLEL_ENVS,
        "stage1_parallel_seed_workers": 2,
        "stage1_parallel_environments_per_seed": 4,
        "periodic_evaluation_episodes": EVAL_EPISODES,
        "checkpoint_frequency_transitions": CHECKPOINT_FREQ,
        "periodic_evaluation_frequency_transitions": EVAL_FREQ,
        "checkpoint_selection_screen_episodes": SCREENING_EPISODES,
        "checkpoint_selection_finalists": FINALIST_COUNT,
        "checkpoint_selection_final_episodes": SELECTION_EPISODES,
        "stage1_independent_seeds": [42, 40],
        "stages_2_to_5_continue_from_previous_selected_checkpoint": True,
        "selection_metric": [
            "hover_success_rate descending", "crash_rate ascending",
            "takeoff_success_rate descending", "mean_absolute_final_altitude_error_m ascending",
            "mean_hover_stable_fraction descending",
            "mean_absolute_vertical_speed_during_flight_mps ascending",
            "mean_return descending (tie-breaker)",
        ],
        "stages": stage_reports,
        "final_model": str(current),
        "final_evaluation": final_report,
        "deployment_eligible": deployment_eligible,
        "simulation_only": True,
        "warning": "Not validated for real flight; validate CPU/TPU physics parity and the physical aircraft model before any real flight.",
    }
    (RUNS / "five-stage-training-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(current, ROOT / "final-model.zip")
    print(json.dumps({
        "training_stages": 5, "final_model": str(ROOT / "final-model.zip"),
        "deployment_eligible": deployment_eligible,
        "final_evaluation": final_report,
    }, indent=2))


if __name__ == "__main__":
    main()
