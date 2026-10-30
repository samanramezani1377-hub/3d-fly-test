"""Five sequential PPO training stages with one final held-out release evaluation."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = 5


def load_profile_ids(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return [p["vehicle_id"] for p in data["profiles"]]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def score(report):
    profiles = report["profiles"]
    return (
        min(x["hover_success_rate"] for x in profiles),
        -max(x["crash_rate"] for x in profiles),
        min(x["takeoff_success_rate"] for x in profiles),
        sum(x["mean_stable_hover_seconds"] for x in profiles) / len(profiles),
        -sum(x.get("mean_motor_saturation_fraction", 1.0) for x in profiles) / len(profiles),
        sum(x.get("mean_return", 0.0) for x in profiles) / len(profiles),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps-per-stage", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--evaluation-episodes", type=int, default=20,
                   help="Final held-out episodes per vehicle profile; only run after stage 5.")
    p.add_argument("--checkpoint-eval-episodes", type=int, default=2,
                   help="Episodes per vehicle profile at each training checkpoint.")
    p.add_argument("--eval-frequency", type=int, default=100_000,
                   help="Training transitions between checkpoint-selection evaluations.")
    p.add_argument("--checkpoint-frequency", type=int, default=20_000,
                   help="Training transitions between resumable latest checkpoints.")
    p.add_argument("--min-final-success-rate", type=float, default=0.50)
    p.add_argument("--max-final-crash-rate", type=float, default=0.25)
    p.add_argument("--min-final-takeoff-rate", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-prefix", default="universal-quad-v5")
    p.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--vehicle-family", type=Path,
                   default=ROOT / "vehicle_models/family/vehicle_profiles.json")
    a = p.parse_args(argv)

    counts = (a.timesteps_per_stage, a.n_envs, a.evaluation_episodes,
              a.checkpoint_eval_episodes, a.eval_frequency, a.checkpoint_frequency)
    if min(counts) <= 0:
        raise SystemExit("Training, checkpoint and evaluation counts must be positive")
    if a.eval_frequency > a.timesteps_per_stage:
        raise SystemExit("--eval-frequency cannot exceed --timesteps-per-stage")
    if a.timesteps_per_stage % (a.n_envs * 500):
        raise SystemExit(
            "Each stage budget must be divisible by n_envs * 500 so PPO stages "
            "finish on complete rollouts."
        )
    if not 0 <= a.min_final_success_rate <= 1 or not 0 <= a.max_final_crash_rate <= 1 or not 0 <= a.min_final_takeoff_rate <= 1:
        raise SystemExit("Acceptance thresholds must be between zero and one")

    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{a.run_prefix}-curriculum-manifest.json"
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "timesteps_per_stage": a.timesteps_per_stage,
        "n_envs": a.n_envs,
        "checkpoint_eval_episodes": a.checkpoint_eval_episodes,
        "eval_frequency": a.eval_frequency,
        "checkpoint_frequency": a.checkpoint_frequency,
        "ppo_n_steps": 500,
        "ppo_batch_size": 250,
        "final_evaluation_episodes": a.evaluation_episodes,
        "min_final_success_rate": a.min_final_success_rate,
        "max_final_crash_rate": a.max_final_crash_rate,
        "min_final_takeoff_rate": a.min_final_takeoff_rate,
        "vehicle_family": str(a.vehicle_family),
        "seed": a.seed,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != config:
            raise RuntimeError(
                "Existing curriculum manifest has different settings. "
                "Refusing an unsafe mixed-configuration resume."
            )
    else:
        manifest = {
            "pipeline_version": PIPELINE_VERSION,
            "config": config,
            "stages": [],
            "checkpoint_handoff": "previous_stage_best_evaluated_checkpoint",
            "resume_policy": "same-stage latest checkpoint only; full fresh budget for each new stage",
            "action_semantics": "collective-thrust-plus-body-torque-allocation-v2",
            "reference_guidance": "privileged-teacher-weight-0.80-0.60-0.40-0.20-0.00",
            "adaptive_profile_sampling": "smoothed-weak-profile-reweighting",
            "checkpoint_selection": "min-hover-success,max-crash,min-takeoff,stable-duration,min-saturation,mean-reward",
            "observation_scaling": "dimensionless-state-history-v1",
            "progressive_initial_recovery": "ground-start-with-stage-scaled-position-and-attitude-perturbations",
            "simulation_only": True,
            "physical_flight_eligible": False,
        }

    heldout_path = out / f"{a.run_prefix}-held-out-domain-randomization.json"
    heldout = json.loads(
        (ROOT / "vehicle_models/family/domain_randomization.json").read_text(encoding="utf-8")
    )
    heldout["description"] = (
        "Independent wider held-out distribution; excluded from all training "
        "and checkpoint selection; evaluated once after stage 5."
    )
    heldout["mass"]["scale"] = [0.76, 1.12]
    heldout["center_of_mass"]["offset_arm_fraction"] = 0.12
    heldout["inertia"]["scale"] = [0.65, 1.45]
    heldout["motor"]["effectiveness"] = [0.86, 1.10]
    heldout["battery"]["voltage_factor"] = [0.92, 1.0]
    heldout["aerodynamics"]["drag_scale"] = [0.65, 1.45]
    heldout["environment"]["wind_component_m_s"] = [-2.5, 2.5]
    heldout.pop("curriculum", None)
    # Held-out evaluation keeps the same stage-5 initial recovery difficulty
    # while using its independent wider physical/environment distribution.
    heldout["active_curriculum_stage"] = {
        "stage": 5, "name": "held-out-stage5-initial-condition"
    }
    write_json(heldout_path, heldout)

    previous_best = None
    for stage in range(1, 6):
        name = f"{a.run_prefix}-stage{stage}"
        stage_dir = out / name
        best = stage_dir / "best" / "best_model.zip"
        final = stage_dir / "final_model.zip"
        latest = stage_dir / "latest_model.zip"
        state_path = stage_dir / "training_state.json"

        entry = next(
            (item for item in manifest["stages"]
             if item.get("stage") == stage and item.get("completed")),
            None,
        )
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        stage_is_complete = (
            state.get("completed") is True
            and state.get("stage") == stage
            and best.is_file()
            and final.is_file()
        )
        if entry and stage_is_complete:
            previous_best = best
            print(f"Stage {stage}/5 already complete; reusing its checkpoint.")
            continue
        if stage_is_complete:
            previous_best = best
            manifest["stages"] = [x for x in manifest["stages"] if x.get("stage") != stage]
            manifest["stages"].append({
                "stage": stage, "completed": True,
                "training_timesteps": int(state["num_timesteps"]),
                "best_checkpoint": str(best), "final_checkpoint": str(final),
                "recovered_from_completed_training_state": True,
            })
            manifest["stages"].sort(key=lambda x: x["stage"])
            write_json(manifest_path, manifest)
            print(f"Stage {stage}/5 training was complete; recovered its manifest entry.")
            continue

        same_stage_resume = (
            latest.is_file()
            and state_path.is_file()
            and state.get("stage") == stage
            and state.get("completed") is False
            and int(state.get("num_timesteps", -1)) > 0
        )
        cmd = [
            sys.executable, str(ROOT / "training/train_universal_ppo.py"),
            "--timesteps", str(a.timesteps_per_stage),
            "--n-envs", str(a.n_envs),
            "--seed", str(a.seed + stage - 1),
            "--run-name", name,
            "--output-dir", str(out),
            "--curriculum-stage", str(stage),
            "--eval-episodes", str(a.checkpoint_eval_episodes),
            "--eval-frequency", str(a.eval_frequency),
            "--checkpoint-frequency", str(a.checkpoint_frequency),
            "--vehicle-family", str(a.vehicle_family),
            "--randomization-config",
            str(ROOT / "vehicle_models/family/domain_randomization.json"),
        ]
        if same_stage_resume:
            cmd.extend(["--load-model", str(latest), "--resume"])
            print(
                f"Resuming stage {stage}/5 from {state['num_timesteps']:,}/"
                f"{a.timesteps_per_stage:,} transitions."
            )
        elif stage > 1:
            if previous_best is None or not previous_best.is_file():
                raise RuntimeError(f"Stage {stage} has no completed previous-stage best checkpoint.")
            cmd.extend(["--load-model", str(previous_best)])
            print(f"Starting stage {stage}/5 from the best checkpoint of stage {stage - 1}.")
        else:
            print("Starting stage 1/5 with a new PPO policy.")

        subprocess.run(cmd, cwd=ROOT, check=True)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("completed") is not True or not best.is_file() or not final.is_file():
            raise RuntimeError(f"Stage {stage} did not complete its full training budget and save both checkpoints.")
        if int(state.get("num_timesteps", -1)) != a.timesteps_per_stage:
            raise RuntimeError(
                f"Stage {stage} recorded {state.get('num_timesteps')} transitions; "
                f"expected exactly {a.timesteps_per_stage}."
            )

        manifest["stages"] = [x for x in manifest["stages"] if x.get("stage") != stage]
        manifest["stages"].append({
            "stage": stage, "completed": True,
            "training_timesteps": int(state["num_timesteps"]),
            "best_checkpoint": str(best), "final_checkpoint": str(final),
            "checkpoint_evaluations": str(stage_dir / "profile-evaluations"),
        })
        manifest["stages"].sort(key=lambda x: x["stage"])
        write_json(manifest_path, manifest)
        previous_best = best

    # Held-out evaluation is run once, after all five actual PPO training stages.
    stage5 = out / f"{a.run_prefix}-stage5"
    release_dir = out / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("final", stage5 / "final_model.zip"),
        ("best", stage5 / "best" / "best_model.zip"),
    ]
    tested = []
    for label, model_path in candidates:
        report_path = stage5 / f"final-test-{label}.json"
        if report_path.is_file():
            print(f"Reusing completed held-out report: {report_path.name}")
        else:
            subprocess.run([
                sys.executable, str(ROOT / "training/evaluate_universal_ppo.py"),
                "--model", str(model_path),
                "--vehicle-family", str(a.vehicle_family),
                "--episodes", str(a.evaluation_episodes),
                "--seed", "990001",
                "--output", str(report_path),
                "--randomization-config", str(heldout_path),
                "--held-out",
            ], cwd=ROOT, check=True)
        result = json.loads(report_path.read_text(encoding="utf-8"))
        expected = set(load_profile_ids(a.vehicle_family))
        actual = {item["vehicle_id"] for item in result["profiles"]}
        if expected != actual:
            raise RuntimeError("Final held-out evaluation did not cover every configured vehicle profile.")
        tested.append({
            "label": label, "model": str(model_path),
            "report": str(report_path), "score": list(score(result)),
            "profiles": result["profiles"],
        })

    winner = max(tested, key=lambda item: tuple(item["score"]))
    shutil.copy2(stage5 / "final_model.zip", release_dir / "final_model.zip")
    shutil.copy2(Path(winner["model"]), release_dir / "best_model.zip")
    min_success = min(x["hover_success_rate"] for x in winner["profiles"])
    max_crash = max(x["crash_rate"] for x in winner["profiles"])
    min_takeoff = min(x["takeoff_success_rate"] for x in winner["profiles"])
    accepted = (
        min_success >= a.min_final_success_rate
        and max_crash <= a.max_final_crash_rate
        and min_takeoff >= a.min_final_takeoff_rate
    )
    release = {
        "pipeline_version": PIPELINE_VERSION,
        "winner": winner["label"],
        "selection_basis": "final held-out evaluation on identical seeds",
        "candidates": tested,
        "minimum_profile_hover_success_rate": min_success,
        "maximum_profile_crash_rate": max_crash,
        "minimum_profile_takeoff_success_rate": min_takeoff,
        "acceptance_thresholds": {
            "minimum_sustained_hover_success_rate": a.min_final_success_rate,
            "maximum_crash_rate": a.max_final_crash_rate,
            "minimum_takeoff_success_rate": a.min_final_takeoff_rate,
        },
        "accepted": accepted,
        "simulation_only": True,
        "physical_flight_eligible": False,
    }
    write_json(release_dir / "release-manifest.json", release)
    manifest["final_test"] = release
    write_json(manifest_path, manifest)
    print(json.dumps(release, indent=2))
    if not accepted:
        raise SystemExit(
            "Final held-out gate failed; all five trained stages, checkpoints and reports were retained."
        )


if __name__ == "__main__":
    main()
