"""Resumable five-stage PPO curriculum with a final held-out release gate."""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
    return (min(x["hover_success_rate"] for x in profiles),
            -max(x["crash_rate"] for x in profiles),
            min(x["takeoff_success_rate"] for x in profiles),
            sum(x["mean_stable_hover_seconds"] for x in profiles) / len(profiles))

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps-per-stage", type=int, default=200000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--evaluation-episodes", type=int, default=20)
    p.add_argument("--checkpoint-eval-episodes", type=int, default=5)
    p.add_argument("--min-final-success-rate", type=float, default=0.50)
    p.add_argument("--max-final-crash-rate", type=float, default=0.25)
    p.add_argument("--min-final-takeoff-rate", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-prefix", default="universal-quad")
    p.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--vehicle-family", type=Path, default=ROOT / "vehicle_models/family/vehicle_profiles.json")
    a = p.parse_args(argv)
    if min(a.timesteps_per_stage, a.n_envs, a.evaluation_episodes, a.checkpoint_eval_episodes) <= 0:
        raise SystemExit("training and evaluation counts must be positive")
    if not 0 <= a.min_final_success_rate <= 1 or not 0 <= a.max_final_crash_rate <= 1 or not 0 <= a.min_final_takeoff_rate <= 1:
        raise SystemExit("acceptance thresholds must be between zero and one")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{a.run_prefix}-curriculum-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "stages": [], "checkpoint_handoff": "previous_stage_balanced_best_checkpoint",
        "simulation_only": True, "physical_flight_eligible": False}
    heldout_path = out / "held-out-domain-randomization.json"
    heldout = json.loads((ROOT / "vehicle_models/family/domain_randomization.json").read_text())
    heldout["description"] = "Independent wider held-out distribution; excluded from training and checkpoint selection."
    heldout["mass"]["scale"] = [0.76, 1.12]
    heldout["center_of_mass"]["offset_arm_fraction"] = 0.12
    heldout["inertia"]["scale"] = [0.65, 1.45]
    heldout["motor"]["effectiveness"] = [0.86, 1.10]
    heldout["battery"]["voltage_factor"] = [0.92, 1.0]
    heldout["aerodynamics"]["drag_scale"] = [0.65, 1.45]
    heldout["environment"]["wind_component_m_s"] = [-2.5, 2.5]
    heldout.pop("curriculum", None)
    write_json(heldout_path, heldout)

    previous = None
    for stage in range(1, 6):
        name = f"{a.run_prefix}-stage{stage}"
        d = out / name
        best, final = d / "best" / "best_model.zip", d / "final_model.zip"
        report = d / "held-out-evaluation.json"
        completed = next((x for x in manifest["stages"] if x.get("stage") == stage and x.get("completed")), None)
        if completed and best.is_file() and final.is_file() and report.is_file():
            previous = best
            continue
        latest = d / "latest_model.zip"
        state = d / "training_state.json"
        resume = latest if latest.is_file() and state.is_file() else previous
        cmd = [sys.executable, str(ROOT / "training/train_universal_ppo.py"),
            "--timesteps", str(a.timesteps_per_stage), "--n-envs", str(a.n_envs),
            "--seed", str(a.seed + stage - 1), "--run-name", name, "--output-dir", str(out),
            "--curriculum-stage", str(stage), "--eval-episodes", str(a.checkpoint_eval_episodes),
            "--vehicle-family", str(a.vehicle_family),
            "--randomization-config", str(ROOT / "vehicle_models/family/domain_randomization.json")]
        if resume:
            cmd.extend(["--load-model", str(resume)])
        subprocess.run(cmd, cwd=ROOT, check=True)
        if not best.is_file() or not final.is_file():
            raise RuntimeError(f"stage {stage} did not save final and best checkpoints")
        subprocess.run([sys.executable, str(ROOT / "training/evaluate_universal_ppo.py"),
            "--model", str(best), "--vehicle-family", str(a.vehicle_family),
            "--episodes", str(a.evaluation_episodes), "--seed", str(90000 + stage * 1000),
            "--output", str(report), "--randomization-config", str(heldout_path), "--held-out"],
            cwd=ROOT, check=True)
        result = json.loads(report.read_text())
        expected, actual = set(load_profile_ids(a.vehicle_family)), {x["vehicle_id"] for x in result["profiles"]}
        if expected != actual:
            raise RuntimeError(f"stage {stage} evaluation profile mismatch")
        entry = {"stage": stage, "completed": True, "final_checkpoint": str(final),
            "selected_best_checkpoint": str(best), "evaluation": str(report),
            "profiles": result["profiles"], "all_profiles_evaluated": True}
        manifest["stages"] = [x for x in manifest["stages"] if x.get("stage") != stage] + [entry]
        manifest["stages"].sort(key=lambda x: x["stage"])
        manifest["timesteps_per_stage"] = a.timesteps_per_stage
        manifest["vehicle_family"] = str(a.vehicle_family)
        manifest["held_out_distribution"] = str(heldout_path)
        write_json(manifest_path, manifest)
        previous = best

    stage5 = out / f"{a.run_prefix}-stage5"
    candidates = [("final", stage5 / "final_model.zip"), ("best", stage5 / "best" / "best_model.zip")]
    tested = []
    for label, model in candidates:
        rp = stage5 / f"final-test-{label}.json"
        subprocess.run([sys.executable, str(ROOT / "training/evaluate_universal_ppo.py"),
            "--model", str(model), "--vehicle-family", str(a.vehicle_family),
            "--episodes", str(a.evaluation_episodes), "--seed", "990001",
            "--output", str(rp), "--randomization-config", str(heldout_path), "--held-out"],
            cwd=ROOT, check=True)
        data = json.loads(rp.read_text())
        tested.append({"label": label, "model": str(model), "report": str(rp),
                       "score": list(score(data)), "profiles": data["profiles"]})
    winner = max(tested, key=lambda x: tuple(x["score"]))
    release_dir = out / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage5 / "final_model.zip", release_dir / "final_model.zip")
    shutil.copy2(Path(winner["model"]), release_dir / "best_model.zip")
    min_success = min(x["hover_success_rate"] for x in winner["profiles"])
    max_crash = max(x["crash_rate"] for x in winner["profiles"])
    min_takeoff = min(x["takeoff_success_rate"] for x in winner["profiles"])
    accepted = (min_success >= a.min_final_success_rate
                and max_crash <= a.max_final_crash_rate
                and min_takeoff >= a.min_final_takeoff_rate)
    release = {"winner": winner["label"], "selection_basis": "final held-out evaluation on identical seeds",
        "candidates": tested, "minimum_profile_hover_success_rate": min_success,
        "maximum_profile_crash_rate": max_crash,
        "minimum_profile_takeoff_success_rate": min_takeoff,
        "acceptance_thresholds": {"minimum_sustained_hover_success_rate": a.min_final_success_rate,
                                  "maximum_crash_rate": a.max_final_crash_rate,
                                  "minimum_takeoff_success_rate": a.min_final_takeoff_rate},
        "accepted": accepted, "simulation_only": True, "physical_flight_eligible": False}
    write_json(release_dir / "release-manifest.json", release)
    manifest["final_test"] = release
    write_json(manifest_path, manifest)
    print(json.dumps(release, indent=2))
    if not accepted:
        raise SystemExit("Final held-out test failed the acceptance gate; all model files and reports were retained.")

if __name__ == "__main__":
    main()
