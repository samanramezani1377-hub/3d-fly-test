"""Five-stage universal-vehicle PPO curriculum with evaluated checkpoint handoff."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_profile_ids(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return [p["vehicle_id"] for p in data["profiles"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps-per-stage", type=int, default=200000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--evaluation-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-prefix", default="universal-quad")
    p.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--vehicle-family", type=Path, default=ROOT / "vehicle_models/family/vehicle_profiles.json")
    args = p.parse_args(argv)
    if args.timesteps_per_stage <= 0 or args.n_envs <= 0 or args.evaluation_episodes <= 0:
        raise SystemExit("timesteps, n-envs and evaluation-episodes must be positive")
    output, previous, stages = args.output_dir.resolve(), None, []
    output.mkdir(parents=True, exist_ok=True)
    heldout_path = output / "held-out-domain-randomization.json"
    heldout = json.loads((ROOT / "vehicle_models/family/domain_randomization.json").read_text(encoding="utf-8"))
    heldout["description"] = "Independent wider held-out distribution; excluded from training and checkpoint selection."
    heldout["mass"]["scale"] = [0.76, 1.24]
    heldout["center_of_mass"]["offset_arm_fraction"] = 0.12
    heldout["inertia"]["scale"] = [0.65, 1.45]
    heldout["motor"]["effectiveness"] = [0.82, 1.13]
    heldout["battery"]["voltage_factor"] = [0.85, 1.0]
    heldout["aerodynamics"]["drag_scale"] = [0.65, 1.45]
    heldout["environment"]["wind_component_m_s"] = [-2.5, 2.5]
    heldout.pop("curriculum", None)
    heldout_path.write_text(json.dumps(heldout, indent=2), encoding="utf-8")
    for stage in range(1, 6):
        name = f"{args.run_prefix}-stage{stage}"
        command = [sys.executable, str(ROOT / "training/train_universal_ppo.py"),
            "--timesteps", str(args.timesteps_per_stage), "--n-envs", str(args.n_envs),
            "--seed", str(args.seed + stage - 1), "--run-name", name,
            "--output-dir", str(output), "--curriculum-stage", str(stage),
            "--vehicle-family", str(args.vehicle_family), "--randomization-config", str(ROOT / "vehicle_models/family/domain_randomization.json")]
        if previous:
            command.extend(["--load-model", str(previous)])
        subprocess.run(command, cwd=ROOT, check=True)
        final_checkpoint = output / name / "final_model.zip"
        previous = output / name / "best" / "best_model.zip"
        if not final_checkpoint.is_file():
            raise RuntimeError(f"stage {stage} final checkpoint missing: {final_checkpoint}")
        if not previous.is_file():
            raise RuntimeError(f"stage {stage} balanced best checkpoint missing: {previous}")
        report = output / name / "held-out-evaluation.json"
        subprocess.run([sys.executable, str(ROOT / "training/evaluate_universal_ppo.py"),
            "--model", str(previous), "--vehicle-family", str(args.vehicle_family),
            "--episodes", str(args.evaluation_episodes), "--seed", str(90000 + stage * 1000), "--output", str(report), "--randomization-config", str(heldout_path), "--held-out"],
            cwd=ROOT, check=True)
        evaluation = json.loads(report.read_text(encoding="utf-8"))
        expected_profiles = set(load_profile_ids(args.vehicle_family))
        evaluated_profiles = {item["vehicle_id"] for item in evaluation.get("profiles", [])}
        if evaluated_profiles != expected_profiles:
            raise RuntimeError(f"stage {stage} evaluation profile mismatch: expected {sorted(expected_profiles)}, got {sorted(evaluated_profiles)}")
        stages.append({"stage":stage,"final_checkpoint":str(final_checkpoint),
                       "selected_best_checkpoint":str(previous),
                       "checkpoint_selection":"balanced_min_profile_success_then_worst_crash_then_mean_return",
                       "evaluation":str(report),"profiles":evaluation["profiles"], "all_profiles_evaluated": len(evaluation["profiles"]) == len(load_profile_ids(args.vehicle_family))})
    manifest = {"stages":stages,"checkpoint_handoff":"previous_stage_balanced_best_checkpoint",
                "timesteps_per_stage":args.timesteps_per_stage,
                "vehicle_family":str(args.vehicle_family),"held_out_distribution":str(heldout_path),"simulation_only":True,
                "physical_flight_eligible":False}
    path = output / f"{args.run_prefix}-curriculum-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Universal curriculum complete: {path}")


if __name__ == "__main__":
    main()
