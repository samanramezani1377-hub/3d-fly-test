"""Four-stage universal-vehicle PPO curriculum with checkpoint handoff."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps-per-stage", type=int, default=200000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-prefix", default="universal-quad")
    p.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--vehicle-family", type=Path, default=ROOT / "vehicle_models/family/vehicle_profiles.json")
    args = p.parse_args(argv)
    if args.timesteps_per_stage <= 0 or args.n_envs <= 0:
        raise SystemExit("timesteps and n-envs must be positive")
    output, previous, stages = args.output_dir.resolve(), None, []
    for stage in range(1, 5):
        name = f"{args.run_prefix}-stage{stage}"
        command = [sys.executable, str(ROOT / "training/train_universal_ppo.py"),
            "--timesteps", str(args.timesteps_per_stage), "--n-envs", str(args.n_envs),
            "--seed", str(args.seed + stage - 1), "--run-name", name,
            "--output-dir", str(output), "--curriculum-stage", str(stage),
            "--vehicle-family", str(args.vehicle_family)]
        if previous:
            command.extend(["--load-model", str(previous)])
        subprocess.run(command, cwd=ROOT, check=True)
        previous = output / name / "final_model.zip"
        if not previous.is_file():
            raise RuntimeError(f"stage {stage} checkpoint missing: {previous}")
        report = output / name / "held-out-evaluation.json"
        subprocess.run([sys.executable, str(ROOT / "training/evaluate_universal_ppo.py"),
            "--model", str(previous), "--vehicle-family", str(args.vehicle_family),
            "--episodes", "20", "--seed", str(90000 + stage * 1000), "--output", str(report)],
            cwd=ROOT, check=True)
        evaluation = json.loads(report.read_text(encoding="utf-8"))
        stages.append({"stage":stage,"checkpoint":str(previous),"evaluation":str(report),
                       "profiles":evaluation["profiles"]})
    manifest = {"stages":stages,"timesteps_per_stage":args.timesteps_per_stage,
                "vehicle_family":str(args.vehicle_family),"simulation_only":True,
                "physical_flight_eligible":False}
    path = output / f"{args.run_prefix}-curriculum-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Universal curriculum complete: {path}")


if __name__ == "__main__":
    main()
