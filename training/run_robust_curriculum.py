"""Run the four-stage smooth2 curriculum, carrying the selected PPO model forward."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "training" / "runs"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps-per-stage", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-prefix", default="robust-smooth2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--software-only-synthetic-calibration", action="store_true")
    args = parser.parse_args(argv)
    if args.timesteps_per_stage <= 0 or args.n_envs <= 0:
        raise SystemExit("timesteps and n-envs must be positive")

    previous_model = None
    stage_manifest = []
    for stage in range(1, 5):
        run_name = f"{args.run_prefix}-stage{stage}"
        command = [
            sys.executable, str(ROOT / "training" / "train_robust_ppo.py"),
            "--timesteps", str(args.timesteps_per_stage),
            "--n-envs", str(args.n_envs),
            "--seed", str(args.seed + stage - 1),
            "--run-name", run_name,
            "--output-dir", str(args.output_dir),
            "--curriculum-stage", str(stage),
        ]
        if previous_model is not None:
            command.extend(["--load-model", str(previous_model)])
        if args.software_only_synthetic_calibration:
            command.append("--software-only-synthetic-calibration")
        subprocess.run(command, cwd=ROOT, check=True)
        previous_model = args.output_dir / run_name / "final_model.zip"
        if not previous_model.is_file():
            raise RuntimeError(f"stage {stage} did not produce {previous_model}")
        stage_manifest.append({
            "stage": stage,
            "run_name": run_name,
            "model": str(previous_model),
            "initial_model": stage_manifest[-1]["model"] if stage_manifest else None,
        })

    import json
    manifest = {
        "vehicle_model": "smooth2-parametric-v1",
        "stages": stage_manifest,
        "timesteps_per_stage": args.timesteps_per_stage,
        "parallel_envs": args.n_envs,
        "synthetic_calibration": args.software_only_synthetic_calibration,
        "simulation_only": True,
        "physical_flight_eligible": False,
    }
    output = args.output_dir / f"{args.run_prefix}-curriculum-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Curriculum complete; manifest: {output}")


if __name__ == "__main__":
    main()
