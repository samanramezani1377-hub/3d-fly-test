"""Diagnose vertical response and PPO actions without changing aircraft configuration.

Runs fixed-throttle probes from the same reset state, then optionally logs a short
deterministic rollout from a saved PPO model. This is simulation diagnostics only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run_throttle_probe(env, throttle: float, steps: int) -> dict:
    obs, _ = env.reset(seed=123)
    initial_position = [float(obs[0]), float(obs[1]), float(obs[2])]
    trace = []
    for step in range(steps):
        action = np.array([0.0, 0.0, 0.0, throttle], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append({
            "step": step + 1,
            "x_m": float(obs[0]),
            "altitude_m": float(obs[1]),
            "z_m": float(obs[2]),
            "vx_mps": float(obs[3]),
            "vertical_speed_mps": float(obs[4]),
            "vz_mps": float(obs[5]),
            "roll_rad": float(obs[6]),
            "pitch_rad": float(obs[7]),
            "yaw_rad": float(obs[8]),
            "throttle": float(throttle),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": bool(info.get("crashed", False)),
        })
        if terminated or truncated:
            break
    last = trace[-1] if trace else {}
    return {
        "throttle": float(throttle),
        "steps_run": len(trace),
        "initial_position_xyz_m": initial_position,
        "final_position_xyz_m": [last.get("x_m"), last.get("altitude_m"), last.get("z_m")],
        "final_velocity_xyz_mps": [last.get("vx_mps"), last.get("vertical_speed_mps"), last.get("vz_mps")],
        "terminated": last.get("terminated"),
        "truncated": last.get("truncated"),
        "crashed": last.get("crashed"),
        "trace": trace,
    }


def run_policy_probe(env, model_path: Path, steps: int) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    obs, _ = env.reset(seed=456)
    trace = []
    terminated = truncated = False
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append({
            "step": step + 1,
            "action_roll": float(action[0]),
            "action_pitch": float(action[1]),
            "action_yaw_rate": float(action[2]),
            "action_throttle": float(action[3]),
            "x_m": float(obs[0]),
            "altitude_m": float(obs[1]),
            "z_m": float(obs[2]),
            "vx_mps": float(obs[3]),
            "vertical_speed_mps": float(obs[4]),
            "vz_mps": float(obs[5]),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": bool(info.get("crashed", False)),
        })
        if terminated or truncated:
            break
    return {
        "model": str(model_path),
        "steps_run": len(trace),
        "final_position_xyz_m": [trace[-1].get(k) for k in ("x_m", "altitude_m", "z_m")] if trace else None,
        "final_vertical_speed_mps": trace[-1].get("vertical_speed_mps") if trace else None,
        "terminated": trace[-1].get("terminated") if trace else None,
        "crashed": trace[-1].get("crashed") if trace else None,
        "trace": trace,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="Optional saved Stable-Baselines3 PPO .zip model")
    parser.add_argument("--steps", type=int, default=100, help="Maximum steps per fixed-throttle probe")
    parser.add_argument("--policy-steps", type=int, default=40, help="Maximum PPO rollout steps when --model is supplied")
    parser.add_argument("--throttles", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--output", type=Path, default=Path("training/runs/control-diagnostics.json"))
    args = parser.parse_args(argv)

    if args.steps <= 0 or args.policy_steps <= 0:
        parser.error("--steps and --policy-steps must be positive")
    if any(t < 0.0 or t > 1.0 for t in args.throttles):
        parser.error("all throttle values must be between 0 and 1")
    if args.model is not None and not args.model.is_file():
        parser.error(f"model not found: {args.model}")

    try:
        from native_env import NativeSkySimEnv
    except ImportError as exc:
        raise SystemExit(
            "Run from the repository root after installing training/requirements.txt "
            "and building the native core with python training/build_native_core.py."
        ) from exc

    env = NativeSkySimEnv()
    try:
        probes = [run_throttle_probe(env, t, args.steps) for t in args.throttles]
        policy = run_policy_probe(env, args.model, args.policy_steps) if args.model else None
    finally:
        env.close()

    report = {
        "diagnostic_only": True,
        "physical_aircraft_model_ready": False,
        "note": "Simulation-only test. Does not validate real-flight safety or sim-to-real accuracy. No drone config values were modified.",
        "environment": {
            "backend": "native-drone-core",
            "reset_position_xyz_m": [0.0, 2.0, 0.0],
            "target_xyz_m": [0.0, 3.0, 0.0],
            "dt_s": 0.02,
        },
        "fixed_throttle_probes": probes,
        "ppo_policy_probe": policy,
    }
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "throttle_summary": [{
            "throttle": p["throttle"],
            "steps_run": p["steps_run"],
            "final_altitude_m": p["final_position_xyz_m"][1],
            "final_vertical_speed_mps": p["final_velocity_xyz_mps"][1],
            "terminated": p["terminated"],
            "crashed": p["crashed"],
        } for p in probes],
        "ppo_summary": None if policy is None else {
            "steps_run": policy["steps_run"],
            "final_position_xyz_m": policy["final_position_xyz_m"],
            "final_vertical_speed_mps": policy["final_vertical_speed_mps"],
            "terminated": policy["terminated"],
            "crashed": policy["crashed"],
            "first_actions": policy["trace"][:5],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
