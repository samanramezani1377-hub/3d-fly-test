"""Diagnose vertical response and PPO actions without changing aircraft configuration.

Runs fixed-throttle probes from the same reset state, then optionally logs a short
deterministic rollout from a saved PPO model. This is simulation diagnostics only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run_throttle_probe(env, throttle_action: float, steps: int) -> dict:
    obs, _ = env.reset(seed=123)
    raw_obs = env.unwrapped._observation()
    initial_position = [float(raw_obs[0]), float(raw_obs[1]), float(raw_obs[2])]
    trace = []
    for step in range(steps):
        action = np.array([0.0, 0.0, 0.0, throttle_action], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        raw_obs = env.unwrapped._observation()
        trace.append({
            "step": step + 1,
            "x_m": float(raw_obs[0]),
            "altitude_m": float(raw_obs[1]),
            "z_m": float(raw_obs[2]),
            "vx_mps": float(raw_obs[3]),
            "vertical_speed_mps": float(raw_obs[4]),
            "vz_mps": float(raw_obs[5]),
            "roll_rad": float(raw_obs[6]),
            "pitch_rad": float(raw_obs[7]),
            "yaw_rad": float(raw_obs[8]),
            "policy_throttle_action": float(throttle_action),
            "physical_throttle": float(info.get("throttle", 0.0)),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": bool(info.get("crashed", False)),
            "rotor_thrust_n": info.get("rotor_thrust_n"),
            "rotor_omega_rads": info.get("rotor_omega_rads"),
        })
        if terminated or truncated:
            break
    last = trace[-1] if trace else {}
    return {
        "policy_throttle_action": float(throttle_action),
        "physical_throttle_final": last.get("physical_throttle"),
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
        raw_obs = env.unwrapped._observation()
        trace.append({
            "step": step + 1,
            "action_roll": float(action[0]),
            "action_pitch": float(action[1]),
            "action_yaw_rate": float(action[2]),
            "action_throttle": float(action[3]),
            "physical_throttle": float(info.get("throttle", 0.0)),
            "x_m": float(raw_obs[0]),
            "altitude_m": float(raw_obs[1]),
            "z_m": float(raw_obs[2]),
            "vx_mps": float(raw_obs[3]),
            "vertical_speed_mps": float(raw_obs[4]),
            "vz_mps": float(raw_obs[5]),
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
    parser.add_argument("--throttles", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0], help="Normalized PPO throttle action values in [-1, 1], not physical throttle")
    parser.add_argument("--output", type=Path, default=Path("training/runs/control-diagnostics.json"))
    args = parser.parse_args(argv)

    if args.steps <= 0 or args.policy_steps <= 0:
        parser.error("--steps and --policy-steps must be positive")
    if any(t < -1.0 or t > 1.0 for t in args.throttles):
        parser.error("all normalized throttle actions must be between -1 and 1")
    if args.model is not None and not args.model.is_file():
        parser.error(f"model not found: {args.model}")

    try:
        from native_env import NativeSkySimEnv, PolicyObservationWrapper
    except ImportError as exc:
        raise SystemExit(
            "Run from the repository root after installing training/requirements.txt "
            "and building the native core with python training/build_native_core.py."
        ) from exc

    # Isolate each throttle point in a fresh core: no motor, ESC, inflow,
    # battery, VRS, or turbulence state may leak between measurements.
    probes = []
    for throttle in args.throttles:
        probe_env = PolicyObservationWrapper(NativeSkySimEnv())
        try:
            probes.append(run_throttle_probe(probe_env, throttle, args.steps))
        finally:
            probe_env.close()

    policy = None
    if args.model:
        policy_env = PolicyObservationWrapper(NativeSkySimEnv())
        try:
            policy = run_policy_probe(policy_env, args.model, args.policy_steps)
        finally:
            policy_env.close()

    report = {
        "diagnostic_only": True,
        "physical_aircraft_model_ready": False,
        "note": "Simulation-only test. Does not validate real-flight safety or sim-to-real accuracy. No drone config values were modified.",
        "environment": {
            "backend": "native-drone-core",
            "reset_position_xyz_m": [0.0, 0.16, 0.0],
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
            "policy_throttle_action": p["policy_throttle_action"],
            "physical_throttle_final": p["physical_throttle_final"],
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
