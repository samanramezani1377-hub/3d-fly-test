"""Browser bridge for the exact training environment; no duplicated flight equations."""
import json
import numpy as np
from training.fast_physics.vehicle_family import VehicleFamilyEnv

_env = None
_last_obs = None
_last_info = None
_profile_id = None
_step_count = 0

def _clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value

def _snapshot(obs, info, reward=0.0, terminated=False, truncated=False):
    plant = _env.inner.plant
    state = plant.state().astype(float).tolist()
    return json.dumps(_clean({
        "state": state,
        "observation": obs,
        "observation_size": int(np.asarray(obs).size),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "step": _step_count,
        "time_s": float(plant.time),
        "dt_s": float(_env.dt),
        "info": info,
        "estimator": _env.inner.estimator.diagnostics(),
        "vehicle_id": _profile_id,
        "mass_kg": float(plant.params.mass_kg),
        "rotor_thrust_n": (info or {}).get("rotor_thrust_n", [0.0]*4),
        "rotor_omega_rad_s": (info or {}).get("rotor_omega_rad_s", [0.0]*4),
        "total_thrust_n": (info or {}).get("total_thrust_n", 0.0),
    }))

def engine_reset(vehicle_id, randomize, curriculum_stage, seed):
    global _env, _last_obs, _last_info, _profile_id, _step_count
    if _env is not None:
        _env.close()
    stage = None if int(curriculum_stage) == 0 else int(curriculum_stage)
    _profile_id = str(vehicle_id)
    _env = VehicleFamilyEnv(
        vehicle_id=_profile_id,
        dt=0.02,
        max_steps=1500,
        history_length=4,
        randomize=bool(randomize),
        curriculum_stage=stage,
    )
    _step_count = 0
    _last_obs, _last_info = _env.reset(seed=int(seed))
    return _snapshot(_last_obs, _last_info)

def engine_step(action_json):
    global _last_obs, _last_info, _step_count
    if _env is None:
        raise RuntimeError("engine_reset must be called before engine_step")
    action = np.asarray(json.loads(action_json), dtype=np.float32)
    _last_obs, reward, terminated, truncated, _last_info = _env.step(action)
    _step_count += 1
    return _snapshot(_last_obs, _last_info, reward, terminated, truncated)

def engine_metadata():
    if _env is None:
        return "{}"
    return json.dumps(_clean({
        "dt_s": _env.dt,
        "max_steps": _env.max_steps,
        "history_length": _env.history_length,
        "observation_size": int(_env.observation_space.shape[0]),
        "action_size": int(_env.action_space.shape[0]),
        "vehicle_id": _profile_id,
        "model": "training.fast_physics.vehicle_family.VehicleFamilyEnv",
        "plant": "training.fast_physics.dynamics.FastQuadrotor",
        "randomization": _env.randomize,
        "curriculum_stage": _env.curriculum_stage,
    }))
