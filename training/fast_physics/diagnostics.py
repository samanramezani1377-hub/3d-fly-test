"""Runtime diagnostics for the shared PPO/Flight Lab physics engine."""
from __future__ import annotations
import math
import numpy as np

def diagnose_environment(env):
    checks=[]
    def add(level, code, message, value=None):
        checks.append({"level":level,"code":code,"message":message,"value":value})
    try:
        dt=float(env.dt)
        add("ok" if math.isfinite(dt) and dt>0 else "error","physics.dt","گام زمانی موتور",dt)
        if abs(dt-0.02)>1e-9: add("warning","physics.dt_expected","گام زمانی با مقدار ۲۰ میلی‌ثانیهٔ Flight Lab/PPO متفاوت است",dt)
        inner=getattr(env,"inner",None)
        if inner is None: add("error","env.inner_missing","محیط داخلی AdaptiveDroneEnv ساخته نشده است"); return checks
        plant=getattr(inner,"plant",None)
        if plant is None: add("error","plant.missing","FastQuadrotor هنوز ساخته نشده است"); return checks
        state=np.asarray(plant.state(),dtype=np.float64)
        add("ok" if state.shape==(12,) else "error","plant.state_shape","ابعاد حالت فیزیکی",list(state.shape))
        add("ok" if np.all(np.isfinite(state)) else "error","plant.state_finite","تمام مؤلفه‌های حالت محدود و متناهی هستند",bool(np.all(np.isfinite(state))))
        params=plant.params; mass=float(params.mass_kg)
        add("ok" if math.isfinite(mass) and mass>0 else "error","plant.mass","جرم مؤثر اپیزود",mass)
        omega=float(params.max_rotor_speed_rad_s); kf=float(params.thrust_coefficient_n_per_rad_s2)
        twr=4.0*kf*omega**2/(mass*9.80665)
        add("ok" if twr>1 else "error","plant.thrust_to_weight","نسبت تراست بیشینه به وزن (باید بیشتر از ۱ باشد)",round(twr,4))
        if twr<1.5: add("warning","plant.low_thrust_margin","حاشیهٔ تراست پایین است؛ برخاست دشوار می‌شود",round(twr,4))
        meta=getattr(env,"model_metadata",{}) or {}; synthetic=bool(meta.get("synthetic_calibration_used",False))
        add("warning" if synthetic else "ok","calibration.synthetic","استفاده از کالیبراسیون تخمینی/مصنوعی",synthetic)
        status=str(meta.get("calibration_status","unknown"))
        add("warning" if status!="measured" else "ok","calibration.status","وضعیت کالیبراسیون",status)
        estimates=list((getattr(env,"active_profile",{}) or {}).get("derived_estimates",[]))
        add("warning" if estimates else "ok","calibration.derived_estimates","پارامترهای مشتق‌شده و تخمینی",estimates)
        action_shape=tuple(env.action_space.shape); obs_shape=tuple(env.observation_space.shape)
        add("ok" if action_shape==(4,) else "error","interface.action_shape","ابعاد ورودی فرمان موتور",action_shape)
        obs=np.asarray(getattr(env,"_last_obs",np.zeros(obs_shape)),dtype=np.float64)
        finite=bool(np.all(np.isfinite(obs)))
        add("ok" if obs.size==int(np.prod(obs_shape)) and finite else "error","interface.observation","ابعاد/مقادیر مشاهدهٔ محیط",{"shape":list(obs.shape),"finite":finite})
    except Exception as exc:
        add("error","diagnostics.exception",f"خطا هنگام عیب‌یابی موتور: {type(exc).__name__}: {exc}")
    return checks
