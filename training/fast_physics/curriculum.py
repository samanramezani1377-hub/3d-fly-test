"""Translate a named curriculum stage into bounded randomization settings."""
from __future__ import annotations

from copy import deepcopy


def randomization_for_stage(config: dict, stage_number: int) -> dict:
    stages = config.get("curriculum", [])
    stage = next((item for item in stages if int(item["stage"]) == int(stage_number)), None)
    if stage is None:
        available = [int(item["stage"]) for item in stages]
        raise ValueError(f"unknown curriculum stage {stage_number}; available stages: {available}")
    result = deepcopy(config)
    result["mass"]["scale"] = stage["mass_scale"]
    center_of_mass = result.setdefault("center_of_mass", {})
    if "cg_offset_arm_fraction" in stage:
        # Universal vehicle-family schema: scale CG uncertainty by mean rotor-arm radius.
        center_of_mass["offset_arm_fraction"] = stage["cg_offset_arm_fraction"]
        center_of_mass.pop("offset_m", None)
    elif "cg_offset_m" in stage:
        # Legacy Smooth2 schema: preserve the configured absolute metre bounds.
        center_of_mass["offset_m"] = stage["cg_offset_m"]
        center_of_mass.pop("offset_arm_fraction", None)
    else:
        raise ValueError(
            f"curriculum stage {stage_number} must define cg_offset_arm_fraction or cg_offset_m"
        )
    result["inertia"]["scale"] = stage["inertia_scale"]
    result["motor"]["effectiveness"] = stage["motor_effectiveness"]
    result["battery"]["voltage_factor"] = stage.get("battery_voltage_factor", [1.0, 1.0])
    result["aerodynamics"]["drag_scale"] = stage.get("drag_scale", [1.0, 1.0])
    result["environment"]["wind_component_m_s"] = stage["wind_m_s"]
    result["active_curriculum_stage"] = {
        "stage": int(stage["stage"]),
        "name": stage["name"],
    }
    return result
