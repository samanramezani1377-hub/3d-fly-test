from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics.hover_reference import HoverReferenceController
from fast_physics.vehicle_family import DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles


def test_oracle_reference_controller_can_hold_nominal_hover_for_every_profile():
    for index, profile in enumerate(load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)):
        env = VehicleFamilyEnv(
            vehicle_family_path=DEFAULT_VEHICLE_FAMILY,
            vehicle_id=profile["vehicle_id"],
            max_steps=1000,
            randomize=False,
            curriculum_stage=1,
        )
        try:
            _, reset_info = env.reset(seed=8100 + index)
            plant = env.inner.plant
            controller = HoverReferenceController(plant.params, target_position=env.inner.target)
            stable_steps = 0
            crashed = False
            for _ in range(env.max_steps):
                action = controller.act(plant.state())
                assert action.shape == (4,)
                assert np.all(np.isfinite(action))
                assert np.all((action >= 0.0) & (action <= 1.0))
                _, _, terminated, truncated, info = env.step(action)
                stable = (
                    abs(float(plant.position[2] - env.inner.target[2])) <= 0.15
                    and float(np.linalg.norm(plant.position[:2] - env.inner.target[:2])) <= 0.20
                    and float(np.max(np.abs(plant.attitude[:2]))) <= 0.25
                    and float(np.linalg.norm(plant.velocity)) <= 0.35
                )
                stable_steps = stable_steps + 1 if stable else 0
                crashed = crashed or bool(info.get("crashed", False))
                if terminated or truncated:
                    break
            assert not crashed, f"{profile['vehicle_id']} crashed under oracle hover control"
            assert stable_steps >= 150, (
                f"{profile['vehicle_id']} failed 3 s sustained hover: "
                f"stable_steps={stable_steps}, z={plant.position[2]:.3f}, "
                f"velocity={plant.velocity.tolist()}, attitude={plant.attitude.tolist()}"
            )
            assert reset_info["calibration_status"] == "provisional_estimate"
        finally:
            env.close()


def test_oracle_reference_cancels_known_fixed_wind_drag_for_every_profile():
    for index, profile in enumerate(load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)):
        env = VehicleFamilyEnv(
            vehicle_family_path=DEFAULT_VEHICLE_FAMILY,
            vehicle_id=profile["vehicle_id"],
            max_steps=1000,
            randomize=False,
            curriculum_stage=1,
        )
        try:
            env.reset(seed=9200 + index)
            plant = env.inner.plant
            plant.position[:] = env.inner.target
            hover_omega = np.sqrt(
                plant.params.mass_kg * 9.80665
                / (4.0 * plant.params.thrust_coefficient_n_per_rad_s2
                   * np.asarray(plant.params.motor_effectiveness, dtype=np.float64))
            )
            plant.motor_omega[:] = hover_omega
            plant.wind[:] = np.array([0.5, -0.4, 0.2], dtype=np.float64)
            controller = HoverReferenceController(
                plant.params, target_position=env.inner.target, wind_m_s=plant.wind,
                dt_s=env.inner.dt,
            )
            stable_steps = 0
            crashed = False
            for _ in range(env.max_steps):
                _, _, terminated, truncated, info = env.step(controller.act(plant.state()))
                stable = (
                    abs(float(plant.position[2] - env.inner.target[2])) <= 0.15
                    and float(np.linalg.norm(plant.position[:2] - env.inner.target[:2])) <= 0.20
                    and float(np.max(np.abs(plant.attitude[:2]))) <= 0.25
                    and float(np.linalg.norm(plant.velocity)) <= 0.35
                )
                stable_steps = stable_steps + 1 if stable else 0
                crashed = crashed or bool(info.get("crashed", False))
                if terminated or truncated:
                    break
            assert not crashed, f"{profile['vehicle_id']} crashed with known fixed wind"
            assert stable_steps >= 150, (
                f"{profile['vehicle_id']} failed fixed-wind hover: "
                f"stable_steps={stable_steps}, position={plant.position.tolist()}, "
                f"velocity={plant.velocity.tolist()}, attitude={plant.attitude.tolist()}"
            )
        finally:
            env.close()
