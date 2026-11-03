from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics.hover_reference import HoverReferenceController
from fast_physics.velocity_reference import VelocityYawReferenceController
from fast_physics.vehicle_family import DEFAULT_VEHICLE_FAMILY, VehicleFamilyEnv, load_vehicle_profiles


def test_velocity_teacher_uses_stronger_feedback_when_braking_or_reversing():
    profile = load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)[0]
    env = VehicleFamilyEnv(
        vehicle_family_path=DEFAULT_VEHICLE_FAMILY,
        vehicle_id=profile["vehicle_id"],
        randomize=False,
        curriculum_stage=5,
        velocity_control=True,
    )
    try:
        env.reset(seed=7301)
        params = env.inner.plant.params
        controller = VelocityYawReferenceController(params, dt_s=env.inner.dt)
        state = np.zeros(12, dtype=np.float64)
        state[3] = 1.2
        controller.desired_wrench(
            state, target_position=np.zeros(3),
            desired_velocity_body=np.zeros(3), desired_yaw_rate_rad_s=0.0,
        )
        diag = controller.last_diagnostics
        assert diag["braking_active_axes"][0]
        assert diag["acceleration_requested_world_m_s2"][0] < 0.0
        assert diag["velocity_feedback_gain"][0] > 1.0

        controller.desired_wrench(
            state, target_position=np.zeros(3),
            desired_velocity_body=np.array([-0.5, 0.0, 0.0]),
            desired_yaw_rate_rad_s=0.0,
        )
        assert controller.last_diagnostics["braking_active_axes"][0]
    finally:
        env.close()


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
            assert reset_info["calibration_status"] in {"cad_bemt_estimate", "provisional_estimate"}
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


def test_velocity_sequence_neutral_tail_separates_pure_braking_from_position_return():
    profile = next(
        p for p in load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)
        if p["vehicle_id"] == "smooth2-a2212-custom-prop"
    )
    for mode, should_reset_target in (("braking", True), ("position_return", False)):
        env = VehicleFamilyEnv(
            vehicle_id=profile["vehicle_id"], randomize=False,
            curriculum_stage=5, velocity_control=True,
        )
        try:
            env.reset(seed=7310, options={
                "velocity_command_sequence": [
                    {"velocity_m_s": [0.5, 0.0, 0.0], "yaw_rate_rad_s": 0.0, "duration_steps": 1},
                    {"velocity_m_s": [0.0, 0.0, 0.0], "yaw_rate_rad_s": 0.2, "duration_steps": 1},
                ],
                "target_position": [0.0, 0.0, 2.0],
                "neutral_tail_mode": mode,
            })
            env.inner.plant.position[:] = [1.25, -0.5, 2.4]
            original_target = env.inner.target.copy()
            env.inner.desired_velocity_body[:] = [0.5, 0.0, 0.0]
            # Advance into a yaw-only command: braking mode resets the old
            # translation target, while position_return must preserve it.
            env.inner._velocity_sequence_index = 1
            env.inner._command_steps_remaining = 0
            env.inner._update_velocity_command()
            if should_reset_target:
                np.testing.assert_allclose(env.inner.target, env.inner.plant.position)
            else:
                np.testing.assert_allclose(env.inner.target, original_target)

            # The final neutral tail must retain the same mode semantics.
            env.inner.desired_velocity_body[:] = [0.0, 0.0, 0.0]
            env.inner._velocity_sequence_index = len(env.inner._velocity_command_sequence)
            env.inner._command_steps_remaining = 0
            env.inner._update_velocity_command()
            if should_reset_target:
                np.testing.assert_allclose(env.inner.target, env.inner.plant.position)
            else:
                np.testing.assert_allclose(env.inner.target, original_target)
            assert env.inner.neutral_tail_mode == mode
        finally:
            env.close()


def test_velocity_teacher_caps_position_hold_velocity_for_large_position_error():
    profile = next(
        p for p in load_vehicle_profiles(DEFAULT_VEHICLE_FAMILY)
        if p["vehicle_id"] == "smooth2-a2212-custom-prop"
    )
    env = VehicleFamilyEnv(
        vehicle_id=profile["vehicle_id"], randomize=False,
        curriculum_stage=5, velocity_control=True,
    )
    try:
        env.reset(seed=7311)
        controller = VelocityYawReferenceController(env.inner.plant.params, dt_s=env.inner.dt)
        state = np.zeros(12, dtype=np.float64)
        controller.desired_wrench(
            state, target_position=np.array([100.0, -100.0, 100.0]),
            desired_velocity_body=np.zeros(3), desired_yaw_rate_rad_s=0.0,
        )
        hold = np.asarray(controller.last_diagnostics["position_hold_velocity_limited_m_s"])
        np.testing.assert_allclose(hold, [1.5, -1.5, 0.75])
    finally:
        env.close()
