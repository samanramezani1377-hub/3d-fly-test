from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from fast_physics import AdaptiveDroneEnv, FastQuadrotor, VehicleParameters, load_vehicle_parameters, sample_scenario, OnlineThrustEstimator, randomization_for_stage
from fast_physics.ppo_action import wrench_to_motor_commands
from fast_physics.vehicle_family import load_vehicle_profiles, profile_to_parameters
from fast_physics.velocity_reference import VelocityYawReferenceController
from fast_physics.dynamics import rotation_matrix, _quaternion_from_euler, _rotation_matrix_from_quaternion
from fast_physics.propeller import BladeElementPropeller
from fast_physics.system_identification import estimate_first_order_time_constant, estimate_body_axis_drag_coefficients
from fast_physics.mass_properties import aggregate_mass_properties
from evaluate_robust_ppo import summarize_episode
AIRFRAME = ROOT / "vehicle_models/smooth2_parametric/base_airframe.json"
CALIBRATION = ROOT / "vehicle_models/smooth2_parametric/physics_calibration.json"
RANDOMIZATION = ROOT / "vehicle_models/smooth2_parametric/domain_randomization.json"


def synthetic_params():
    return VehicleParameters(
        mass_kg=0.6,
        center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(0.004, 0.006, 0.004),
        rotor_positions_m=((0.082, 0.082, 0.0), (0.082, -0.082, 0.0),
                           (-0.082, -0.082, 0.0), (-0.082, 0.082, 0.0)),
        spin_signs=(1.0, -1.0, 1.0, -1.0),
        max_rotor_speed_rad_s=1100.0,
        thrust_coefficient_n_per_rad_s2=2.0e-6,
        torque_to_thrust_m=0.012,
    )


def _write_incomplete_calibration(tmp_path):
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    physical = calibration["physical_properties"]
    motor = calibration["motor_propeller"]
    for key in ("takeoff_mass_kg", "center_of_mass_m", "inertia_tensor_kg_m2"):
        physical[key] = None
    for key in ("max_rotor_speed_rad_s", "thrust_coefficient_n_per_rad_s2", "torque_to_thrust_m"):
        motor[key] = None
    path = tmp_path / "incomplete-calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    return path


def test_unknown_calibration_is_not_silently_filled(tmp_path):
    incomplete = _write_incomplete_calibration(tmp_path)
    with pytest.raises(ValueError, match="Physical calibration is incomplete"):
        load_vehicle_parameters(AIRFRAME, incomplete)


def test_synthetic_calibration_requires_explicit_opt_in(tmp_path):
    incomplete = _write_incomplete_calibration(tmp_path)
    params, metadata = load_vehicle_parameters(
        AIRFRAME, incomplete, allow_synthetic_calibration=True
    )
    assert metadata["synthetic_calibration_used"] is True
    assert params.mass_kg == pytest.approx(0.6)
    assert params.battery_nominal_voltage_v == pytest.approx(11.1)
    assert params.battery_capacity_ah == pytest.approx(2.2)


def test_airframe_geometry_is_fixed_and_four_rotors_are_present():
    data = json.loads(AIRFRAME.read_text(encoding="utf-8"))
    assert data["vehicle_id"] == "smooth2-parametric-v1"
    assert len(data["rotors"]) == 4
    assert all(np.allclose(r["axis_body"], [0, 0, 1]) for r in data["rotors"])
    assert data["geometry"]["model_scale"] == pytest.approx(1.414028)
    assert data["uncertain_physical_properties"]["takeoff_mass_kg"] is None


def test_dynamics_hover_force_balances_weight_at_expected_command():
    params = synthetic_params()
    plant = FastQuadrotor(params)
    plant.reset(position=(0, 0, 1))
    # A full-scale command is not a hover command; this test checks finite response.
    state, telemetry = plant.step(np.ones(4, dtype=np.float32) * 0.5)
    assert np.all(np.isfinite(state))
    assert len(telemetry["rotor_thrust_n"]) == 4
    assert telemetry["total_thrust_n"] > 0


def test_domain_randomization_is_reproducible_and_changes_mass_cg_inertia():
    params = synthetic_params()
    config = json.loads(RANDOMIZATION.read_text(encoding="utf-8"))
    a, details_a = sample_scenario(params, np.random.default_rng(123), config)
    b, details_b = sample_scenario(params, np.random.default_rng(123), config)
    assert a == b
    assert details_a == details_b
    assert a.mass_kg != params.mass_kg
    assert a.center_of_mass_m != params.center_of_mass_m
    assert a.inertia_kg_m2 != params.inertia_kg_m2


def test_env_observation_contains_history_but_not_privileged_parameters():
    env = AdaptiveDroneEnv(
        params=synthetic_params(), randomization_path=RANDOMIZATION,
        history_length=4, max_steps=20,
    )
    try:
        obs, info = env.reset(seed=9)
        assert obs.shape == (66,)
        assert info["policy_observation_excludes_privileged_parameters"] is True
        assert "mass_kg" not in info
        next_obs, reward, terminated, truncated, step_info = env.step(np.full(4, 0.5))
        assert next_obs.shape == (66,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "mass_kg" in step_info
    finally:
        env.close()


def test_randomized_env_seed_repeats_scenario():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION)
    try:
        _, info_a = env.reset(seed=77)
        _, info_b = env.reset(seed=77)
        assert info_a["scenario"] == info_b["scenario"]
        assert info_a["wind_m_s"] == info_b["wind_m_s"]
    finally:
        env.close()


def test_action_shape_is_validated():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION)
    try:
        env.reset(seed=1)
        with pytest.raises(ValueError, match="four motor commands"):
            env.step(np.zeros(3))
    finally:
        env.close()


def test_full_inertia_tensor_is_supported_and_positive_definite():
    params = synthetic_params()
    full = VehicleParameters(**{**params.__dict__, "inertia_kg_m2": (0.02, 0.001, 0.0, 0.001, 0.035, 0.0, 0.0, 0.0, 0.02)})
    full.validate()
    assert full.inertia_matrix.shape == (3, 3)
    invalid = VehicleParameters(**{**params.__dict__, "inertia_kg_m2": (1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0)})
    with pytest.raises(ValueError):
        invalid.validate()


def test_online_estimator_learns_effective_collective_response_from_observations():
    estimator = OnlineThrustEstimator()
    dt = 0.02
    vz = 0.0
    estimator.reset(vz)
    for index in range(600):
        command = 0.15 + 0.75 * ((index % 30) / 29.0)
        acceleration = 0.4 + 14.0 * command ** 2 - 9.80665
        vz += acceleration * dt
        estimator.update(np.full(4, command), vz, dt)
    result = estimator.diagnostics()
    assert np.isfinite(result["effective_lift_gain_m_s2"])
    assert result["samples"] == 600
    assert result["confidence"] > 0.0
    assert abs(result["effective_lift_gain_m_s2"] - 14.0) < 2.0


def test_curriculum_stage_one_is_nominal_and_stage_four_uses_wider_bounds():
    config = json.loads(RANDOMIZATION.read_text(encoding="utf-8"))
    stage1 = randomization_for_stage(config, 1)
    stage4 = randomization_for_stage(config, 4)
    assert stage1["mass"]["scale"] == [1.0, 1.0]
    assert stage1["environment"]["wind_component_m_s"] == [0.0, 0.0]
    assert stage4["mass"]["scale"] == [0.80, 1.10]
    assert stage4["inertia"]["scale"] == [0.70, 1.40]
    assert stage4["battery"]["voltage_factor"] == [0.90, 1.0]
    assert stage4["motor"]["effectiveness"] == [0.95, 1.10]
    assert stage4["environment"]["wind_component_m_s"] == [-2.0, 2.0]


def test_env_exposes_estimator_context_but_not_ground_truth_parameters():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION)
    try:
        obs, _ = env.reset(seed=12)
        assert obs.shape == (66,)
        assert np.all(np.isfinite(obs[-2:]))
        _, _, _, _, info = env.step(np.full(4, 0.4))
        assert "online_estimator" in info
        assert "mass_kg" not in info["online_estimator"]
    finally:
        env.close()


def test_full_cg_offset_is_randomized_in_all_three_body_axes():
    params = synthetic_params()
    config = json.loads(RANDOMIZATION.read_text(encoding="utf-8"))
    sampled, details = sample_scenario(params, np.random.default_rng(31), config)
    assert len(details["center_of_mass_offset_m"]) == 3
    assert sampled.center_of_mass_m[2] != params.center_of_mass_m[2]


def test_sensor_noise_is_reproducible_with_same_seed():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION)
    try:
        obs_a, _ = env.reset(seed=202)
        obs_b, _ = env.reset(seed=202)
        assert np.allclose(obs_a, obs_b)
    finally:
        env.close()


def test_held_out_evaluator_requires_sustained_hover_and_no_crash():
    stable_samples = [
        {
            "taken_off": True, "stable": True, "crashed": False,
            "altitude_error_m": 0.05, "horizontal_error_m": 0.05,
        }
        for _ in range(300)
    ]
    good = summarize_episode(stable_samples, 500.0)
    assert good["hover_success"] is True
    assert good["crashed"] is False

    crashed_samples = stable_samples[:-1] + [{
        "taken_off": True, "stable": False, "crashed": True,
        "altitude_error_m": 0.5, "horizontal_error_m": 0.5,
    }]
    bad = summarize_episode(crashed_samples, 100.0)
    assert bad["hover_success"] is False
    assert bad["crashed"] is True


def test_online_estimator_uses_sensor_measurement_not_simulator_truth():
    env = AdaptiveDroneEnv(params=synthetic_params(), randomization_path=RANDOMIZATION, randomize=False)
    try:
        env.reset(seed=4)
        original_measure = env._measure_state

        def biased_sensor(state):
            measured = original_measure(state)
            measured[5] += 0.123
            return measured

        env._measure_state = biased_sensor
        env.step(np.full(4, 0.4))
        assert env.estimator.previous_vertical_velocity == pytest.approx(
            env.plant.velocity[2] + 0.123
        )
    finally:
        env.close()



def test_smooth2_cad_frame_is_converted_to_forward_left_up_body_frame():
    params, metadata = load_vehicle_parameters(
        AIRFRAME, CALIBRATION, allow_synthetic_calibration=True
    )
    assert metadata["source_coordinate_frame"] == "CAD_EXPORT_Z_UP"
    assert metadata["physical_coordinate_frame"] == "CAD_EXPORT_Z_UP"
    assert metadata["body_coordinate_frame"] == "BODY_FORWARD_LEFT_UP"
    expected = np.array([
        [-0.125141478, 0.125141478, 0.0],
        [0.125141478, 0.125141478, 0.0],
        [0.125141478, -0.125141478, 0.0],
        [-0.125141478, -0.125141478, 0.0],
    ])
    assert np.allclose(params.rotor_positions_m, expected)
    # The 90-degree frame rotation swaps CAD X/Y principal inertias.
    assert np.allclose(params.inertia_matrix, np.diag([0.003935447909363858, 0.004031247909363858, 0.007722739818727713]))


def test_smooth2_family_profile_coordinate_conversion_is_idempotent():
    profiles = load_vehicle_profiles(ROOT / "vehicle_models/family/vehicle_profiles.json")
    profile = next(p for p in profiles if p["vehicle_id"] == "smooth2-a2212-custom-prop")
    first = profile_to_parameters(profile)
    second = profile_to_parameters(profile)
    assert profile["coordinate_frame"] == "BODY_FORWARD_LEFT_UP"
    assert profile["source_coordinate_frame"] == "CAD_EXPORT_Z_UP"
    assert first.rotor_positions_m == second.rotor_positions_m
    assert first.inertia_kg_m2 == second.inertia_kg_m2
    assert np.allclose(first.rotor_positions_m[0], [-0.125141478, 0.125141478, 0.0])


def test_body_yaw_velocity_axes_match_forward_left_and_world_heading():
    forward = np.array([1.0, 0.0, 0.0])
    left = np.array([0.0, 1.0, 0.0])
    assert np.allclose(AdaptiveDroneEnv._body_yaw_velocity_to_world(forward, 0.0), [1, 0, 0])
    assert np.allclose(AdaptiveDroneEnv._body_yaw_velocity_to_world(left, 0.0), [0, 1, 0])
    assert np.allclose(
        AdaptiveDroneEnv._body_yaw_velocity_to_world(forward, np.pi / 2),
        [0, 1, 0], atol=1e-7,
    )
    world = np.array([0.0, 2.0, 0.3])
    body = AdaptiveDroneEnv._world_velocity_to_body_yaw(world, np.pi / 2)
    assert np.allclose(body, [2.0, 0.0, 0.3], atol=1e-7)


def test_rotor_mixer_produces_correct_longitudinal_lateral_and_yaw_reactions():
    params = VehicleParameters(
        mass_kg=0.6, center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(0.004, 0.006, 0.004),
        rotor_positions_m=(
            (-0.082, 0.082, 0.0), (0.082, 0.082, 0.0),
            (0.082, -0.082, 0.0), (-0.082, -0.082, 0.0),
        ),
        spin_signs=(1.0, -1.0, 1.0, -1.0),
        max_rotor_speed_rad_s=1100.0,
        thrust_coefficient_n_per_rad_s2=2.0e-6,
        torque_to_thrust_m=0.012,
    )
    hover_thrust = params.mass_kg * 9.80665 / 4.0
    hover_omega = np.sqrt(hover_thrust / params.thrust_coefficient_n_per_rad_s2)

    def run_wrench(wrench):
        plant = FastQuadrotor(params, dt=0.01)
        plant.reset(position=(0.0, 0.0, 1.0))
        plant.motor_omega[:] = hover_omega
        commands = wrench_to_motor_commands(wrench, params)
        states = []
        for _ in range(12):
            state, telemetry = plant.step(commands)
            states.append(state.copy())
        return np.asarray(states), telemetry

    pitch_states, pitch_telemetry = run_wrench([params.mass_kg * 9.80665, 0.0, 0.004, 0.0])
    roll_states, roll_telemetry = run_wrench([params.mass_kg * 9.80665, 0.004, 0.0, 0.0])
    yaw_states, yaw_telemetry = run_wrench([params.mass_kg * 9.80665, 0.0, 0.0, 0.001])
    assert pitch_telemetry["body_torque_nm"][1] > 0.0
    assert roll_telemetry["body_torque_nm"][0] > 0.0
    assert yaw_telemetry["body_torque_nm"][2] > 0.0
    assert pitch_states[-1, 3] > 0.0  # positive pitch moves nose-forward (+body X)
    assert roll_states[-1, 4] < 0.0   # positive roll accelerates toward body-right (-Y)
    assert yaw_states[-1, 11] > 0.0


def test_velocity_teacher_requests_correct_pitch_roll_and_yaw_directions():
    params = VehicleParameters(
        mass_kg=0.6, center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(0.004, 0.006, 0.004),
        rotor_positions_m=(
            (-0.082, 0.082, 0.0), (0.082, 0.082, 0.0),
            (0.082, -0.082, 0.0), (-0.082, -0.082, 0.0),
        ),
        spin_signs=(1.0, -1.0, 1.0, -1.0),
        max_rotor_speed_rad_s=1100.0,
        thrust_coefficient_n_per_rad_s2=2.0e-6,
        torque_to_thrust_m=0.012,
    )
    teacher = VelocityYawReferenceController(params)
    state = np.zeros(12)
    state[2] = 1.0
    target = np.array([0.0, 0.0, 1.0])
    forward = teacher.desired_wrench(
        state, target_position=target,
        desired_velocity_body=np.array([0.5, 0.0, 0.0]),
        desired_yaw_rate_rad_s=0.0,
    )
    left = teacher.desired_wrench(
        state, target_position=target,
        desired_velocity_body=np.array([0.0, 0.5, 0.0]),
        desired_yaw_rate_rad_s=0.0,
    )
    turn_left = teacher.desired_wrench(
        state, target_position=target,
        desired_velocity_body=np.zeros(3), desired_yaw_rate_rad_s=0.3,
    )
    turn_right = teacher.desired_wrench(
        state, target_position=target,
        desired_velocity_body=np.zeros(3), desired_yaw_rate_rad_s=-0.3,
    )
    assert forward[2] > 0.0
    assert left[1] < 0.0
    assert turn_left[3] > 0.0 and turn_right[3] < 0.0


def test_quaternion_plant_integration_stays_finite_under_combined_body_rates():
    params = synthetic_params()
    plant = FastQuadrotor(params, dt=0.01)
    plant.reset(position=(0.0, 0.0, 1.0))
    plant.attitude[:] = [0.3, -0.25, 0.7]
    plant.angular_velocity[:] = [1.5, -1.2, 2.0]
    for _ in range(500):
        state, telemetry = plant.step(np.full(4, 0.5))
        assert np.all(np.isfinite(state))
        assert np.isclose(np.linalg.norm(telemetry["attitude_quaternion_wxyz"]), 1.0, atol=1e-8)


def test_body_axis_drag_rotates_with_heading_and_opposes_airflow():
    params = VehicleParameters(
        mass_kg=0.6, center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(0.004, 0.006, 0.004),
        rotor_positions_m=((0.082, 0.082, 0.0), (0.082, -0.082, 0.0),
                           (-0.082, -0.082, 0.0), (-0.082, 0.082, 0.0)),
        spin_signs=(1.0, -1.0, 1.0, -1.0),
        max_rotor_speed_rad_s=1100.0,
        thrust_coefficient_n_per_rad_s2=2.0e-6,
        torque_to_thrust_m=0.012,
        linear_drag_body_kg_s=(1.0, 2.0, 3.0),
        quadratic_drag_body_kg_m=(0.0, 0.0, 0.0),
    )
    plant = FastQuadrotor(params, dt=0.01)
    plant.reset(position=(0.0, 0.0, 1.0))
    plant.attitude[2] = np.pi / 2
    plant.velocity[:] = [0.0, 1.0, 0.0]  # body-forward airspeed at yaw=+90 degrees
    _, telemetry = plant.step(np.zeros(4))
    assert np.allclose(telemetry["drag_world_n"], [0.0, -1.0, 0.0], atol=1e-7)


def test_quaternion_rotation_matches_existing_euler_rotation_convention():
    angles = np.array([0.31, -0.27, 1.12])
    quaternion = _quaternion_from_euler(angles)
    assert np.allclose(
        _rotation_matrix_from_quaternion(quaternion),
        rotation_matrix(angles), atol=1e-10,
    )


def test_optional_rotor_gyroscopic_coupling_is_applied():
    params = VehicleParameters(
        mass_kg=0.6, center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(0.004, 0.006, 0.004),
        rotor_positions_m=((0.082, 0.082, 0.0), (0.082, -0.082, 0.0),
                           (-0.082, -0.082, 0.0), (-0.082, 0.082, 0.0)),
        spin_signs=(1.0, -1.0, 1.0, -1.0),
        max_rotor_speed_rad_s=1100.0,
        thrust_coefficient_n_per_rad_s2=2.0e-6,
        torque_to_thrust_m=0.012,
        rotor_inertia_kg_m2=1.0e-5,
    )
    plant = FastQuadrotor(params, dt=0.01)
    plant.reset(position=(0.0, 0.0, 1.0))
    plant.motor_omega[:] = [900.0, 800.0, 900.0, 800.0]
    plant.angular_velocity[:] = [1.0, 0.0, 0.0]
    commands = plant.motor_omega / params.max_rotor_speed_rad_s
    _, telemetry = plant.step(commands)
    assert telemetry["rotor_gyroscopic_torque_nm"][1] < 0.0


def test_a2212_catalog_reference_is_recorded_without_claiming_physical_calibration():
    motor = json.loads(
        (ROOT / "vehicle_models/components/a2212_1000kv_motor.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    assert motor["identity"]["kv_rpm_per_volt"] == 1000
    assert motor["confirmed_or_repeated_catalog_values"]["no_load_current_a_at_10v"] == pytest.approx(0.5)
    assert motor["confirmed_or_repeated_catalog_values"]["internal_resistance_ohm_catalog"] == pytest.approx(0.09)
    assert len(motor["published_static_test_points"]) >= 2
    assert calibration["motor_propeller"]["kv_rpm_per_volt"] == 1000
    assert calibration["motor_propeller"]["max_rotor_speed_rad_s"] == pytest.approx(1051.8)
    assert calibration["motor_propeller"]["thrust_coefficient_n_per_rad_s2"] == pytest.approx(5.53666e-6)
    assert calibration["motor_propeller"]["torque_to_thrust_m"] == pytest.approx(0.01294)
    assert calibration["motor_propeller"]["motor_time_constant_s"] == pytest.approx(0.045)
    assert calibration["motor_propeller"]["electrical_model_status"] == "bemt_and_generic_motor_screening_estimate_not_bench_calibrated"
    params, metadata = load_vehicle_parameters(AIRFRAME, CALIBRATION)
    assert metadata["synthetic_calibration_used"] is False


def test_electrical_telemetry_estimates_current_and_battery_consumption_without_claiming_voltage_sag():
    params = replace(
        synthetic_params(),
        battery_nominal_voltage_v=11.1,
        battery_capacity_ah=2.2,
    )
    plant = FastQuadrotor(params)
    reference_plant = FastQuadrotor(synthetic_params())
    plant.reset(position=(0.0, 0.0, 1.0))
    reference_plant.reset(position=(0.0, 0.0, 1.0))
    telemetry = None
    for _ in range(150):
        state_with_diagnostics, telemetry = plant.step(np.full(4, 0.7, dtype=np.float32))
        state_without_diagnostics, _ = reference_plant.step(np.full(4, 0.7, dtype=np.float32))
        assert np.allclose(state_with_diagnostics, state_without_diagnostics, atol=1e-12)
    assert telemetry["estimated_battery_current_a"] > 0.0
    assert len(telemetry["estimated_motor_current_a"]) == 4
    assert telemetry["estimated_electrical_power_w"] > 0.0
    assert telemetry["battery_consumed_ah_estimate"] > 0.0
    assert 0.0 < telemetry["battery_state_of_charge_estimate"] < 1.0
    assert telemetry["battery_voltage_sag_modelled"] is False
    assert telemetry["ESC_current_limit_modelled"] is False
    assert "excludes_no_load" in telemetry["electrical_estimate_status"]
    plant.reset(position=(0.0, 0.0, 1.0))
    _, reset_telemetry = plant.step(np.zeros(4, dtype=np.float32))
    assert reset_telemetry["battery_consumed_ah_estimate"] == 0.0
    assert reset_telemetry["battery_state_of_charge_estimate"] == pytest.approx(1.0)


def test_vehicle_profiles_expose_nominal_battery_capacity_for_telemetry():
    for profile in load_vehicle_profiles():
        params = profile_to_parameters(profile)
        assert params.battery_nominal_voltage_v == pytest.approx(profile["propulsion"]["nominal_battery_voltage_v"])
        assert params.battery_capacity_ah == pytest.approx(profile["propulsion"]["battery_capacity_ah"])
        assert profile["propulsion"]["battery_internal_resistance_ohm"] is None
        assert profile["propulsion"]["esc_current_limit_a_per_motor"] is None


def test_first_order_motor_time_constant_can_be_identified_from_rpm_step_log():
    time = np.linspace(0.0, 0.5, 251)
    target = np.full_like(time, 800.0)
    tau = 0.045
    measured = target * (1.0 - np.exp(-time / tau))
    result = estimate_first_order_time_constant(time, target, measured)
    assert result["motor_time_constant_s"] == pytest.approx(tau, rel=0.03)
    assert result["fit_r_squared"] > 0.99
    assert result["scope"] == "motor_speed_response_only_no_thrust_or_propeller_inference"


def test_component_mass_properties_use_parallel_axis_theorem():
    result = aggregate_mass_properties([
        {"component_id": "left", "mass_kg": 1.0, "center_m": [-0.5, 0.0, 0.0]},
        {"component_id": "right", "mass_kg": 1.0, "center_m": [0.5, 0.0, 0.0]},
    ])
    assert result["takeoff_mass_kg"] == pytest.approx(2.0)
    assert result["center_of_mass_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert np.asarray(result["inertia_tensor_kg_m2"]) == pytest.approx(np.diag([0.0, 0.5, 0.5]))
    assert result["coordinate_frame"] == "CAD_EXPORT_Z_UP"


def test_body_axis_drag_identification_recovers_linear_and_quadratic_coefficients():
    rng = np.random.default_rng(19)
    velocity = rng.uniform(-8.0, 8.0, size=(300, 3))
    expected_linear = np.array([0.12, 0.20, 0.31])
    expected_quadratic = np.array([0.018, 0.027, 0.041])
    force = (
        -velocity * expected_linear[None, :]
        -np.abs(velocity) * velocity * expected_quadratic[None, :]
    )
    result = estimate_body_axis_drag_coefficients(velocity, force)
    assert result["linear_drag_body_kg_s"] == pytest.approx(expected_linear, rel=1e-8)
    assert result["quadratic_drag_body_kg_m"] == pytest.approx(expected_quadratic, rel=1e-8)
    assert result["fit_r_squared_by_axis"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-10)


def test_smooth2_runtime_bemt_uses_cad_blade_stations_and_reports_model_status():
    params, metadata = load_vehicle_parameters(AIRFRAME, CALIBRATION)
    assert params.propeller_bemt is not None
    assert metadata["synthetic_calibration_used"] is False
    propeller = BladeElementPropeller(params.propeller_bemt)
    thrust, torque = propeller.evaluate(params.max_rotor_speed_rad_s)
    assert thrust == pytest.approx(6.125, rel=0.12)
    assert torque == pytest.approx(0.07927, rel=0.15)
    assert propeller.evaluate(0.0) == (0.0, 0.0)

    plant = FastQuadrotor(params)
    plant.motor_omega[:] = params.max_rotor_speed_rad_s
    _, telemetry = plant.step(np.ones(4, dtype=np.float64))
    assert telemetry["thrust_model_status"].startswith("runtime_CAD_station_BEMT")
    assert telemetry["total_thrust_n"] == pytest.approx(4.0 * thrust, rel=0.03)
    assert len(telemetry["rotor_shaft_torque_nm"]) == 4
