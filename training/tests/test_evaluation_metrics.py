from training.evaluate import is_hover_success, summarize_flight_metrics


def test_flight_metrics_exclude_all_grounded_samples():
    samples = [
        {
            "has_taken_off": False,
            "altitude_error_m": -2.84,
            "horizontal_error_m": 0.0,
            "velocity_xyz_mps": [0.0, 0.0, 0.0],
            "hover_stable": False,
        },
        {
            "has_taken_off": False,
            "altitude_error_m": -2.80,
            "horizontal_error_m": 0.1,
            "velocity_xyz_mps": [0.0, 0.0, 0.0],
            "hover_stable": False,
        },
    ]
    metrics = summarize_flight_metrics(samples)
    assert metrics["flight_steps"] == 0
    assert metrics["hover_stable_fraction"] == 0.0
    assert metrics["mean_absolute_altitude_error_m"] is None
    assert metrics["mean_horizontal_error_m"] is None
    assert metrics["mean_absolute_vertical_speed_mps"] is None


def test_flight_metrics_use_only_samples_after_takeoff():
    samples = [
        {
            "has_taken_off": False,
            "altitude_error_m": -2.84,
            "horizontal_error_m": 9.0,
            "velocity_xyz_mps": [0.0, 0.0, 0.0],
            "hover_stable": False,
        },
        {
            "has_taken_off": True,
            "altitude_error_m": -0.4,
            "horizontal_error_m": 0.2,
            "velocity_xyz_mps": [0.0, 0.4, 0.0],
            "hover_stable": True,
        },
        {
            "has_taken_off": True,
            "altitude_error_m": -0.2,
            "horizontal_error_m": 0.4,
            "velocity_xyz_mps": [0.0, -0.2, 0.0],
            "hover_stable": False,
        },
    ]
    metrics = summarize_flight_metrics(samples)
    assert metrics["flight_steps"] == 2
    assert metrics["hover_stable_steps"] == 1
    assert metrics["hover_stable_fraction"] == 0.5
    assert abs(metrics["mean_absolute_altitude_error_m"] - 0.3) < 1e-9
    assert abs(metrics["mean_horizontal_error_m"] - 0.3) < 1e-9
    assert abs(metrics["mean_absolute_vertical_speed_mps"] - 0.3) < 1e-9


def test_hover_success_requires_takeoff_survival_and_five_seconds_stable():
    metrics = {
        "flight_steps": 400,
        "hover_stable_steps": 300,
        "hover_stable_fraction": 0.75,
    }
    assert not is_hover_success(
        metrics, crashed=False, takeoff_seen=False, min_hover_stable_steps=250
    )
    assert not is_hover_success(
        metrics, crashed=True, takeoff_seen=True, min_hover_stable_steps=250
    )
    assert not is_hover_success(
        metrics, crashed=False, takeoff_seen=True, min_hover_stable_steps=301
    )
    metrics["hover_stable_fraction"] = 0.8
    assert is_hover_success(
        metrics, crashed=False, takeoff_seen=True, min_hover_stable_steps=250
    )

def test_production_pipeline_stays_at_four_environments():
    from training.run_five_stage import PARALLEL_ENVS, STEPS

    assert PARALLEL_ENVS == 4
    assert STEPS == 100_000


def test_quality_gate_does_not_claim_comparability_without_flight_data():
    from training.benchmark_parallel_envs_quality import quality_gate

    baseline = {
        "hover_success_rate": 0.0,
        "crash_rate": 0.0,
        "takeoff_success_rate": 0.0,
        "mean_absolute_altitude_error_during_flight_m": None,
        "mean_absolute_vertical_speed_during_flight_mps": None,
    }
    candidate = dict(baseline)
    result = quality_gate(candidate, baseline)
    assert not result["checks"]["altitude_error_available_for_comparison"]
    assert not result["checks"]["vertical_speed_available_for_comparison"]
    assert not result["passed"]
