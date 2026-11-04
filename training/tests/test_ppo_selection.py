from training.ppo_selection import checkpoint_selection_score


def report(*, success=0.0, crash=1.0, takeoff=1.0, complete=0.0,
           velocity_coverage=0.0, yaw_coverage=0.0, velocity_error=None,
           yaw_error=None, saturation=0.0, reward=0.0):
    return {
        "success_rate": success,
        "crash_rate": crash,
        "takeoff_success_rate": takeoff,
        "full_command_coverage_episode_rate": complete,
        "mean_velocity_direction_coverage": velocity_coverage,
        "mean_yaw_command_coverage": yaw_coverage,
        "mean_complete_episode_velocity_tracking_error_m_s": velocity_error,
        "mean_complete_episode_yaw_rate_tracking_error_rad_s": yaw_error,
        "mean_motor_saturation_fraction": saturation,
        "mean_return": reward,
    }


def test_missing_tracking_samples_are_not_scored_as_zero_error():
    missing = checkpoint_selection_score([report()])
    measured = checkpoint_selection_score([report(
        complete=1.0, velocity_coverage=26, yaw_coverage=30,
        velocity_error=0.4, yaw_error=0.2,
    )])
    assert measured > missing
    assert missing[6] == -1_000_000.0
    assert missing[7] == -1_000_000.0


def test_full_command_coverage_precedes_tracking_error_and_return():
    partial_but_low_error = report(
        complete=0.0, velocity_coverage=2, yaw_coverage=2,
        velocity_error=0.01, yaw_error=0.01, reward=1000,
    )
    full_coverage = report(
        complete=1.0, velocity_coverage=26, yaw_coverage=30,
        velocity_error=0.8, yaw_error=0.4, reward=-1000,
    )
    assert checkpoint_selection_score([full_coverage]) > checkpoint_selection_score([partial_but_low_error])


def test_crash_and_success_remain_higher_priority_than_coverage():
    safe = report(success=0.0, crash=0.5, complete=0.0)
    crashed = report(success=0.0, crash=1.0, complete=1.0)
    assert checkpoint_selection_score([safe]) > checkpoint_selection_score([crashed])


def test_score_is_finite_and_json_serializable_for_missing_metrics():
    import json
    score = checkpoint_selection_score([report()])
    json.dumps(score, allow_nan=False)

def test_tracking_errors_require_full_coverage_evidence_for_every_profile():
    first = report(complete=1.0, velocity_coverage=26, yaw_coverage=30,
                   velocity_error=0.2, yaw_error=0.1)
    second = report(complete=0.0, velocity_coverage=3, yaw_coverage=3)
    score = checkpoint_selection_score([first, second])
    assert score[6] == -1_000_000.0
    assert score[7] == -1_000_000.0
