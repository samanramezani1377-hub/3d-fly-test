"""Coverage-aware lexicographic PPO checkpoint selection.

Missing measurements are never converted to zero error. Tracking errors only
participate after complete command coverage has been demonstrated.
"""
from __future__ import annotations

import math


def checkpoint_selection_score(profile_reports, *, velocity_control=True):
    if not profile_reports:
        raise ValueError("at least one profile report is required")
    mean_return = sum(float(p.get("mean_return", 0.0)) for p in profile_reports) / len(profile_reports)
    mean_saturation = sum(float(p.get("mean_motor_saturation_fraction", 0.0)) for p in profile_reports) / len(profile_reports)
    common = (
        min(float(p.get("success_rate", 0.0)) for p in profile_reports),
        -max(float(p.get("crash_rate", 1.0)) for p in profile_reports),
        min(float(p.get("takeoff_success_rate", 0.0)) for p in profile_reports),
    )
    if not velocity_control:
        return common + (
            1.0, 1.0, 1.0, 0.0, 0.0, -mean_saturation,
            sum(float(p.get("mean_stable_hover_seconds", 0.0)) for p in profile_reports) / len(profile_reports),
        )
    complete_rate = min(float(p.get("full_command_coverage_episode_rate", 0.0)) for p in profile_reports)
    velocity_coverage = min(
        min(1.0, max(0.0, float(p.get("mean_velocity_direction_coverage", 0.0)) / 26.0))
        for p in profile_reports
    )
    yaw_coverage = min(
        min(1.0, max(0.0, float(p.get("mean_yaw_command_coverage", 0.0)) / 30.0))
        for p in profile_reports
    )
    complete_velocity_errors = [
        float(p["mean_complete_episode_velocity_tracking_error_m_s"])
        for p in profile_reports
        if p.get("mean_complete_episode_velocity_tracking_error_m_s") is not None
        and math.isfinite(float(p["mean_complete_episode_velocity_tracking_error_m_s"]))
    ]
    complete_yaw_errors = [
        float(p["mean_complete_episode_yaw_rate_tracking_error_rad_s"])
        for p in profile_reports
        if p.get("mean_complete_episode_yaw_rate_tracking_error_rad_s") is not None
        and math.isfinite(float(p["mean_complete_episode_yaw_rate_tracking_error_rad_s"]))
    ]
    # A large finite penalty keeps JSON standards-compliant and ensures absent
    # full-coverage evidence can never masquerade as a zero tracking error.
    velocity_error_score = (
        -(sum(complete_velocity_errors) / len(complete_velocity_errors))
        if len(complete_velocity_errors) == len(profile_reports) else -1_000_000.0
    )
    yaw_error_score = (
        -(sum(complete_yaw_errors) / len(complete_yaw_errors))
        if len(complete_yaw_errors) == len(profile_reports) else -1_000_000.0
    )
    return common + (
        complete_rate, velocity_coverage, yaw_coverage,
        velocity_error_score, yaw_error_score, -mean_saturation, mean_return,
    )
