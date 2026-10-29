"""Regression tests for hover reward shaping and safe checkpoint ranking."""
from __future__ import annotations

import unittest

import numpy as np

from training.native_env import NativeSkySimEnv
from training.run_five_stage import rank as chained_rank
from training.select_checkpoint import rank as checkpoint_rank


class RewardDesignTests(unittest.TestCase):
    def reward(self, position, *, has_taken_off, ground_wait_steps=0, velocity=(0, 0, 0)):
        reward, components, crashed, reason = NativeSkySimEnv.calculate_reward(
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64),
            attitude=np.zeros(3),
            angular_velocity=np.zeros(3),
            target=np.asarray([0.0, 3.0, 0.0]),
            action=np.zeros(4),
            throttle=0.35,
            throttle_center=0.35,
            has_taken_off=has_taken_off,
            ground_wait_steps=ground_wait_steps,
        )
        return reward, components, crashed, reason

    def test_target_hover_rewards_more_than_grounded_state(self):
        target_reward, _, target_crashed, _ = self.reward(
            [0.0, 3.0, 0.0], has_taken_off=True
        )
        ground_reward, _, ground_crashed, _ = self.reward(
            [0.0, 0.16, 0.0], has_taken_off=False, ground_wait_steps=100
        )
        self.assertFalse(target_crashed)
        self.assertFalse(ground_crashed)
        self.assertGreater(target_reward, ground_reward)
        self.assertGreater(ground_reward, 0.0)

    def test_crash_is_worse_than_safe_grounded_state(self):
        ground_reward, _, _, _ = self.reward(
            [0.0, 0.16, 0.0], has_taken_off=False, ground_wait_steps=100
        )
        crash_reward, _, crashed, reason = self.reward(
            [0.0, 0.21, 0.0], has_taken_off=True
        )
        self.assertTrue(crashed)
        self.assertEqual(reason, "ground_contact_after_takeoff")
        self.assertLess(crash_reward, ground_reward)

    def test_excess_velocity_reduces_reward(self):
        calm, _, _, _ = self.reward([0.0, 3.0, 0.0], has_taken_off=True)
        fast, _, _, _ = self.reward(
            [0.0, 3.0, 0.0], has_taken_off=True, velocity=(4.0, 0.0, 4.0)
        )
        self.assertGreater(calm, fast)

    def test_checkpoint_rank_prefers_safe_model_over_takeoff_then_crash(self):
        safe_grounded = {
            "hover_success_rate": 0.0,
            "takeoff_success_rate": 0.0,
            "crash_rate": 0.0,
            "mean_absolute_final_altitude_error_m": 2.85,
            "mean_hover_stable_fraction": 0.0,
            "mean_absolute_vertical_speed_during_flight_mps": 0.0,
            "mean_return": 100.0,
        }
        takeoff_then_crash = {
            **safe_grounded,
            "takeoff_success_rate": 1.0,
            "crash_rate": 1.0,
            "mean_absolute_final_altitude_error_m": 2.5,
            "mean_absolute_vertical_speed_during_flight_mps": 0.3,
            "mean_return": -800.0,
        }
        self.assertGreater(chained_rank(safe_grounded), chained_rank(takeoff_then_crash))
        self.assertGreater(checkpoint_rank(safe_grounded), checkpoint_rank(takeoff_then_crash))

    def test_stable_hover_always_outranks_non_hover(self):
        baseline = {
            "hover_success_rate": 0.0,
            "takeoff_success_rate": 0.0,
            "crash_rate": 0.0,
            "mean_absolute_final_altitude_error_m": 2.85,
            "mean_hover_stable_fraction": 0.0,
            "mean_absolute_vertical_speed_during_flight_mps": 0.0,
            "mean_return": 100.0,
        }
        stable = {
            **baseline,
            "hover_success_rate": 1.0,
            "takeoff_success_rate": 1.0,
            "mean_absolute_final_altitude_error_m": 0.1,
            "mean_hover_stable_fraction": 0.95,
            "mean_return": 2000.0,
        }
        self.assertGreater(chained_rank(stable), chained_rank(baseline))
        self.assertGreater(checkpoint_rank(stable), checkpoint_rank(baseline))


if __name__ == "__main__":
    unittest.main()
