# Smooth2 body-relative velocity and yaw-rate control

## Control contract

The flight-stability policy receives four normalized command values:

- **Forward velocity (m/s):** positive along the aircraft nose, negative backward.
- **Left velocity (m/s):** positive to the left of the nose, negative to the right.
- **Vertical velocity (m/s):** positive world-up (climb), negative world-down (descent).
- **Yaw rate (rad/s):** positive turns the nose left (counter-clockwise viewed from above); negative turns it right.

Forward and left axes rotate with the current yaw. Vertical velocity remains aligned with world-up. Yaw rate is an independent angular-rate reference, so the aircraft can rotate in place with all three translational commands set to zero, or rotate while translating in any combination of axes. This is a yaw-aligned command frame, not the fully tilted rigid-body frame.

The software-training translational command envelope is [1.5, 1.5, 0.75] m/s. The yaw-rate envelope grows by stage: 20, 35, 50, 70, and 90 degrees/second. Internally yaw rate is represented in radians/second. The virtual altitude reference is bounded between 0.60 m and 8.0 m; a vertical command toward a reached bound is neutralized. These are simulation curriculum limits only, not measured safe limits for the physical aircraft.

## Camera-model interface

The camera/perception model remains a separate component. Its future output should be a desired command tuple:
`[forward_m_s, left_m_s, up_m_s, yaw_rate_rad_s]`.
The first three values are clamped to the translational envelope and the last to the active yaw-rate envelope. It must not output motor commands. The flight controller remains responsible for attitude stabilization, disturbance rejection, velocity tracking, angular-rate tracking, and safe stopping.

## Training curriculum

The existing hover checkpoint is used as a warm start. The policy observation appends four normalized command values. The migration preserves the complete legacy observation order (including the 12 nominal hardware features) and expands the actor and critic first-layer matrices by four zero-initialized columns. The old hover policy therefore retains its original input mapping at initialization. The PPO optimizer state is intentionally reinitialized because the observation shape changed.

All five stages use teacher-free movement control; a hover-only teacher is not allowed to override commanded motion:

1. Single-axis low-speed translation or yaw-only rotation.
2. Single-axis medium-speed translation or yaw.
3. Two-axis combinations including simultaneous translation and yaw.
4. Two/three-axis combinations.
5. Continuous random 3D velocity plus yaw-rate commands.

Episodes begin with a two-second zero-command takeoff/hover interval. Command segments last 1–3 seconds during training and include neutral intervals so the policy learns linear braking, angular braking, stopping, reversal, and hover recovery. Initial heading randomization expands from a narrow range in stage one to the full heading range from stage three onward.

The translational reference integrates forward/left velocity using current yaw; the vertical reference follows world-up velocity. Yaw is not integrated into a target heading: the policy tracks the requested yaw rate directly, which avoids imposing an arbitrary absolute heading when the task is to rotate in place.

## Evaluation and checkpoint selection

Every 200,000 aggregate transitions, ten teacher-free episodes are evaluated. Each episode tests all 26 translational command directions, including combined yaw-rate commands and immediate linear/yaw reversals, then four pure in-place yaw commands (both directions at full and half stage rate). Each command is held for 35 simulation steps (0.7 seconds at 50 Hz), followed by a neutral interval to measure linear and angular braking. This sequence plus the 2-second bootstrap fits within the 1,500-step episode budget.

A mission is successful only when takeoff succeeds, all 26 translation commands and all 30 yaw-bearing commands are covered, mean translational tracking error is at most 0.50 m/s, worst translational command error is at most 0.85 m/s, mean yaw-rate error is at most 0.30 rad/s, worst yaw-command error is at most 0.50 rad/s, and both linear speed (0.25 m/s) and yaw rate (0.15 rad/s) settle within 3.5 seconds after the neutral command. Checkpoint selection prioritizes per-profile mission success, crash rate, takeoff success, velocity tracking error, motor saturation, and return.

The five-stage plan remains in the existing manually dispatched workflow. No additional workflow or test workflow is introduced, and no Actions run is triggered as part of code changes.

## Important limits

This remains a simulation-trained research controller. The fast plant is not CFD; sensor/actuator models, calibration, and latency characterization are incomplete. No physical-flight safety claim is made. Before any real flight, measured mass/inertia/propulsion calibration, realistic sensor delays/noise, estimator validation, independent low-level failsafes, hardware-in-the-loop testing, and controlled tethered-flight validation are required. Do not flash or fly this checkpoint as a flight-ready controller.
