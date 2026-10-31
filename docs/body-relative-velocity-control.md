# Smooth2 body-relative 3D velocity control

## Control contract

The PPO policy is conditioned on a three-component velocity command in a yaw-aligned frame:

- X: forward along the aircraft nose (positive forward, negative backward).
- Y: left relative to the nose (positive left, negative right).
- Z: world-up (positive climb, negative descent).

The horizontal axes rotate with yaw. The vertical command remains aligned with world-up so a pitch/roll manoeuvre does not redefine what "up" means. This is a yaw-aligned body frame, not the fully tilted rigid-body frame.

The software-training command envelope is currently `[1.5, 1.5, 0.75] m/s` for forward/left/up. The virtual altitude reference is bounded between 0.60 m and 8.0 m; a vertical command toward a reached bound is neutralized. These are simulation curriculum limits only; they are not measured safe flight speeds or safe operating altitudes for the physical Smooth2 aircraft.

## Camera-model interface

The camera model is intentionally separate from the flight-stability policy. Its future output should be a desired velocity vector in this same frame, `[forward_m_s, left_m_s, up_m_s]`, with each component normalized/clamped to the controller's accepted range. It must not output motor commands. The flight controller remains responsible for attitude stabilization, disturbance rejection, velocity tracking and safe stopping.

## Training curriculum

The existing hover checkpoint is used as a warm start. The observation receives three normalized command values. The migration copies every compatible actor/critic weight and expands the two first-layer matrices by three zero-initialized columns. This preserves the previous hover policy response at initialization; the PPO optimizer state is intentionally reinitialized because the network input shape changed.

All five velocity stages run with the privileged hover teacher disabled:

1. Single-axis commands at low speed.
2. Single-axis commands at medium speed.
3. Two-axis combinations.
4. Two- and three-axis combinations.
5. Continuous random 3D velocity vectors.

Episodes begin with a two-second zero-command takeoff/hover interval. Command segments then last 1–3 seconds and include neutral intervals so the policy learns braking, stopping, reversal and hover recovery. Yaw initialization expands from a narrow range in stage one to the full heading range from stage three onward.

The reference path integrates the requested forward/left velocity using current yaw, while altitude follows the world-up command. The reward combines velocity tracking, path/altitude tracking, attitude/rate stability and smooth motor usage.

## Evaluation and checkpoint selection

Every 200,000 aggregate transitions, ten teacher-free episodes are evaluated. Each episode covers all 26 directional combinations: 6 cardinal, 12 two-axis and 8 three-axis commands. Every direction is immediately followed by its reverse. Each command is held for 45 simulation steps (0.9 seconds at 50 Hz), followed by a neutral interval to measure braking. Evaluation speed scales increase with the curriculum stage.

An episode counts as a successful 3D command mission only when all 26 directions are covered, takeoff succeeds, the mean velocity-tracking error is at most 0.50 m/s, the worst direction's mean error is at most 0.85 m/s, and speed remains below 0.25 m/s for at least 0.5 seconds within 3.5 seconds after the neutral command. Checkpoint selection prioritizes per-profile mission success, crash rate, takeoff success, mean 3D velocity-tracking error, motor saturation and return. The five-stage workflow remains the existing manually dispatched workflow; no additional test workflow is introduced.

## Important limits

This remains a simulation-trained research controller. The fast plant is not CFD, sensor/actuator models and calibration are incomplete, and no physical-flight safety claim is made. Before any real flight, measured mass/inertia/propulsion calibration, realistic sensor delays/noise, estimator validation, independent low-level failsafes, hardware-in-the-loop testing and controlled tethered flight validation are required. Do not flash or fly this checkpoint as a flight-ready controller.
