# Smooth2 body-relative 3D velocity control

## Control contract

The PPO policy is conditioned on a three-component velocity command in a yaw-aligned frame:

- X: forward along the aircraft nose (positive forward, negative backward).
- Y: left relative to the nose (positive left, negative right).
- Z: world-up (positive climb, negative descent).

The horizontal axes rotate with yaw. The vertical command remains aligned with world-up so a pitch/roll manoeuvre does not redefine what "up" means. This is a yaw-aligned body frame, not the fully tilted rigid-body frame.

The software-training command envelope is currently `[1.5, 1.5, 0.75] m/s` for forward/left/up. These are simulation curriculum limits only; they are not measured safe flight speeds for the physical Smooth2 aircraft.

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

Every 200,000 aggregate transitions, ten teacher-free episodes are evaluated. Checkpoint selection prioritizes per-profile mission success, crash rate, takeoff success, mean 3D velocity-tracking error, motor saturation and return. The five-stage workflow remains the existing manually dispatched workflow; no additional test workflow is introduced.

## Important limits

This remains a simulation-trained research controller. The fast plant is not CFD, sensor/actuator models and calibration are incomplete, and no physical-flight safety claim is made. Before any real flight, measured mass/inertia/propulsion calibration, realistic sensor delays/noise, estimator validation, independent low-level failsafes, hardware-in-the-loop testing and controlled tethered flight validation are required. Do not flash or fly this checkpoint as a flight-ready controller.
