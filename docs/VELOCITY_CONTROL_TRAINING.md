# Smooth2 3D body-relative velocity control

## Control contract

The PPO flight controller receives a desired velocity vector in m/s:

- X: forward/backward relative to the aircraft nose.
- Y: left/right relative to the aircraft nose.
- Z: up/down relative to world vertical (ENU Z-up).

The horizontal command frame follows current yaw. A positive lateral command means left when looking out of the nose. Positive vertical means up. The vector is normalized by the configured per-axis command envelope before it is appended to the policy observation.

Initial software-training command envelope: `[1.5, 1.5, 0.75] m/s` for forward/lateral/vertical. These are curriculum limits for this simulator, not validated flight limits or a claim about safe physical performance.

The controller still outputs collective/roll/pitch/yaw body-wrench requests. The action wrapper allocates those requests to the four motors. The camera/perception model is deliberately separate: it may later produce velocity or waypoint requests, but it does not replace the low-level stabilizing controller.

## Training curriculum

The existing manually dispatched `universal-ppo-five-stage.yml` workflow is the only training workflow used for this task. It is not triggered by code changes.

1. Stage 1: takeoff/hover bootstrap, then low-speed single-axis commands.
2. Stage 2: medium-speed single-axis commands.
3. Stage 3: two-axis combinations.
4. Stage 4: two- and three-axis combinations.
5. Stage 5: continuous random 3D commands.

Commands are held for randomized 1–3 second intervals during training. Neutral commands are included so the policy learns to brake and return to a stable hover. The command is expressed in the current yaw-aligned frame, so forward/left directions rotate with the nose.

The same Smooth2 CAD geometry and rotor locations are retained. Episode-level physical variation is randomized and hidden from the policy. The velocity-control workflow uses no reference teacher in training or periodic evaluation.

## Checkpoint migration

The previous hover checkpoint has three fewer observation values. The migration routine preserves all compatible actor/critic weights and expands only their first input layers by three columns. The new command columns are initialized to zero, so the initial migrated policy retains its prior hover response. A fresh PPO optimizer is created because its parameter-state tensors cannot be reused safely after the input-layer expansion.

A migration is rejected unless the observation dimension differs by exactly three values and exactly the expected actor and critic input layers are expanded.

## Periodic evaluation

Every 200,000 aggregate transitions, the existing callback evaluates the policy with the teacher disabled. Each evaluation episode covers:

- 6 cardinal directions;
- 12 two-axis combinations;
- 8 three-axis combinations;
- immediate reverse transitions for every tested direction;
- a neutral tail to measure braking and stopping.

Each command is held for 45 scheduler steps (about 0.9 seconds at 50 Hz), followed by a neutral braking interval. The command order and speed scale vary between evaluation episodes while preserving all 26 direction combinations.

Checkpoint selection considers per-profile mission success, crash rate, takeoff success, mean velocity tracking error, motor saturation and return. An episode cannot count as a successful 3D command mission unless it covers all 26 directions, keeps mean tracking error at or below 0.50 m/s, keeps the worst direction's mean error at or below 0.85 m/s, and holds speed below 0.25 m/s for at least 0.5 seconds within 3.5 seconds after the neutral command.

## Scope and safety

This is a simulation-training controller, not an aircraft-certified autopilot. The model is not approved for physical flight. The simulator is a simplified rigid-body model and cannot establish robustness to every real-world condition. Before any physical test, the project still needs validated vehicle parameters, independent safety supervision, motor/ESC limits, sensor-failure handling, hardware-in-the-loop validation and controlled flight testing.
