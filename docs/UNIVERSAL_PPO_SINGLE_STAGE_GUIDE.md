# Universal PPO — five-stage 3D velocity and yaw-rate training

> **This guide describes the current executable workflow, not the older single-stage hover experiment.** The legacy filename is retained so existing links do not break. The source of truth is `.github/workflows/universal-ppo-five-stage.yml` and the trainer implementation it invokes.

## 1. Purpose and current operating mode

The manual GitHub Actions workflow trains one hardware-conditioned PPO policy across four quadrotor profiles. The policy is trained for:

- 3D translational velocity tracking, with horizontal commands expressed relative to the current nose heading;
- independent yaw-rate control, including in-place left/right turns with zero translational velocity;
- combined translation and yaw rotation;
- progressive command and physical-domain randomization across five chained curriculum stages.

This is **five active stages**, not the former single-stage takeoff/hover-only workflow. Each stage has two phases: teacher-assisted followed by teacher-free. The next stage starts from the best evaluated teacher-free checkpoint of the preceding stage.

The workflow is simulation-only. The four profiles are explicitly provisional estimates. Successful training or evaluation does not establish real-aircraft performance or flight readiness.

## 2. Exact GitHub Actions configuration

Workflow: [`.github/workflows/universal-ppo-five-stage.yml`](../.github/workflows/universal-ppo-five-stage.yml)

Launch it manually from **Actions → Universal quadrotor PPO - Five-stage 3D velocity and yaw-rate curriculum → Run workflow**, selecting `main`.

| Setting | Current value |
|---|---:|
| Trigger | Manual `workflow_dispatch` only |
| Training stages | 5, all active |
| Phases per stage | 2 (teacher-assisted, then teacher-free) |
| Timesteps per phase | 400,000 aggregate transitions |
| Timesteps per stage | 800,000 |
| Total new training budget | 4,000,000 aggregate transitions |
| Parallel environments | 100 |
| PPO rollout length | 500 per environment |
| Aggregate transitions per rollout | 50,000 |
| PPO minibatch size | 250 |
| Device | CPU-only |
| Teacher blend in phase A | 0.80 |
| Teacher blend in phase B | 0.00 (teacher disabled) |
| Periodic evaluation | Every 200,000 aggregate transitions, at rollout boundaries |
| Episodes per profile per periodic evaluation | 10 |
| Latest checkpoint interval | 50,000 aggregate transitions, at rollout boundaries |
| Maximum job duration | 360 minutes |
| Run output directory | `training/runs/` |
| Published model directory | `training/models/universal-quad-four-vehicle-five-stage/` |

A timestep is an aggregate environment transition. With 100 parallel environments, one vectorized step contributes 100 transitions. A rollout contains 500 vectorized steps, or 50,000 aggregate transitions. Therefore each 400,000-transition phase consists of eight complete PPO rollouts.

The workflow installs CPU-only PyTorch and constrains common numerical libraries to one thread per process. It does not request a GPU.

### Mandatory teacher-only capability gate (runs before checkpoint restore and PPO)

Before the bootstrap checkpoint is downloaded and before any PPO optimization starts, the same workflow runs `training/preflight_velocity_teacher.py`. This is a closed-loop simulator test, not merely a Python import or finite-output check.

- It runs the privileged velocity/yaw teacher at **100% weight**; PPO contributes exactly zero action.
- It tests all four vehicle profiles, with three deterministic Stage-5 randomized episodes per profile (12 episodes total).
- Each episode exercises the same 26 translation-command cases, 30 signed yaw-command cases (including pure in-place left/right turns), and neutral braking tail used by the PPO evaluator.
- It checks takeoff, no crash, full command coverage, mean velocity error ≤ 0.50 m/s, worst per-direction mean error ≤ 0.85 m/s, mean yaw-rate error ≤ 0.30 rad/s, worst per-yaw-command mean error ≤ 0.50 rad/s, and neutral braking within 3.5 seconds.
- **All 12 episodes must pass.** A failed profile/scenario fails the step and stops the job before checkpoint restoration or any training compute.
- The JSON report is uploaded as the `universal-teacher-capability-preflight` Actions artifact even when the gate fails, and is also copied into the final published model directory on a successful run.

This is a finite, seeded sample across the configured Stage-5 randomization bounds; it cannot prove success in every mathematically possible or physically real condition. It is intended to catch an incapable or broken oracle teacher before spending the much larger PPO training budget.


## 3. Initial checkpoint and stage chaining

The workflow downloads the selected teacher-assisted 1,000,000-step checkpoint from Actions run [41158518207](https://github.com/samanramezani1377-hub/3d-fly-test/actions/runs/41158518207), artifact:

`universal-ppo-stage2-teacher-to-teacher-free-backup`

Before training, it verifies the checkpoint files and selection metadata, checks that the selected timestep count is exactly 1,000,000, and confirms the expected four vehicle IDs and their `provisional_estimate` calibration status.

The new velocity/yaw policy interface adds four command inputs to the older hover checkpoint. The loader preserves compatible policy weights, initializes the new command-input weights to zero, remaps the legacy roll/pitch action outputs for the required body-frame convention, and initializes a fresh optimizer. This is a warm start, not a same-architecture optimizer-state resume.

For every stage:

1. Phase A loads the preceding checkpoint and trains 400,000 transitions with teacher weight 0.80.
2. The best evaluated phase-A checkpoint is required; the terminal snapshot is not substituted.
3. Phase B loads that phase-A best checkpoint and trains 400,000 transitions with the teacher disabled and weight 0.00.
4. The best evaluated phase-B checkpoint becomes the starting checkpoint for the next stage.

The five stage pairs are therefore sequential and form one checkpoint lineage. If a phase fails to produce its required evaluated best checkpoint, the workflow stops rather than silently continuing from an unevaluated model.

## 4. What the policy is commanded to do

The action interface combines nose-relative 3D velocity commands and a separate yaw-rate command.

- Body-horizontal axes are nose-forward and nose-left.
- Vertical velocity is world-up.
- Positive yaw rate turns the nose left (counter-clockwise when viewed from above); negative yaw rate turns it right.
- The velocity limits are 1.5 m/s forward, 1.5 m/s lateral, and 0.75 m/s vertical.
- Yaw-rate command limits increase by stage: 20, 35, 50, 70, then 90 degrees/second.

The evaluation sequence includes 26 translational direction combinations, paired with yaw commands, plus four pure-yaw commands (both directions at full and half stage rate). Pure-yaw commands have zero translational velocity, so the evaluator explicitly exercises in-place rotation. Commands are held for 35 simulation steps each, followed by a neutral braking period.

| Stage | Maximum yaw-rate command | Curriculum focus |
|---|---:|---|
| 1 | 20°/s | Low-speed single-axis motion and yaw-only turns |
| 2 | 35°/s | Medium-speed single-axis motion with yaw |
| 3 | 50°/s | Two-axis translation/yaw combinations |
| 4 | 70°/s | Two- and three-axis translation/yaw combinations |
| 5 | 90°/s | Continuous randomized 3D velocity and yaw-rate commands |

The curriculum is a progressive training distribution; it is not a guarantee that the policy will meet every command-tracking threshold.

## 5. Vehicle family and domain randomization

The policy is shared across exactly these four profiles in `vehicle_models/family/vehicle_profiles.json`:

- `smooth2-a2212-custom-prop`
- `generic-5inch-quad`
- `generic-7inch-long-range`
- `generic-3inch-cinewhoop`

Each profile is marked `provisional_estimate`. The Smooth2 profile combines CAD-derived rotor geometry with provisional physical and propulsion coefficients. The other profiles are generic class estimates, not verified specifications for particular commercial aircraft.

The active randomization configuration is `vehicle_models/family/domain_randomization.json`. It varies, within stage-dependent bounds:

- mass, centre of mass and inertia;
- motor effectiveness, motor response time, maximum rotor speed and thrust/torque coefficients;
- battery voltage factor;
- aerodynamic drag;
- world-frame wind;
- sensor/observation noise for position, velocity, attitude and angular rate.

The Stage 5 outer bounds include mass scale 0.8–1.06, inertia scale 0.7–1.4, drag scale 0.7–1.4 and wind components from -2 to +2 m/s. These are simulator randomization ranges, not measured uncertainty bounds or safe operating limits.

**Important distinction:** wind and sensor noise are already represented in the current randomization configuration. A more detailed physical wind/aerodynamic model, realistic sensor dynamics and estimator stack, and real measured calibration are still future fidelity improvements; they are not prerequisites to running this software-training experiment.

## 6. Evaluation and checkpoint selection

Periodic evaluations run without the teacher. Each evaluation checks all four profiles for 10 episodes each, using the current stage's velocity/yaw command sequence. The evaluation records include takeoff/crash outcomes, velocity and yaw-rate tracking errors, direction/command coverage, neutral braking, motor saturation and estimated battery telemetry.

The balanced best-checkpoint selection prioritizes the weakest vehicle profile. The current lexicographic order is:

1. maximize minimum per-profile velocity-mission success;
2. minimize worst per-profile crash rate;
3. maximize minimum per-profile takeoff success;
4. minimize minimum-profile velocity-tracking error;
5. minimize yaw-rate tracking error;
6. minimize motor saturation;
7. maximize mean episode return as the final tie-breaker.

The final independent evaluation runs after stage 5, with the teacher disabled, a separate deterministic seed range beginning at 987654, and 10 episodes per profile (40 episodes total). The workflow verifies that all four profiles are present and writes `final-evaluation-summary.json`.

Evaluation results are simulator measurements. Inspect per-episode reports and all profile-level metrics; a successful workflow run alone is not evidence that the learned policy has met the mission objective.

## 7. Checkpoints, logs and published outputs

During a phase, its run directory is under `training/runs/<run-name>/` and contains, as applicable:

- `latest_model.zip` and `training_state.json` for the latest resumable snapshot;
- `final_model.zip` for the terminal phase snapshot;
- `best/best_model.zip` and `best/selection.json` for the best evaluated checkpoint;
- profile evaluation JSON reports, sampling weights, metadata and TensorBoard/log output.

The workflow publishes the chained stage directories, initial checkpoint and checksum, final selected model, independent evaluation summary, and the randomization configuration under:

`training/models/universal-quad-four-vehicle-five-stage/`

The workflow commits the published outputs to the branch on which it was dispatched, if there are changes. The final selected model is also retained with the run output. Do not confuse a phase's terminal `final_model.zip` with its best evaluated `best/best_model.zip`; stage handoff uses the latter.

The trainer emits machine-searchable throughput records:

```text
[PPO_SPEED] timesteps=... rollout_steps=... rollout_seconds=... rollout_steps_per_sec=... average_steps_per_sec=...
```

Rates are aggregate transitions per second across all 100 environments, not per-environment speed. Evaluation work can cause visible throughput dips.

## 8. How to launch and monitor

1. Open the [Actions workflow](https://github.com/samanramezani1377-hub/3d-fly-test/actions/workflows/universal-ppo-five-stage.yml).
2. Select the `main` branch.
3. Select **Run workflow** once.
4. Monitor the single job and confirm it advances through all ten phase runs (five teacher phases and five teacher-free phases).
5. Confirm the final independent evaluation and publish step complete.
6. Inspect the published model directory and the run logs before interpreting the result.

The workflow has a concurrency group with `cancel-in-progress: false`; do not manually start duplicate runs. The workflow is manually dispatched and is not automatically started by ordinary commits or pull requests.

## 9. Current prerequisites versus future fidelity work

### Already implemented for this simulation-training run

- Four-profile shared-policy training and profile-conditioned observations.
- Nose-relative 3D velocity commands and independent positive/negative yaw-rate commands.
- Explicit pure-yaw / in-place-turn evaluation and combined translation-plus-yaw evaluation.
- Five-stage curriculum with increasing yaw command limits.
- Teacher-assisted then teacher-free training in every stage.
- Best-checkpoint selection and sequential checkpoint handoff.
- Randomization of mass, CG, inertia, actuator/propulsion parameters, battery voltage factor, drag, wind and sensor noise.
- Per-profile periodic evaluation and a separate final evaluation.

### Not required to start this training, but required for stronger physical claims

- Measured complete takeoff mass, centre of gravity and full inertia tensor for each real aircraft.
- Measured motor/propeller thrust and reaction-torque curves across RPM and voltage, plus motor response data.
- Actual ESC current limits, battery internal resistance, voltage sag, thermal behavior and electrical dynamics.
- More realistic aerodynamic/propeller-flow effects, detailed wind disturbances, and a physical sensor/estimator model with latency, bias, drift and correlated noise.
- Verified rotor spin directions, motor numbering, actuator allocation and hardware failsafes.
- Independent hardware-in-the-loop, bench and controlled flight validation.

The current randomization and estimated battery/current telemetry do not constitute a validated electrical model. In particular, current profiles contain null values for several real electrical properties and explicitly describe their electrical model as provisional.

## 10. Simulation-only safety boundary

All four vehicle profiles are provisional estimates. The fast-physics plant is a simplified 6-DoF research model, not a validated digital twin. The teacher is privileged simulator-side guidance; it is not available as a real-world safety mechanism. The PPO policy and its checkpoints are not certified or approved flight controllers.

Do not connect this policy directly to a real aircraft or use the training score as flight clearance. Real hardware use would require aircraft-specific measured parameters, verified control/coordinate conventions, electrical and thermal limits, failsafes, independent bench testing and a separately designed, supervised flight-validation process.

## 11. Source of truth

When documentation and executable settings disagree, use these files as the authority:

- `.github/workflows/universal-ppo-five-stage.yml` — workflow trigger, CPU setup, stage budgets, checkpoint bootstrap, chaining and publication;
- `training/train_universal_ppo.py` — PPO hyperparameters, velocity/yaw command generation, teacher-free evaluation and checkpoint selection;
- `training/fast_physics/ppo_action.py` — command/action mapping and teacher blending;
- `training/fast_physics/velocity_reference.py` — privileged velocity/yaw reference controller;
- `training/fast_physics/vehicle_family.py` — profile loading, sampling and hardware-conditioned observations;
- `vehicle_models/family/vehicle_profiles.json` — four profile definitions and calibration provenance;
- `vehicle_models/family/domain_randomization.json` — stage-specific randomization ranges.

This guide must be updated whenever those executable settings change.
