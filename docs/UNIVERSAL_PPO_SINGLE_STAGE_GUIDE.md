# Universal PPO — single-stage takeoff and hover training

## 1. Current operating mode

The existing workflow, `.github/workflows/ppo-cpu-two-stage.yml`, is temporarily configured for a **single active training stage**. The immediate objective is deliberately narrow:

1. learn to leave the ground reliably;
2. reach the target altitude without crashing;
3. maintain a stable hover for at least five continuous seconds under the current simulator's stability definition.

Only curriculum stage 1 is executed. Stages 2, 3, 4 and 5 remain in the source code and are **disabled, not deleted**. They are not trained, resumed, evaluated or used for checkpoint handoff while `--active-stages 1` is selected. The final held-out release evaluation is also deferred until all five stages are explicitly re-enabled.

This is a temporary training focus, not a change to the long-term five-stage design.

## 2. Exact GitHub Actions configuration

A manual `workflow_dispatch` on the existing workflow runs the following command:

```bash
python training/run_universal_curriculum.py \
  --active-stages 1 \
  --timesteps-per-stage 2000000 \
  --n-envs 32 \
  --run-prefix universal-quad-v5-foundation \
  --checkpoint-eval-episodes 2 \
  --eval-frequency 100000 \
  --checkpoint-frequency 20000 \
  --evaluation-episodes 20
```

| Setting | Current value | Meaning |
|---|---:|---|
| Active stages | 1 | Only stage 1 runs |
| PPO transitions | 2,000,000 | Aggregate environment transitions for the active stage |
| Parallel environments | 32 | 32 simulator instances collect data for one shared PPO policy |
| PPO rollout length | 500 per environment | Existing PPO configuration |
| Aggregate transitions per rollout | 16,000 | 32 × 500 |
| PPO minibatch size | 250 | Existing PPO configuration |
| Checkpoint evaluation interval | 100,000 requested | Checked at rollout boundaries |
| Episodes per profile per checkpoint evaluation | 2 | Balanced profile evaluation |
| Latest-checkpoint interval | 20,000 requested | Saved at a rollout boundary |
| Teacher contribution | 0.80 in stage 1 | Privileged simulator reference contributes to action blending in this stage |
| Final held-out evaluation | Deferred | Not run in single-stage mode |

The 2,000,000-step budget is exactly 125 complete rollouts of 16,000 transitions. Therefore, it is aligned to the PPO rollout boundary and should not require a partial final rollout.

**Meaning of “step”:** the configured timestep budget is the aggregate number of transitions across all 32 environments, not two million transitions per environment. A complete vectorized step advances all 32 environments and contributes 32 aggregate transitions.

## 3. What stage 1 is trying to learn

Stage 1 uses the current takeoff/hover task and the stage-1 initial-condition/randomization configuration. It does not attempt to solve the later curriculum's progressively harder conditions.

The training objective is assessed through the existing balanced per-profile evaluator. The most important signals are:

- takeoff success rate;
- five-second sustained-hover success rate;
- crash rate and crash reasons;
- longest continuous stable-hover duration;
- altitude and horizontal-position error;
- motor saturation and zero-command fractions;
- episode return.

A sustained-hover success is not merely reaching the target altitude or surviving until timeout. The evaluator requires takeoff, no crash, and a continuous five-second interval satisfying the environment's current bounds:

- altitude error at most 0.15 m;
- horizontal position error at most 0.20 m;
- translational speed at most 0.35 m/s;
- absolute roll and pitch at most 0.25 rad;
- angular-rate norm at most 0.5 rad/s.

These thresholds are simulator evaluation criteria. They are not real-aircraft acceptance limits.

## 4. Teacher guidance in stage 1

The stage-1 policy uses the body-wrench action wrapper. Its PPO action represents collective thrust and body-axis roll, pitch and yaw torque requests; the wrapper allocates the requested wrench to motor commands.

In stage 1, the privileged hover-reference controller contributes with weight 0.80. This means the early policy is teacher-guided rather than fully autonomous. The reference controller can use the simulator's true plant state and episode parameters inside the wrapper. Those privileged values are not appended to the PPO observation.

Stages 2–5 retain their existing teacher schedule in code (0.60, 0.40, 0.20 and 0.00 respectively), but they are dormant in the current workflow. The current stage-1 run therefore does **not** demonstrate teacher-free policy performance. That must be evaluated after the teacher is reduced or removed in a later training phase.

## 5. Parallel environments and throughput

The 32 environments belong to one shared PPO learner; they do not train 32 separate policies. Each PPO rollout collects 500 transitions from each environment, giving 16,000 aggregate transitions per rollout.

The callback measures elapsed wall-clock time with a monotonic high-resolution timer and prints one machine-searchable record at every rollout boundary:

```text
[PPO_SPEED] timesteps=16000 rollout_steps=16000 rollout_seconds=... rollout_steps_per_sec=... average_steps_per_sec=...
```

Fields:

- `timesteps`: cumulative aggregate transitions reported by Stable-Baselines3;
- `rollout_steps`: aggregate transitions collected since the previous speed record;
- `rollout_seconds`: wall-clock seconds for that interval, including environment collection and PPO work since the preceding rollout-end callback;
- `rollout_steps_per_sec`: interval aggregate transitions divided by interval seconds;
- `average_steps_per_sec`: aggregate transitions since the start of this `learn()` call divided by elapsed wall-clock seconds.

The speed is **aggregate environment transitions per second across all workers**, not steps/second per individual environment. The cumulative average restarts when an interrupted job resumes because the timer measures the current process lifetime; the checkpoint timestep counter itself is retained.

The records are written to the GitHub Actions console and streamed to `single-stage-training.log` using `tee`. The workflow prints the log tail at the end and uploads the full log as part of the workflow artifact. This allows later comparison of throughput across runs. Compare runs using the same hardware class, Python/PyTorch dependencies, environment count and evaluation frequency; evaluation work can create visible dips in the interval speed.

## 6. Checkpointing, resume and artifact locations

The active run prefix is `universal-quad-v5-foundation`. It intentionally isolates this foundation run from older `universal-quad-v5` runs and earlier v4 checkpoints.

During execution, working artifacts are placed under:

```text
training/runs/
  universal-quad-v5-foundation-stage1/
    latest_model.zip
    training_state.json
    final_model.zip
    best/
      best_model.zip
      selection.json
    profile-evaluations/
    profile-sampling-weights.json
    metadata.json
  universal-quad-v5-foundation-curriculum-manifest.json
```

The workflow restores previously committed foundation artifacts from:

```text
training/models/universal-quad-v5-foundation/
```

At rollout boundaries, the trainer saves a resumable latest checkpoint and a state file. The callback's checkpoint threshold is 20,000 aggregate transitions, but saves occur only at complete rollout boundaries. With 16,000 transitions per rollout, the first eligible boundary is 32,000 transitions. The evaluation threshold is 100,000 and is likewise checked at rollout boundaries, so the first eligible boundary is 112,000 transitions. These are threshold-based intervals, not exact wall-clock schedules.

If the workflow is interrupted, rerunning the same manual workflow restores the foundation directory and the runner resumes the incomplete stage from its latest checkpoint, provided the manifest, stage state and configuration agree. Do not change the stage budget, environment count, vehicle-family file, seed or other manifest settings when resuming an existing run. The runner intentionally refuses a mismatched manifest rather than mixing incompatible runs.

At completion, `final_model.zip` is the terminal training snapshot. `best/best_model.zip` is the best checkpoint selected by the periodic balanced profile evaluation. They are distinct artifacts and must not be assumed to be identical.

The workflow copies the stage artifacts back into `training/models/universal-quad-v5-foundation/` and commits them to the current branch when the job has produced changes. It also uploads a 90-day Actions artifact backup. TensorBoard directories and Monitor CSV files are excluded from the repository checkpoint copy to avoid unnecessary repository growth; the workflow artifact retains the run directory.

## 7. Checkpoint selection

At each checkpoint evaluation, the policy is evaluated deterministically against every configured vehicle profile. Two episodes are run per profile. The selection score is compared lexicographically in this order:

1. maximize the minimum per-profile sustained-hover success rate;
2. minimize the worst per-profile crash rate;
3. maximize the minimum per-profile takeoff success rate;
4. maximize mean stable-hover duration;
5. minimize mean motor saturation fraction;
6. maximize mean episode return as the final tie-breaker.

This ranking intentionally protects the weakest profile instead of allowing a strong result on one profile to hide a failure on another. With only two episodes per profile, these rates are noisy early indicators, not statistically conclusive estimates. Inspect the episode-level reports and later run a larger evaluation before drawing conclusions.

The adaptive profile sampler updates profile weights from observed weaknesses. This changes how often profiles are sampled in subsequent training episodes; it does not change the vehicle parameters or create separate policies.

## 8. How to launch the training

1. Open the repository's **Actions** tab.
2. Select **Universal PPO - PR validation / manual single-stage training**.
3. Select the `main` branch.
4. Choose **Run workflow**.

A pull request follows the existing validation path only. It does not launch PPO training. The 2-million-transition training run starts only through the manual `workflow_dispatch` event.

Do not start duplicate manual runs concurrently. The workflow concurrency group prevents overlapping runs on the same ref from proceeding independently.

## 9. How to inspect the result

After the run, inspect the following in order:

1. `single-stage-training.log`: search for `[PPO_SPEED]`, warnings, exceptions and the final timestep count.
2. `training_state.json`: confirm `completed: true` and `num_timesteps: 2000000`.
3. `best/selection.json`: see the selected checkpoint's score and evaluation report.
4. `profile-evaluations/profile-evaluation-*.json`: compare all profiles and inspect individual episode outcomes.
5. `metadata.json`: verify the configuration, profile provenance, teacher schedule and simulation-only warning.
6. `universal-quad-v5-foundation-curriculum-manifest.json`: confirm that only stage 1 is listed as active and stages 2–5 are recorded as deferred.

A completed job only proves that the configured simulation training budget ran and artifacts were saved. It does not prove that the policy has learned reliable takeoff or hover. The evaluation metrics must show that. If the weakest profile remains unsuccessful, investigate its episode-level crash reasons, takeoff behavior, motor saturation, reward terms, initial conditions and reference-controller performance before increasing compute or activating later stages.

## 10. Re-enabling the later curriculum stages

Stages 2–5 are retained in the runner. To reactivate the full chain later, change the existing workflow argument from:

```bash
--active-stages 1
```

to:

```bash
--active-stages 5
```

Do this only after reviewing stage-1 takeoff and hover results and deciding that the foundation checkpoint is suitable for curriculum continuation. Keep the same isolated run prefix if the intent is to continue the same curriculum lineage. The runner then proceeds through stages 1–5 and enables the final held-out evaluation after stage 5. The five-stage final held-out gate is not part of the current single-stage run.

Do not point the later stages at a checkpoint from a different action/observation version or incompatible PPO rollout configuration. The v5 foundation prefix is intentionally separate from older checkpoint families.

## 11. Existing workflow and test policy

No additional workflow was created. The existing workflow still has two event paths:

- pull request: runs the pre-existing physics, family and hover regression tests;
- manual dispatch: runs the one-stage PPO training.

No test suite was added for this documentation change. Do not use a PR validation run as a substitute for the manual training run, and do not repeatedly trigger training merely to inspect code changes.

## 12. Simulation-only and safety boundary

The example vehicle profiles and many physical parameters are provisional estimates. The simplified fast-physics plant is not a validated digital twin of the final aircraft. It does not establish accurate motor/ESC electrical behavior, battery dynamics, custom-propeller aerodynamics, sensor/estimator behavior, controller timing or sim-to-real transfer.

The privileged teacher is a simulator-side training aid. Neither the teacher nor the PPO checkpoint is an approved flight controller. Even a perfect simulated takeoff/hover score does not authorize a physical flight. Aircraft-specific mass, CG, inertia, rotor geometry and spin directions, motor/ESC/propeller/battery performance, current/thermal limits, failsafes and independent bench/flight validation are prerequisites for any separate real-world safety assessment.

**Never connect this policy directly to a real aircraft based only on these training results.**

## 13. Source of truth

The current runtime settings are defined by:

- `.github/workflows/ppo-cpu-two-stage.yml` — GitHub Actions entry point and run arguments;
- `training/run_universal_curriculum.py` — active-stage selection, budgets, resume and deferred final evaluation;
- `training/train_universal_ppo.py` — PPO configuration, evaluation/checkpoint callback and throughput logging;
- `training/fast_physics/ppo_action.py` — policy body-wrench action mapping and stage teacher blending;
- `training/fast_physics/vehicle_family.py` — vehicle-profile selection and adaptive sampling;
- `training/evaluate_universal_ppo.py` — standalone per-profile evaluator.

If this document and executable settings disagree, the workflow and source code are authoritative; update this guide when those settings change.
