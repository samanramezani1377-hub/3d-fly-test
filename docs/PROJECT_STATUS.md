# 3D Fly Test — Project Goals, Architecture, Status, and Roadmap

**Status date:** 2026-10-30  
**Repository:** https://github.com/samanramezani1377-hub/3d-fly-test  
**Upstream simulator:** https://github.com/vishwagw/Sky-Sim-drone-simulator-3.0

## 1. Mission

Build a repeatable, browser-accessible flight-test workspace for the user's quadcopter project. The intended workflow is:

1. Keep the simulator configuration, controller code, and optional 3D model in this repository.
2. Edit files and push changes to `main`, or manually run the GitHub Actions workflow.
3. GitHub Actions obtains a pinned SkySim source revision, applies the repository's configuration, builds the browser physics core, and deploys the site to GitHub Pages.
4. Open the published site, run the available flight test, and inspect the behaviour.
5. Change the model, physical parameters, wind, or controller and repeat the test.

The long-term goal is to compare controller changes and different drone designs under repeatable simulated conditions before any real hardware flight.

## 2. Important boundary: visual model vs. flight physics

A GLB file describes visible geometry; it does **not** reliably provide all data needed to simulate a real aircraft. In the current implementation, the optional file at `models/current/drone.glb` is intended to replace the visible mesh only.

The simplified SkySim quad-X physics still needs explicit parameters in `config/drone.json`, such as:

- total take-off mass and payload
- motor-to-motor / arm geometry and rotor size
- battery voltage, cell count, capacity, and internal resistance
- moments of inertia
- motor and propeller thrust/torque characteristics (not yet fully parameterized from measured data)
- centre of gravity and aerodynamic properties (not currently inferred from the mesh)
- wind conditions

Changing the GLB does not yet change rotor positions, collision geometry, mass, inertia, thrust curves, or aerodynamic forces automatically. Until those are represented, a flight test is a test of the configured simplified quad-X, not a validated physical simulation of the uploaded CAD assembly.

## 3. What has been added to this repository

The following files were committed:

| Path | Purpose |
|---|---|
| `README.md` | Short start guide and key caveats |
| `config/drone.json` | Editable physical and visual parameters |
| `controllers/controller.py` | Starter Python policy for browser experiments |
| `models/current/README.md` | Instructions for the replaceable GLB |
| `scripts/prepare_site.py` | Patches a pinned SkySim checkout with project settings |
| `sim_support/battery.hpp` | Compatibility battery model for a missing upstream header |
| `THIRD_PARTY_NOTICES.md` | Upstream attribution and licence reference |
| `.github/workflows/pages.yml` | Build and deploy workflow for GitHub Pages |

The workflow checks out the configuration repository and the pinned SkySim revision, sets up Emscripten, applies configuration/controller/model files, builds the WebAssembly core, assembles a Pages artifact, and deploys it.

## 4. Current state and verification

The repository files and GitHub Actions runs were inspected during setup. An early build exposed a missing battery-header dependency in the pinned upstream source; a compatibility header was added and the workflow was updated to use it. The next run successfully completed source checkout, Emscripten setup, configuration patching, WebAssembly build, Pages artifact preparation, and artifact upload. Deployment then failed because GitHub Pages has not yet been enabled for this repository.

**Current blocker: enable GitHub Pages.** Open [Settings → Pages](https://github.com/samanramezani1377-hub/3d-fly-test/settings/pages) and configure **Build and deployment → Source: GitHub Actions**. A later run must finish successfully, and the published page must be opened and tested, before deployment is considered verified. The successful compilation alone does not prove that browser loading, controller execution, or flight behaviour works correctly.

Use these live links to check status:

- [Repository](https://github.com/samanramezani1377-hub/3d-fly-test)
- [GitHub Actions runs](https://github.com/samanramezani1377-hub/3d-fly-test/actions)
- [GitHub Pages settings](https://github.com/samanramezani1377-hub/3d-fly-test/settings/pages)
- Expected Pages URL: https://samanramezani1377-hub.github.io/3d-fly-test/

The Pages URL is only usable after Pages has been enabled and a deployment succeeds.

## 5. Current parameter file

Edit `config/drone.json` to change the model's configured physics and fallback appearance. All dimensions are SI units unless the key explicitly says otherwise.

- `mass_kg`: total simulated take-off mass, kg
- `rotor_radius_m`: propeller radius, metres (not diameter)
- `arm_length_m`: distance from the body centre to each motor, metres
- `motor_kv_rpm_per_volt`: motor KV rating
- `max_voltage_v`: configured pack voltage, volts
- `battery_cells`: cell count
- `battery_capacity_mah`: battery capacity, mAh
- `battery_internal_resistance_ohm`: simplified pack resistance, ohms
- `inertia_kg_m2`: diagonal inertia values about simulator body axes
- `ground_contact_radius_m`: simplified ground-contact parameter
- `wind_mps`: configured world-frame wind vector
- `visual`: fallback mesh dimensions
- `model_scale`: scale multiplier for a supplied GLB

The values currently in the file are starting placeholders, not measured values for the user's actual aircraft. Before interpreting results, replace them with credible measurements or documented component data.

## 6. Controller interface

The current browser controller contract documented in the repository is:

```python
def policy(obs):
    # obs = [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    # return [roll, pitch, yaw_rate, throttle]
    return [0.0, 0.0, 0.0, 0.34]
```

The starter controller is a basic example, not a tuned or certified flight controller. Verify the observation coordinate frame, units, sign conventions, control ranges, update rate, and throttle-to-motor mapping against the actual browser implementation before developing a more advanced policy. Do not assume that a controller that runs without errors is stable or physically realistic.

## 7. Development phases

### Phase A — Build and deployment reliability
- [ ] Confirm the workflow succeeds from a clean run.
- [ ] Confirm GitHub Pages is configured to use GitHub Actions.
- [ ] Open the deployed page and verify that the JavaScript and WebAssembly assets load without console errors.
- [ ] Verify the simulator can reset, arm, and run its baseline demo.
- [ ] Verify that every intended file change triggers a deployment.

### Phase B — Repeatable baseline tests
- [ ] Define a repeatable hover test (initial position, target altitude, duration, initial battery state).
- [ ] Add explicit reset and deterministic initial conditions.
- [ ] Record altitude, position, attitude, angular rates, motor commands, and battery values over time if available.
- [ ] Add a machine-readable results export (CSV or JSON).
- [ ] Store test definitions separately from the controller.

### Phase C — Use the real drone model
- [ ] Upload the current CAD export as GLB.
- [ ] Check scale, axes, origin, orientation, and rotor placement in the browser.
- [ ] Measure total mass and centre of gravity.
- [ ] Estimate or calculate inertia from the CAD assembly and verify units.
- [ ] Record actual motor, ESC, propeller, and battery specifications.
- [ ] Fit or obtain thrust/torque curves and account for voltage sag.
- [ ] Update the physics model so rotor positions and dynamics match the real design.
- [ ] Add collision geometry only if the simulator supports it; visual geometry is not collision geometry.
- [ ] Compare simulated static thrust and hover throttle against real component data.

### Phase D — Controller experiments
- [ ] Start with a baseline controller and a stable hover test.
- [ ] Tune altitude hold before adding horizontal position control.
- [ ] Test small disturbances and bounded wind values.
- [ ] Add multiple test scenarios and repeat each scenario after controller changes.
- [ ] Compare performance metrics rather than relying on visual impression.
- [ ] Add failsafes for NaN values, excessive attitude, low battery, and out-of-bounds motion where the simulator supports them.

### Phase E — Model comparison and automation
- [ ] Make each test load a named model/configuration version.
- [ ] Store controller commit SHA, configuration hash, simulator revision, and test settings with results.
- [ ] Add a summary report showing settling time, altitude error, drift, peak tilt, energy use, and crashes where measurable.
- [ ] Consider automated parameter sweeps only after a single run is reliable.
- [ ] Keep the upstream simulator revision pinned and review upstream changes before updating it.

## 8. Structured flight-test logging

The browser page now includes a recorder that samples simulator telemetry at approximately 20 Hz and can export:

- **JSON**: run metadata plus samples, including scenario name, UTC start/stop timestamps, repository commit SHA, pinned upstream simulator revision, controller source, full configured drone parameters, and optional GLB SHA-256.
- **CSV**: time-series telemetry for spreadsheet analysis.

Captured samples include simulation time, position and velocity, attitude and angular rates, altitude, vertical speed, thrust, commanded control values, armed state, and controller mode. This feature is implemented in the preparation script and must still be verified in a successful deployed browser session. It does not yet calculate pass/fail or derived performance metrics automatically.

### Recorder correction and controller-axis findings (2026-10-29)

The recorder patch now captures the source text from the JS editor when **Run script** compiles it and from the Python editor when **Run Python** successfully compiles it. JSON metadata uses the active controller source rather than always copying the build-time Python default. Samples record the effective execution path (JS, Python, or keyboard fallback), plus the source mode. The duplicated build-time `controller_source` field is excluded from `drone_config` to avoid presenting it as the active source.

This is a source-code correction only. It is **not verified until a new Pages build succeeds and a fresh browser recording is inspected**. The existing `baseline-hover` JSON was created before this correction and must not be treated as corrected retroactively.

The pinned upstream physics source documents a world frame with **Y up**, **X right**, and **-Z forward**. The observation vector is position XYZ, velocity XYZ, Euler roll/pitch/yaw, and body angular rates. Therefore the current Python policy's use of `obs[1]` / `obs[4]` for altitude / vertical speed is consistent with the inspected source's world-Y-up convention. The current default JS demo policy, however, returns zero roll, pitch, and yaw-rate commands; it primarily tests altitude response and does not actively hold horizontal position. Horizontal drift in that JS demo is therefore not evidence of a defect in our custom airframe.

**Physical target model remains intentionally deferred.** Do not change the placeholder values in `config/drone.json` or claim they represent the user's aircraft until the real physical model and its measurements are ready.

## 8A. PPO training pipeline status (2026-10-30)

The production CPU training workflow is `.github/workflows/ppo-cpu-two-stage.yml`, which runs the five-stage chain in `training/run_five_stage.py`. The production settings are four parallel environments per training process. Stage 1 compares independent seeds 42 and 40 concurrently (four environments per seed, eight simulator environments total); stages 2–5 each continue from the previous selected checkpoint using four environments. The five-stage workflow does not invoke the standalone 1/4/8-environment benchmark.

Evaluation metrics named "during flight" are calculated only from steps after the environment has latched takeoff. A successful-hover episode must take off, avoid a crash, accumulate at least five seconds of stable-hover samples, and have stable-hover samples in at least 80% of its airborne steps. If no takeoff occurs, flight-only error metrics are null rather than misleadingly measuring grounded steps. The checkpoint selector treats unavailable flight metrics as worst-case for ranking, and the standalone parallel-environment quality gate refuses to compare a flight-only metric when either side has no airborne data.

The five-stage workflow artifact list includes both Stage 1 seed folders (42 and 40) and no longer requests the removed benchmark outputs. The standalone benchmark scripts and workflow remain available for explicitly requested experiments; they do not change production settings.

These changes improve software measurement and checkpoint selection. They do not establish physical-model accuracy, native/WASM numerical parity, or real-flight readiness. The drone configuration still contains placeholders pending verified aircraft measurements.

## 9. Known limitations and risks

1. **GLB import is visual-only.** It does not turn the mesh into a complete physics model.
2. **Simplified quad-X physics.** Arbitrary motor/prop layouts, prop guards, ducts, body drag, structural flex, and detailed aerodynamic interactions are not necessarily represented.
3. **Starter controller is not validated.** It may require changes after checking the actual observation axes and controls.
4. **Battery model is approximate.** The compatibility header is a deterministic simplified model, not a validated electrochemical battery model.
5. **No automatic extraction from CAD yet.** Mass, centre of gravity, inertia, rotor locations, and motor properties must be supplied or computed separately.
6. **No guarantee of physical fidelity.** Simulation outcomes depend on model assumptions and parameter quality.
7. **No real-flight safety claim.** A successful browser simulation does not certify a real drone as safe to fly.

## 10. Definition of done for the first usable milestone

The first milestone is complete only when all of these are true:

- GitHub Actions finishes successfully.
- GitHub Pages serves the simulator.
- The browser loads the physics engine without errors.
- A baseline flight test can be started and reset.
- The configured controller actually executes.
- Changing one supported parameter and pushing a commit changes the deployed build.
- The user can replace the GLB and see the new visual model.
- The UI/test result clearly states that the GLB is visual-only and the physics remain simplified.

After that milestone, the next priority is to make the model's physical parameters and test outputs more faithful and measurable.


## 8B. Robust/adaptive fast-physics backend (2026-10-30)

An additive experimental backend has been introduced for the fixed `smooth2-parametric-v1` airframe. It contains a simplified 6-DoF plant, explicit physical calibration record, bounded domain randomization, hidden-parameter Gymnasium observations with four-frame state/action history, a separate PPO entry point, and unit tests.

The v1 geometry is held fixed; mass, CG, inertia, motor effectiveness, voltage factor, drag and wind vary between episodes. Unknown aircraft values remain null in the calibration record. The software-only synthetic parameter set requires explicit opt-in. The new backend does not replace or modify the existing NativeSkySimEnv/SkySim training chain.

Implementation status and limitations: [Robust/adaptive architecture](ROBUST_ADAPTIVE_FLIGHT_ARCHITECTURE.md).

**Important:** the fast plant is a research approximation, not CFD or a validated model of the real aircraft. It has not yet been calibrated against thrust-stand measurements, verified mass properties or flight telemetry. Automatic curriculum scheduling, a dedicated online estimator, native-SkySim parity and physical validation remain future work. No model from this backend is cleared for real flight.
