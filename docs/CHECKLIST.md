# 3D Fly Test — Master Checklist

Use this checklist as the source of truth for project progress. Check an item only after verifying it; creating a file or workflow is not the same as proving it works.

## A. Repository and deployment

- [x] Create the `3d-fly-test` repository.
- [x] Add project overview in `README.md`.
- [x] Add editable drone configuration at `config/drone.json`.
- [x] Add starter Python controller at `controllers/controller.py`.
- [x] Add GLB drop-location instructions.
- [x] Add a script to prepare the pinned SkySim source.
- [x] Add GitHub Actions workflow for build and GitHub Pages deployment.
- [x] Add a compatibility battery header for the missing upstream dependency.
- [ ] Confirm GitHub Pages source is set to **GitHub Actions**.
- [x] Confirm the source checkout, Emscripten setup, configuration patch, WebAssembly build, Pages artifact preparation, and artifact upload succeed in GitHub Actions.
- [ ] Enable GitHub Pages with **Source: GitHub Actions** (the previous run stopped at deployment because Pages is not enabled yet).
- [ ] Get a fully successful GitHub Actions run after enabling Pages.
- [ ] Open the deployed site on desktop and Android.
- [ ] Confirm no JavaScript, WebAssembly, or network errors in browser developer tools.
- [ ] Confirm a push to `main` starts a deployment.
- [ ] Confirm the manual **Run workflow** action works.
- [ ] Confirm the published page corresponds to the latest commit.

## B. Baseline simulator behaviour

- [ ] Load the page from a fresh browser session.
- [ ] Confirm the 3D scene and drone appear.
- [ ] Confirm reset returns the drone to the same initial state.
- [ ] Confirm arm/disarm controls work.
- [ ] Confirm the stock/demo flight can run without a controller.
- [ ] Confirm the Python editor contains `controllers/controller.py`.
- [ ] Confirm the Python controller can start and stop.
- [ ] Confirm the controller observation vector and output ranges match the implementation.
- [ ] Confirm no invalid numbers or runaway values occur during a short test.
- [ ] Save a baseline screenshot and record the tested commit SHA.

## C. Real model upload

- [ ] Export the CAD model as GLB.
- [ ] Check that the GLB opens in an independent viewer before committing it.
- [ ] Put the file at `models/current/drone.glb`.
- [ ] Check orientation and forward direction.
- [ ] Check units and visual scale; adjust `model_scale` if needed.
- [ ] Check that the model origin is at the intended body reference point.
- [ ] Check that all four arms/motors/propellers are positioned as intended.
- [ ] Confirm the GLB loads on the deployed page after a successful build.
- [ ] Confirm replacing the GLB replaces the previous visual mesh.
- [ ] Record the model revision or file hash for each test.

**Important:** completing this section only validates visual loading. It does not validate the physics of the model.

## D. Physical properties

- [ ] Measure or calculate total take-off mass, including battery and payload.
- [ ] Record overall dimensions and motor-to-motor geometry.
- [ ] Record propeller diameter and radius.
- [ ] Record the centre of gravity in the model coordinate frame.
- [ ] Calculate/estimate the three principal moments of inertia with correct units.
- [ ] Record motor KV, ESC limits, and battery voltage.
- [ ] Record battery cell count, capacity, and internal resistance if known.
- [ ] Obtain motor/propeller thrust and torque data across the expected operating range.
- [ ] Define the coordinate frame and rotation/sign conventions.
- [ ] Update `config/drone.json` from verified values rather than placeholders.
- [ ] Check that simulator rotor locations match the actual CAD design.
- [ ] Determine which forces and effects are not represented (body drag, propeller interaction, prop guards, ground effect, etc.).
- [ ] Do not claim physical accuracy until the missing effects have been assessed.

## E. Controller and flight tests

- [ ] Confirm a baseline controller runs before changing gains.
- [ ] Define the desired hover altitude and initial state.
- [ ] Measure altitude error over a fixed-duration test.
- [ ] Measure horizontal drift.
- [ ] Measure roll/pitch/yaw excursions.
- [ ] Test reset and repeatability.
- [ ] Add a small, bounded wind test.
- [ ] Test multiple wind directions and speeds within the simulator's validated range.
- [ ] Test a change in mass and verify the expected qualitative response.
- [ ] Test low-battery/voltage-sag behaviour if supported by the model.
- [ ] Add guardrails for invalid outputs and unstable motion.
- [ ] Compare results from controller versions using the same initial conditions.
- [ ] Export numerical logs; do not judge performance from animation alone.

## F. Automation and reporting

- [ ] Add named test scenarios in version control.
- [ ] Add an output format such as CSV/JSON.
- [ ] Record commit SHA, simulator revision, model hash, configuration, controller version, and test scenario.
- [ ] Add a concise run summary with pass/fail criteria.
- [ ] Define metrics: altitude error, settling time, drift, peak tilt, battery use, and crash/limit events.
- [ ] Add repeat runs to distinguish consistent behaviour from one-off results.
- [ ] Add parameter sweeps only after individual tests are stable.
- [ ] Add regression checks for known baseline scenarios.

## G. Documentation and release hygiene

- [x] Document current goals, architecture, limitations, and roadmap.
- [x] Create this master checklist.
- [ ] Update checklist after each verified milestone.
- [ ] Keep a short changelog of significant model/controller changes.
- [ ] Review third-party license notices whenever the upstream simulator revision changes.
- [ ] Keep the upstream revision pinned until an update has been tested.
- [ ] Clearly label unverified or placeholder values.
- [ ] Never treat browser simulation as a real-flight safety certification.

## Next 5 actions (recommended order)

1. [ ] Open [GitHub Pages settings](https://github.com/samanramezani1377-hub/3d-fly-test/settings/pages) and set **Source: GitHub Actions**.
2. [ ] Open [GitHub Actions](https://github.com/samanramezani1377-hub/3d-fly-test/actions) and confirm the new deployment run succeeds.
3. [ ] Open the published site and verify the baseline simulator and controller manually.
4. [ ] Fix any build or runtime issues before adding the real CAD model.
5. [ ] Export the current drone CAD as GLB and upload it to `models/current/drone.glb`; then begin measuring the real physical parameters.
