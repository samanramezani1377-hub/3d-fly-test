# 3D Fly Test — SkySim

This repository prepares a customizable SkySim browser flight-test environment.

## What this repository does

- Deploys the upstream SkySim WebAssembly flight simulator to this repository's GitHub Pages.
- Builds the physics core in GitHub Actions whenever `main` changes, or on manual workflow dispatch.
- Loads the Python controller from `controllers/controller.py` as the default policy in the browser.
- Records repeatable flight-test telemetry in the browser and exports JSON (metadata + samples) or CSV (time series).
- Applies physical parameters from `config/drone.json` to the simulator before building.
- If `models/current/drone.glb` exists, loads it as a **visual model** in the browser. GLB appearance does not automatically change aerodynamics; physical values must also be entered in `config/drone.json`.

## Quick start

1. Open **Settings → Pages** and set **Build and deployment → Source** to **GitHub Actions**.
2. Open **Actions → Deploy SkySim flight test → Run workflow** (or push a commit to `main`).
3. Wait for the workflow to finish, then open the GitHub Pages URL shown in the deployment.
4. Select **Python** in the simulator and press **Run Python ▶**. The controller in `controllers/controller.py` is preloaded by the deployment.

## Record and compare flight tests

1. Deploy the latest commit and open the simulator.
2. Enter a scenario name in **Flight test recorder** (for example `baseline-hover-v1`).
3. Press **Start recording**, run the desired test, then press **Stop**.
4. Download both JSON and CSV. JSON includes the scenario, timestamps, repository commit, pinned simulator revision, controller source, full configured parameters, optional GLB SHA-256, and telemetry samples. CSV contains the time series for spreadsheet analysis.
5. Keep each run's files together when comparing controller or drone revisions. Use the same scenario and initial conditions for fair comparisons.

Samples are captured at approximately 20 Hz while the simulator page is running. Available fields include position/velocity, attitude and angular rates, altitude, vertical speed, thrust, commanded roll/pitch/yaw-rate/throttle, armed state, and controller mode. This is simulation telemetry; it does not by itself establish physical accuracy or flight safety.

## Change the drone

Edit `config/drone.json`. Values are SI units (metres, kilograms, volts, metres/second). Initial values are SkySim's stock quad defaults and are **placeholders**, not measurements of your physical aircraft.

Key fields:
- `mass_kg`: total takeoff mass, including battery and payload
- `rotor_radius_m`: propeller radius, not diameter
- `arm_length_m`: centre-to-motor distance
- `motor_kv_rpm_per_volt`, `max_voltage_v`
- `battery_cells`, `battery_capacity_mah`, `battery_internal_resistance_ohm`
- `inertia_kg_m2`: diagonal moments of inertia in simulator body axes
- `wind_mps`: world-frame wind vector
- `visual`: dimensions for the simple fallback model
- `model_scale`: visual scale multiplier when a GLB is supplied

To use a different model, replace `models/current/drone.glb`, then update its physical parameters in `config/drone.json`. A GLB is currently used for visualisation only; the simulator still uses the configured simplified quad-X physical model. It does not derive mass, inertia, motor thrust curves, or aerodynamic properties from the mesh.

## Change the controller

Edit `controllers/controller.py`. The browser policy contract is:

```python
def policy(obs):
    # obs = [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    # return [roll, pitch, yaw_rate, throttle]
    return [0.0, 0.0, 0.0, 0.34]
```

The browser tier runs Python through Pyodide and is intended for classical/lightweight control policies, not GPU-heavy model training. Use the on-screen controls to arm/reset and run the policy.

## Notes and limitations

- This is a simulation and software-development setup, not proof that a real aircraft is safe to fly.
- The browser simulator currently models a quad-X. Changing the GLB changes the displayed shape only. The rotor layout and physical parameters remain a simplified model.
- Verify mass, rotor size, motor/prop performance, battery, inertia and centre of gravity before interpreting results as representative of a real drone.
- The upstream SkySim source is fetched at a pinned commit during deployment; see `.github/workflows/pages.yml`. The pinned upstream revision omits its documented battery header, so `sim_support/battery.hpp` supplies a small deterministic compatibility battery model with configurable capacity, cell count and internal resistance.
- SkySim is MIT-licensed; see `THIRD_PARTY_NOTICES.md`.

## Project documentation

- [Full project status, architecture, limitations, and roadmap](docs/PROJECT_STATUS.md)
- [Master checklist](docs/CHECKLIST.md)

## Upstream

- [SkySim source](https://github.com/vishwagw/Sky-Sim-drone-simulator-3.0)
- [SkySim browser demo](https://vishwagw.github.io/Sky-Sim-drone-simulator-3.0/)
