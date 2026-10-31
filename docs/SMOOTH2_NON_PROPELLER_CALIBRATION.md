# Smooth2 non-propeller physical calibration plan

**Scope:** complete the physical-model estimate using the latest synchronized CAD and component references. A provisional CAD-station BEMT thrust/torque model is now active in the Smooth2 simulator and PPO profile. It evaluates per-rotor axial inflow and blade-station loads. The estimate is explicitly not a bench measurement.

## Already represented in software

- CAD-derived frame convention, body geometry, rotor positions and motor mount locations from 3D/kilo@38d4820fc5f7e2208e6ce502ca6b54a83989196a.
- Latest CAD propeller geometry: four blades, approximately 159.0 mm diameter and 98.004 mm nominal geometric pitch. Geometry is not a measured thrust curve.
- Four-motor 6-DoF rigid-body equations, quaternion attitude integration, rotor lag, BEMT-derived rotor thrust/reaction torque with axial inflow, gyroscopic coupling and body-axis drag.\n- Full symmetric 3x3 inertia tensor support; the Smooth2 prior is diagonal under nominal symmetry assumptions.\n- Provisional anisotropic body-axis drag coefficients from CAD-envelope projected-area estimates; not CFD or wind-tunnel calibrated.
- Provisional reference motor: generic A2212/13T 1000KV. Exact manufacturer and SKU are unknown.
- Provisional battery reference: 3S, 11.1 V nominal, 2.2 Ah. Exact pack identity and discharge rating are unknown.
- Model-based electrical telemetry and battery-consumption integration, clearly marked as estimates.
- Utilities for assembling mass/CG/inertia and fitting motor lag and aerodynamic drag from suitable measurement logs.

## Remaining measurements — no values are silently filled

| Item | Current status | What is needed |
|---|---|---|
| Complete takeoff mass | Unknown for the CAD airframe; family training uses a provisional 0.6 kg | Weigh the complete aircraft with installed battery, wiring and payload |
| Centre of gravity | Unknown; software baseline uses a geometric-centre assumption | Measure in the documented CAD_EXPORT_Z_UP frame |
| Full inertia tensor | A provisional symmetric 3x3 tensor and shifted CG are active from an explicit component-lumped BOM estimate; not mesh integration | Component masses, component CGs and centroidal inertias; cross-check the assembled result |
| Motor identity | Generic A2212/13T 1000KV reference only | Manufacturer, exact SKU, winding and physical motor mass |
| Motor time constant | 0.045 s software estimate; no step-response measurement | Synchronized motor command target and measured RPM log |
| Battery identity | Provisional 3S 11.1 V / 2.2 Ah reference | Exact pack label, cell count, capacity and discharge rating |
| Battery internal resistance | Unknown | Pack-level resistance measurement with method and test conditions recorded |
| ESC model and limits | Unknown | Exact ESC model/firmware, continuous and burst current limits, voltage range and thermal limits |
| Motor current/thermal behavior | Not characterized | Exact motor/ESC/pack combination with current, voltage and temperature logs |
| Body-axis drag | Anisotropic CAD-envelope projected-area estimate is active; coefficients remain uncalibrated | Wind-relative body velocity and isolated aerodynamic force samples |

## Available tools

### 1. Mass, centre of gravity and inertia

Fill `vehicle_models/smooth2_parametric/component_mass_measurements_template.csv`.
Coordinates and local inertia tensors must use `CAD_EXPORT_Z_UP`.

Run:

```bash
PYTHONPATH=training python -m fast_physics.mass_properties \
  vehicle_models/smooth2_parametric/component_mass_measurements_template.csv \
  /tmp/smooth2-mass-properties.json
```

The utility applies the parallel-axis theorem and emits total mass, CG and the full symmetric inertia tensor. It does not guess missing parts, material density or component mass.

### 2. Motor response

Use `fast_physics.system_identification.estimate_first_order_time_constant`
with synchronized arrays of time (s), target motor speed (rad/s) and measured
motor speed (rad/s). Record the sample rate, voltage and test conditions
alongside the returned fit. This identifies rotor-speed lag only.

### 3. Body-axis drag

Use `fast_physics.system_identification.estimate_body_axis_drag_coefficients`
only after isolating aerodynamic force from gravity, rotor force and other
known forces. It returns separate linear and quadratic coefficients for body
forward, body left and body up axes. Do not fit it directly to raw acceleration.

### 4. Battery and ESC

The current simulator intentionally does not apply voltage sag, ESC current
limiting or thermal derating. The telemetry estimate excludes no-load current,
copper losses, ESC losses and wiring losses. It is not suitable for selecting
an ESC, wiring, battery discharge rating or estimating flight time. Enable
those dynamics only after the exact pack resistance and ESC/motor limits are
available.

## Merge and training policy

- Keep every unknown field explicitly null or labelled as an estimate.\n- Runtime BEMT is enabled only for Smooth2; the three generic vehicle profiles retain their established quadratic thrust fallback.\n- The BEMT model is screening-level and must be validated against a thrust stand before physical claims.
- Do not convert generic catalog maxima into safe continuous operating limits.
- Keep calibration data separate from CAD geometry.
- Do not launch a long PPO run solely to validate these bookkeeping changes.
- Do not merge changes that alter the policy observation dimensions or the
  existing flight-control command contract.
- The simulator thrust-generation implementation and its configured thrust coefficient, torque proxy, motor lag and rotor-speed limits remain unchanged by this CAD synchronization.
- CAD/BEMT screening estimates must not be treated as measured propeller thrust or torque.
