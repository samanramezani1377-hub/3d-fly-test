# Smooth2 non-propeller physical calibration plan

**Scope:** complete the non-propeller physical model using existing CAD and component references. Propeller geometry, thrust curves and torque curves are explicitly out of scope and remain untouched.

## Already represented in software

- CAD-derived frame convention, body geometry, rotor positions and motor mount locations.
- Four-motor 6-DoF rigid-body equations, quaternion attitude integration, rotor lag, rotor reaction torque, gyroscopic coupling and body-axis drag.
- Provisional reference motor: generic A2212/13T 1000KV. Exact manufacturer and SKU are unknown.
- Provisional battery reference: 3S, 11.1 V nominal, 2.2 Ah. Exact pack identity and discharge rating are unknown.
- Model-based electrical telemetry and battery-consumption integration, clearly marked as estimates.
- Utilities for assembling mass/CG/inertia and fitting motor lag and aerodynamic drag from suitable measurement logs.

## Remaining measurements — no values are silently filled

| Item | Current status | What is needed |
|---|---|---|
| Complete takeoff mass | Unknown for the CAD airframe; family training uses a provisional 0.6 kg | Weigh the complete aircraft with installed battery, wiring and payload |
| Centre of gravity | Unknown; software baseline uses a geometric-centre assumption | Measure in the documented CAD_EXPORT_Z_UP frame |
| Full inertia tensor | Unknown for Smooth2; generic family values are provisional | Component masses, component CGs and centroidal inertias; cross-check the assembled result |
| Motor identity | Generic A2212/13T 1000KV reference only | Manufacturer, exact SKU, winding and physical motor mass |
| Motor time constant | 0.045 s software estimate; no step-response measurement | Synchronized motor command target and measured RPM log |
| Battery identity | Provisional 3S 11.1 V / 2.2 Ah reference | Exact pack label, cell count, capacity and discharge rating |
| Battery internal resistance | Unknown | Pack-level resistance measurement with method and test conditions recorded |
| ESC model and limits | Unknown | Exact ESC model/firmware, continuous and burst current limits, voltage range and thermal limits |
| Motor current/thermal behavior | Not characterized | Exact motor/ESC/pack combination with current, voltage and temperature logs |
| Body-axis drag | Scalar software regularizers only; no measured anisotropic coefficients | Wind-relative body velocity and isolated aerodynamic force samples |

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

- Keep every unknown field explicitly null or labelled as an estimate.
- Do not convert generic catalog maxima into safe continuous operating limits.
- Keep calibration data separate from CAD geometry.
- Do not launch a long PPO run solely to validate these bookkeeping changes.
- Do not merge changes that alter the policy observation dimensions or the
  existing flight-control command contract.
- No propeller design, geometry, thrust coefficient or torque coefficient is
  changed by this calibration work.
