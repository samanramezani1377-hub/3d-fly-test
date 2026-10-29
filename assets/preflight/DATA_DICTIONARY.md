# Preflight data dictionary

All values must carry a unit, source, date/version and confidence label. Never store an unknown value as zero.

## Required records

| Record | Fields | Units / notes |
|---|---|---|
| Vehicle | `vehicle_id`, `total_mass`, `cg_body`, `inertia_body` | kg, m, kg·m²; inertia is a symmetric 3×3 tensor |
| Component | `component_id`, `step_path`, `transform`, `material`, `density`, `mass` | SI units; density and mass need traceable source |
| Rotor | `rotor_id`, `position_body`, `axis_body`, `spin_direction`, `propeller_id` | m; unit vector; CW/CCW viewed from declared direction |
| Propeller | `diameter`, `hub_diameter`, `blade_count`, `chord_stations`, `twist_stations`, `airfoil_sections` | m, dimensionless ratios, degrees; retain radial station coordinates |
| Motor | `manufacturer`, `model`, `kv`, `voltage_range`, `current_limit`, `power_limit`, `rpm_range` | KV in rpm/V; voltage V; current A; power W; RPM |
| ESC | `manufacturer`, `model`, `firmware`, `current_limit`, `timing` | Must match motor and battery |
| Battery | `chemistry`, `cell_count`, `capacity`, `voltage_loaded`, `internal_resistance` | Ah, V, ohm; log state of charge and temperature |
| Environment | `air_density`, `temperature`, `pressure`, `wind_vector` | kg/m³, K, Pa, m/s |
| Rotor test point | `rpm`, `thrust`, `torque`, `voltage`, `current`, `temperature` | rpm, N, N·m, V, A, °C |
| Evidence | `value_status`, `source_uri`, `method`, `uncertainty`, `tested_at` | status: measured, manufacturer, CAD-derived, analytical, CFD, simulated, unknown |

## Evidence status

- `measured`: directly measured with instrument details and calibration.
- `manufacturer`: published by the manufacturer for the exact part/version.
- `cad_derived`: extracted from a validated CAD solid; does not imply mass unless density is known.
- `analytical`: calculated from documented equations and inputs.
- `cfd`: numerical flow result with solver/mesh/model metadata.
- `simulated`: output from a flight simulator.
- `unknown`: missing; must remain missing.

## Frame conventions

The CAD assembly declares exported +Z as vertical, -Y as camera/front and -Z as underside. The flight simulator currently uses its own world-Godot and body-FRD conventions. A dedicated, tested transform must map CAD export coordinates to the simulation body frame. Do not assume that matching axis labels means matching handedness.

## Initial motor lookup input

When the user provides the exact motor brand and model (including suffix/version), research the manufacturer's datasheet and independent thrust-test data. Record the exact propeller, cell count/voltage and test conditions. If these do not match this aircraft, mark the data as reference-only and request a bench test.
