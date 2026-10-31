# Smooth2 CAD/BEMT provisional physics update

This revision replaces the Smooth2 software-only thrust constants with a geometry-driven **screening estimate** and makes the nominal estimate active in the existing universal PPO vehicle family. It does not alter the simulator's thrust-generation equations, propeller geometry, blade station data, action mapping, reward, or training-stage count.

## CAD inputs

Pinned geometry source: `3D/kilo@38d4820fc5f7e2208e6ce502ca6b54a83989196a`.

- Four rotors; diagonal motor span: 0.353954 m.
- Four-blade propeller diameter: 0.158999 m.
- Hub radius used by the aerodynamic integration: approximately 0.010 m.
- Propeller blade station chord and twist values: `cad/parametric/model.params.json`.
- Motor reference: generic A2212/13T 1000KV; 3S 11.1 V.
- Generic motor screening assumptions: 0.12 ohm winding resistance and 0.5 A no-load current.

## Propeller BEMT screening estimate

The existing `cad/parametric/prop_bemt.py` annular blade-element/momentum model was evaluated using the four-blade CAD station geometry, Prandtl tip loss, sea-level density 1.225 kg/m³, and the generic A2212 motor operating-point model.

| Quantity | Nominal estimate |
|---|---:|
| Loaded full-throttle speed | 10,392 rpm / 1,088.3 rad/s |
| Static thrust per rotor | 4.635 N |
| Static torque per rotor | 0.05153 N·m |
| Shaft power per rotor | 56.08 W |
| Estimated current per motor | 5.90 A |
| Fitted thrust coefficient | 3.9133e-6 N/(rad/s)² |
| Fitted torque coefficient | 4.351e-8 N·m/(rad/s)² |
| Torque/thrust ratio | 0.01112 m |
| Four-rotor static thrust | 18.54 N |
| T/W at nominal 0.60 kg | 3.15 |

The RPM/thrust/torque operating points are also recorded in `vehicle_models/smooth2_parametric/physics_calibration.json`. These are model outputs, not measured stand data. BEMT uncertainty includes airfoil polar approximation, Reynolds number, blade flexibility, guard interference, inflow, manufacturing variation and motor vendor differences.

## Mass properties

The PPO nominal mass remains 0.60 kg as an explicit assembly/BOM-level engineering prior, not a measured or mesh-integrated mass. The nominal CG is the CAD geometric origin. The provisional diagonal inertia is updated to [0.0038, 0.0038, 0.0075] kg·m² using the CAD rotor locations, generic 53 g motor references and a central lumped frame/electronics/battery approximation. Because CAD exports do not assign verified material densities or exact electronics/battery masses, these are intentionally broad priors.

## PPO integration

- `vehicle_models/family/vehicle_profiles.json` now supplies the BEMT-derived thrust coefficient, torque ratio and loaded RPM baseline for Smooth2.
- The existing universal PPO workflow continues to consume this same family profile; the workflow's preflight assertion now verifies the Smooth2 estimate is active.
- The existing family domain-randomization file broadens the final-stage mass, inertia, thrust coefficient, torque ratio, motor speed and motor time-constant variation to expose the nominal model uncertainty.
- No new workflow or training run is created or triggered by this change.

## Still approximate / not flight-validated

The 0.60 kg mass, CG, inertia, motor lag (0.045 s), scalar drag, battery pack identity/resistance, ESC model/current/thermal limits, rotor polar inertia and motor electrical behavior remain approximate. The assumed 9 A ESC/motor software cap is not a verified rating. The BEMT curve is suitable as a simulation prior for adaptation research only; it is not a safe operating envelope or a claim of actual aircraft thrust.

The next improvement can use the existing component STL/STEP exports with verified material assignments and a complete BOM to integrate solid volumes, component masses, CG and inertia. The aerodynamic baseline should eventually be replaced by thrust-stand measurements for the exact assembled motor/ESC/battery/propeller combination.
