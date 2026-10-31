# Smooth2 CAD/BEMT provisional physics update

This revision replaces the Smooth2 software-only thrust constants with a geometry-driven **screening estimate** and makes the nominal estimate active in the existing universal PPO vehicle family. The runtime simulator now evaluates the pinned CAD blade stations with a blade-element/momentum annulus solve, Prandtl tip loss and local axial inflow at each rotor. Reaction torque is taken from the BEMT torque output; motor-effectiveness and thrust/torque uncertainty scales remain randomized. Generic vehicle profiles retain their existing quadratic coefficient fallback.

## CAD inputs

Pinned geometry source: `3D/kilo@38d4820fc5f7e2208e6ce502ca6b54a83989196a`.

- Four rotors; diagonal motor span: 0.353954 m.
- Four-blade propeller diameter: 0.158999 m.
- Hub radius used by the aerodynamic integration: approximately 0.010 m.
- Propeller blade station chord and twist values: `cad/parametric/model.params.json`.
- Motor reference: generic A2212/13T 1000KV; 3S 11.1 V.
- Generic motor screening assumptions: 0.12 ohm winding resistance and 0.5 A no-load current.

## Propeller BEMT screening estimate

The existing `cad/parametric/prop_bemt.py` annular blade-element/momentum model was evaluated using the four-blade CAD station geometry, Prandtl tip loss, sea-level density 1.225 kg/m³, and the generic A2212 motor operating-point model, including the CAD airfoil camber correction (4% camber at 40% chord; thin-airfoil zero-lift angle −4.1545°).

| Quantity | Nominal estimate |
|---|---:|
| Loaded full-throttle speed | 10,044 rpm / 1,051.8 rad/s |
| Static thrust per rotor | 6.125 N |
| Static torque per rotor | 0.07927 N·m |
| Shaft power per rotor | 83.38 W |
| Estimated current per motor | 8.80 A |
| Fitted thrust coefficient | 5.5367e-6 N/(rad/s)² |
| Fitted torque coefficient | 7.1658e-8 N·m/(rad/s)² |
| Torque/thrust ratio | 0.01294 m |
| Four-rotor static thrust | 24.50 N |
| T/W at nominal 0.60 kg | 4.16 |

The RPM/thrust/torque operating points are also recorded in `vehicle_models/smooth2_parametric/physics_calibration.json`. These are model outputs, not measured stand data. BEMT uncertainty includes airfoil polar approximation, Reynolds number, blade flexibility, guard interference, inflow, manufacturing variation and motor vendor differences.

## Mass properties

The PPO nominal mass remains 0.60 kg as an explicit assembly/BOM-level engineering prior, not a measured or mesh-integrated mass. The nominal CG is the CAD geometric origin. The provisional diagonal inertia is updated to [0.0038, 0.0038, 0.0075] kg·m² using the CAD rotor locations, generic 53 g motor references and a central lumped frame/electronics/battery approximation. Because CAD exports do not assign verified material densities or exact electronics/battery masses, these are intentionally broad priors.

## Runtime simulator integration\n\n`training/fast_physics/propeller.py` implements the runtime blade-element/momentum calculation using the CAD station radii, chord and twist. `FastQuadrotor.step()` now computes per-rotor axial inflow from body-relative velocity plus the angular-rate velocity at each rotor station, then evaluates thrust and shaft torque. The Smooth2 family profile and the default `physics_calibration.json` loader both carry the same pinned BEMT station configuration. Other generic family profiles keep the legacy coefficient model. Runtime results are labelled `runtime_CAD_station_BEMT_with_axial_inflow_and_randomized_effectiveness`; they remain a screening model, not CFD or measured data.\n\n## PPO integration

- `vehicle_models/family/vehicle_profiles.json` now supplies the BEMT-derived thrust coefficient, torque ratio and loaded RPM baseline for Smooth2.
- The existing universal PPO workflow continues to consume this same family profile; the workflow's preflight assertion now verifies the Smooth2 estimate is active.
- The existing family domain-randomization file broadens the final-stage mass, inertia, thrust coefficient, torque ratio, motor speed and motor time-constant variation to expose the nominal model uncertainty.
- No new workflow or training run is created or triggered by this change.

## Still approximate / not flight-validated

The 0.60 kg mass, CG, inertia and anisotropic drag values are still geometry/BOM-envelope engineering estimates, not integrated STEP-solid mass properties or CFD/wind-tunnel outputs. The current full symmetric inertia tensor is diagonal because nominal bilateral symmetry is assumed; products of inertia are not experimentally verified. Motor lag (0.045 s), battery pack identity/resistance, ESC model/current/thermal limits, rotor polar inertia and motor electrical behavior remain approximate. The assumed 9 A ESC/motor software cap is not a verified rating. The BEMT curve is suitable as a simulation prior for adaptation research only; it is not a safe operating envelope or a claim of actual aircraft thrust.

The next improvement is to use the component STL/STEP solids with verified material assignments and a complete BOM to integrate solid volumes, component masses, CG and inertia. The aerodynamic baseline should eventually be replaced by thrust-stand measurements for the exact assembled motor/ESC/battery/propeller combination.
