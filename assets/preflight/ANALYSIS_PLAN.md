# Preflight aerodynamic and flight-dynamics analysis plan

## Purpose

Build a repeatable engineering preflight pipeline for the actual quadcopter CAD model before starting or tuning PPO. The pipeline must distinguish measured inputs, manufacturer data, CAD-derived geometry, analytical estimates, CFD results and experimentally validated quantities.

It must not label a design as flight-ready based only on a STEP file or a successful software simulation.

## Pipeline

### Gate 0 — asset integrity

- Rebuild the full STEP assembly from the six committed chunks.
- Verify ISO-10303-21 envelope markers and file size.
- Confirm all manifest STEP paths exist.
- Reopen assembly and individual parts with OpenCascade/CadQuery.
- Record units, solid count, validity, bounding boxes and per-part placement.
- Reconcile discrepancies in the current report: approximately 247 mm vs 271.89 mm footprint; propeller diameter 63.5 mm vs 74.802 mm.
- Confirm whether exported files are assembly placements or already world-positioned parts to avoid double transforms.

### Gate 1 — geometry and mass properties

Extract and store, per component:

- stable component ID and STEP source path;
- local and world transforms;
- bounding box and solid count;
- material/density source and confidence;
- mass, center of mass and inertia tensor.

CAD volume alone is not mass. A density must be assigned from a documented material or a measured component mass. Printed parts also require print orientation, infill, shell count and material assumptions where structural behavior is assessed.

### Gate 2 — rotor and motor characterization

For each rotor:

- propeller diameter, hub diameter, blade count, radial chord distribution, twist/pitch distribution and airfoil sections;
- rotation direction and rotor axis;
- motor make/model, KV, winding/version, max voltage/current/power and thermal limits;
- ESC model, timing, firmware and current limit;
- battery cell count, voltage under load, internal resistance and discharge capability.

Find manufacturer thrust/RPM/current/power data for the exact motor + propeller + voltage combination where available. A motor KV or free-running RPM estimate is not a thrust curve. If reliable data are unavailable, collect guarded thrust-stand measurements over the safe RPM range. Record RPM, thrust, torque if available, voltage, current, temperature and air density.

### Gate 3 — propeller aerodynamics

Use a fidelity ladder rather than running expensive CFD for every design iteration:

1. Geometry extraction and blade-element/momentum-theory (BEMT) estimates.
2. Airfoil polar data at relevant Reynolds numbers; document where XFOIL assumptions are valid or invalid.
3. Rotor interaction / duct or guard effects using a suitable CFD setup (for example OpenFOAM or SU2) only after boundary conditions and mesh independence are checked.
4. Correlate predicted thrust, torque and power against measured bench data.

The guard ring and nearby arm/motor geometry may affect inflow. Isolated-propeller results must not be assumed to represent the installed rotor.

### Gate 4 — 6-DoF vehicle dynamics

Build the vehicle model from verified quantities:

- total mass and center of gravity;
- full inertia tensor in a declared body frame;
- rotor locations and axes;
- per-rotor thrust and reaction torque maps versus RPM, voltage and inflow;
- aerodynamic body forces/moments;
- battery voltage sag and actuator dynamics;
- ground effect, rotor-rotor and rotor-body interaction where supported by evidence.

Document coordinate conventions explicitly. Keep world and body frames separate and test every transform with known vectors.

### Gate 5 — baseline control before PPO

Implement a conventional controller (attitude/rate inner loop and altitude/position outer loop) using the same verified plant. Test:
- trim/hover operating point;
- takeoff and landing transitions;
- axis decoupling and actuator saturation;
- disturbances and wind;
- mass and battery variation;
- sensor noise, delay and estimator error.

Set quantitative acceptance limits before running the tests. Do not silently tune plant parameters to make the controller pass.

### Gate 6 — PPO integration

PPO may begin only when:
- the plant has a documented parameter set and uncertainty bounds;
- the baseline controller passes repeatable simulation tests;
- action-to-RPM/thrust mapping and actuator limits are validated;
- training and evaluation use the same documented physics version;
- no unresolved geometry or thrust-curve blockers remain.

PPO success in simulation is not evidence of real-aircraft safety. Physical flight testing requires a separate staged test plan, hardware checks, containment, failsafes and qualified supervision.

## Tool roles

- CadQuery/OpenCascade: STEP import, BREP validity, assembly transforms, volumes and mass properties (only after assigning density).
- BEMT: efficient rotor performance estimates with documented airfoil polar inputs.
- XFOIL: section polar estimates within its applicability limits.
- OpenFOAM/SU2: higher-fidelity flow analysis when a validated setup is justified.
- Existing SkySim native core: 6-DoF simulation and controller/PPO integration after parameter calibration.

No CFD solver should be treated as a black-box truth source. Mesh, turbulence model, boundary conditions, rotating reference frame/overset strategy and convergence must be documented.

## Current blockers

- Exact motor make/model has not yet been supplied.
- RPM operating range and motor/ESC/battery limits are not yet known.
- The exported CAD report and README contain inconsistent footprint and propeller-diameter values.
- Current imported directory contains STEP files; STL availability is not assumed.
- Physical mass, material densities, CG and inertia are not yet confirmed.
- The propeller hub/shaft interface is explicitly documented as not modelled/verified.
- The screw/nut geometry is simplified; real M3 hardware and physical fit are not verified.

Until these blockers are resolved, status is **INCOMPLETE — NOT CLEARED FOR PPO PLANT CALIBRATION OR PHYSICAL FLIGHT**.
