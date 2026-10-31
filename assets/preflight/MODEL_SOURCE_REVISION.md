# Authoritative quadcopter model source

- Repository: https://github.com/samanramezani1377-hub/3D
- Branch: `kilo`
- Source commit: `38d4820fc5f7e2208e6ce502ca6b54a83989196a`
- Parametric CAD source: `cad/parametric/`
- Pinned CAD-builder dependencies: `cad/parametric/requirements.txt`
- Generated assembly and per-part exports: `output/smooth2-parametric/`

The destination CAD source and generated exports are synchronized together from this single immutable source revision.
The geometry package's README, assembly manifest, parameter snapshot, validation report and all per-part STEP/STL exports are retained from the same source revision.

The CAD model now contains four-blade propellers with approximately 159.0 mm diameter and 98.004 mm nominal geometric pitch. This is a geometry synchronization only: the simulator's thrust-generation implementation, thrust coefficient, torque proxy, motor lag and rotor-speed limits are intentionally unchanged. CAD/BEMT screening estimates are not measured propeller performance.

This synchronization does not constitute physical mass-property calibration, propeller thrust testing, motor/ESC characterization, print-fit validation, or flight-safety approval. Mass, CG, inertia, exact motor identity, ESC/battery limits and measured thrust/torque curves remain unknown until measured.
