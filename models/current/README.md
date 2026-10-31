# Current drone model

The authoritative source is the pinned Smooth2 parametric CAD package:

- Repository: https://github.com/samanramezani1377-hub/3D
- Branch: `kilo`
- Revision: `38d4820fc5f7e2208e6ce502ca6b54a83989196a`
- Committed source and exports: `cad/parametric/` and `output/smooth2-parametric/`

The browser-ready `drone.glb` is generated from all 26 committed component STL files during the existing Flight Lab Pages deployment. It is published at `models/current/drone.glb`; it is not a manually maintained binary copy. The four propellers remain separate named nodes so the Flight Lab can animate them using the active profile's rotor commands and spin directions.

The export converts CAD millimetres to metres, rotates CAD Z-up into the browser's Y-up frame, and places the rotor plane at the visual origin. The model is visual geometry; physical parameters are still read from the vehicle profile and calibration files.

The simulator profile geometry is checked against the pinned CAD parameters by `scripts/validate_smooth2_model_sync.py`. The Smooth2 profile remains provisional until mass, centre of gravity, inertia, exact motor/ESC/battery identity and measured thrust/torque curves are supplied. The GLB does not infer collision geometry, mass, inertia or aerodynamics.
