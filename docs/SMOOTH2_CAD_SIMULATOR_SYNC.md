# Smooth2 CAD-to-simulator integration

The pinned visual and geometric source is `3D/kilo@38d4820fc5f7e2208e6ce502ca6b54a83989196a`.
The committed STEP/STL package, assembly manifest and validation report in this repository are synchronized from that revision.

## Geometry contract

- Four motors, with each motor station at a radial distance of approximately 0.176977 m from the rotor-plane origin.
- Diagonal motor-to-motor span: approximately 0.353954 m.
- Four-blade propellers, 0.158999 m diameter (0.079499 m radius).
- The profile uses the CAD-export coordinate frame and is converted by the Python physics loader into BODY_FORWARD_LEFT_UP. The browser renderer maps the same raw profile to Three.js coordinates.
- The rotor reaction-torque signs alternate by diagonal. The propeller blade handedness is mirrored accordingly; the browser animation reverses reaction-torque sign to obtain physical rotor spin direction.

`scripts/validate_smooth2_model_sync.py` checks the pinned source revision, assembly/report input hashes, four rotor stations, motor span, propeller diameter, reaction signs, and the legacy simulator geometry fields. It runs inside the existing physics/CAD validation workflow; it does not create another workflow or a training run.

## Browser GLB

`scripts/export_smooth2_glb.py` builds `_site/models/current/drone.glb` from the exact 26 committed per-part STL exports. It keeps all components named and exports the four propellers as separate nodes so the Flight Lab can animate them independently. CAD millimetres are converted to metres and the CAD Z-up frame is rotated into the Flight Lab's Y-up frame. The rotor plane is used as the visual origin.

The Pages workflow generates the GLB during the existing deploy job and publishes it beside the Flight Lab. No binary model is hand-copied or allowed to drift from the pinned CAD exports.

## Physical-model boundary

The new CAD geometry is connected to the vehicle profile and browser visual. It does not establish real mass properties or measured propulsion performance. The profile remains `provisional_estimate`; the separate calibration record intentionally keeps takeoff mass, CG, inertia and measured motor/ESC/propeller curves unknown. The existing simulator thrust coefficient, torque proxy, motor lag, RPM limits and PPO control/reward logic are not changed by this integration. Do not interpret the GLB or BEMT screening estimates as a physical thrust test or flight-safety approval.
