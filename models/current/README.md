# Current drone model

When you are ready to test the real model's appearance, upload a file named exactly:

`models/current/drone.glb`

The deployment will load this GLB into the browser scene as the drone's **visual mesh**. Adjust `model_scale` in `config/drone.json` if the size/orientation needs tuning.

Important: the mesh does not automatically supply physical properties. Update the mass, rotor radius, motor arm length, motor KV, battery voltage, inertia and wind in `config/drone.json` as well. The current SkySim browser core uses a simplified quad-X physical model; GLB import is visual only and does not create collision geometry or infer aerodynamics.
