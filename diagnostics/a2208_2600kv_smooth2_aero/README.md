# A2208/8T 2600KV × Smooth2 — isolated aerodynamic diagnostic

This branch is deliberately separate from PR #23 and from the production PPO workflow.

## Scope
- Analyze the Smooth2 two-blade propeller geometry against the A2208/8T 2600KV motor.
- Solve a static blade-element/momentum (BEMT) propeller load and approximate motor torque equilibrium.
- Report static thrust, loaded-RPM estimate, torque, current, input power and Reynolds range.
- Do not modify active vehicle profiles, PPO workflows, training checkpoints, CAD source repositories, or the propeller geometry.

## Inputs
- Geometry snapshot is copied into this branch as a read-only diagnostic input from the Smooth2 CAD snapshot previously reviewed in PR #23.
- Motor values are the published candidate values discussed for A2208/8T 2600KV: 2600KV, 62mΩ winding resistance, 1.3A no-load current at 10V, and 20A/60s current ceiling. Exact manufacturer/variant must be verified.
- Default battery is 3S nominal (11.1V); 2S is also supported by the script.

## Model and interpretation
The solver uses annular blade elements, induced axial velocity, Prandtl tip/root loss and a motor electrical torque balance. It currently has a clearly marked placeholder low-Re polar, not measured NACA 4410 data. Therefore its output is a *diagnostic estimate*, not an exact physical thrust result. The polar CSV input allows replacing the placeholder with externally validated alpha/CL/CD data.

Run:
```bash
python diagnostics/a2208_2600kv_smooth2_aero/model.py --cells 3
python diagnostics/a2208_2600kv_smooth2_aero/model.py --cells 2 3 --battery-sag-v 0.6
```

Polar CSV format: `alpha_deg,cl,cd`. For a defensible high-confidence result, provide validated low-Re polar tables spanning the Reynolds numbers printed by the model, then validate against a guarded thrust-stand test measuring RPM, voltage, current, thrust and temperature.

## Explicit exclusions
No change to PR #23, no merge to main, no PPO execution, no modification to production physics constants, and no modification to the source CAD repository.
