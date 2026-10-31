# Smooth2 CAD propulsion screening

## Scope

This report extracts the propeller geometry from the pinned Smooth2 CAD parameter snapshot and compares a deliberately simplified static blade-element / actuator-disk estimate with the provisional vehicle profile. It is a screening calculation, not a motor-propeller match, a validated thrust prediction, or proof that the propeller is defective.

The source CAD repository and its `kilo` branch are read-only inputs. The snapshot in this repository records the source blob SHA.

## Geometry recovered

- Two blades.
- Final diameter: approximately 74.8 mm (2.94 in).
- Nominal geometric pitch: 53.086 mm (2.09 in).
- Pitch-to-diameter ratio: approximately 0.71.
- Eleven radial chord stations.
- NACA 4410-style parameterization: 4% camber at 40% chord and 10% thickness.
- Twist follows the CAD builder's constant-pitch law, `atan(P/(2*pi*r))`, capped at 42 degrees.
- The motor is identified in the profile as an A2212/13T 1000KV reference on nominal 3S (11.1 V). CAD motor dimensions are only a simplified envelope.

## Calculation method

The script couples blade-element lift/drag to an annular actuator-disk momentum balance. It evaluates four RPM points at 50%, 65%, 80%, and 95% of the lower of the ideal KV-times-voltage no-load RPM and the configured simulation rotor-speed cap. These are sensitivity points only; they are not predicted loaded motor RPMs.

For reproducibility, the preliminary section model uses a 2*pi-per-radian lift slope, CL clipped to +/-1.2, and CD = 0.012 + 0.01*CL^2. It reports local Reynolds-number bounds for every RPM point.

## Critical caveats

The CAD's NACA 4410-style geometry is not accompanied by airfoil polars. The estimated Reynolds numbers are extremely low, so classical thin-airfoil lift slope and the assumed drag polar are not dependable. The model also omits tip/root loss, induced swirl, propeller/guard interaction, ground effect, forward flight, and motor electrical/thermal behavior. KV and battery voltage alone cannot determine loaded RPM or thrust.

Consequently, the estimate must not be used to overwrite the active simulation Kf or to declare the propeller faulty. It only flags a potentially serious thrust shortfall that must be checked with low-Re airfoil data and a real thrust stand using the actual motor, ESC, battery, and propeller.

## Next physical validation

1. Obtain manufacturer thrust/current/RPM data for the exact A2212 motor and candidate propeller, or measure them on a guarded thrust stand.
2. Measure loaded RPM, voltage, current, thrust, and motor temperature at several throttle points.
3. Obtain low-Reynolds-number polars for the actual section, or replace the analytical section model with validated XFOIL/CFD/experimental data where applicable.
4. Only after these measurements, identify Kf and Kq and decide whether the propeller geometry or motor pairing should change.

The active vehicle profile remains unchanged. No PPO training was started.
