# Default browser policy for SkySim (runs in Pyodide).
# obs = [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
# return [roll, pitch, yaw_rate, throttle] with throttle in [0, 1].
#
# This is a simple demonstration controller, not a flight-certified controller.
TARGET_ALT_M = 5.0
HOVER_FEED_FORWARD = 0.34

def _clip(value, low, high):
    return max(low, min(high, value))

def policy(obs):
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy, vz = obs[3], obs[4], obs[5]

    # Simple altitude PD plus gentle position damping.
    throttle = HOVER_FEED_FORWARD + 0.15 * (TARGET_ALT_M - y) - 0.08 * vy
    roll = _clip(-(0.35 * x + 0.45 * vx), -0.30, 0.30)
    pitch = _clip(-(0.35 * z + 0.45 * vz), -0.30, 0.30)

    return [roll, pitch, 0.0, _clip(throttle, 0.0, 1.0)]
