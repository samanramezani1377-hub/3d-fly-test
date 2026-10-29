#!/usr/bin/env python3
"""Apply deterministic reset, frame-contract, and telemetry fixes to pinned SkySim."""
from __future__ import annotations
import argparse
from pathlib import Path

def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match for {label} in {path}; found {count}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    root = parser.parse_args().upstream.resolve()
    header = root / "web/core/drone_core.hpp"
    replace_once(header,
"""        _s = RigidBodyState{};
        _s.position = {x, y, z};
        _s.mass = _mass;
        _armed = false;
        _direct = false;
        _setpoints = {};
        _time = 0.0;
        if (_fc) _fc->reset();
        _battery.reset();
        publish_battery();""",
"""        _s = RigidBodyState{};
        _s.position = {x, y, z};
        _s.mass = _mass;
        _armed = false;
        _direct = false;
        _direct_thr.clear();
        _setpoints = {};
        _time = 0.0;
        _wind = {};
        _t = TelemetryLite{};
        if (_fc) _fc->reset();
        if (_rotors) _rotors->reset();
        if (_effects) _effects->reset();
        _atm.reset_runtime();
        _battery.reset();
        publish_battery();""", "complete episode reset")
    replace_once(header,
"""    // out[12] = pos(3), vel(3), euler(3, Godot rpy rad), body_rates(3)
    void get_obs(double* out) const {
        Vec3d rpy = _s.orientation.to_euler_rpy();
        out[0]=_s.position.x; out[1]=_s.position.y; out[2]=_s.position.z;
        out[3]=_s.velocity.x; out[4]=_s.velocity.y; out[5]=_s.velocity.z;
        out[6]=rpy.x; out[7]=rpy.y; out[8]=rpy.z;
        out[9]=_s.angular_velocity.x; out[10]=_s.angular_velocity.y; out[11]=_s.angular_velocity.z;
    }""",
"""    // World-Godot position/velocity followed by body-FRD Euler angles/rates.
    // Attitude and body rates match the FlightController's FRD feedback and setpoints.
    void get_obs(double* out) const {
        const Quat q_frd = frames::godot_to_ned(_s.orientation);
        const Vec3d rpy_frd = q_frd.to_euler_rpy();
        const Vec3d rates_frd = frames::godot_to_ned(_s.angular_velocity);
        out[0]=_s.position.x; out[1]=_s.position.y; out[2]=_s.position.z;
        out[3]=_s.velocity.x; out[4]=_s.velocity.y; out[5]=_s.velocity.z;
        out[6]=rpy_frd.x; out[7]=rpy_frd.y; out[8]=rpy_frd.z;
        out[9]=rates_frd.x; out[10]=rates_frd.y; out[11]=rates_frd.z;
    }""", "FRD observation contract")
    replace_once(header,
"""    const Battery&        battery() const { return _battery; }
    double time() const { return _time; }""",
"""    const Battery&        battery() const { return _battery; }
    double rotor_thrust(size_t i) const {
        return (_rotors && i < _rotors->states().size()) ? _rotors->states()[i].thrust : 0.0;
    }
    double rotor_omega(size_t i) const {
        return (_rotors && i < _rotors->states().size()) ? _rotors->states()[i].omega : 0.0;
    }
    double time() const { return _time; }""", "rotor diagnostic accessors")

    blade = root / "include/aero/blade_element.hpp"
    replace_once(blade,
"""    const RotorConfig& config() const noexcept { return _cfg; }

    // Runtime supply-voltage scale""",
"""    void reset() noexcept {
        _prev_vi = 0.0;
        _last_thrust = 0.0;
        _supply_scale = 1.0;
    }

    const RotorConfig& config() const noexcept { return _cfg; }

    // Runtime supply-voltage scale""", "blade element reset")
    replace_once(blade,
"""    void add_rotor(RotorConfig cfg) {
        _solvers.emplace_back(std::move(cfg));
        _states.emplace_back();
    }

    // Set normalised throttle""",
"""    void add_rotor(RotorConfig cfg) {
        _solvers.emplace_back(std::move(cfg));
        _states.emplace_back();
    }

    void reset() noexcept {
        for (auto& state : _states) state = RotorState{};
        for (auto& solver : _solvers) solver.reset();
    }

    // Set normalised throttle""", "rotor array reset")

    atmosphere = root / "include/aero/atmosphere.hpp"
    replace_once(atmosphere,
"""    void set_turbulence(DrydenParams p) noexcept { _turb = p; }
    void set_wind_global(Vec3d w)       noexcept { _wind_global = w; }""",
"""    void set_turbulence(DrydenParams p) noexcept { _turb = p; }
    void set_wind_global(Vec3d w)       noexcept { _wind_global = w; }
    // Clear episode-local turbulence and restart its deterministic RNG.
    void reset_runtime(uint64_t seed = 0xDEADBEEFCAFEBABEULL) noexcept {
        _turb_state = {};
        _lcg_state = seed;
    }""", "atmosphere reset")

    effects = root / "include/aero/aero_effects.hpp"
    replace_once(effects,
"""private:
    double _severity{0.0};
    double _noise_phase{0.0};
};""",
"""    void reset() noexcept {
        _severity = 0.0;
        _noise_phase = 0.0;
    }

private:
    double _severity{0.0};
    double _noise_phase{0.0};
};""", "VRS reset")
    replace_once(effects,
"""    explicit AeroEffectsBundle(double rotor_radius)
        : ground_effect(rotor_radius) {}
};""",
"""    explicit AeroEffectsBundle(double rotor_radius)
        : ground_effect(rotor_radius) {}

    void reset() noexcept { vrs.reset(); }
};""", "aero bundle reset")

    cpp = root / "web/core/drone_core.cpp"
    replace_once(cpp,
"""    _t.altitude = alt; _t.vertical_speed = _s.velocity.y; _t.ground_speed = lateral_speed;""",
"""    _t.altitude = _s.position.y;
    _t.vertical_speed = _s.velocity.y;
    _t.ground_speed = std::sqrt(_s.velocity.x*_s.velocity.x + _s.velocity.z*_s.velocity.z);""",
    "post-integration telemetry")
    capi = root / "web/core/capi.cpp"
    replace_once(capi,
"""KA double skysim_thrust(DroneCore* d)  { return d->telem().total_thrust; }""",
"""KA double skysim_thrust(DroneCore* d)  { return d->telem().total_thrust; }
KA double skysim_rotor_thrust(DroneCore* d,int i) { return i >= 0 ? d->rotor_thrust(static_cast<size_t>(i)) : 0.0; }
KA double skysim_rotor_omega(DroneCore* d,int i) { return i >= 0 ? d->rotor_omega(static_cast<size_t>(i)) : 0.0; }""",
    "per-rotor C API")
    print("Applied native SkySim reset, FRD observation, and telemetry patches.")

if __name__ == "__main__":
    main()
