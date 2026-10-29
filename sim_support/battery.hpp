#pragma once
// Compatibility battery model for SkySim's browser core.
// The pinned upstream revision documents include/power/battery.hpp but omits
// that file from the repository tree. This deterministic first-order LiPo model
// provides the API expected by DroneCore without claiming cell-level accuracy.
#include <algorithm>
#include <cmath>

namespace dronesim {

class Battery {
public:
    void configure(double capacity_mah, int cells, double resistance_ohm, double nominal_voltage) {
        _capacity_mah = std::max(100.0, capacity_mah);
        _cells = std::max(1, cells);
        _resistance_ohm = std::max(0.0, resistance_ohm);
        _nominal_voltage = std::max(1.0, nominal_voltage);
        reset();
    }

    void reset() {
        _soc = 1.0;
        _mah_used = 0.0;
        _current = 0.0;
        _ocv = ocv_for_soc(_soc);
        _terminal = _ocv;
        _low_warning = false;
        _cutoff = false;
    }

    double terminal_voltage() const { return _terminal; }
    double ocv_voltage() const { return _ocv; }
    double current() const { return _current; }
    double soc() const { return _soc; }
    double mah_used() const { return _mah_used; }
    bool low_warning() const { return _low_warning; }
    bool cutoff() const { return _cutoff; }

    void update(double power_w, double dt) {
        if (dt <= 0.0) return;
        const double voc = ocv_for_soc(_soc);
        const double r = _resistance_ohm;
        double amps = 0.0;
        if (power_w > 0.0) {
            if (r > 1e-9) {
                const double disc = std::max(0.0, voc * voc - 4.0 * r * power_w);
                amps = (voc - std::sqrt(disc)) / (2.0 * r);
            } else {
                amps = power_w / std::max(voc, 1e-6);
            }
        }
        _current = std::max(0.0, amps);
        _terminal = std::max(0.0, voc - _current * r);
        _mah_used += _current * dt / 3.6;
        _soc = std::clamp(_soc - (_current * dt) / (_capacity_mah * 3.6), 0.0, 1.0);
        _ocv = ocv_for_soc(_soc);
        const double scale = (_nominal_voltage / static_cast<double>(_cells)) / 3.7;
        const double cell_v = _terminal / static_cast<double>(_cells) / std::max(scale, 1e-6);
        _low_warning = cell_v <= 3.50;
        _cutoff = cell_v <= 3.30 || _soc <= 0.0;
    }

private:
    double ocv_for_soc(double soc) const {
        // Piecewise-linear LiPo-shaped per-cell curve, scaled to configured nominal V.
        constexpr double xs[] = {0.00, 0.04, 0.20, 0.50, 0.80, 1.00};
        constexpr double ys[] = {3.30, 3.30, 3.50, 3.70, 4.00, 4.20};
        double cell = ys[0];
        if (soc >= 1.0) cell = ys[5];
        else if (soc > 0.0) {
            for (int i = 1; i < 6; ++i) {
                if (soc <= xs[i]) {
                    const double f = (soc - xs[i-1]) / (xs[i] - xs[i-1]);
                    cell = ys[i-1] + f * (ys[i] - ys[i-1]);
                    break;
                }
            }
        }
        const double scale = (_nominal_voltage / static_cast<double>(_cells)) / 3.7;
        return cell * scale * static_cast<double>(_cells);
    }

    double _capacity_mah{4000.0};
    int _cells{4};
    double _resistance_ohm{0.030};
    double _nominal_voltage{14.8};
    double _soc{1.0};
    double _mah_used{0.0};
    double _current{0.0};
    double _ocv{16.8};
    double _terminal{16.8};
    bool _low_warning{false};
    bool _cutoff{false};
};

} // namespace dronesim
