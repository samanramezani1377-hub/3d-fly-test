import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "smooth2_propulsion", ROOT / "scripts/derive_smooth2_propulsion.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_smooth2_cad_propulsion_estimate():
    report = MODULE.derive()
    cases = report["BEMT_model"]["rpm_cases"]
    assert report["source"]["source_repository_modified"] is False
    assert abs(report["propeller"]["diameter_m"] - 0.0748020812) < 1e-9
    assert report["propeller"]["blades"] == 2
    assert len(report["propeller"]["airfoil"]["radial_stations"]) == 11
    assert report["motor_reference"]["KV_rpm_per_volt"] == 1000
    assert report["motor_reference"]["loaded_operating_rpm_known"] is False
    assert len(cases) == 4
    assert all(a["thrust_N_per_rotor"] < b["thrust_N_per_rotor"]
               for a, b in zip(cases, cases[1:]))
    assert all(case["Reynolds_min"] > 0 for case in cases)
    assert report["active_vehicle_profile_modified"] is False
    assert report["physical_flight_eligible"] is False
