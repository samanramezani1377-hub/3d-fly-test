import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("smooth2_propulsion", ROOT / "scripts/derive_smooth2_propulsion.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_smooth2_cad_propulsion_estimate():
    report = MODULE.derive()
    assert report["source"]["source_repository_modified"] is False
    assert report["propeller"]["diameter_m"] == pytest.approx(0.0748020812)
    assert report["motor"]["KV"] == 1000
    assert report["active_vehicle_profile_modified"] is False
    assert report["physical_flight_eligible"] is False
