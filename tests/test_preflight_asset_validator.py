import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_preflight_assets.py"
SPEC = importlib.util.spec_from_file_location("validate_preflight_assets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_authoritative_asset_contract_is_explicit():
    assert MODULE.EXPECTED_COMPONENTS == 26
    assert MODULE.ASSETS.parent.name == "output"
    assert MODULE.ASSETS.name == "smooth2-parametric"


def test_step_envelope_check_rejects_non_step(tmp_path):
    errors = []
    bad = tmp_path / "not-a-step.step"
    bad.write_text("not a STEP file " * 20, encoding="ascii")
    MODULE.check_step(bad, errors)
    assert any("Missing ISO-10303-21 header" in item for item in errors)
    assert any("Missing END-ISO-10303-21 trailer" in item for item in errors)


def test_step_envelope_check_accepts_minimal_envelope(tmp_path):
    errors = []
    good = tmp_path / "minimal.step"
    good.write_bytes(b"ISO-10303-21;\nHEADER;\n/* " + (b"fixture " * 20) + b"*/\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    MODULE.check_step(good, errors)
    assert errors == []


def test_binary_stl_size_and_triangle_count_are_checked(tmp_path):
    errors = []
    good = tmp_path / "one-triangle.stl"
    good.write_bytes(b"fixture".ljust(80, b" ") + (1).to_bytes(4, "little") + bytes(50))
    MODULE.check_stl(good, errors)
    assert errors == []


def test_stl_envelope_check_rejects_corrupt_file(tmp_path):
    errors = []
    bad = tmp_path / "broken.stl"
    bad.write_bytes(b"not an STL" * 20)
    MODULE.check_stl(bad, errors)
    assert any("STL size/header is inconsistent" in item for item in errors)
