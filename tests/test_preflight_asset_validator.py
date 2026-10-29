import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_preflight_assets.py"
SPEC = importlib.util.spec_from_file_location("validate_preflight_assets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_expected_asset_contract_is_explicit():
    assert MODULE.EXPECTED_CHUNKS == 6
    assert MODULE.EXPECTED_COMPONENTS == 26
    assert MODULE.EXPECTED_CHUNK_BYTES == 5_970_685


def test_step_envelope_check_rejects_non_step(tmp_path):
    errors = []
    bad = tmp_path / "not-a-step.step"
    bad.write_text("not a STEP file", encoding="ascii")
    MODULE.check_step(bad, errors)
    assert any("Missing ISO-10303-21 header" in item for item in errors)
    assert any("Missing END-ISO-10303-21 trailer" in item for item in errors)


def test_step_envelope_check_accepts_minimal_envelope(tmp_path):
    errors = []
    good = tmp_path / "minimal.step"
    good.write_bytes(b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    MODULE.check_step(good, errors)
    assert errors == []
