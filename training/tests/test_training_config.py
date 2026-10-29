import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "train_ppo.py"
SPEC = importlib.util.spec_from_file_location("train_ppo", MODULE_PATH)
train_ppo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_ppo)


def test_defaults_are_prototype_only():
    args = train_ppo.parse_args([])
    assert args.timesteps == 200_000
    assert args.seed == 42
    assert args.port == 5557
    assert args.run_name == "prototype-hover"
    assert train_ppo.UPSTREAM_REVISION == "a31864f999418531775980b879979ce4c57e791d"
    assert "not verified equivalent" in train_ppo.BACKEND_WARNING


def test_arguments_can_be_overridden():
    args = train_ppo.parse_args([
        "--timesteps", "1000",
        "--seed", "7",
        "--target-y", "2.5",
        "--run-name", "smoke-test",
    ])
    assert args.timesteps == 1000
    assert args.seed == 7
    assert args.target_y == 2.5
    assert args.run_name == "smoke-test"
