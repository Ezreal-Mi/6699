import importlib.util
from pathlib import Path


def load_phase1_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_fixed.py"
    spec = importlib.util.spec_from_file_location("run_phase1_fixed", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase1_runner_parses_seed_lists():
    module = load_phase1_script()

    assert module.parse_seed_list("0,1,4") == [0, 1, 4]


def test_phase1_runner_quick_mode_overrides_expensive_defaults():
    module = load_phase1_script()
    args = module.parse_args(["--quick", "--epochs", "30000", "--seeds", "0,1,2"])

    assert args.seeds == [0]
    assert args.epochs == 1000
    assert args.max_restarts == 0
    assert args.log_every == 0
