"""Canonical experiment facts for reports and consistency checks.

This module is the small source of truth for claims that are easy to let drift
across the README, paper, slides, and generated tables. It reads saved result
files when available and keeps the conservative interpretation of the current
project explicit.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    depth: int
    width: int
    params: int


@dataclass(frozen=True)
class H1Facts:
    deep: ModelSpec
    shallow: ModelSpec
    deep_mse_mean: float
    deep_mse_std: float
    shallow_mse_mean: float
    shallow_mse_std: float
    ratio_shallow_over_deep: float
    interpretation: str


@dataclass(frozen=True)
class RobustnessBaselineFacts:
    deep: ModelSpec
    shallow: ModelSpec
    interpretation: str


@dataclass(frozen=True)
class Phase6Facts:
    k_values: tuple[int, ...]
    valid_trend_ks: tuple[int, ...]
    collapsed_ks: tuple[int, ...]
    no_gap_ks: tuple[int, ...]
    interpretation: str


def load_h1_facts(table_dir: Path = TABLE_DIR) -> H1Facts:
    """Load the current Phase 1 W=16 rescue facts from saved CSV output."""
    rows: dict[str, dict[str, str]] = {}
    with (table_dir / "phase1_summary.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["model"]] = row

    deep_mean = float(rows["deep"]["final_mse_mean"])
    shallow_mean = float(rows["shallow"]["final_mse_mean"])
    ratio = shallow_mean / deep_mean

    return H1Facts(
        deep=ModelSpec("deep", depth=9, width=16, params=int(rows["deep"]["params"])),
        shallow=ModelSpec("shallow", depth=2, width=651, params=int(rows["shallow"]["params"])),
        deep_mse_mean=deep_mean,
        deep_mse_std=float(rows["deep"]["final_mse_std"]),
        shallow_mse_mean=shallow_mean,
        shallow_mse_std=float(rows["shallow"]["final_mse_std"]),
        ratio_shallow_over_deep=ratio,
        interpretation="partially supported; moderate depth advantage, below the original 10x target",
    )


def load_robustness_baseline_facts() -> RobustnessBaselineFacts:
    """Return the fixed Phase 2--5 baseline architecture facts."""
    return RobustnessBaselineFacts(
        deep=ModelSpec("deep", depth=9, width=8, params=529),
        shallow=ModelSpec("shallow", depth=2, width=176, params=529),
        interpretation="separate W=8 robustness baseline; do not merge with the W=16 H1 rescue",
    )


def load_phase6_facts(table_dir: Path = TABLE_DIR) -> Phase6Facts:
    """Load Phase 6 scaling facts and preserve the conservative interpretation."""
    data = json.loads((table_dir / "phase6_summary.json").read_text())
    return Phase6Facts(
        k_values=tuple(int(k) for k in data["k_values"]),
        valid_trend_ks=(3, 4, 5),
        collapsed_ks=(6,),
        no_gap_ks=(2,),
        interpretation=(
            "qualitative complexity-robustness trend only; not a precise scaling law "
            "because k=2 has no clean gap, k=5 is marginal, and k=6 collapses"
        ),
    )


def load_all_facts() -> dict[str, object]:
    """Load all canonical facts used by report consistency checks."""
    return {
        "h1": load_h1_facts(),
        "robustness_baseline": load_robustness_baseline_facts(),
        "phase6": load_phase6_facts(),
    }
