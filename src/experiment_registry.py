"""Canonical experiment facts for the report and presentation.

This module reads saved result files and exposes the small set of numbers and
interpretations reused across the README, paper, slides, and generated tables.
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
class Phase2Facts:
    k: int
    theoretical_sigma_star: float
    empirical_sigma_star: float
    h2_supported: bool


@dataclass(frozen=True)
class Phase3Facts:
    deep_degradation_at_max_sigma: float
    shallow_degradation_at_max_sigma: float
    h3_supported: bool


@dataclass(frozen=True)
class Phase4Facts:
    theory_eps_star: float
    fgsm_eps_star: float
    pgd_eps_star: float
    h4_supported: bool


@dataclass(frozen=True)
class Phase5Facts:
    deep_lip_fd_mean: float
    shallow_lip_fd_mean: float
    lip_fd_ratio: float
    deep_lip_higher_all_seeds: bool
    h5_supported: bool


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


def load_phase2_facts(table_dir: Path = TABLE_DIR) -> Phase2Facts:
    """Load Phase 2 input-noise crossover facts."""
    data = json.loads((table_dir / "phase2_summary.json").read_text())
    return Phase2Facts(
        k=int(data["k"]),
        theoretical_sigma_star=float(data["theoretical_sigma_star"]),
        empirical_sigma_star=float(data["empirical_sigma_star"]),
        h2_supported=bool(data["H2_supported"]),
    )


def load_phase3_facts(table_dir: Path = TABLE_DIR) -> Phase3Facts:
    """Load Phase 3 label-noise degradation facts."""
    data = json.loads((table_dir / "phase3_summary.json").read_text())
    return Phase3Facts(
        deep_degradation_at_max_sigma=float(data["deep_degradation_at_max_sigma"]),
        shallow_degradation_at_max_sigma=float(data["shallow_degradation_at_max_sigma"]),
        h3_supported=bool(data["H3_supported"]),
    )


def load_phase4_facts(table_dir: Path = TABLE_DIR) -> Phase4Facts:
    """Load Phase 4 adversarial crossover facts."""
    data = json.loads((table_dir / "phase4_summary.json").read_text())
    return Phase4Facts(
        theory_eps_star=float(data["theory_eps_star"]),
        fgsm_eps_star=float(data["fgsm_eps_star"]),
        pgd_eps_star=float(data["pgd_eps_star"]),
        h4_supported=bool(data["H4_supported"]),
    )


def load_phase5_facts(table_dir: Path = TABLE_DIR) -> Phase5Facts:
    """Load Phase 5 Lipschitz diagnostic facts."""
    data = json.loads((table_dir / "phase5_summary.json").read_text())
    return Phase5Facts(
        deep_lip_fd_mean=float(data["deep_lip_fd_mean"]),
        shallow_lip_fd_mean=float(data["shallow_lip_fd_mean"]),
        lip_fd_ratio=float(data["lip_fd_ratio"]),
        deep_lip_higher_all_seeds=bool(data["deep_lip_higher_all_seeds"]),
        h5_supported=bool(data["H5_supported"]),
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
    """Load all canonical facts used by the report-facing summaries."""
    return {
        "h1": load_h1_facts(),
        "robustness_baseline": load_robustness_baseline_facts(),
        "phase2": load_phase2_facts(),
        "phase3": load_phase3_facts(),
        "phase4": load_phase4_facts(),
        "phase5": load_phase5_facts(),
        "phase6": load_phase6_facts(),
    }
