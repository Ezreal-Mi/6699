"""Check that report-facing claims match saved experiment facts.

This is a lightweight guardrail for the current project state. It does not
re-run experiments; it checks that the main prose does not drift away from the
saved result tables and the conservative H1/Phase 6 interpretation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_registry import load_all_facts

DOCS = [
    ROOT / "README.md",
    ROOT / "report" / "paper.tex",
    ROOT / "slides" / "presentation_outline.md",
    ROOT.parent / "The_Expressive_Power_of_Depth_and_Robustness_Formatted" / "main.tex",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def check_h1_numbers(text_by_path: dict[Path, str], errors: list[str]) -> None:
    facts = load_all_facts()["h1"]
    assert hasattr(facts, "ratio_shallow_over_deep")

    required_tokens = [
        str(facts.deep.params),
        str(facts.shallow.params),
        "W=16",
        "2.9",
    ]
    for path, text in text_by_path.items():
        if path.name == "presentation_outline.md":
            required = required_tokens[:3]
        else:
            required = required_tokens
        for token in required:
            _require(token in text, f"{path}: missing H1 token {token!r}", errors)


def check_phase6_conservatism(text_by_path: dict[Path, str], errors: list[str]) -> None:
    forbidden = [
        "Phase 6) confirms",
        "confirms\nthat both crossovers scale as",
        "both crossovers scale as $2^{-k}$",
        "yielding a quantitative",
        "trade-off law",
        "threshold confirmed",
    ]
    for path, text in text_by_path.items():
        for phrase in forbidden:
            _require(phrase not in text, f"{path}: stale strong Phase 6/H4 phrase {phrase!r}", errors)

    paper = text_by_path[ROOT / "report" / "paper.tex"]
    _require("qualitative" in paper, "report/paper.tex: Phase 6 should say qualitative", errors)
    _require("not strong enough to call this a precise law" in paper, "report/paper.tex: missing conservative Phase 6 caveat", errors)
    _require("k=6" in paper and "collapses" in paper, "report/paper.tex: missing k=6 collapse caveat", errors)


def check_latex_list_typos(text_by_path: dict[Path, str], errors: list[str]) -> None:
    for path, text in text_by_path.items():
        if path.suffix != ".tex":
            continue
        _require("\\\\item" not in text, f"{path}: contains literal \\\\item typo", errors)
        _require("\\  \\item" not in text, f"{path}: contains escaped-space item typo", errors)


def main() -> int:
    missing = [path for path in DOCS if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing document: {path}")
        return 2

    text_by_path = {path: _read(path) for path in DOCS}
    errors: list[str] = []
    check_h1_numbers(text_by_path, errors)
    check_phase6_conservatism(text_by_path, errors)
    check_latex_list_typos(text_by_path, errors)

    if errors:
        print("report consistency check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("report consistency check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
