from src.experiment_registry import load_all_facts, load_h1_facts, load_phase6_facts


def test_h1_registry_matches_conservative_rescue_run():
    facts = load_h1_facts()

    assert facts.deep.params == 1953
    assert facts.shallow.params == 1954
    assert facts.deep.width == 16
    assert facts.shallow.width == 651
    assert 2.8 < facts.ratio_shallow_over_deep < 3.0
    assert "partially supported" in facts.interpretation


def test_phase6_registry_keeps_conservative_interpretation():
    facts = load_phase6_facts()

    assert facts.k_values == (2, 3, 4, 5, 6)
    assert facts.valid_trend_ks == (3, 4, 5)
    assert facts.no_gap_ks == (2,)
    assert facts.collapsed_ks == (6,)
    assert "not a precise scaling law" in facts.interpretation


def test_registry_loads_all_phase_facts():
    facts = load_all_facts()

    assert set(facts) == {
        "h1",
        "robustness_baseline",
        "phase2",
        "phase3",
        "phase4",
        "phase5",
        "phase6",
    }
    assert facts["phase2"].h2_supported is True
    assert facts["phase3"].h3_supported is True
    assert facts["phase4"].h4_supported is True
    assert facts["phase5"].h5_supported is True
