from scripts.rules_playtest import run_rules_matrix


def test_coc_rules_interactive_matrix_passes():
    report = run_rules_matrix()
    assert report["passed"] is True
    assert report["case_count"] == 7
    assert {case["case"] for case in report["cases"]} == {
        "accept_failure",
        "spend_luck",
        "push_success",
        "push_failure",
        "opposed_combat_bonus_die",
        "sanity_check",
        "major_wound",
    }
