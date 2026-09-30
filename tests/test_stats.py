import math

import pytest

from rlt_rsi.stats import mean_ci95, t_critical_95


def test_t_critical_known_values():
    assert t_critical_95(2) == pytest.approx(4.303)
    assert t_critical_95(9) == pytest.approx(2.262)
    assert t_critical_95(1000) == pytest.approx(1.960)


def test_t_critical_is_monotone_non_increasing():
    values = [t_critical_95(df) for df in range(1, 200)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_t_critical_rejects_zero_df():
    with pytest.raises(ValueError):
        t_critical_95(0)


def test_mean_ci95_matches_hand_computation():
    low, high = mean_ci95([0.0, 0.1, 0.2])
    half = 4.303 * math.sqrt(0.01 / 3)
    assert low == pytest.approx(0.1 - half)
    assert high == pytest.approx(0.1 + half)


def test_mean_ci95_undefined_for_single_value():
    assert mean_ci95([0.3]) == (None, None)


def test_mean_ci95_empty_input_has_no_interval():
    assert mean_ci95([]) == (None, None)


def test_mean_ci95_zero_variance_collapses_to_the_mean():
    low, high = mean_ci95([0.05, 0.05, 0.05])
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.05)


def test_mean_ci95_two_values_uses_df_one():
    low, high = mean_ci95([0.0, 0.2])
    half = 12.706 * math.sqrt(0.02 / 2)
    assert low == pytest.approx(0.1 - half)
    assert high == pytest.approx(0.1 + half)


def test_t_critical_between_table_rows_is_not_narrower_than_the_next_row_below():
    # True values: df=35 -> 2.030, df=50 -> 2.009, df=100 -> 1.984. The lookup must never be narrower.
    assert t_critical_95(35) >= 2.030
    assert t_critical_95(50) >= 2.009
    assert t_critical_95(100) >= 1.984
    assert t_critical_95(30) == pytest.approx(2.042)
    assert t_critical_95(40) == pytest.approx(2.021)


def test_paired_ci_is_reported_in_payload_and_report(tmp_path):
    from rlt_rsi.train import main
    import json

    out = tmp_path / "run.json"
    assert main([
        "--backend", "numpy", "--seeds", "7,42,123", "--train-size", "32", "--dev-size", "16",
        "--heldout-size", "16", "--epochs", "1", "--output", str(out),
    ]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    for row in payload["results"]:
        if row["model"] == "baseline":
            continue
        pd = row["paired_delta_vs_baseline"]["heldout_accuracy"]
        expected = mean_ci95(pd["per_seed"])
        assert (pd["ci95_low"], pd["ci95_high"]) == pytest.approx(expected)
        assert pd["ci95_low"] <= pd["mean"] <= pd["ci95_high"]
    report = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "95% CI (t)" in report


def test_report_tolerates_payloads_written_before_ci_fields_existed(tmp_path):
    import copy
    import json

    from rlt_rsi.train import main, write_report

    out = tmp_path / "run.json"
    assert main([
        "--backend", "numpy", "--seeds", "7,42", "--train-size", "32", "--dev-size", "16",
        "--heldout-size", "16", "--epochs", "1", "--output", str(out),
    ]) == 0
    payload = copy.deepcopy(json.loads(out.read_text(encoding="utf-8")))
    for row in payload["results"]:
        for values in row["paired_delta_vs_baseline"].values():
            values.pop("ci95_low", None)
            values.pop("ci95_high", None)
    path = tmp_path / "legacy.md"
    write_report(path, payload)
    assert "n/a (N<2)" in path.read_text(encoding="utf-8")
