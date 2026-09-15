"""Report-generator honesty tests.

These tests call :func:`rlt_rsi.train.write_report` with payloads derived from a
real (tiny) NumPy run and then mutated, so the assertions are about the report
generator's behaviour rather than about any particular experiment result.
"""

import copy
import json

import pytest

from rlt_rsi.train import main, write_report


@pytest.fixture(scope="module")
def base_payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("payload") / "run.json"
    assert main([
        "--backend", "numpy",
        "--seeds", "7,42",
        "--train-size", "32",
        "--dev-size", "16",
        "--heldout-size", "16",
        "--epochs", "2",
        "--output", str(out),
    ]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _render(payload, tmp_path, name="report.md"):
    path = tmp_path / name
    write_report(path, payload)
    return path.read_text(encoding="utf-8")


def _row(md, prefix):
    for line in md.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"row starting with {prefix!r} not found in report")


def _looped_row(md, loops=2):
    return _row(md, f"| looped | {loops} |")


def _h1_row(md):
    return _row(md, "| H1 |")


@pytest.mark.parametrize("decision", ["improvement", "flat", "regression"])
def test_decision_words_come_from_the_payload(tmp_path, base_payload, decision):
    payload = copy.deepcopy(base_payload)
    for row in payload["results"]:
        if row["model"] != "baseline":
            row["decision"] = decision
    for hyp in payload["hypotheses"]:
        if hyp["id"] == "H1":
            for config in hyp["per_config"]:
                config["result"] = decision
            hyp["result"] = decision

    md = _render(payload, tmp_path, f"{decision}.md")

    assert _looped_row(md).rstrip().endswith(f"| {decision} |")
    assert _h1_row(md).rstrip().endswith(f"| {decision} |")


def test_numpy_payload_never_claims_pytorch_is_missing_when_metadata_says_present(tmp_path, base_payload):
    payload = copy.deepcopy(base_payload)
    payload["environment"]["torch"] = "9.9.9"
    payload["environment"]["torch_available"] = True

    md = _render(payload, tmp_path, "torch-present.md")

    assert "PyTorch 9.9.9 is available in this environment" in md
    assert "PyTorch is not installed" not in md
    assert "not importable" not in md


def test_report_claims_pytorch_is_missing_only_when_metadata_says_so(tmp_path, base_payload):
    payload = copy.deepcopy(base_payload)
    payload["environment"]["torch"] = None
    payload["environment"]["torch_available"] = False

    md = _render(payload, tmp_path, "torch-absent.md")

    assert "PyTorch is not installed in this environment" in md
    assert "is available in this environment" not in md


def test_torch_payload_reports_the_device_actually_used(tmp_path, base_payload):
    payload = copy.deepcopy(base_payload)
    payload["backend"] = "torch"
    payload["training"]["device_requested"] = "auto"
    payload["training"]["device"] = "mps"
    payload["training"]["dtype"] = "float32"
    payload["training"]["optimizer"] = "AdamW"
    payload["environment"]["torch"] = "2.14.0"
    payload["environment"]["torch_available"] = True

    md = _render(payload, tmp_path, "torch-mps.md")

    assert "was used for this run on device `mps`" in md
    assert "`auto` -> resolved `mps`" in md
    assert "AdamW" in md
    assert "float32" in md


def test_report_contains_no_stale_hardcoded_claims(tmp_path, base_payload):
    payload = copy.deepcopy(base_payload)
    payload["environment"]["torch"] = "2.14.0"
    payload["environment"]["torch_available"] = True

    md = _render(payload, tmp_path, "stale.md")

    # Removed unconditional claims from the previous generator.
    assert "PyTorch is not installed" not in md
    assert "yield **flat**" not in md
    assert "held-out accuracy remains near the random chance threshold" not in md
    assert "4,656 parameters" not in md

    total = payload["results"][0]["parameter_counts"]["total"]
    assert total == payload["results"][0]["parameter_counts"]["trainable"] + payload["results"][0]["parameter_counts"]["frozen"]
    assert f"{total:,}" in md
    assert _row(md, "| baseline | 1 |").startswith(f"| baseline | 1 | {total:,} |")

    # Scope and honest-protocol statements must be present.
    assert "operational decision rule only" in md
    assert "does **not** establish preregistration" in md
    assert "sequential block applications" in md
    assert "different architectures" in md


def test_report_includes_h2_and_paired_per_seed_deltas(tmp_path, base_payload):
    payload = copy.deepcopy(base_payload)
    md = _render(payload, tmp_path, "h2.md")

    assert _row(md, "| H2 |")
    assert "length_generalization_drop" in md
    assert "in_distribution_accuracy(dev only, lengths 4-8) - heldout_accuracy" in md
    assert "Paired per-seed differences vs the baseline" in md
    # Every non-baseline configuration reports its per-seed paired deltas.
    for row in payload["results"]:
        if row["model"] == "baseline":
            continue
        assert row["paired_delta_vs_baseline"]["heldout_accuracy"]["per_seed"]
