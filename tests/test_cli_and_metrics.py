from __future__ import annotations

import json

import pytest
from conftest import FakeModel, write_csv
from prometheus_client import generate_latest

from sentiment_pipeline import cli
from sentiment_pipeline.metrics import Metrics, build_registry


@pytest.fixture
def use_fake_model(monkeypatch):
    model = FakeModel(poison=["BOOM"])
    monkeypatch.setattr(cli, "load_model", lambda args: model)
    return model


def run_cli(tmp_path, *extra: str) -> tuple[int, dict]:
    output = tmp_path / "out" / "predictions.csv"
    code = cli.main(["--input", str(tmp_path / "in.csv"), "--output", str(output), *extra])
    summary = json.loads(output.with_suffix(".metrics.json").read_text())
    return code, summary


def test_successful_run_exits_zero_and_writes_summary(tmp_path, use_fake_model):
    write_csv(tmp_path / "in.csv", ["text"], [["good"], ["bad"]])

    code, summary = run_cli(tmp_path)

    assert code == cli.EXIT_OK
    assert summary["status"] == "completed"
    assert summary["rows_read"] == 2
    assert summary["predictions_by_label"] == {"positive": 1, "negative": 1}
    # Provenance: the output can be traced to an exact input, model and run.
    assert len(summary["run"]["input_sha256"]) == 64
    assert summary["run"]["model_revision"]
    assert summary["run"]["run_id"]


def test_failure_rate_above_threshold_exits_two_but_publishes_output(tmp_path, use_fake_model):
    write_csv(tmp_path / "in.csv", ["text"], [["good"], ["BOOM"], [""], ["bad"]])

    code, summary = run_cli(tmp_path, "--max-failure-rate", "0.25")

    assert code == cli.EXIT_FAILURE_RATE
    assert summary["status"] == "completed_with_errors"
    assert summary["failure_rate"] == 0.5
    assert (tmp_path / "out" / "predictions.csv").exists()


def test_bad_input_exits_one_and_still_writes_summary(tmp_path, use_fake_model):
    write_csv(tmp_path / "in.csv", ["review"], [["good"]])

    code, summary = run_cli(tmp_path)

    assert code == cli.EXIT_FATAL
    assert summary["status"] == "failed"
    assert not (tmp_path / "out" / "predictions.csv").exists()


def test_invalid_batch_size_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--input", "x.csv", "--batch-size", "0"])


def test_prometheus_exposition_reflects_metrics():
    metrics = Metrics(info={"run_id": "abc123"})
    for score in (0.99, 0.95, 0.55):
        metrics.record_prediction("positive", score)
    metrics.record_failure("empty_text")
    metrics.record_batch(0.5, fallback=True)

    text = generate_latest(build_registry(metrics)).decode()

    assert 'sentiment_run_info{run_id="abc123"} 1.0' in text
    assert 'sentiment_predictions_total{label="positive"} 3.0' in text
    assert "sentiment_low_confidence_predictions_total 1.0" in text
    assert 'sentiment_rows_total{status="succeeded"} 3.0' in text
    assert 'sentiment_rows_total{status="failed"} 1.0' in text
    assert 'sentiment_row_failures_total{reason="empty_text"} 1.0' in text
    assert 'sentiment_row_failures_total{reason="inference_error"} 0.0' in text
    assert "sentiment_batch_fallbacks_total 1.0" in text
    assert "sentiment_inference_seconds_total 0.5" in text
