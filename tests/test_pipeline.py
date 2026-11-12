from __future__ import annotations

import csv

import pytest
from conftest import FakeModel, read_output, write_csv

from sentiment_pipeline.csv_io import InputSchemaError
from sentiment_pipeline.metrics import Metrics
from sentiment_pipeline.pipeline import PipelineAborted, PipelineConfig, run


def make_config(tmp_path, **overrides) -> PipelineConfig:
    defaults = dict(
        input_path=tmp_path / "in.csv",
        output_path=tmp_path / "out" / "predictions.csv",
        batch_size=4,
    )
    return PipelineConfig(**{**defaults, **overrides})


def test_processes_all_rows_in_batches_and_preserves_order(tmp_path, fake_model):
    texts = [f"good review {i}" if i % 2 else f"bad review {i}" for i in range(10)]
    write_csv(tmp_path / "in.csv", ["text"], [[t] for t in texts])
    config = make_config(tmp_path)
    metrics = Metrics()

    run(config, lambda: fake_model, metrics)

    assert [len(c) for c in fake_model.calls] == [4, 4, 2]
    out = read_output(config.output_path)
    assert [r["text"] for r in out] == texts
    assert [r["row_number"] for r in out] == [str(i) for i in range(1, 11)]
    assert out[0]["predicted_label"] == "negative"
    assert out[1]["predicted_label"] == "positive"
    assert all(r["status"] == "ok" for r in out)
    assert metrics.summary()["rows_succeeded"] == 10


def test_empty_text_fails_without_calling_model(tmp_path, fake_model):
    write_csv(tmp_path / "in.csv", ["text"], [["good"], [""], ["   "], ["bad"]])
    config = make_config(tmp_path)
    metrics = Metrics()

    run(config, lambda: fake_model, metrics)

    assert fake_model.calls == [["bad", "good"]]  # sorted by length within the window
    out = read_output(config.output_path)
    assert [r["status"] for r in out] == ["ok", "failed", "failed", "ok"]
    assert out[1]["error"].startswith("empty_text")
    assert metrics.summary()["failures_by_reason"] == {"empty_text": 2}


def test_poison_row_is_isolated_and_rest_of_batch_succeeds(tmp_path):
    model = FakeModel(poison=["BOOM"])
    write_csv(tmp_path / "in.csv", ["text"], [["good 1"], ["BOOM"], ["bad 3"], ["good 4"]])
    config = make_config(tmp_path)
    metrics = Metrics()

    run(config, lambda: model, metrics)

    out = read_output(config.output_path)
    assert [r["status"] for r in out] == ["ok", "failed", "ok", "ok"]
    assert out[1]["error"].startswith("inference_error: ValueError")
    summary = metrics.summary()
    assert summary["batch_fallbacks"] == 1
    assert summary["failures_by_reason"] == {"inference_error": 1}


def test_wrong_prediction_count_is_treated_as_batch_failure(tmp_path):
    class DroppingModel(FakeModel):
        def predict(self, texts):
            predictions = super().predict(texts)
            return predictions[:-1] if len(texts) > 1 else predictions

    write_csv(tmp_path / "in.csv", ["text"], [["a"], ["b"], ["c"]])
    config = make_config(tmp_path)
    metrics = Metrics()

    run(config, lambda: DroppingModel(), metrics)

    assert all(r["status"] == "ok" for r in read_output(config.output_path))
    assert metrics.summary()["batch_fallbacks"] == 1


def test_every_input_record_yields_exactly_one_output_record(tmp_path):
    (tmp_path / "in.csv").write_text(
        "id,text\n"
        "1,good\n"
        "2,\n"
        "3,too,many,fields\n"
        '4,"multi\nline good"\n'
        "\n"  # blank line: not a record
        "5,BOOM\n",
        encoding="utf-8",
    )
    config = make_config(tmp_path, id_column="id")
    metrics = Metrics()

    run(config, lambda: FakeModel(poison=["BOOM"]), metrics)

    out = read_output(config.output_path)
    summary = metrics.summary()
    assert len(out) == summary["rows_read"] == 5
    assert summary["rows_read"] == summary["rows_succeeded"] + summary["rows_failed"]
    assert [r["id"] for r in out] == ["1", "2", "3", "4", "5"]
    assert [r["status"] for r in out] == ["ok", "failed", "failed", "ok", "failed"]
    assert out[2]["error"] == "malformed_csv: expected 2 fields, got 4"


def test_csv_parse_error_is_recorded_and_reading_continues(tmp_path, fake_model):
    old_limit = csv.field_size_limit()
    csv.field_size_limit(100)
    try:
        write_csv(tmp_path / "in.csv", ["text"], [["good"], ["x" * 500], ["bad"]])
        config = make_config(tmp_path)
        run(config, lambda: fake_model, Metrics())
    finally:
        csv.field_size_limit(old_limit)

    out = read_output(config.output_path)
    assert [r["status"] for r in out] == ["ok", "failed", "ok"]
    assert out[1]["error"].startswith("malformed_csv: field larger than field limit")


def test_missing_text_column_fails_fast_without_creating_output(tmp_path, fake_model):
    write_csv(tmp_path / "in.csv", ["review"], [["good"]])
    config = make_config(tmp_path)

    with pytest.raises(InputSchemaError, match="column 'text' not found"):
        run(config, lambda: fake_model, Metrics())

    assert not config.output_path.exists()
    assert not config.output_path.with_name("predictions.csv.partial").exists()


def test_missing_input_file_raises_schema_error(tmp_path, fake_model):
    with pytest.raises(InputSchemaError, match="not found"):
        run(make_config(tmp_path), lambda: fake_model, Metrics())


def test_utf8_bom_header_is_handled(tmp_path, fake_model):
    (tmp_path / "in.csv").write_bytes("﻿text\ngood\n".encode())
    config = make_config(tmp_path)

    run(config, lambda: fake_model, Metrics())

    assert read_output(config.output_path)[0]["status"] == "ok"


def test_systemic_model_failure_aborts_and_does_not_publish_output(tmp_path):
    write_csv(tmp_path / "in.csv", ["text"], [[f"row {i}"] for i in range(40)])
    config = make_config(tmp_path, max_consecutive_failed_batches=3)
    model = FakeModel(always_fail=True)

    with pytest.raises(PipelineAborted):
        run(config, lambda: model, Metrics())

    # 3 batches x (1 batch call + 4 single-row retries), then stop.
    assert len(model.calls) == 15
    assert not config.output_path.exists()
    assert config.output_path.with_name("predictions.csv.partial").exists()


def test_accuracy_is_computed_when_label_column_given(tmp_path, fake_model):
    write_csv(
        tmp_path / "in.csv",
        ["text", "label"],
        [["good", "positive"], ["bad", "Negative "], ["good", "negative"], ["bad", ""]],
    )
    config = make_config(tmp_path, label_column="label")
    metrics = Metrics()

    run(config, lambda: fake_model, metrics)

    assert metrics.summary()["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert "label" in read_output(config.output_path)[0]


def test_non_english_rows_are_rejected_without_calling_model(tmp_path, fake_model):
    write_csv(
        tmp_path / "in.csv",
        ["text"],
        [["a good product"], ["Das Essen war wirklich schrecklich"], ["这个产品很好"]],
    )
    config = make_config(tmp_path)
    metrics = Metrics()

    run(config, lambda: fake_model, metrics)

    out = read_output(config.output_path)
    assert [r["status"] for r in out] == ["ok", "failed", "failed"]
    assert out[1]["error"].startswith("non_english: detected 'de'")
    assert out[2]["error"] == "non_english: text is mostly in a non-Latin script"
    assert fake_model.calls == [["a good product"]]
    assert metrics.summary()["failures_by_reason"] == {"non_english": 2}


def test_language_check_can_be_disabled(tmp_path, fake_model):
    write_csv(tmp_path / "in.csv", ["text"], [["Das Essen war wirklich schrecklich"]])
    config = make_config(tmp_path, language_check=False)

    run(config, lambda: fake_model, Metrics())

    assert read_output(config.output_path)[0]["status"] == "ok"


def test_changing_the_language_check_invalidates_a_checkpoint(tmp_path):
    config = make_config(tmp_path)
    assert config.fingerprint("abc") != make_config(tmp_path, language_check=False).fingerprint(
        "abc"
    )
