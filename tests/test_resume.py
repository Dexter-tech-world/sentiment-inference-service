"""Crash safety and resume.

The core property: a run that dies at any point and is then resumed produces
output byte-identical to a run that never died.
"""

from __future__ import annotations

import json
import threading

import pytest
from conftest import FakeModel, write_csv

from sentiment_pipeline.metrics import Metrics
from sentiment_pipeline.output import OutputLocked, ResumeError, output_lock
from sentiment_pipeline.pipeline import PipelineConfig, RunInterrupted, run


class Crash(BaseException):
    """Stands in for SIGKILL / OOM: no `except Exception` handler can catch it."""


class CrashingModel(FakeModel):
    def __init__(self, crash_on_call: int):
        super().__init__()
        self.crash_on_call = crash_on_call

    def predict(self, texts):
        if len(self.calls) + 1 == self.crash_on_call:
            raise Crash
        return super().predict(texts)


# Different lengths so that length-sorting actually reorders rows within a window.
TEXTS = [("good " if i % 3 else "bad ") + "x" * (i * 7 % 13) for i in range(25)]


@pytest.fixture
def input_csv(tmp_path):
    return write_csv(tmp_path / "in.csv", ["text"], [[t] for t in TEXTS])


def config_for(tmp_path, name: str, **overrides) -> PipelineConfig:
    # 25 rows, windows of 10: commits after rows 10 and 20, then the final 5.
    return PipelineConfig(
        input_path=tmp_path / "in.csv",
        output_path=tmp_path / name / "predictions.csv",
        batch_size=4,
        window_size=10,
        **overrides,
    )


@pytest.fixture
def reference_output(tmp_path, input_csv) -> bytes:
    config = config_for(tmp_path, "reference")
    run(config, FakeModel, Metrics())
    return config.output_path.read_bytes()


def crash_during_second_window(config: PipelineConfig) -> None:
    # Window 1 (10 rows, batches of 4) makes 3 calls; call 5 is inside window 2.
    with pytest.raises(Crash):
        run(config, lambda: CrashingModel(crash_on_call=5), Metrics())
    checkpoint = json.loads(
        config.output_path.with_name("predictions.csv.checkpoint.json").read_text()
    )
    assert checkpoint["rows_committed"] == 10
    assert not config.output_path.exists()


def test_resumed_run_matches_uninterrupted_run(tmp_path, reference_output):
    config = config_for(tmp_path, "crashed")
    crash_during_second_window(config)

    model = FakeModel()
    metrics = Metrics()
    run(config, lambda: model, metrics)

    assert config.output_path.read_bytes() == reference_output
    assert sum(len(c) for c in model.calls) == 15  # only rows 11-25 were re-inferred
    summary = metrics.summary()
    assert summary["rows_read"] == 25
    assert summary["rows_resumed"] == 10
    assert summary["predictions_by_label"] == {"positive": 16, "negative": 9}


def test_resume_discards_bytes_written_after_the_last_commit(tmp_path, reference_output):
    config = config_for(tmp_path, "torn")
    crash_during_second_window(config)
    with config.output_path.with_name("predictions.csv.partial").open("a") as f:
        f.write("11,half-written ro")  # what a kill mid-write leaves behind

    run(config, FakeModel, Metrics())

    assert config.output_path.read_bytes() == reference_output


def test_resume_refuses_a_checkpoint_from_a_different_input(tmp_path, input_csv):
    config = config_for(tmp_path, "changed")
    crash_during_second_window(config)
    write_csv(input_csv, ["text"], [["completely different"]])

    with pytest.raises(ResumeError, match="different input"):
        run(config, FakeModel, Metrics())

    fresh = PipelineConfig(**{**config.__dict__, "resume": False})
    run(fresh, FakeModel, Metrics())
    assert config.output_path.exists()


def test_stop_request_commits_current_window_then_resume_completes(tmp_path, reference_output):
    config = config_for(tmp_path, "stopped")
    stop = threading.Event()
    stop.set()  # as if SIGTERM arrived during the first window

    with pytest.raises(RunInterrupted, match="stopped after 10 rows"):
        run(config, FakeModel, Metrics(), stop)

    run(config, FakeModel, Metrics())
    assert config.output_path.read_bytes() == reference_output


def test_completed_run_leaves_no_checkpoint_or_partial(tmp_path, input_csv):
    config = config_for(tmp_path, "clean")
    run(config, FakeModel, Metrics())

    leftovers = sorted(p.name for p in config.output_path.parent.iterdir())
    assert leftovers == ["predictions.csv", "predictions.csv.lock"]


def test_second_run_on_the_same_output_is_rejected(tmp_path, input_csv):
    config = config_for(tmp_path, "locked")
    with output_lock(config.output_path), pytest.raises(OutputLocked):
        run(config, FakeModel, Metrics())


def test_batches_are_length_sorted_but_output_keeps_input_order(tmp_path, input_csv):
    config = config_for(tmp_path, "order")
    model = FakeModel()
    run(config, lambda: model, Metrics())

    first_window = [t for call in model.calls[:3] for t in call]
    assert first_window == sorted(TEXTS[:10], key=len)
    with config.output_path.open() as f:
        written = [line.split(",")[1] for line in f.read().splitlines()[1:]]
    assert written == TEXTS
