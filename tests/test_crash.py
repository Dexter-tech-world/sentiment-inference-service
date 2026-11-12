"""Crash test with real process kills.

test_resume.py simulates crashes in-process. This test runs the pipeline in a
child process and kills it with SIGKILL, so no Python code (no finally block,
no exception handler, no buffered-file flush) gets to run: the same thing
that happens on an OOM kill or a host failure. The job is killed three times
at different points, then completed, and the result must be byte-identical to
an uninterrupted run.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from conftest import FakeModel, write_csv

from sentiment_pipeline.metrics import Metrics
from sentiment_pipeline.pipeline import PipelineConfig, run

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX SIGKILL")

ROWS = 1200
WINDOW = 50
TESTS_DIR = Path(__file__).parent

CHILD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {tests_dir!r})
    from conftest import FakeModel
    from pathlib import Path
    from sentiment_pipeline.metrics import Metrics
    from sentiment_pipeline.pipeline import PipelineConfig, run

    class SlowModel(FakeModel):
        def predict(self, texts):
            time.sleep(0.01)  # slow enough to be killed mid-run
            return super().predict(texts)

    config = PipelineConfig(input_path=Path({input!r}), output_path=Path({output!r}),
                            batch_size=8, window_size={window})
    run(config, SlowModel, Metrics())
    """
)


def config_for(input_path: Path, output_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input_path=input_path, output_path=output_path, batch_size=8, window_size=WINDOW
    )


def committed_rows(checkpoint: Path) -> int:
    try:
        return json.loads(checkpoint.read_text())["rows_committed"]
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def start_and_kill(input_path: Path, output_path: Path, kill_after_rows: int) -> int:
    """Run the pipeline in a child process and SIGKILL it once enough rows are committed."""
    script = CHILD.format(
        tests_dir=str(TESTS_DIR), input=str(input_path), output=str(output_path), window=WINDOW
    )
    child = subprocess.Popen([sys.executable, "-c", script], stderr=subprocess.PIPE)
    checkpoint = output_path.with_name(output_path.name + ".checkpoint.json")
    deadline = time.monotonic() + 60
    while committed_rows(checkpoint) < kill_after_rows:
        if child.poll() is not None:
            pytest.fail(f"child exited early ({child.returncode}): {child.stderr.read().decode()}")
        if time.monotonic() > deadline:
            child.kill()
            pytest.fail("child made no progress")
        time.sleep(0.005)
    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    child.stderr.close()
    assert child.returncode == -signal.SIGKILL
    return committed_rows(checkpoint)


def test_repeated_sigkill_then_resume_is_byte_identical(tmp_path):
    texts = [
        f"{'good' if i % 3 else 'bad'} review number {i} " + "x" * (i % 17) for i in range(ROWS)
    ]
    input_path = write_csv(tmp_path / "in.csv", ["text"], [[t] for t in texts])

    reference = tmp_path / "reference" / "predictions.csv"
    run(config_for(input_path, reference), FakeModel, Metrics())

    output = tmp_path / "crashed" / "predictions.csv"
    progress = []
    for kill_after in (150, 500, 900):  # three kills, each after resuming the previous one
        progress.append(start_and_kill(input_path, output, kill_after))
        assert not output.exists(), "nothing may be published before the job completes"

    assert progress == sorted(progress) and progress[-1] < ROWS  # each run resumed and advanced

    metrics = Metrics()
    run(config_for(input_path, output), FakeModel, metrics)

    assert output.read_bytes() == reference.read_bytes()
    summary = metrics.summary()
    assert summary["rows_read"] == ROWS
    assert summary["rows_resumed"] == progress[-1]
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "predictions.csv",
        "predictions.csv.lock",
    ]
