"""Crash-safe, resumable CSV output.

Files kept next to the output path while a run is in progress:

    <output>.partial          rows written so far (appended to)
    <output>.checkpoint.json  last durable commit: rows done, byte length of .partial, counters
    <output>.lock             flock held for the lifetime of the run

Commit protocol, once per window: append rows -> flush + fsync .partial ->
atomically replace the checkpoint. Whenever the process dies, the checkpoint
on disk describes a prefix of .partial, so resuming means "truncate .partial
to that length and skip that many input records". No exception handler has to
run for this to hold, so it also survives SIGKILL and the OOM killer.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

CHECKPOINT_VERSION = 1


class OutputLocked(RuntimeError):
    """Another process is already writing this output."""


class ResumeError(RuntimeError):
    """A checkpoint exists but cannot be used safely."""


@dataclass(frozen=True)
class Checkpoint:
    rows_committed: int
    output_bytes: int
    metrics: dict


def sibling(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def fsync_dir(path: Path) -> None:
    """Make a rename inside `path` durable: without this, power loss can undo it."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, data: dict) -> None:
    """Readers see either the old file or the new one, never a torn write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sibling(path, ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


@contextmanager
def output_lock(path: Path) -> Iterator[None]:
    """Fail fast if another run targets the same output.

    flock is released by the kernel when the process dies, so a crash never
    leaves a stale lock. The lock file itself is deliberately not deleted:
    unlinking it would let a new process lock a fresh inode while an old one
    still holds the previous one.
    """
    lock_path = sibling(path, ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OutputLocked(f"another run is already writing {path}") from None
        yield


class CheckpointedCsvWriter:
    def __init__(self, path: Path, fieldnames: Sequence[str], fingerprint: str):
        self.path = path
        self.partial_path = sibling(path, ".partial")
        self.checkpoint_path = sibling(path, ".checkpoint.json")
        self.fieldnames = list(fieldnames)
        self.fingerprint = fingerprint
        self._file = None

    def load_checkpoint(self) -> Checkpoint | None:
        if not self.checkpoint_path.exists():
            return None
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if data.get("version") != CHECKPOINT_VERSION or data.get("fingerprint") != self.fingerprint:
            raise ResumeError(
                f"{self.checkpoint_path} belongs to a different input, model or column "
                "configuration; rerun with --fresh to discard it"
            )
        output_bytes = data["output_bytes"]
        if not self.partial_path.exists() or self.partial_path.stat().st_size < output_bytes:
            raise ResumeError(
                f"{self.partial_path} is missing or shorter than its checkpoint; "
                "rerun with --fresh to start over"
            )
        return Checkpoint(data["rows_committed"], output_bytes, data["metrics"])

    def open(self, checkpoint: Checkpoint | None) -> CheckpointedCsvWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint is None:
            self.checkpoint_path.unlink(missing_ok=True)
            self._file = self.partial_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
        else:
            # Drop anything written after the last commit (a crash mid-window).
            with self.partial_path.open("r+b") as f:
                f.truncate(checkpoint.output_bytes)
            self._file = self.partial_path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        return self

    def write_rows(self, rows: Sequence[dict]) -> None:
        self._writer.writerows(rows)

    def commit(self, rows_committed: int, metrics_state: dict) -> None:
        self._file.flush()
        os.fsync(self._file.fileno())
        atomic_write_json(
            self.checkpoint_path,
            {
                "version": CHECKPOINT_VERSION,
                "fingerprint": self.fingerprint,
                "rows_committed": rows_committed,
                "output_bytes": os.fstat(self._file.fileno()).st_size,
                "metrics": metrics_state,
            },
        )

    def publish(self) -> None:
        """Make the output durable and move it into place."""
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        # Checkpoint first: dying between the two steps leaves a .partial with no
        # checkpoint, which the next run simply redoes. The reverse order would
        # leave a checkpoint pointing at a file that no longer exists.
        self.checkpoint_path.unlink(missing_ok=True)
        os.replace(self.partial_path, self.path)
        fsync_dir(self.path.parent)

    def __enter__(self) -> CheckpointedCsvWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()
