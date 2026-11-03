"""Streaming CSV input: memory stays flat regardless of file size."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


class InputSchemaError(ValueError):
    """The input file cannot be processed at all (missing file, header or column)."""


@dataclass(frozen=True)
class InputRow:
    row_number: int  # 1-based index of the data record, header excluded
    text: str | None
    id: str | None = None
    label: str | None = None
    error: str | None = None  # set when the record itself could not be parsed


def read_rows(
    path: Path,
    text_column: str,
    id_column: str | None = None,
    label_column: str | None = None,
    encoding: str = "utf-8-sig",
) -> Iterator[InputRow]:
    """Validate the header now, then lazily yield one InputRow per CSV record.

    Header problems raise InputSchemaError immediately (before any output is
    created). After that, every record yields exactly one InputRow, including
    broken ones (with `error` set), so output can always be reconciled with
    input. `utf-8-sig` strips the BOM that Excel adds; undecodable bytes are
    replaced rather than aborting a long run on one bad byte.
    """
    if not path.is_file():
        raise InputSchemaError(f"input file not found: {path}")

    f = path.open(newline="", encoding=encoding, errors="replace")
    try:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise InputSchemaError(f"input file is empty: {path}")
        index = {name.strip(): i for i, name in enumerate(header)}
        for column in (text_column, id_column, label_column):
            if column is not None and column not in index:
                raise InputSchemaError(
                    f"column {column!r} not found in {path}; available columns: {sorted(index)}"
                )
    except BaseException:
        f.close()
        raise

    columns = (index[text_column], _opt(index, id_column), _opt(index, label_column))
    return _iter_records(f, reader, len(header), *columns)


def _opt(index: dict[str, int], column: str | None) -> int | None:
    return None if column is None else index[column]


def _iter_records(
    f: TextIO,
    reader: Iterator[list[str]],
    width: int,
    text_i: int,
    id_i: int | None,
    label_i: int | None,
) -> Iterator[InputRow]:
    with f:
        row_number = 0
        while True:
            try:
                row = next(reader)
            except StopIteration:
                return
            except csv.Error as exc:
                # e.g. an unbalanced quote swallowing following lines until the
                # field size limit trips. Record it and keep going.
                row_number += 1
                yield InputRow(row_number, text=None, error=str(exc))
                continue

            if not row:  # blank line
                continue
            row_number += 1
            row_id = row[id_i] if id_i is not None and id_i < len(row) else None
            if len(row) != width:
                yield InputRow(
                    row_number,
                    text=None,
                    id=row_id,
                    error=f"expected {width} fields, got {len(row)}",
                )
                continue
            yield InputRow(
                row_number,
                text=row[text_i],
                id=row_id,
                label=row[label_i] if label_i is not None else None,
            )
