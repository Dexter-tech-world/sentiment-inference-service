"""Compare two prediction files: same rows, same labels, scores within tolerance.

Byte equality is the wrong test across platforms: different CPU math kernels
(Accelerate on macOS, oneDNN/MKL on Linux) differ around 1e-6, which flips the
4th decimal of some scores. Labels and scores-within-tolerance are what matter.

    python scripts/compare_predictions.py expected.csv actual.csv
"""

import csv
import sys

SCORE_TOLERANCE = 1e-3


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(expected_path: str, actual_path: str) -> int:
    expected, actual = load(expected_path), load(actual_path)
    if len(expected) != len(actual):
        print(f"row count differs: {len(expected)} vs {len(actual)}")
        return 1
    label_diffs = [
        e["row_number"]
        for e, a in zip(expected, actual, strict=True)
        if e["predicted_label"] != a["predicted_label"] or e["status"] != a["status"]
    ]
    max_score_diff = max(
        (
            abs(float(e["score"]) - float(a["score"]))
            for e, a in zip(expected, actual, strict=True)
            if e["score"] and a["score"]
        ),
        default=0.0,
    )
    print(f"rows={len(expected)} label_mismatches={len(label_diffs)}", end=" ")
    print(f"max_score_diff={max_score_diff:.6f}")
    if label_diffs:
        print(f"first mismatching rows: {label_diffs[:10]}")
    return 1 if label_diffs or max_score_diff > SCORE_TOLERANCE else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
