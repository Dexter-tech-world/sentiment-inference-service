"""Download a suggested dataset and normalise it to CSV: id,source,text,label.

Labels are normalised to "negative"/"positive" so they compare directly with
the model's output labels.

    python scripts/prepare_data.py uci --out data/uci_sentences.csv
    python scripts/prepare_data.py sentiment140 --limit 20000 --out data/sentiment140_20k.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
SENTIMENT140_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
UCI_LABELS = {"0": "negative", "1": "positive"}
SENTIMENT140_LABELS = {"0": "negative", "4": "positive"}  # 2 (neutral) is absent from train


def download(url: str) -> zipfile.ZipFile:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        return zipfile.ZipFile(io.BytesIO(response.read()))


def uci_rows(archive: zipfile.ZipFile):
    for source in ("amazon_cells", "imdb", "yelp"):
        member = f"sentiment labelled sentences/{source}_labelled.txt"
        with archive.open(member) as raw:
            # Iterate the text stream rather than str.splitlines(): the imdb file
            # contains U+0085 (NEL) inside sentences, which splitlines() would
            # treat as a line break.
            for line_number, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8"), start=1):
                line = line.strip()
                if not line:
                    continue
                text, label = line.rsplit("\t", 1)
                yield f"{source}-{line_number}", source, text.strip(), UCI_LABELS[label]


def sentiment140_rows(archive: zipfile.ZipFile, limit: int, seed: int):
    # The file is sorted by label (800k negative, then 800k positive), so taking
    # the first N rows would give a single-class sample. Reservoir sampling gives
    # a uniform random sample in one pass with O(limit) memory.
    rng = random.Random(seed)
    sample: list[list[str]] = []
    member = "training.1600000.processed.noemoticon.csv"
    with archive.open(member) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="latin-1"))
        for i, row in enumerate(reader):
            if i < limit:
                sample.append(row)
            elif (j := rng.randint(0, i)) < limit:
                sample[j] = row
    for polarity, tweet_id, _date, _query, _user, text in sample:
        yield tweet_id, "sentiment140", text, SENTIMENT140_LABELS[polarity]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=["uci", "sentiment140"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20_000, help="sentiment140 sample size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "uci":
        rows = uci_rows(download(UCI_URL))
    else:
        rows = sentiment140_rows(download(SENTIMENT140_URL), args.limit, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source", "text", "label"])
        for count, row in enumerate(rows, start=1):  # noqa: B007
            writer.writerow(row)
    print(f"wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
