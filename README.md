# Sentiment Batch Inference Pipeline

A batch inference service that classifies the text in a CSV file as `positive` or `negative`
using a pre-trained DistilBERT model (SST-2) and writes the predictions to a CSV file.

Key properties:

- **Resumable.** Progress is checkpointed. A run that is killed at any point resumes from the
  last checkpoint, and the resumed output is byte-identical to the output of an uninterrupted run.
- **Reproducible.** The base image is pinned by digest. All Python dependencies, including
  transitive ones, are pinned and hash-verified. The model is pinned to a specific Hub commit.
- **Observable.** Structured JSON logs, a JSON run report with provenance, and an optional
  Prometheus `/metrics` endpoint.
- **Fault-isolating.** Invalid rows and inference errors are recorded for each row and never
  abort the run. Systemic failures stop the run early.

## Contents

- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Design](#design)
- [Failure Handling](#failure-handling)
- [Observability](#observability)
- [Performance](#performance)
- [Design Decisions](#design-decisions)
- [Security](#security)
- [Testing and CI](#testing-and-ci)
- [Datasets](#datasets)
- [Limitations and Roadmap](#limitations-and-roadmap)

## Results

| Dataset | Rows | Throughput | Failed rows | Accuracy |
|---|---|---|---|---|
| UCI Sentiment Labelled Sentences | 3,000 | 470 rows/s | 0 | 92.8% |
| Sentiment140 (seeded random sample) | 20,000 | 354 rows/s | 0 | 70.4% |

Measured on CPU only (4 PyTorch threads, Apple M-series); throughput varies by about 10% between
runs. Peak memory is about 1 GB and does not grow with input size.

Prediction files are included in [output/](output/), each with its `*.metrics.json` run report.
`edge_cases_predictions.csv` shows how invalid rows are reported; that run exits with code `2`
because 3 of its 10 rows are deliberately invalid.

## Getting Started

### Prerequisites

- Docker, for the containerised workflow
- Python 3.12 and [uv](https://docs.astral.sh/uv/), for local development

### Run with Docker

```bash
docker build -t sentiment-pipeline .

docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/data:ro" -v "$PWD/output:/output" \
  sentiment-pipeline \
  --input /data/uci_sentences.csv \
  --output /output/predictions.csv \
  --id-column id --label-column label
```

The model weights are built into the image, so the container needs no network access at
runtime. The `--user` flag makes the output files belong to the host user.

To expose metrics during a run, add `-p 8000:8000` to `docker run` and `--metrics-port 8000`
to the pipeline arguments. `make docker-run` runs this configuration.

### Local Development

| Command | Description |
|---|---|
| `make install` | Create `.venv` and install hash-verified dependencies from `requirements.lock` |
| `make test` | Run the test suite with a stub model (no PyTorch needed, under 1 second) |
| `make test-model` | Run the integration tests against the real model |
| `make lint` | Run Ruff lint and format checks |
| `make run` | Run the pipeline on the UCI dataset with human-readable logs |
| `make data` | Download and normalise the datasets |
| `make lock` | Re-resolve `requirements.in` into `requirements.lock` |

## Usage

```bash
python -m sentiment_pipeline --input FILE [options]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--input` | required | Input CSV path |
| `--output` | `output/predictions.csv` | Output CSV path |
| `--text-column` | `text` | Column containing the text to classify |
| `--id-column` | none | Column carried through to the output as a stable identifier |
| `--label-column` | none | Ground-truth column; enables the accuracy metric |
| `--batch-size` | `32` | Rows per forward pass |
| `--window-size` | `1024` | Rows sorted by length before batching; also the checkpoint interval |
| `--max-length` | `512` | Maximum tokens per text; longer texts are truncated |
| `--max-failure-rate` | `0.05` | Failure-rate threshold above which the run exits with code 2 |
| `--fresh` | off | Discard any existing checkpoint and start from the first row |
| `--model-name`, `--model-revision` | pinned SST-2 commit | Hugging Face model and revision |
| `--encoding` | `utf-8-sig` | Input encoding (`utf-8-sig` also strips an Excel BOM) |
| `--metrics-port` | `0` (off) | Port for the Prometheus `/metrics` endpoint |
| `--metrics-file` | `<output>.metrics.json` | Path of the run report |
| `--log-format` | `json` | `json` or `text` |
| `--log-level` | `INFO` | Standard logging level |

### Exit Codes

| Code | Meaning | Output published |
|---|---|---|
| `0` | Completed; failure rate within threshold | Yes |
| `1` | Fatal error: invalid input or schema, output locked by another run, incompatible checkpoint, model load failure, or circuit breaker tripped | No |
| `2` | Completed; failure rate above `--max-failure-rate` | Yes |
| `130` | Interrupted by SIGINT or SIGTERM; progress is checkpointed | No |

The run report is written for every outcome. Its `status` field is `completed`,
`completed_with_errors`, `failed` or `interrupted`.

### Output Schema

Each input record produces exactly one output row, in input order.

| Column | Description |
|---|---|
| `row_number` | 1-based record index in the input, excluding the header |
| `id` | Present when `--id-column` is set |
| `text` | Input text |
| `label` | Present when `--label-column` is set |
| `predicted_label` | `positive` or `negative`; empty for failed rows |
| `score` | Softmax probability of the predicted label, rounded to 4 decimal places |
| `status` | `ok` or `failed` |
| `error` | `<reason>: <detail>`, where `reason` is `empty_text`, `malformed_csv` or `inference_error` |

### Resuming a Run

Rerun the same command. The pipeline detects the checkpoint, validates it against the current
input and configuration, and continues from the last committed row. Use `--fresh` to discard
the checkpoint and start again.

## Design

### Components

| Module | Responsibility |
|---|---|
| [`csv_io.py`](src/sentiment_pipeline/csv_io.py) | Streaming CSV reader. Validates the header before any work starts and yields one record per input row, including rows that fail to parse. |
| [`pipeline.py`](src/sentiment_pipeline/pipeline.py) | Orchestration: windowing, validation, length bucketing, batch inference with row-level fallback, circuit breaker, checkpoint cadence, and the resume fingerprint. |
| [`output.py`](src/sentiment_pipeline/output.py) | Durable output: checkpoint commit protocol, atomic publish, and a single-writer lock. |
| [`model.py`](src/sentiment_pipeline/model.py) | `SentimentModel` protocol and the Hugging Face implementation. |
| [`metrics.py`](src/sentiment_pipeline/metrics.py) | Thread-safe counters, the JSON run report, and the Prometheus collector. Counter state is persisted in checkpoints. |
| [`cli.py`](src/sentiment_pipeline/cli.py) | Argument parsing, structured logging, signal handling, and exit codes. |

The pipeline depends only on the `SentimentModel` protocol (`predict(texts) -> predictions`).
That separation lets the test suite run with a stub model, and it is the single extension point
for alternative backends such as ONNX Runtime or a remote model server.

### Processing Model

1. The input is streamed in windows of `--window-size` rows. Memory use does not depend on file size.
2. Rows that fail validation (empty text, malformed record) are marked failed and are not sent to the model.
3. The remaining rows in the window are sorted by text length and split into batches of
   `--batch-size`. This keeps padding to a minimum.
4. If a batch fails, its rows are retried one at a time so that a single bad input affects only its own row.
5. Results are written back in the original input order, and the window is committed.

### Durability and Resume

Output is written to `<output>.partial`. After each window, the pipeline:

1. flushes and fsyncs the partial file;
2. writes a checkpoint (rows committed, partial file length, counter state, configuration
   fingerprint) to a temporary file, fsyncs it, and atomically renames it into place.

On resume, the partial file is truncated to the length recorded in the checkpoint, and the same
number of input rows is skipped. This removes any bytes written after the last commit. On
completion, the checkpoint is deleted and the partial file is atomically renamed to the final
output path. Directory entries are fsynced after each rename.

Correctness does not depend on cleanup code running, so the protocol holds under SIGKILL,
OOM termination and host failure.

### Guarantees

| Guarantee | Mechanism |
|---|---|
| Every input record appears exactly once in the output | Each record, including unparseable ones, yields one output row; `rows_read = rows_succeeded + rows_failed` |
| The final output file is never partially written | Write to `.partial`, then atomic rename |
| A crashed run resumes with identical output | Checkpoint commit protocol; verified by SIGKILL and resume against an uninterrupted run |
| A checkpoint is reused only for the same job | Fingerprint of the input SHA-256, column mapping, encoding, model name, revision and maximum length, and output schema |
| One writer per output path | Non-blocking `flock`, released by the kernel if the process dies |
| Cheap validation happens before expensive work | Order of operations: schema check, lock, input hash, model load |

Batch size and window size are excluded from the fingerprint because they affect throughput,
not results. A run that runs out of memory can therefore be resumed with a smaller batch size.

Rows in an uncommitted window (at most `--window-size`) are inferred again after a crash. Each
row is still written to the output exactly once.

## Failure Handling

| Condition | Behaviour |
|---|---|
| Missing input file, missing column or empty file | Exit `1` before any output is created; the error lists the available columns |
| Empty or whitespace-only text | Row marked `empty_text`; not sent to the model |
| Wrong field count or unparseable record | Row marked `malformed_csv`; reading continues with the next record |
| Undecodable bytes | Replaced with U+FFFD |
| Text longer than `--max-length` tokens | Truncated and classified |
| Batch inference error, or wrong number of results | Batch retried row by row; only the failing rows are marked `inference_error` |
| All rows failing in 3 consecutive batches | Circuit breaker aborts with exit `1`; progress is retained for resume |
| Failure rate above threshold | Output published; exit `2` |
| SIGTERM or SIGINT | Current window completes and is committed; exit `130`. A second signal exits immediately. |
| SIGKILL, OOM termination, host failure | Resume from the last committed window on the next run |
| Concurrent run on the same output | Second run exits `1` immediately |
| Input or configuration changed since the checkpoint | Exit `1`; requires `--fresh` |

## Observability

### Run Report

`<output>.metrics.json` is written for every outcome. It contains row and failure counts,
prediction distribution, throughput and accuracy, plus provenance fields: `run_id`, package
version, model name and revision, input SHA-256, and start time.

### Logs

JSON lines on stderr. Every line includes the `run_id`. Progress is logged every 10 windows,
and a summary is logged at the end of the run. The pipeline does not log input text.

### Prometheus Metrics

| Metric | Type | Purpose |
|---|---|---|
| `sentiment_rows_total{status}` | Counter | Rows processed, by outcome |
| `sentiment_row_failures_total{reason}` | Counter | Separates data-quality failures from model failures |
| `sentiment_batch_fallbacks_total` | Counter | Batches retried row by row; an early indicator of bad inputs or model degradation |
| `sentiment_predictions_total{label}` | Counter | Prediction distribution; detects models that fail without raising errors |
| `sentiment_low_confidence_predictions_total` | Counter | Predictions with a score below 0.7; an indicator of drift or out-of-domain input |
| `sentiment_inference_seconds_total` | Counter | Time spent in model inference |
| `sentiment_rows_per_second` | Gauge | Throughput of the current session |
| `sentiment_run_info` | Info | Run metadata for correlating metrics with a specific run |

All label values come from small fixed sets. Exception messages are never used as label values.

### Recommended Alerts

- Non-zero exit code
- Failure rate above 5%
- Any batch fallbacks
- Share of predicted labels outside the baseline range for the dataset
- Share of low-confidence predictions above twice the baseline
- Throughput below 50% of the baseline

## Performance

### Length Bucketing

Sorting each window by text length before batching reduces padding overhead.

| Dataset | Without bucketing | With bucketing | Speedup |
|---|---|---|---|
| UCI (mixed lengths) | 315 rows/s | 470–520 rows/s | 1.5–1.6x |
| Sentiment140 (short texts) | 251 rows/s | 354–375 rows/s | 1.4–1.5x |

Accuracy is unchanged to 4 decimal places. A full-file sort gave a similar speedup, so sorting
within a window captures most of the benefit while memory use stays constant.

### Batch Size

| Batch size | 8 | 32 | 128 |
|---|---|---|---|
| Throughput (UCI, with bucketing) | 426 rows/s | 520 rows/s | 496 rows/s |

### Capacity Planning

- The full Sentiment140 dataset (1.6M rows) takes about 75 minutes in a single 4-core container.
- Peak memory is about 1 GB. A container memory limit of 1.5–2 GB is recommended.
- To scale horizontally, split the input across containers, each with its own output path and
  checkpoint. No state is shared between them.
- In CPU-limited containers, set `OMP_NUM_THREADS` to the CPU limit. PyTorch otherwise sizes its
  thread pool from the host core count.

## Design Decisions

| Decision | Alternatives considered | Rationale | Revisit when |
|---|---|---|---|
| Single-process batch CLI | Service with a queue; Spark or Ray | The workload is bounded and offline. A single process is the simplest option to operate and test. | Sustained volume above ~10M rows/day, or a latency SLA |
| Python's built-in `csv` module, streamed | pandas, pyarrow | Constant memory, and per-record handling of parse errors | Input moves to a columnar format |
| CSV output | Parquet, database | Meets the requirement, and any tool can read it | Consumers need typed or columnar data |
| Checkpointed single output file | One output file per shard; SQLite | Durable and simple to reason about | Parallel workers are introduced |
| Model built into the image, pinned by commit | Download at runtime; mounted volume | Immutable artefact; no runtime network dependency; the build fails if the model cannot load | Models are updated frequently |
| PyTorch, called directly | `transformers.pipeline()`, ONNX Runtime | Every step is explicit. ONNX Runtime is faster on CPU but needs model conversion and parity testing. | Compute cost becomes significant |
| Row-by-row fallback on batch failure | Bisecting the batch | Simpler; the extra cost is incurred only when a batch fails | Batch failures become frequent |
| Resume enabled by default | Opt-in resume | Rerunning the same command is the expected operator and scheduler behaviour; the fingerprint prevents unsafe reuse | — |
| Pull-based metrics plus a JSON report | Prometheus Pushgateway | No extra infrastructure; the report covers short-lived runs | Scheduled production runs |
| Hash-locked dependencies | Pin only top-level dependencies | Prevents transitive dependencies changing between builds and detects tampered packages | — |

## Security

- The container runs as a non-root user and needs no network access at runtime. CI runs it with
  `--network none` and a read-only root filesystem.
- Supply chain: the base image is pinned by digest, packages are hash-verified, and the model is
  pinned by commit. Weights are loaded from `safetensors`, and `trust_remote_code` is never enabled.
- Input text appears in the output file but is not logged by the pipeline.
- Output text is written unescaped. Values starting with `=`, `+`, `-` or `@` may be interpreted
  as formulas by spreadsheet applications. Escaping them would alter the data for programmatic
  consumers, so this is left to the consumer.

## Testing and CI

The test suite contains 23 behavioural tests that use a stub model, plus 2 integration tests
against the real model.

- **Core behaviour:** batching, validation, isolation of bad rows, malformed CSV recovery, BOM
  handling, the circuit breaker, exit codes and metrics output.
- **Durability:** simulated crashes that bypass all exception handlers, partially written
  output, input changed between runs, graceful stop, and lock contention. The main assertion is
  that the output of a crashed and resumed run is byte-identical to that of an uninterrupted run.
- The suite was checked by deliberately breaking the code: removing the truncation on resume or
  the length sort causes the corresponding tests to fail.

The CI pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs:

1. Lint, format check and tests.
2. A check that `requirements.lock` matches `requirements.in`.
3. A Docker build and an end-to-end run with no network and a read-only filesystem, which must
   reach at least 90% accuracy and match the reference predictions.
4. An edge-case run that must exit with code `2` and still publish its output.

Predictions are compared on labels, with scores allowed to differ by up to 1e-3
([`scripts/compare_predictions.py`](scripts/compare_predictions.py)). Different CPU math
libraries produce differences in the order of 1e-6, which can change the fourth decimal place,
so byte comparison is not reliable across platforms.

## Datasets

| Dataset | Preparation |
|---|---|
| UCI Sentiment Labelled Sentences | The three source files are merged and labels normalised to `positive`/`negative`. The IMDB file contains U+0085 characters inside sentences, so it is read as a text stream rather than with `str.splitlines()`, which would split on them. |
| Sentiment140 | The source file is sorted by label, so the first N rows would all be negative. A seeded reservoir sample produces a uniform random subset in one pass. |

[`data/edge_cases.csv`](data/edge_cases.csv) covers quoted delimiters, multi-line values,
non-ASCII text, empty values, text above the token limit, and a malformed record.

### Model Behaviour

- Accuracy is higher on UCI (92.8%) than on Sentiment140 (70.4%). SST-2 consists of reviews,
  which closely match UCI. Tweets are a different domain, and Sentiment140 labels are inferred
  from emoticons and are therefore noisy.
- On Sentiment140 the model predicts 59.6% negative against a true rate of 49.6%. This bias
  shows up in the prediction-distribution metric but not in the accuracy figure.
- The model supports English only. Non-English input receives confident but unreliable predictions.

## Limitations and Roadmap

1. Parallel execution across sharded inputs, with a merge step.
2. ONNX Runtime with int8 quantisation, validated against the reference predictions.
3. Skipping the run when a completed output with a matching fingerprint already exists.
4. Pushing final metrics to a Prometheus Pushgateway for scheduled runs.
5. A metric counting replaced (undecodable) characters.
6. A three-class model if a `neutral` label is needed. The binary model always assigns
   `positive` or `negative`.
