# Sentiment Inference Service

Classifies English text as `positive` or `negative` using a pre-trained DistilBERT model
(SST-2). It provides two entry points that share one core:

- a **batch pipeline** that processes a CSV file and writes predictions to a CSV file;
- an **HTTP API** (Flask, served by gunicorn) for real-time predictions.

Key properties:

- **Resumable.** Batch progress is checkpointed. A run that is killed at any point resumes from
  the last checkpoint, and the resumed output is byte-identical to the output of an
  uninterrupted run. This is verified by tests that kill real processes with `SIGKILL`.
- **English-only by design.** The model only understands English, so text that is clearly not
  English is rejected rather than given a confident but meaningless label.
- **Reproducible.** The base image is pinned by digest. All Python dependencies, including
  transitive ones, are pinned and hash-verified. The model is pinned to a specific Hub commit.
- **Observable.** Structured JSON logs, a JSON run report with provenance, and Prometheus metrics.
- **Fault-isolating.** Invalid input and inference errors are recorded per row (or per item in
  the API) and never abort a run or a request. Systemic failures stop a batch run early.

## Contents

- [Results](#results)
- [Getting Started](#getting-started)
- [Batch Pipeline](#batch-pipeline)
- [HTTP API](#http-api)
- [English Language Check](#english-language-check)
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

| Dataset | Rows | Throughput | Rejected rows | Accuracy |
|---|---|---|---|---|
| UCI Sentiment Labelled Sentences | 3,000 | 489 rows/s | 0 | 92.8% |
| Sentiment140 (seeded random sample) | 20,000 | 320 rows/s | 27 (non-English) | 70.4% |

Measured on CPU only (4 PyTorch threads on a machine with 4 performance cores, Apple M-series).
Throughput varies by up to 15% between runs, because other load competes for the same cores.
Peak memory is about 1 GB and does not grow with input size. Accuracy is measured over the rows
that were classified.

Prediction files are included in [output/](output/), each with its `*.metrics.json` run report.
`edge_cases_predictions.csv` shows how invalid rows are reported. That run exits with code `2`
because 4 of its 10 rows are deliberately invalid: two empty, one malformed, and one not English.

## Getting Started

### Prerequisites

- Docker, for the containerised workflow
- Python 3.12 and [uv](https://docs.astral.sh/uv/), for local development

### Run with Docker

```bash
docker build -t sentiment-pipeline .

# Batch: classify a CSV file
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/data:ro" -v "$PWD/output:/output" \
  sentiment-pipeline \
  --input /data/uci_sentences.csv \
  --output /output/predictions.csv \
  --id-column id --label-column label

# API: serve predictions on http://localhost:8080
docker run --rm -p 8080:8080 --entrypoint sentiment-api sentiment-pipeline
```

The model weights are built into the image, so the container needs no network access at
runtime. The `--user` flag makes the output files belong to the host user.

### Local Development

| Command | Description |
|---|---|
| `make install` | Create `.venv` and install hash-verified dependencies from `requirements.lock` |
| `make test` | Run the test suite with a stub model (no PyTorch needed, a few seconds) |
| `make test-model` | Run the integration tests against the real model |
| `make crash-test` | Kill the pipeline with `SIGKILL` three times mid-run, resume, and compare output |
| `make lint` | Run Ruff lint and format checks |
| `make run` | Run the batch pipeline on the UCI dataset with human-readable logs |
| `make serve` | Run the API on `http://127.0.0.1:8080` |
| `make docker-build` | Build the image |
| `make docker-run` / `make docker-serve` | Run the batch pipeline / the API from the image |
| `make docker-test` | Run the crash test and API smoke test against the image |
| `make data` | Download and normalise the datasets |
| `make lock` | Re-resolve `requirements.in` into `requirements.lock` |

## Batch Pipeline

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
| `--no-language-check` | off | Classify every row, including text detected as clearly not English |
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
| `error` | `<reason>: <detail>`, where `reason` is `empty_text`, `non_english`, `malformed_csv` or `inference_error` |

### Resuming a Run

Rerun the same command. The pipeline detects the checkpoint, validates it against the current
input and configuration, and continues from the last committed row. Use `--fresh` to discard
the checkpoint and start again.

## HTTP API

```bash
sentiment-api [--host 127.0.0.1] [--port 8080] [--workers 1] [options]
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/predict` | Classify up to 64 texts |
| `GET` | `/healthz` | Liveness: the process is serving requests |
| `GET` | `/readyz` | Readiness: the model is loaded; reports the model version |
| `GET` | `/metrics` | Prometheus metrics |

### Request and Response

```bash
curl -s -X POST http://localhost:8080/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["I absolutely love this phone", "Ce film était magnifique", "   "]}'
```

```json
{
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english@714eb0fa...",
  "predictions": [
    {"label": "positive", "score": 0.9999, "status": "ok", "error": null},
    {"label": null, "score": null, "status": "failed", "error": "non_english: detected 'fr', P(English)=0.0001"},
    {"label": null, "score": null, "status": "failed", "error": "empty_text: text is empty"}
  ]
}
```

Results are returned in request order. A problem with one text's content (empty, not English,
or an inference error) is reported for that item with `status: failed` and never fails the
request. The rules are the same as in the batch pipeline, because both use the same code.

### Errors

Requests that break the contract are rejected as a whole, with a JSON body of the form
`{"error": {"code": 400, "name": "Bad Request", "message": "..."}}`.

| Status | Cause |
|---|---|
| `400` | Malformed JSON; `texts` missing, empty or not an array; an item that is not a string; more than 64 texts; a text longer than 10,000 characters |
| `413` | Request body larger than 1 MiB, rejected before it is parsed |
| `415` | `Content-Type` is not `application/json` |
| `404` / `405` | Unknown path / wrong method |
| `500` | Unexpected error; details are logged, not returned |

Inference errors report only the exception type, because exception messages can echo input text
or internal details.

### Options

| Option | Environment variable | Default | Description |
|---|---|---|---|
| `--host` | `API_HOST` | `127.0.0.1` (`0.0.0.0` in the image) | Bind address |
| `--port` | `PORT` | `8080` | Bind port |
| `--workers` | `API_WORKERS` | `1` | gunicorn worker processes; each loads its own model (~1 GB) |
| `--timeout` | | `60` | Seconds before an unresponsive worker is restarted |
| `--no-language-check` | | off | Classify every text, including clearly non-English text |
| `--model-name`, `--model-revision` | `MODEL_NAME`, `MODEL_REVISION` | pinned SST-2 commit | Model to serve |
| `--log-format`, `--log-level` | | `json`, `INFO` | Logging |

### Operating Notes

- **Scaling.** Run one worker per container and scale by adding containers. PyTorch already uses
  every core for a single forward pass, so the API runs one inference at a time per process.
  Extra concurrency within a process would only compete for the same cores.
- **Metrics are per process.** With `--workers` above 1, each Prometheus scrape reaches one
  worker and sees only that worker's counters. One worker per container avoids this.
- **Model loading.** The model is loaded in each worker after the fork, never in the gunicorn
  master. Forking after PyTorch has started its thread pools can deadlock.
- **Requests are traced.** Each response carries an `X-Request-ID` header (propagated from the
  request when it is valid, generated otherwise). The request ID also appears in the JSON log
  line for that request.
- **Shutdown.** `SIGTERM` (for example from `docker stop`) lets in-flight requests finish
  within a 30-second grace period.
- **Latency.** A request with 32 short texts takes about 55–75 ms on the measurement machine.

## English Language Check

The model is trained on English only. On other languages it returns predictions with high
confidence scores that carry no meaning. The pipeline and the API therefore reject text that is
clearly not English, recording it as `non_english`.

### Decision Rule

Rejecting real English is a worse mistake than letting foreign text through, because the latter
only reproduces the behaviour without the check. The rule therefore rejects text only on strong
evidence:

1. Remove URLs, `@mentions`, `#hashtags` and HTML entities. Lowercase, shorten repeated letters
   (`yayyyyy` becomes `yayy`) and collapse whitespace.
2. If most letters are in a non-Latin script (for example Chinese, Arabic or Cyrillic), reject.
3. If fewer than 10 letters remain, accept. Very short texts such as `lol ok` cannot be
   identified reliably.
4. Otherwise, reject only if the probability of English is below 0.01.

Language identification uses [py3langid](https://github.com/adbar/py3langid), restricted to 26
common languages.

### Measured Results

| Test set | Result |
|---|---|
| UCI sentences (all English) | 0 of 3,000 rejected |
| Short informal English (`lol ok`, `gr8 day!!`, `@john thx!`, ...) | 30 of 30 accepted |
| Short sentences in 12 other languages | 16 of 20 rejected |
| Sentiment140 tweets | 27 of 20,000 rejected; all 27 confirmed non-English on manual review |

The foreign texts that pass are short phrases such as `Muy caro` and `C'est nul`. These fall
under the minimum-length rule and are classified as before. A looser threshold (0.05) catches
one more of them but starts rejecting real English (`wow i so dumb.`), so it was not adopted.
Use `--no-language-check` to disable the check.

## Design

### Components

| Module | Responsibility |
|---|---|
| [`csv_io.py`](src/sentiment_pipeline/csv_io.py) | Streaming CSV reader. Validates the header before any work starts and yields one record per input row, including rows that fail to parse. |
| [`inference.py`](src/sentiment_pipeline/inference.py) | Shared by both entry points: text validation and batch inference with row-level fallback. |
| [`language.py`](src/sentiment_pipeline/language.py) | The English language check. |
| [`pipeline.py`](src/sentiment_pipeline/pipeline.py) | Batch orchestration: windowing, length bucketing, circuit breaker, checkpoint cadence and the resume fingerprint. |
| [`output.py`](src/sentiment_pipeline/output.py) | Durable output: checkpoint commit protocol, atomic publish, and a single-writer lock. |
| [`api.py`](src/sentiment_pipeline/api.py) | Flask application, request validation, error handling, HTTP metrics, and the gunicorn launcher. |
| [`model.py`](src/sentiment_pipeline/model.py) | `SentimentModel` protocol and the Hugging Face implementation. |
| [`metrics.py`](src/sentiment_pipeline/metrics.py) | Thread-safe counters, the JSON run report, and the Prometheus collector. Counter state is persisted in checkpoints. |
| [`cli.py`](src/sentiment_pipeline/cli.py) | Batch argument parsing, structured logging, signal handling, and exit codes. |

Both entry points depend only on the `SentimentModel` protocol (`predict(texts) -> predictions`).
That separation lets the test suite run with a stub model, and it is the single extension point
for alternative backends such as ONNX Runtime or a remote model server.

### Processing Model (Batch)

1. The input is streamed in windows of `--window-size` rows. Memory use does not depend on file size.
2. Rows that fail validation (malformed record, empty text, not English) are marked failed and
   are not sent to the model.
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
| A crashed run resumes with identical output | Checkpoint commit protocol; verified by killing real processes and containers |
| A checkpoint is reused only for the same job | Fingerprint of the input SHA-256, column mapping, encoding, model name, revision and maximum length, language check setting, and output schema |
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
| Text that is clearly not English | Row marked `non_english`; not sent to the model |
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

`<output>.metrics.json` is written for every batch outcome. It contains row and failure counts,
prediction distribution, throughput and accuracy, plus provenance fields: `run_id`, package
version, model name and revision, input SHA-256, and start time.

### Logs

JSON lines on stderr. Every line includes the `run_id` of the batch run or API instance. The
batch pipeline logs progress every 10 windows and a summary at the end. The API logs one line
per request, with its request ID, status and latency. Input text is not logged.

### Prometheus Metrics

| Metric | Type | Purpose |
|---|---|---|
| `sentiment_rows_total{status}` | Counter | Rows (or API items) processed, by outcome |
| `sentiment_row_failures_total{reason}` | Counter | Separates data-quality failures (`empty_text`, `non_english`, `malformed_csv`) from model failures (`inference_error`) |
| `sentiment_batch_fallbacks_total` | Counter | Batches retried row by row; an early indicator of bad inputs or model degradation |
| `sentiment_predictions_total{label}` | Counter | Prediction distribution; detects models that fail without raising errors |
| `sentiment_low_confidence_predictions_total` | Counter | Predictions with a score below 0.7; an indicator of drift or out-of-domain input |
| `sentiment_inference_seconds_total` | Counter | Time spent in model inference |
| `sentiment_rows_per_second` | Gauge | Throughput since start |
| `sentiment_run_info` | Info | Metadata for correlating metrics with a specific run or instance |
| `sentiment_http_requests_total{method,endpoint,status}` | Counter | API only: requests by route and status code |
| `sentiment_http_request_duration_seconds{endpoint}` | Histogram | API only: request latency |

All label values come from small fixed sets. Unknown API paths are recorded as `unmatched`, and
exception messages are never used as label values.

### Recommended Alerts

- Non-zero exit code (batch) or 5xx rate above 1% (API)
- Failure rate above 5%
- Any batch fallbacks
- Share of predicted labels outside the baseline range for the dataset
- Share of low-confidence predictions above twice the baseline
- Share of `non_english` rejections above the baseline for the input source
- Throughput below 50% of the baseline, or p95 API latency above the target

## Performance

### Length Bucketing

Sorting each window by text length before batching reduces padding overhead.

| Dataset | Without bucketing | With bucketing | Speedup |
|---|---|---|---|
| UCI (mixed lengths) | 315 rows/s | 460–520 rows/s | 1.5–1.6x |
| Sentiment140 (short texts) | 251 rows/s | 320–375 rows/s | 1.3–1.5x |

Accuracy is unchanged to 4 decimal places. A full-file sort gave a similar speedup, so sorting
within a window captures most of the benefit while memory use stays constant.

### Batch Size

| Batch size | 8 | 32 | 128 |
|---|---|---|---|
| Throughput (UCI, with bucketing) | 426 rows/s | 520 rows/s | 496 rows/s |

The language check processes about 40,000 texts per second, so it adds no measurable cost.

### Capacity Planning

- The full Sentiment140 dataset (1.6M rows) takes about 70–85 minutes in a single 4-core container.
- Peak memory is about 1 GB per process (batch run or API worker). A container memory limit of
  1.5–2 GB is recommended.
- To scale batch work horizontally, split the input across containers, each with its own output
  path and checkpoint. No state is shared between them.
- In CPU-limited containers, set `OMP_NUM_THREADS` to the CPU limit. PyTorch otherwise sizes its
  thread pool from the host core count.

## Design Decisions

| Decision | Alternatives considered | Rationale | Revisit when |
|---|---|---|---|
| Single-process batch CLI | Service with a queue; Spark or Ray | The workload is bounded and offline. A single process is the simplest option to operate and test. | Sustained volume above ~10M rows/day |
| Flask served by gunicorn | FastAPI with uvicorn | Inference is CPU-bound and runs one at a time per process, so an async framework adds no throughput. Flask with a pre-fork server is simple and mature. | Requests spend most of their time waiting on I/O |
| One validation and inference module for both entry points | Separate implementations | Batch and API results cannot drift apart | — |
| Per-item failures in the API | Fail the whole request | One bad text in a batch of 64 should not discard 63 valid results | — |
| Asymmetric English check (py3langid) | lingua; no check | Measured: equally accurate, 35x faster and 60x smaller than lingua. Rejects only on strong evidence. | Non-English traffic becomes significant (route it to a multilingual model) |
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

- The container runs as a non-root user. The batch pipeline needs no network access at runtime.
  CI runs the batch job with `--network none`, and both the batch job and the API with a
  read-only root filesystem.
- The API binds to `127.0.0.1` by default; only the container image binds to all interfaces.
  Request bodies are limited to 1 MiB and checked before parsing, and request contents are
  validated against fixed limits.
- Supply chain: the base image is pinned by digest, packages are hash-verified, and the model is
  pinned by commit. Weights are loaded from `safetensors`, and `trust_remote_code` is never enabled.
- Input text appears in the batch output file but is not logged. API error responses never
  include exception messages or stack traces.
- Batch output text is written unescaped. Values starting with `=`, `+`, `-` or `@` may be
  interpreted as formulas by spreadsheet applications. Escaping them would alter the data for
  programmatic consumers, so this is left to the consumer.

## Testing and CI

The suite has 66 tests that use a stub model, so they need no PyTorch and run in a few
seconds, plus 2 integration tests against the real model (`make test-model`).

- **Batch behaviour:** batching, validation, isolation of bad rows, malformed CSV recovery, BOM
  handling, the circuit breaker, exit codes and metrics output.
- **English check:** informal and very short English is accepted; clearly foreign and
  non-Latin-script text is rejected; surrounding whitespace cannot change a verdict.
- **API:** per-item results, every error status, size limits, request IDs, health endpoints,
  and metrics label cardinality.
- **Resume:** simulated crashes that bypass all exception handlers, partially written output,
  input changed between runs, graceful stop, and lock contention.
- **Crash test** ([`tests/test_crash.py`](tests/test_crash.py)): runs the pipeline in a child
  process, kills it with `SIGKILL` three times at different points, completes the job, and
  requires output byte-identical to an uninterrupted run.

The tests were checked by deliberately introducing bugs. The two kinds of crash test catch
different ones, so both are kept:

| Bug introduced | Real `SIGKILL` test | In-process resume tests |
|---|---|---|
| Resume does not skip committed rows | Fails | Fails |
| Resume does not truncate bytes written after the last commit | Passes | Fails |
| Checkpoint written before the data is flushed to disk | Fails | Passes |

The last case can only be caught by a real kill: in-process tests unwind normally, and Python
flushes the file buffers on the way out.

The CI pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs:

1. Lint, format check and the full stub-model suite, including the `SIGKILL` crash test.
2. A check that `requirements.lock` matches `requirements.in`.
3. A Docker build and a batch run with no network and a read-only filesystem, which must reach
   at least 90% accuracy and match the reference predictions.
4. An edge-case run that must exit with code `2`, still publish its output, and report each
   failure reason with the expected count.
5. A crash test with the real model in the real image
   ([`scripts/crash_test_docker.sh`](scripts/crash_test_docker.sh)): the container is killed
   with `SIGKILL`, resumed, stopped with `docker stop` (which must exit `130`), and resumed to
   completion. The output must be byte-identical to an uninterrupted run.
6. An API smoke test ([`scripts/api_smoke_test.sh`](scripts/api_smoke_test.sh)) with a
   read-only filesystem: readiness, predictions, the English check, error statuses, metrics,
   and a clean shutdown.

Predictions are compared across platforms on labels, with scores allowed to differ by up to
1e-3 ([`scripts/compare_predictions.py`](scripts/compare_predictions.py)). Different CPU math
libraries produce differences in the order of 1e-6, which can change the fourth decimal place,
so byte comparison is used only on the same machine and image.

## Datasets

| Dataset | Preparation |
|---|---|
| UCI Sentiment Labelled Sentences | The three source files are merged and labels normalised to `positive`/`negative`. The IMDB file contains U+0085 characters inside sentences, so it is read as a text stream rather than with `str.splitlines()`, which would split on them. |
| Sentiment140 | The source file is sorted by label, so the first N rows would all be negative. A seeded reservoir sample produces a uniform random subset in one pass. |

[`data/edge_cases.csv`](data/edge_cases.csv) covers quoted delimiters, multi-line values,
non-English text, empty values, text above the token limit, and a malformed record.

### Data Sources

The datasets in [`data/`](data/) are redistributed for evaluation only and remain under their
original terms.

- **Sentiment Labelled Sentences**: D. Kotzias, M. Denil, N. de Freitas and P. Smyth, "From
  Group to Individual Labels using Deep Features", KDD 2015. Available from the
  [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Labels were renamed from
  `0`/`1` to `negative`/`positive`.
- **Sentiment140**: A. Go, R. Bhayani and L. Huang, "Twitter Sentiment Classification using
  Distant Supervision", Stanford University, 2009. Available from
  [Sentiment140](http://help.sentiment140.com/for-students). `data/sentiment140_20k.csv` is a
  seeded random sample of 20,000 tweets.

### Model Behaviour

- Accuracy is higher on UCI (92.8%) than on Sentiment140 (70.4%). SST-2 consists of reviews,
  which closely match UCI. Tweets are a different domain, and Sentiment140 labels are inferred
  from emoticons and are therefore noisy.
- On Sentiment140 the model predicts 59.6% negative against a true rate of 49.6%. This bias
  shows up in the prediction-distribution metric but not in the accuracy figure.
- Some Sentiment140 tweets are double-encoded (for example Vietnamese or Russian text that
  appears as `Ð\x94Ð°...`), a known defect of the dataset. The English check rejects them.
- The English check removes the one confidently wrong prediction in the edge-case file, a
  French and Japanese review.

## Limitations and Roadmap

1. Parallel batch execution across sharded inputs, with a merge step.
2. Batching across concurrent API requests, to raise throughput under load.
3. ONNX Runtime with int8 quantisation, validated against the reference predictions.
4. Routing non-English text to a multilingual model instead of rejecting it, if demand exists.
5. Skipping a batch run when a completed output with a matching fingerprint already exists.
6. Pushing final batch metrics to a Prometheus Pushgateway for scheduled runs.
7. A metric counting replaced (undecodable) characters.
8. A three-class model if a `neutral` label is needed. The binary model always assigns
   `positive` or `negative`.
