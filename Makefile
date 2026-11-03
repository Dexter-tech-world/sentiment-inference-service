IMAGE ?= sentiment-pipeline
VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: install lock test test-model lint data run docker-build docker-run

install:            ## local dev environment (needs uv)
	uv venv --python 3.12 --allow-existing $(VENV)
	uv pip sync --python $(PY) --require-hashes --torch-backend cpu requirements.lock
	uv pip install --python $(PY) -r requirements-dev.txt
	uv pip install --python $(PY) --no-deps -e .

lock:               ## re-resolve requirements.in into the hashed, cross-platform lock
	uv pip compile requirements.in --universal --python-version 3.12 --torch-backend cpu \
		--generate-hashes --custom-compile-command "make lock" -o requirements.lock

test:               ## fast tests (fake model, no torch needed)
	$(PY) -m pytest

test-model:         ## tests against the real model (downloads it once)
	RUN_MODEL_TESTS=1 $(PY) -m pytest tests/test_model_integration.py

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

data:               ## re-download and normalise the datasets
	$(PY) scripts/prepare_data.py uci --out data/uci_sentences.csv
	$(PY) scripts/prepare_data.py sentiment140 --limit 20000 --out data/sentiment140_20k.csv

run:
	$(PY) -m sentiment_pipeline --input data/uci_sentences.csv --output output/predictions.csv \
		--id-column id --label-column label --log-format text

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --user "$$(id -u):$$(id -g)" -p 8000:8000 \
		-v "$$PWD/data:/data:ro" -v "$$PWD/output:/output" $(IMAGE) \
		--input /data/uci_sentences.csv --output /output/predictions.csv \
		--id-column id --label-column label --metrics-port 8000
