#!/usr/bin/env bash
# Crash test against the real image and the real model.
#
#   1. Uninterrupted reference run.
#   2. Start the same job, SIGKILL the container mid-run (no cleanup code runs).
#   3. Rerun it, stop it gracefully with `docker stop` mid-run (must exit 130).
#   4. Rerun it to completion.
# The output must be byte-identical to the reference: same image, same machine,
# same batch composition, so not even float noise may differ.
#
#   scripts/crash_test_docker.sh [image]
set -euo pipefail

IMAGE="${1:-sentiment-pipeline}"
NAME="sentiment-crash-test-$$"
WORK="$(mktemp -d)"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT
mkdir -p "$WORK/reference" "$WORK/crashed"

# Small windows give many commit points to interrupt between.
ARGS=(--input /data/uci_sentences.csv --output /output/predictions.csv
      --id-column id --label-column label --window-size 128)

docker_run() {  # docker_run <output dir> [extra docker args...]
  local out="$1"; shift
  docker run --network none --user "$(id -u):$(id -g)" \
    -v "$PWD/data:/data:ro" -v "$out:/output" "$@" "$IMAGE" "${ARGS[@]}"
}

committed() {
  jq -r '.rows_committed' "$WORK/crashed/predictions.csv.checkpoint.json" 2>/dev/null || echo 0
}

wait_for_rows() {  # block until the checkpoint records at least $1 rows
  for _ in $(seq 1 600); do
    [ "$(committed)" -ge "$1" ] && return 0
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ] || {
      echo "container stopped before reaching $1 rows"; docker logs "$NAME" | tail -5; return 1; }
    sleep 0.1
  done
  echo "timed out waiting for $1 committed rows"; return 1
}

echo "== reference run"
docker_run "$WORK/reference" --rm > /dev/null 2>&1

echo "== run 1: SIGKILL mid-run"
docker_run "$WORK/crashed" -d --name "$NAME" > /dev/null
wait_for_rows 512
docker kill --signal KILL "$NAME" > /dev/null
docker wait "$NAME" > /dev/null; docker rm "$NAME" > /dev/null
echo "   killed with $(committed) rows committed"
test ! -e "$WORK/crashed/predictions.csv"   # nothing published

echo "== run 2: resume, then graceful stop (SIGTERM)"
start=$(committed)
docker_run "$WORK/crashed" -d --name "$NAME" > /dev/null
wait_for_rows $((start + 512))
docker stop --time 30 "$NAME" > /dev/null
code=$(docker wait "$NAME"); docker rm "$NAME" > /dev/null
echo "   stopped with $(committed) rows committed, exit code $code"
test "$code" -eq 130
test ! -e "$WORK/crashed/predictions.csv"

echo "== run 3: resume to completion"
docker_run "$WORK/crashed" --rm > /dev/null 2>&1
jq -e '.status == "completed" and .rows_read == 3000 and .rows_resumed > 0' \
  "$WORK/crashed/predictions.metrics.json" > /dev/null

cmp "$WORK/reference/predictions.csv" "$WORK/crashed/predictions.csv"
echo "PASS: output after SIGKILL + SIGTERM + resume is byte-identical to the uninterrupted run"
