#!/usr/bin/env bash
# Smoke test of the HTTP API in the real image, with a read-only root filesystem.
#
#   scripts/api_smoke_test.sh [image] [host port]
set -euo pipefail

IMAGE="${1:-sentiment-pipeline}"
PORT="${2:-18080}"
NAME="sentiment-api-smoke-$$"
URL="http://localhost:$PORT"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

docker run -d --name "$NAME" -p "$PORT:8080" --read-only --tmpfs /tmp \
  --entrypoint sentiment-api "$IMAGE" > /dev/null

for _ in $(seq 1 120); do
  curl -fs "$URL/readyz" > /dev/null 2>&1 && break
  sleep 0.5
done
curl -fs "$URL/readyz" | jq -e '.status == "ready"' > /dev/null || { docker logs "$NAME"; exit 1; }
curl -fs "$URL/healthz" | jq -e '.status == "ok"' > /dev/null

response="$(curl -fs -X POST "$URL/v1/predict" -H 'Content-Type: application/json' \
  -d '{"texts": ["I absolutely love this phone", "Worst purchase I have ever made",
                 "Ce film était magnifique, je le recommande", "   "]}')"
echo "$response" | jq .
echo "$response" | jq -e '
  .predictions[0].label == "positive" and
  .predictions[1].label == "negative" and
  (.predictions[2].error | startswith("non_english")) and
  (.predictions[3].error | startswith("empty_text"))' > /dev/null

test "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/v1/predict" -d 'not json')" = 415
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/v1/predict" \
  -H 'Content-Type: application/json' -d '{"texts": [1]}')" = 400

curl -fs "$URL/metrics" | grep -q 'sentiment_http_requests_total{endpoint="/v1/predict",method="POST",status="200"}'

docker stop --time 30 "$NAME" > /dev/null
test "$(docker inspect -f '{{.State.ExitCode}}' "$NAME")" -eq 0
echo "PASS: API serves predictions, enforces its contract, exposes metrics and shuts down cleanly"
