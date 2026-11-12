from __future__ import annotations

import pytest
from conftest import FakeModel

from sentiment_pipeline.api import MAX_TEXT_CHARS, MAX_TEXTS_PER_REQUEST, create_app


@pytest.fixture
def client():
    app = create_app(FakeModel(poison=["BOOM"]), model_version="fake@1")
    return app.test_client()


def predict(client, texts):
    return client.post("/v1/predict", json={"texts": texts})


def test_predicts_in_order_with_per_item_results(client):
    response = predict(client, ["a good day", "a bad day"])

    assert response.status_code == 200
    body = response.get_json()
    assert body["model"] == "fake@1"
    assert [p["label"] for p in body["predictions"]] == ["positive", "negative"]
    assert all(p["status"] == "ok" and p["error"] is None for p in body["predictions"])


def test_content_problems_fail_per_item_not_per_request(client):
    response = predict(client, ["good", "   ", "Das Essen war wirklich schrecklich", "BOOM", "bad"])

    assert response.status_code == 200
    results = response.get_json()["predictions"]
    assert [r["status"] for r in results] == ["ok", "failed", "failed", "failed", "ok"]
    assert results[1]["error"].startswith("empty_text")
    assert results[2]["error"].startswith("non_english")
    # Only the exception type is exposed, never its message (which can echo input).
    assert results[3]["error"] == "inference_error: ValueError"


def test_language_check_can_be_disabled():
    client = create_app(FakeModel(), language_check=False).test_client()
    result = predict(client, ["Das Essen war wirklich schrecklich"]).get_json()["predictions"][0]
    assert result["status"] == "ok"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"text": "singular key"}, '"texts" must be a non-empty array'),
        ({"texts": []}, '"texts" must be a non-empty array'),
        ({"texts": "not a list"}, '"texts" must be a non-empty array'),
        ({"texts": ["ok", 42]}, '"texts[1]" must be a string'),
        ({"texts": ["x"] * (MAX_TEXTS_PER_REQUEST + 1)}, "at most"),
        ({"texts": ["x" * (MAX_TEXT_CHARS + 1)]}, "exceeds"),
        (["not", "an", "object"], "must be a JSON object"),
    ],
)
def test_contract_violations_return_400_with_json_error(client, body, message):
    response = client.post("/v1/predict", json=body)

    assert response.status_code == 400
    assert message in response.get_json()["error"]["message"]


def test_non_json_body_returns_415(client):
    response = client.post("/v1/predict", data="texts=hello", content_type="text/plain")
    assert response.status_code == 415


def test_malformed_json_returns_400(client):
    response = client.post("/v1/predict", data="{not json", content_type="application/json")
    assert response.status_code == 400


def test_oversized_body_returns_413_before_parsing(client):
    response = client.post("/v1/predict", data="x" * (2 << 20), content_type="application/json")
    assert response.status_code == 413


def test_unknown_route_and_wrong_method_return_json_errors(client):
    assert client.get("/nope").get_json()["error"]["code"] == 404
    assert client.get("/v1/predict").status_code == 405


def test_health_and_readiness(client):
    assert client.get("/healthz").get_json() == {"status": "ok"}
    ready = client.get("/readyz").get_json()
    assert ready["status"] == "ready" and ready["model"] == "fake@1"


def test_request_id_is_propagated_or_generated(client):
    assert (
        client.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"]
        == "abc-123"
    )
    generated = client.get("/healthz", headers={"X-Request-ID": "bad id with spaces"})
    assert generated.headers["X-Request-ID"] != "bad id with spaces"


def test_metrics_cover_http_traffic_and_model_behaviour(client):
    predict(client, ["good", "bad", ""])
    client.get("/nope")

    text = client.get("/metrics").get_data(as_text=True)

    assert (
        'sentiment_http_requests_total{endpoint="/v1/predict",method="POST",status="200"} 1.0'
        in text
    )
    # Unknown paths collapse into one label value, so cardinality stays bounded.
    assert 'endpoint="unmatched"' in text and "/nope" not in text
    assert 'sentiment_predictions_total{label="positive"} 1.0' in text
    assert 'sentiment_row_failures_total{reason="empty_text"} 1.0' in text
    assert "sentiment_http_request_duration_seconds_bucket" in text
