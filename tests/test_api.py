import pytest
from fastapi.testclient import TestClient

from src.serving.api import app

pytestmark = pytest.mark.integration


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_history_endpoint_shape():
    client = TestClient(app)
    r = client.get("/history?limit=5")
    assert r.status_code == 200
    payload = r.json()
    assert "total" in payload
    assert "items" in payload
    data = payload["items"]
    assert isinstance(data, list)
    if data:
        row = data[0]
        assert "id" in row
        assert "mode" in row
        assert "prompt" in row
        assert "generated_text" in row


def test_history_filters_and_delete_not_found():
    client = TestClient(app)
    r = client.get("/history?limit=5&mode=pure&q=test")
    assert r.status_code == 200
    assert isinstance(r.json()["items"], list)

    r2 = client.delete("/history/999999999")
    assert r2.status_code == 404


def test_retrieve_debug_shape():
    client = TestClient(app)
    r = client.get("/retrieve_debug?q=hello&k=2")
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("query") == "hello"
    assert "hits" in payload and isinstance(payload["hits"], list)


def test_invalid_params_and_exports():
    client = TestClient(app)
    r = client.post("/generate", json={"prompt": "   "})
    assert r.status_code == 422

    r2 = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 9999})
    assert r2.status_code == 422

    e1 = client.get("/history/export?fmt=json")
    assert e1.status_code == 200
    assert "items" in e1.json()

    e2 = client.get("/history/export?fmt=csv")
    assert e2.status_code == 200
    assert "id,mode,prompt" in e2.text
