from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Set at start of main() for clearer connection-refused messages.
_LAST_SMOKE_BASE_URL = "http://127.0.0.1:8080"


def _connection_refused(exc: BaseException) -> bool:
    parts = [str(exc).lower()]
    if isinstance(exc, urllib.error.URLError) and exc.reason is not None:
        parts.append(str(exc.reason).lower())
    blob = " ".join(parts)
    return (
        "actively refused" in blob
        or "connection refused" in blob
        or "errno 111" in blob  # Linux refused
        or "10061" in blob  # Windows refused
    )


def request_json(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return resp.getcode(), json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"error": raw}
        return e.code, parsed


def request_text(url: str, method: str = "GET") -> tuple[int, str]:
    req = urllib.request.Request(url=url, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.getcode(), resp.read().decode("utf-8")


def assert_ok(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    global _LAST_SMOKE_BASE_URL
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    _LAST_SMOKE_BASE_URL = base

    print("[1/7] Checking /health")
    code, health = request_json(f"{base}/health")
    assert_ok(code == 200, f"/health failed with status {code}")
    assert_ok(health.get("status") == "ok", f"/health unexpected payload: {health}")

    print("[2/7] Checking /generate")
    code, pure = request_json(
        f"{base}/generate",
        method="POST",
        body={"prompt": "Explain retrieval augmentation briefly.", "max_new_tokens": 20},
    )
    assert_ok(code == 200, f"/generate failed with status {code}: {pure}")
    assert_ok("generated_text" in pure and "latency_ms" in pure, f"/generate payload invalid: {pure}")

    print("[3/7] Checking /hybrid_generate")
    code, hybrid = request_json(
        f"{base}/hybrid_generate",
        method="POST",
        body={"prompt": "Explain retrieval augmentation briefly.", "retrieval_top_k": 2, "max_new_tokens": 20},
    )
    assert_ok(code == 200, f"/hybrid_generate failed with status {code}: {hybrid}")
    assert_ok(
        "generated_text" in hybrid and "contexts" in hybrid and "latency_ms" in hybrid,
        f"/hybrid_generate payload invalid: {hybrid}",
    )

    print("[4/7] Checking /history pagination")
    code, history = request_json(f"{base}/history?limit=5&offset=0")
    assert_ok(code == 200, f"/history failed with status {code}: {history}")
    assert_ok(
        all(k in history for k in ("total", "limit", "offset", "items")),
        f"/history payload invalid: {history}",
    )

    print("[5/7] Checking /history/export JSON")
    code, export_json = request_json(f"{base}/history/export?fmt=json")
    assert_ok(code == 200, f"/history/export?fmt=json failed with status {code}")
    assert_ok("items" in export_json, f"/history export json payload invalid: {export_json}")

    print("[6/7] Checking /history/export CSV")
    code, csv_text = request_text(f"{base}/history/export?fmt=csv")
    assert_ok(code == 200, f"/history/export?fmt=csv failed with status {code}")
    assert_ok("id,mode,prompt" in csv_text, "CSV export missing expected header")

    print("[7/7] Checking /retrieve_debug")
    code, dbg = request_json(f"{base}/retrieve_debug?q=test&k=3")
    assert_ok(code == 200, f"/retrieve_debug failed with status {code}: {dbg}")
    assert_ok("hits" in dbg and isinstance(dbg["hits"], list), f"/retrieve_debug payload invalid: {dbg}")

    print("Smoke release checks passed.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        if _connection_refused(exc):
            print(
                "Smoke release failed: nothing is listening at "
                f"{_LAST_SMOKE_BASE_URL}.\n"
                "Start the API first in another terminal (from the project root), then retry:\n"
                "  .\\scripts\\run_api.ps1\n"
                "or:\n"
                "  $env:PYTHONPATH='.'; .\\.venv\\Scripts\\python.exe -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8080\n"
                "(First load can take about 1-2 minutes on CPU.)\n"
                f"Original error: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"Smoke release failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Smoke release failed: {exc}")
        sys.exit(1)
