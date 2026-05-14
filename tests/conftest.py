"""Pytest configuration for NovaLM-Hybrid."""

from __future__ import annotations

import os


def pytest_ignore_collect(collection_path, config):
    """
    tests/test_api.py imports src.serving.api, which loads the checkpoint at import time.
    Skip collecting that module when integration is excluded or on GitHub Actions without weights.
    """
    if collection_path.name != "test_api.py":
        return False
    if os.environ.get("RUN_INTEGRATION_API", "").lower() in ("1", "true", "yes"):
        return False
    markexpr = (getattr(config.option, "markexpr", None) or "").replace(" ", "").lower()
    return "notintegration" in markexpr
