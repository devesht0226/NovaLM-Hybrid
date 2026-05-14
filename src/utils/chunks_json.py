from __future__ import annotations

import json
from pathlib import Path


def load_chunks_json(path: Path | str) -> list[str]:
    """Load retrieval chunks from JSON: bare list or {"chunks": [...]} / legacy shapes."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid retrieval chunks JSON: {p}") from exc

    if isinstance(data, dict):
        arr = data.get("chunks", data.get("items", []))
    elif isinstance(data, list):
        arr = data
    else:
        raise RuntimeError(f"Retrieval chunks must be list or object: {p}")

    if not isinstance(arr, list):
        raise RuntimeError(f"Retrieval chunks must be a list: {p}")

    out: list[str] = []
    for item in arr:
        if isinstance(item, dict):
            txt = str(item.get("text", "")).strip()
        else:
            txt = str(item).strip()
        if txt:
            out.append(txt)
    return out
