from __future__ import annotations

from pathlib import Path


def load_raw_texts(raw_dir: str) -> list[str]:
    texts: list[str] = []
    for path in sorted(Path(raw_dir).rglob("*.txt")):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return texts
