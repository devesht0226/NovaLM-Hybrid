from __future__ import annotations

import json
from pathlib import Path


def build_chunks(
    corpus_lines: list[str],
    chunk_size: int = 4,
    overlap_lines: int = 0,
) -> list[str]:
    if chunk_size < 1:
        chunk_size = 1
    overlap_lines = max(0, min(int(overlap_lines), chunk_size - 1))
    step = max(1, chunk_size - overlap_lines)
    chunks: list[str] = []
    i = 0
    while i < len(corpus_lines):
        window = corpus_lines[i : i + chunk_size]
        chunk = " ".join(window).strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def save_chunks(chunks: list[str], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
