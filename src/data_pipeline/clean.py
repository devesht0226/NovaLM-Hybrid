from __future__ import annotations

import re


def clean_text_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        s = re.sub(r"\s+", " ", line.strip().lower())
        if len(s) >= 3:
            lines.append(s)
    return lines
