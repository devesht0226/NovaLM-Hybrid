from __future__ import annotations


def deduplicate(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out
