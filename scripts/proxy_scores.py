"""
Fill artifacts/reports/scores.csv using lightweight heuristics from compare_rows.json.

These are transparent proxies (lexical overlap, repetition, context overlap) - not a
human rubric. Replace with manual scores if your course requires blind evaluation.

Usage:
  python scripts/run_compare.py          # writes compare_rows.json + final_report.md
  python scripts/proxy_scores.py
  python scripts/score_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from src.utils.config import ROOT


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _coherence_score(text: str) -> int:
    w = _words(text)
    if len(w) < 2:
        return 1
    ratio = len(set(w)) / len(w)
    if "⁇" in text or "\ufffd" in text:
        ratio *= 0.55
    return int(min(5, max(1, round(1 + 4 * ratio))))


def _relevance_score(prompt: str, text: str) -> int:
    pw = set(_words(prompt))
    tw = set(_words(text))
    if not pw:
        return 2
    hit = len(pw & tw) / len(pw)
    return int(min(5, max(1, round(1 + 4 * hit))))


def _hybrid_groundedness(text: str, contexts: list[dict]) -> int:
    if not contexts:
        return 1
    ctx: set[str] = set()
    for c in contexts:
        ctx |= set(_words(str(c.get("text", ""))))
    tw = set(_words(text))
    if not tw:
        return 1
    hit = len(tw & ctx) / len(tw)
    return int(min(5, max(1, round(1 + 4 * hit))))


def _pure_groundedness(coh: int, rel: int) -> int:
    return int(min(5, max(1, round((coh + rel) / 2))))


def score_row(prompt: str, pure: str, hybrid: str, contexts: list[dict]) -> tuple[dict, dict]:
    pc, pr = _coherence_score(pure), _relevance_score(prompt, pure)
    pg = _pure_groundedness(pc, pr)
    hc, hr = _coherence_score(hybrid), _relevance_score(prompt, hybrid)
    hg = _hybrid_groundedness(hybrid, contexts)
    pure_d = {
        "coherence": pc,
        "relevance": pr,
        "groundedness": pg,
        "notes": "proxy heuristic (proxy_scores.py); replace with human rubric if required",
    }
    hybrid_d = {
        "coherence": hc,
        "relevance": hr,
        "groundedness": hg,
        "notes": "proxy heuristic (proxy_scores.py); replace with human rubric if required",
    }
    return pure_d, hybrid_d


def write_scores_csv(path: Path, rows_json: list[dict]) -> None:
    out_rows: list[list[str]] = [
        ["prompt_id", "prompt", "mode", "coherence", "relevance", "groundedness", "tags", "notes"],
    ]
    for row in rows_json:
        pid = str(row["id"])
        prompt = str(row["prompt"])
        ctx = row.get("contexts") or []
        pure_d, hybrid_d = score_row(prompt, str(row["pure"]), str(row["hybrid"]), ctx)
        out_rows.append(
            [pid, prompt, "pure", str(pure_d["coherence"]), str(pure_d["relevance"]), str(pure_d["groundedness"]), "", pure_d["notes"]]
        )
        out_rows.append(
            [
                pid,
                prompt,
                "hybrid",
                str(hybrid_d["coherence"]),
                str(hybrid_d["relevance"]),
                str(hybrid_d["groundedness"]),
                "",
                hybrid_d["notes"],
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(out_rows)


def _inject_table_scores(report_text: str, scored: dict[int, tuple[int, int, int, int, int, int]]) -> str:
    """Replace trailing six '| - |' score slots on each prompt row."""
    tail = "| - | - | - | - | - | - |"
    lines_out: list[str] = []
    for line in report_text.splitlines():
        s = line.rstrip()
        if "| P.coh |" in line or s.startswith("|---") or not s.endswith(tail):
            lines_out.append(line)
            continue
        prefix = s[: -len(tail)].rstrip()
        m_id = re.match(r"^\|\s*(\d+)\s*\|", prefix)
        if not m_id:
            lines_out.append(line)
            continue
        pid = int(m_id.group(1))
        if pid not in scored:
            lines_out.append(line)
            continue
        pc, pr, pg, hc, hr, hg = scored[pid]
        lines_out.append(f"{prefix} | {pc} | {pr} | {pg} | {hc} | {hr} | {hg} |")
    return "\n".join(lines_out)


def inject_into_report(report_path: Path, rows_json: list[dict]) -> None:
    scored: dict[int, tuple[int, int, int, int, int, int]] = {}
    for row in rows_json:
        pure_d, hybrid_d = score_row(str(row["prompt"]), str(row["pure"]), str(row["hybrid"]), row.get("contexts") or [])
        scored[int(row["id"])] = (
            pure_d["coherence"],
            pure_d["relevance"],
            pure_d["groundedness"],
            hybrid_d["coherence"],
            hybrid_d["relevance"],
            hybrid_d["groundedness"],
        )
    text = report_path.read_text(encoding="utf-8")
    text = _inject_table_scores(text, scored)
    marker = "Heuristic score injection (scripts/proxy_scores.py)"
    if marker not in text and "## Best Cases" in text:
        note = (
            f"\n\n*{marker}: table scores are auto-filled. "
            "Edit `artifacts/reports/scores.csv` manually and run `python scripts/score_report.py` for rubric-only workflows.*\n"
        )
        text = text.replace("\n## Best Cases", note + "\n## Best Cases", 1)
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-json", default="artifacts/reports/compare_rows.json")
    p.add_argument("--scores", default="artifacts/reports/scores.csv")
    p.add_argument("--report", default="artifacts/reports/final_report.md")
    args = p.parse_args()

    jpath = Path(args.compare_json)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    rows = json.loads(jpath.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("compare_rows.json must contain a list of row objects.")

    scores_path = Path(args.scores)
    if not scores_path.is_absolute():
        scores_path = ROOT / scores_path
    write_scores_csv(scores_path, rows)
    print(f"Wrote heuristic scores to {scores_path}")

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    if report_path.exists():
        inject_into_report(report_path, rows)
        print(f"Injected score columns into {report_path}")


if __name__ == "__main__":
    main()
