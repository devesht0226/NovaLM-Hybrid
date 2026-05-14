from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def load_scores(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON score file must be a list of objects.")
        return data
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError("Unsupported score file format. Use .csv or .json")


def compute_averages(rows: list[dict]) -> dict[str, float]:
    buckets = {
        "pure_coherence": [],
        "pure_relevance": [],
        "pure_groundedness": [],
        "hybrid_coherence": [],
        "hybrid_relevance": [],
        "hybrid_groundedness": [],
    }
    for r in rows:
        mode = str(r.get("mode", "")).strip().lower()
        coh = _to_float(str(r.get("coherence", "")))
        rel = _to_float(str(r.get("relevance", "")))
        grd = _to_float(str(r.get("groundedness", "")))
        if mode == "pure":
            buckets["pure_coherence"].append(coh)
            buckets["pure_relevance"].append(rel)
            buckets["pure_groundedness"].append(grd)
        elif mode == "hybrid":
            buckets["hybrid_coherence"].append(coh)
            buckets["hybrid_relevance"].append(rel)
            buckets["hybrid_groundedness"].append(grd)

    def avg(vals: list[float]) -> float:
        return sum(vals) / max(1, len(vals))

    out = {k: avg(v) for k, v in buckets.items()}
    out["pure_overall"] = (out["pure_coherence"] + out["pure_relevance"] + out["pure_groundedness"]) / 3.0
    out["hybrid_overall"] = (
        out["hybrid_coherence"] + out["hybrid_relevance"] + out["hybrid_groundedness"]
    ) / 3.0
    return out


def upsert_summary(report_path: str, averages: dict[str, float]) -> None:
    path = Path(report_path)
    text = path.read_text(encoding="utf-8") if path.exists() else "# NovaLM-Hybrid Final Report\n"

    summary = "\n".join(
        [
            "## Score Averages",
            f"- Pure coherence: {averages['pure_coherence']:.2f}",
            f"- Pure relevance: {averages['pure_relevance']:.2f}",
            f"- Pure groundedness: {averages['pure_groundedness']:.2f}",
            f"- Pure overall: {averages['pure_overall']:.2f}",
            f"- Hybrid coherence: {averages['hybrid_coherence']:.2f}",
            f"- Hybrid relevance: {averages['hybrid_relevance']:.2f}",
            f"- Hybrid groundedness: {averages['hybrid_groundedness']:.2f}",
            f"- Hybrid overall: {averages['hybrid_overall']:.2f}",
            (
                "- Winner: comparable"
                if abs(averages["hybrid_overall"] - averages["pure_overall"]) < 0.2
                else "- Winner: hybrid" if averages["hybrid_overall"] > averages["pure_overall"] else "- Winner: pure"
            ),
            "",
        ]
    )

    marker_start = "## Score Averages"
    if marker_start in text:
        head = text.split(marker_start)[0].rstrip()
        tail = ""
        # Keep content after next section header if exists
        rest = text.split(marker_start, 1)[1]
        idx = rest.find("\n## ", 1)
        if idx != -1:
            tail = rest[idx:]
        text = f"{head}\n\n{summary}{tail}"
    else:
        text = f"{text.rstrip()}\n\n{summary}"

    path.write_text(text, encoding="utf-8")


def write_template_csv(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prompts = [
        "What is self-attention in transformers?",
        "Why do language models use tokenization?",
        "Difference between top-k and top-p sampling?",
        "What causes hallucinations in LMs?",
        "What is perplexity and why does it matter?",
        "Write a short motivational message for students.",
        "Explain recursion with a simple analogy.",
        "Draft a polite email asking for project feedback.",
        "Give a 5-point plan to learn deep learning quickly.",
        "Write a 4-line poem about coding at night.",
    ]
    rows = [
        ["prompt_id", "prompt", "mode", "coherence", "relevance", "groundedness", "tags", "notes"],
    ]
    for i, prompt in enumerate(prompts, start=1):
        rows.append([str(i), prompt, "pure", "", "", "", "", ""])
        rows.append([str(i), prompt, "hybrid", "", "", "", "", ""])
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="artifacts/reports/scores.csv")
    parser.add_argument("--report", default="artifacts/reports/final_report.md")
    parser.add_argument("--init-template", action="store_true")
    args = parser.parse_args()

    if args.init_template:
        write_template_csv(args.scores)
        print(f"Template score sheet created at {args.scores}")
        return

    rows = load_scores(args.scores)
    averages = compute_averages(rows)
    upsert_summary(args.report, averages)
    print(f"Updated {args.report} with score averages.")


if __name__ == "__main__":
    main()
