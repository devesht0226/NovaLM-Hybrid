from __future__ import annotations

import argparse
import json
import yaml

from src.data_pipeline.build_index import build_chunks, save_chunks
from src.data_pipeline.clean import clean_text_lines
from src.data_pipeline.dedup import deduplicate
from src.data_pipeline.ingest import load_raw_texts
from src.data_pipeline.split import split_lines
from src.utils.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild processed/splits/index even if processed file already exists.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
    processed = ROOT / cfg["processed_path"]
    if processed.exists() and not args.force:
        print(f"Skipping preprocess: existing processed file found at {processed}. Use --force to rebuild.")
        return

    texts = load_raw_texts(str(ROOT / cfg["raw_dir"]))
    if not texts:
        raise SystemExit(
            "No raw text files found. Add .txt files under data/raw/ before preprocessing "
            "(running with an empty folder would wipe downstream artifacts)."
        )
    lines = []
    for t in texts:
        lines.extend(clean_text_lines(t))
    lines = deduplicate(lines)
    train, val, test = split_lines(lines, seed=int(cfg["seed"]))

    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / cfg["train_path"]).write_text("\n".join(train), encoding="utf-8")
    (ROOT / cfg["val_path"]).write_text("\n".join(val), encoding="utf-8")
    (ROOT / cfg["test_path"]).write_text("\n".join(test), encoding="utf-8")
    (ROOT / "data/splits/splits.json").write_text(
        json.dumps({"train": train, "val": val, "test": test}, ensure_ascii=False),
        encoding="utf-8",
    )

    chunk_size = int(cfg.get("chunk_size", 4))
    overlap = int(cfg.get("chunk_overlap_lines", 0))
    chunks = build_chunks(lines, chunk_size=chunk_size, overlap_lines=overlap)
    save_chunks(chunks, str(ROOT / cfg["retrieval_chunks_path"]))
    print(f"Processed {len(lines)} unique lines. Saved {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
