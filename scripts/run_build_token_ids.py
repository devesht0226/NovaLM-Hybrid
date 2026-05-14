import json
from pathlib import Path

import sentencepiece as spm

from src.utils.config import ROOT

SPLITS_JSON = ROOT / "data/splits/splits.json"
SP_MODEL = ROOT / "artifacts/tokenizers/tokenizer.model"
OUT_JSON = ROOT / "data/processed/token_ids.json"


def main():
    if not SPLITS_JSON.exists():
        raise SystemExit(f"Missing {SPLITS_JSON}; run scripts/run_preprocess.py first.")
    if not SP_MODEL.exists():
        raise SystemExit(f"Missing {SP_MODEL}; run scripts/run_tokenizer.py first.")

    splits = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))

    train_text = "\n".join(splits["train"])
    val_text = "\n".join(splits["val"])

    train_ids = sp.encode(train_text, out_type=int)
    val_ids = sp.encode(val_text, out_type=int)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"train_ids": train_ids, "val_ids": val_ids}),
        encoding="utf-8",
    )
    print(f"Saved {OUT_JSON} | train_ids={len(train_ids)} val_ids={len(val_ids)}")


if __name__ == "__main__":
    main()
