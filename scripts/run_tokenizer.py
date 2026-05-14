import yaml

from src.tokenization.train_tokenizer import train_sentencepiece
from src.utils.config import ROOT


if __name__ == "__main__":
    model_cfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
    vocab_size = int(model_cfg["vocab_size"])
    train_txt = ROOT / "data/splits/train.txt"
    if not train_txt.exists():
        raise SystemExit(f"Missing {train_txt}; run scripts/run_preprocess.py first.")

    path = train_sentencepiece(
        input_path=str(train_txt),
        output_prefix=str(ROOT / "artifacts/tokenizers/tokenizer"),
        vocab_size=vocab_size,
    )
    print(f"Tokenizer written to: {path}")
