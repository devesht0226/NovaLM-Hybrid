from __future__ import annotations

import json
import torch

from src.models.transformer_lm import MiniTransformerLM, TransformerConfig
from src.training.evaluate import evaluate_loss
from src.training.metrics import perplexity_from_loss
from src.tokenization.tokenizer_utils import TokenizerWrapper
from src.utils.checkpoint import load_pt_checkpoint
from src.utils.config import ROOT


if __name__ == "__main__":
    tokenizer = TokenizerWrapper(str(ROOT / "artifacts/tokenizers/tokenizer.model"))
    ckpt_path = ROOT / "artifacts/checkpoints/best.pt"
    ckpt = load_pt_checkpoint(ckpt_path, map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])
    model = MiniTransformerLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    seq_len = int(cfg.max_seq_len)

    tokens_json = ROOT / "data/processed/token_ids.json"
    if tokens_json.exists():
        obj = json.loads(tokens_json.read_text(encoding="utf-8"))
        val_ids = torch.tensor(obj["val_ids"], dtype=torch.long)
        val_loss = evaluate_loss(model, val_ids, seq_len=seq_len)
        print(
            f"val_loss={val_loss:.4f} val_ppl={perplexity_from_loss(val_loss):.2f} "
            f"(recomputed on val_ids from {tokens_json.name})"
        )

    test_text = (ROOT / "data/splits/test.txt").read_text(encoding="utf-8")
    test_ids = torch.tensor(tokenizer.encode(test_text), dtype=torch.long)
    loss = evaluate_loss(model, test_ids, seq_len=seq_len)
    print(f"test_loss={loss:.4f} test_ppl={perplexity_from_loss(loss):.2f}")
