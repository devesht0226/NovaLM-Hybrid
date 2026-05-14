from __future__ import annotations

import torch
import yaml

from src.models.hybrid_generator import HybridGenerator
from src.models.retriever import TfidfRetriever
from src.models.transformer_lm import MiniTransformerLM, TransformerConfig
from src.utils.checkpoint import load_pt_checkpoint
from src.utils.chunks_json import load_chunks_json
from src.utils.config import ROOT, load_all_configs


def safe_console_text(s: str) -> str:
    return s.encode("cp1252", errors="replace").decode("cp1252")


if __name__ == "__main__":
    prompt = "Explain transformer attention simply."
    infer_cfg = yaml.safe_load((ROOT / "configs/infer.yaml").read_text(encoding="utf-8"))
    retrieval_cfg = load_all_configs().get("retrieval") or {}
    ckpt = load_pt_checkpoint(ROOT / "artifacts/checkpoints/best.pt", map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])
    model = MiniTransformerLM(cfg)
    model.load_state_dict(ckpt["model_state"])

    hybrid = HybridGenerator(
        model=model,
        sp_model_path=str(ROOT / "artifacts/tokenizers/tokenizer.model"),
        device="cpu",
    )
    top_p = infer_cfg.get("top_p")
    top_pf = float(top_p) if top_p is not None else None
    rep = float(infer_cfg.get("repetition_penalty", 1.0))
    pure = hybrid.generate_pure(
        prompt=prompt,
        max_new_tokens=int(infer_cfg["max_new_tokens"]),
        temperature=float(infer_cfg["temperature"]),
        top_k=int(infer_cfg["top_k"]),
        top_p=top_pf,
        repetition_penalty=rep,
    )
    print("PURE:\n", safe_console_text(pure))

    retriever = TfidfRetriever(retrieval_cfg)
    retriever.fit(load_chunks_json(ROOT / "data/index/chunks.json"))
    _hi = retrieval_cfg.get("hybrid_instruction")
    if isinstance(_hi, str) and not _hi.strip():
        _hi = None
    hybrid_with_retrieval = HybridGenerator(
        model=model,
        sp_model_path=str(ROOT / "artifacts/tokenizers/tokenizer.model"),
        retriever=retriever,
        device="cpu",
        hybrid_instruction=_hi,
    )
    out = hybrid_with_retrieval.generate_hybrid(
        prompt,
        retrieval_top_k=int(infer_cfg["retrieval_top_k"]),
        max_new_tokens=int(infer_cfg["max_new_tokens"]),
        temperature=float(infer_cfg["temperature"]),
        top_k=int(infer_cfg["top_k"]),
        top_p=top_pf,
        repetition_penalty=rep,
    )
    print("\nHYBRID CONTEXTS:\n", out["contexts"])
    print("\nHYBRID TEXT:\n", safe_console_text(out["generated_text"]))
