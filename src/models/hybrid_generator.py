from typing import List, Optional

import sentencepiece as spm
import torch

from src.models.retriever import TfidfRetriever
from src.models.transformer_lm import MiniTransformerLM

_DEFAULT_HYBRID_INSTRUCTION = (
    "Use the numbered excerpts only if they clearly help answer the question.\n"
    "If they are unrelated or off-topic, answer without relying on them and do not invent "
    "current news, dates, or facts that are not supported by the excerpts.\n\n"
    "Excerpts:\n{excerpts}\n\nQuestion: {prompt}\nAnswer:"
)


class HybridGenerator:
    def __init__(
        self,
        model: MiniTransformerLM,
        sp_model_path: str,
        retriever: Optional[TfidfRetriever] = None,
        device: str = "cpu",
        hybrid_instruction: str | None = None,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.retriever = retriever
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.hybrid_instruction = hybrid_instruction

    def _encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def _decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def _clean_text(self, text: str) -> str:
        # SentencePiece unknown markers can look noisy in UI; trim lightly.
        return text.replace("⁇", "").strip()

    def generate_pure(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        input_ids = self._encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        full_ids = out[0].tolist()
        new_ids = full_ids[len(input_ids) :]
        if not new_ids:
            return ""
        return self._clean_text(self._decode(new_ids))

    def generate_hybrid(
        self,
        prompt: str,
        retrieval_top_k: int = 3,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ):
        contexts = []
        if self.retriever is not None:
            contexts = self.retriever.top_k(prompt, k=retrieval_top_k)

        if contexts:
            lines = "\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(contexts))
            tmpl = (self.hybrid_instruction or _DEFAULT_HYBRID_INSTRUCTION).strip()
            try:
                hybrid_prompt = tmpl.format(excerpts=lines, prompt=prompt)
            except (KeyError, ValueError):
                hybrid_prompt = _DEFAULT_HYBRID_INSTRUCTION.format(excerpts=lines, prompt=prompt)
        else:
            hybrid_prompt = prompt

        text = self.generate_pure(
            hybrid_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return {
            "generated_text": text,
            "contexts": [{"text": c.text, "score": c.score} for c in contexts],
        }
