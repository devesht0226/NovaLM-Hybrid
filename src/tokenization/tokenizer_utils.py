from __future__ import annotations

import sentencepiece as spm


class TokenizerWrapper:
    def __init__(self, model_path: str) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()
