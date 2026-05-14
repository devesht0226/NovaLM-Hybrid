from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


def train_sentencepiece(input_path: str, output_prefix: str, vocab_size: int = 16000) -> str:
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    return f"{output_prefix}.model"
