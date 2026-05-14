import torch

from src.models.transformer_lm import MiniTransformerLM, TransformerConfig


def test_transformer_lm_output_shape():
    cfg = TransformerConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=2, ff_dim=64, max_seq_len=16)
    model = MiniTransformerLM(cfg)
    x = torch.randint(0, 100, (2, 16))
    logits, _ = model(x)
    assert logits.shape == (2, 16, 100)


def test_generate_sampling_runs():
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=50, d_model=32, n_heads=4, n_layers=2, ff_dim=64, max_seq_len=16)
    model = MiniTransformerLM(cfg)
    idx = torch.randint(0, 50, (1, 6))
    out = model.generate(
        idx,
        max_new_tokens=4,
        temperature=1.0,
        top_k=10,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    assert out.shape[1] == 10
