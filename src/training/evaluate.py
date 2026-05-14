from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def evaluate_loss(model: nn.Module, data: torch.Tensor, seq_len: int, device: str = "cpu") -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    count = 0
    for i in range(0, data.numel() - seq_len - 1, seq_len):
        x = data[i : i + seq_len].unsqueeze(0).to(device)
        y = data[i + 1 : i + seq_len + 1].unsqueeze(0).to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
        count += 1
    return total / max(1, count)
