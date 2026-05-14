import argparse
import json
import math
import os
from dataclasses import fields
from pathlib import Path

import sentencepiece as spm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from src.models.transformer_lm import MiniTransformerLM, TransformerConfig
from src.utils.config import ROOT, load_all_configs


class TokenDataset(Dataset):
    def __init__(self, token_ids, block_size=256):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.token_ids) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_token_ids(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["train_ids"], obj["val_ids"]


def _resolve_vocab_size(cli_vocab: int | None, sp_path: Path | None, fallback: int) -> int:
    if cli_vocab is not None:
        return int(cli_vocab)
    if sp_path is not None and sp_path.exists():
        return int(spm.SentencePieceProcessor(model_file=str(sp_path)).vocab_size())
    return int(fallback)


def _optimizer_steps_per_epoch(num_batches: int, accum_steps: int) -> int:
    return max(1, (num_batches + accum_steps - 1) // accum_steps)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    if not losses:
        return float("inf"), float("inf")
    avg = sum(losses) / len(losses)
    ppl = math.exp(min(avg, 20))
    return avg, ppl


def _transformer_config_from_yaml(model_cfg: dict, vocab_size: int) -> TransformerConfig:
    allowed = {f.name for f in fields(TransformerConfig)}
    merged = {k: v for k, v in model_cfg.items() if k in allowed}
    merged["vocab_size"] = vocab_size
    return TransformerConfig(**merged)


def main():
    bundles = load_all_configs()
    model_yaml = bundles["model"]
    train_yaml = bundles["train"]

    parser = argparse.ArgumentParser(description="Train MiniTransformerLM from token_ids JSON.")
    parser.add_argument("--tokens_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="artifacts/checkpoints")
    parser.add_argument("--sp_model", type=str, default=None, help="SentencePiece model for vocab_size when --vocab_size omitted.")
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sp_default = ROOT / "artifacts/tokenizers/tokenizer.model"
    sp_path = Path(args.sp_model) if args.sp_model else sp_default
    if not sp_path.is_absolute():
        sp_path = ROOT / sp_path

    batch_size = args.batch_size if args.batch_size is not None else int(train_yaml["batch_size"])
    epochs = args.epochs if args.epochs is not None else int(train_yaml["epochs"])
    lr = args.lr if args.lr is not None else float(train_yaml["lr"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(train_yaml["weight_decay"])
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else int(model_yaml["max_seq_len"])
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(train_yaml.get("warmup_steps", 0))
    grad_clip = args.grad_clip if args.grad_clip is not None else float(train_yaml.get("grad_clip", 1.0))
    accum_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else int(train_yaml.get("gradient_accumulation_steps", 1))
    )
    accum_steps = max(1, accum_steps)
    patience = args.early_stopping_patience if args.early_stopping_patience is not None else int(train_yaml["early_stopping_patience"])
    min_delta = (
        args.early_stopping_min_delta
        if args.early_stopping_min_delta is not None
        else float(train_yaml.get("early_stopping_min_delta", 0.0))
    )

    vocab_size = _resolve_vocab_size(args.vocab_size, sp_path, model_yaml["vocab_size"])
    cfg = _transformer_config_from_yaml({**model_yaml, "max_seq_len": max_seq_len}, vocab_size)

    tokens_path = Path(args.tokens_json)
    if not tokens_path.is_absolute():
        tokens_path = ROOT / tokens_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    train_ids, val_ids = load_token_ids(str(tokens_path))
    train_ds = TokenDataset(train_ids, block_size=cfg.max_seq_len)
    val_ds = TokenDataset(val_ids, block_size=cfg.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    if len(train_loader) == 0:
        raise SystemExit(
            "No training batches (dataset too small for batch_size and block_size). "
            "Add more text, lower configs/train.yaml batch_size, or reduce configs/model.yaml max_seq_len."
        )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MiniTransformerLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    steps_per_epoch = _optimizer_steps_per_epoch(len(train_loader), accum_steps)
    total_steps = max(1, steps_per_epoch * epochs)

    warmup_eff = min(max(0, warmup_steps), total_steps)
    if warmup_eff > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_eff)
        cosine_steps = max(1, total_steps - warmup_eff)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_eff],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        optimizer.zero_grad(set_to_none=True)

        def _optimizer_step() -> None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _, batch_loss = model(x, y)
                loss = batch_loss / accum_steps

            scaler.scale(loss).backward()
            running += batch_loss.item()

            if batch_idx % accum_steps == 0:
                _optimizer_step()

        if len(train_loader) % accum_steps != 0:
            _optimizer_step()

        train_loss = running / max(1, len(train_loader))
        val_loss, val_ppl = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
        )

        ckpt_path = out_dir / f"epoch_{epoch}.pt"
        torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, ckpt_path)

        if val_loss + 1e-12 < best_val - min_delta:
            best_val = val_loss
            bad_epochs = 0
            best_path = out_dir / "best.pt"
            torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
