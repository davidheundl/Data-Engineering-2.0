#!/usr/bin/env python3
"""Fine-tune BERT / RoBERTa to predict L1 discourse-sense distributions.

KL-divergence loss between predicted softmax over 4 senses and the target
distribution stored in `label_distribution` of each training example.
Saves model weights, tokenizer, and config to --out-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SENSES = ["temporal", "contingency", "comparison", "expansion"]


class SenseDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        r = self.examples[idx]
        enc = self.tokenizer(
            r["arg1"], r["arg2"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = torch.tensor([r["label_distribution"][s] for s in SENSES], dtype=torch.float32)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_distribution": label,
        }


class SenseClassifier(torch.nn.Module):
    def __init__(self, backbone_name: str, n_classes: int = len(SENSES), dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))


def kl_loss(logits: torch.Tensor, target_dist: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """KL(target || pred)."""
    pred_log = F.log_softmax(logits, dim=-1)
    target = target_dist.clamp(min=eps)
    target = target / target.sum(dim=-1, keepdim=True)
    return F.kl_div(pred_log, target, reduction="batchmean")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--backbone", default="bert-base-uncased")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-frac", type=float, default=0.10)
    parser.add_argument("--init-from", default=None,
                        help="Optional path to a checkpoint (model.pt) whose weights "
                             "are loaded before training. Backbone must match.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}  backbone: {args.backbone}  train: {args.train_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    train_ds = SenseDataset(args.train_jsonl, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = SenseClassifier(args.backbone).to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded checkpoint from {args.init_from}")
        if missing:
            print(f"    missing keys: {len(missing)}")
        if unexpected:
            print(f"    unexpected keys: {len(unexpected)}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_frac * total_steps),
        num_training_steps=total_steps,
    )

    losses = []
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label_distribution"].to(device)
            logits = model(input_ids, attention_mask)
            loss = kl_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n_batches += 1
        avg = running / max(1, n_batches)
        losses.append(avg)
        print(f"  Epoch {epoch+1}/{args.epochs}  KL loss={avg:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    tokenizer.save_pretrained(out_dir)
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "seed": args.seed,
            "warmup_frac": args.warmup_frac,
            "train_jsonl": args.train_jsonl,
            "n_train_examples": len(train_ds),
            "device_used": device,
            "loss_history": losses,
            "init_from": args.init_from,
        }, f, indent=2)
    print(f"Saved model + config to {out_dir}")


if __name__ == "__main__":
    main()
