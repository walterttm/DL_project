"""
从头训练 vs 微调  对比实验
用法:
    python train.py --exp A1   # ResNeXt50  scratch
    python train.py --exp A2   # ResNeXt50  finetune
    python train.py --exp B1   # DenseNet121 scratch
    python train.py --exp B2   # DenseNet121 finetune
"""
import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import build_dataloaders
from models import build_model

# ----------------------------- 实验配置 -----------------------------
EXPERIMENTS = {
    "A1": dict(arch="resnext50", pretrained=False, lr=1e-2, name="ResNeXt50-Scratch"),
    "A2": dict(arch="resnext50", pretrained=True,  lr=1e-3, name="ResNeXt50-Finetune"),
    "B1": dict(arch="densenet121", pretrained=False, lr=1e-2, name="DenseNet121-Scratch"),
    "B2": dict(arch="densenet121", pretrained=True,  lr=1e-3, name="DenseNet121-Finetune"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, required=True, choices=list(EXPERIMENTS.keys()))
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=102)  # Flowers102
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = EXPERIMENTS[args.exp]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.exp}] {cfg['name']} | device={device}")

    # 数据
    train_loader, val_loader = build_dataloaders(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # 模型
    model = build_model(cfg["arch"], num_classes=args.num_classes,
                        pretrained=cfg["pretrained"]).to(device)

    # 损失 & 优化器
    criterion = nn.CrossEntropyLoss()
    # 微调时给新分类头更大学习率（差分学习率）
    if cfg["pretrained"]:
        head_params, backbone_params = [], []
        for n, p in model.named_parameters():
            if "fc" in n or "classifier" in n:
                head_params.append(p)
            else:
                backbone_params.append(p)
        optimizer = torch.optim.SGD(
            [{"params": backbone_params, "lr": cfg["lr"]},
             {"params": head_params,     "lr": cfg["lr"] * 10}],
            momentum=0.9, weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                    momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 日志
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.exp}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"])

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ts = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        dt = time.time() - ts
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[{args.exp}] ep {epoch:02d}/{args.epochs} "
              f"| tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} "
              f"| va_loss {va_loss:.4f} va_acc {va_acc:.4f} "
              f"| lr {cur_lr:.5f} | {dt:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_acc, va_loss, va_acc, cur_lr, dt])

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, f"{args.exp}_best.pt"))

    total_min = (time.time() - t0) / 60
    print(f"[{args.exp}] DONE | best_val_acc={best_acc:.4f} | total {total_min:.1f} min")


if __name__ == "__main__":
    main()
