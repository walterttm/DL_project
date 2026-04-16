"""
读 logs/*.csv 画对比图，输出到 figures/
"""
import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt

LOG_DIR = "./logs"
FIG_DIR = "./figures"
EXPS = {
    "A1": ("ResNeXt50-Scratch",  "tab:red",   "--"),
    "A2": ("ResNeXt50-Finetune", "tab:red",   "-"),
    "B1": ("DenseNet121-Scratch","tab:blue",  "--"),
    "B2": ("DenseNet121-Finetune","tab:blue", "-"),
}


def load_csv(path):
    rows = {"epoch": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "lr": [], "time_s": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in rows:
                rows[k].append(float(r[k]))
    return rows


def main():
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    data = {}
    for exp in EXPS:
        path = os.path.join(LOG_DIR, f"{exp}.csv")
        if os.path.exists(path):
            data[exp] = load_csv(path)
        else:
            print(f"missing {path}, skip")

    # 1) val acc 对比
    plt.figure(figsize=(8, 5))
    for exp, rows in data.items():
        name, color, ls = EXPS[exp]
        plt.plot(rows["epoch"], rows["val_acc"], label=name, color=color, linestyle=ls, lw=2)
    plt.xlabel("Epoch"); plt.ylabel("Val Top-1 Acc")
    plt.title("Validation Accuracy: Scratch vs Finetune")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "val_acc.png"), dpi=150)
    plt.close()

    # 2) val loss 对比
    plt.figure(figsize=(8, 5))
    for exp, rows in data.items():
        name, color, ls = EXPS[exp]
        plt.plot(rows["epoch"], rows["val_loss"], label=name, color=color, linestyle=ls, lw=2)
    plt.xlabel("Epoch"); plt.ylabel("Val Loss")
    plt.title("Validation Loss: Scratch vs Finetune")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "val_loss.png"), dpi=150)
    plt.close()

    # 3) train vs val acc 过拟合分析(每个实验一个子图)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (exp, rows) in zip(axes.flatten(), data.items()):
        name = EXPS[exp][0]
        ax.plot(rows["epoch"], rows["train_acc"], label="train", lw=2)
        ax.plot(rows["epoch"], rows["val_acc"],   label="val",   lw=2)
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("Acc")
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "overfit.png"), dpi=150)
    plt.close()

    # 4) 汇总表
    print("\n===== Summary =====")
    print(f"{'Exp':<6}{'Name':<24}{'Best ValAcc':<14}{'Final ValAcc':<14}{'Total min':<10}")
    for exp, rows in data.items():
        name = EXPS[exp][0]
        best = max(rows["val_acc"])
        final = rows["val_acc"][-1]
        total_min = sum(rows["time_s"]) / 60
        print(f"{exp:<6}{name:<24}{best:<14.4f}{final:<14.4f}{total_min:<10.1f}")

    print(f"\nfigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
