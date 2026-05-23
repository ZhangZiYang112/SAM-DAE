import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ITER_RE = re.compile(
    r"(?:epoch\s+(?P<epoch>\d+)\s+)?iteration\s+(?P<iter>\d+)\s+:\s+"
    r"loss:\s+(?P<loss>[-+0-9.eE]+).*?"
    r"loss_l:\s+(?P<loss_l>[-+0-9.eE]+).*?"
    r"loss_u:\s+(?P<loss_u>[-+0-9.eE]+).*?"
    r"sam_loss:\s+(?P<sam_loss>[-+0-9.eE]+).*?"
    r"dae_loss:\s+(?P<dae_loss>[-+0-9.eE]+)"
)
PRETRAIN_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)\s+:\s+loss:\s+(?P<loss>[-+0-9.eE]+),\s+"
    r"loss_dice:\s+(?P<loss_dice>[-+0-9.eE]+),\s+loss_ce:\s+(?P<loss_ce>[-+0-9.eE]+)"
)
DICE_RE = re.compile(r"iter_(?P<iter>\d+)_dice_(?P<dice>[-+0-9.]+)")


def parse_log(path):
    train_rows = []
    pretrain_rows = []
    dice_rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                train_rows.append({k: float(v) if k != "epoch" and k != "iter" else int(v)
                                   for k, v in m.groupdict(default="0").items()})
                continue
            m = PRETRAIN_RE.search(line)
            if m:
                pretrain_rows.append({k: float(v) if k != "iter" else int(v)
                                      for k, v in m.groupdict().items()})
            for m in DICE_RE.finditer(line):
                dice_rows.append({"iter": int(m.group("iter")), "dice": float(m.group("dice"))})
    return pd.DataFrame(train_rows), pd.DataFrame(pretrain_rows), pd.DataFrame(dice_rows)


def save_line_plot(df, x, ys, ylabel, title, out_path):
    if df.empty:
        return False
    plt.figure(figsize=(7.0, 4.2), dpi=300)
    for y in ys:
        if y in df.columns:
            plt.plot(df[x], df[y], linewidth=1.8, label=y)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to training log.txt")
    parser.add_argument("--out_dir", default="figures/training_curves")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_df, pretrain_df, dice_df = parse_log(args.log)

    if not train_df.empty:
        train_df.to_csv(os.path.join(args.out_dir, "parsed_self_train_log.csv"), index=False)
        save_line_plot(
            train_df,
            "iter",
            ["loss", "loss_l", "loss_u"],
            "Loss",
            "Self-training Loss Curves",
            os.path.join(args.out_dir, "loss_curves.png"),
        )
        save_line_plot(
            train_df,
            "iter",
            ["sam_loss", "dae_loss"],
            "Loss",
            "SAM and DAE Consistency Loss",
            os.path.join(args.out_dir, "sam_dae_curves.png"),
        )

    if not pretrain_df.empty:
        pretrain_df.to_csv(os.path.join(args.out_dir, "parsed_pretrain_log.csv"), index=False)
        save_line_plot(
            pretrain_df,
            "iter",
            ["loss", "loss_dice", "loss_ce"],
            "Loss",
            "Pre-training Loss Curves",
            os.path.join(args.out_dir, "pretrain_loss_curves.png"),
        )

    if not dice_df.empty:
        dice_df = dice_df.drop_duplicates().sort_values("iter")
        dice_df.to_csv(os.path.join(args.out_dir, "parsed_dice_checkpoints.csv"), index=False)
        save_line_plot(
            dice_df,
            "iter",
            ["dice"],
            "Dice",
            "Validation Dice Checkpoints",
            os.path.join(args.out_dir, "dice_checkpoints.png"),
        )

    print(f"Saved figures and parsed CSV files to: {args.out_dir}")


if __name__ == "__main__":
    main()
