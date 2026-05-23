import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = ["dice", "jaccard", "hd95", "asd"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with columns: method,dice,jaccard,hd95,asd")
    parser.add_argument("--out", default="figures/metric_bars.png")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "method" not in df.columns:
        raise ValueError("CSV must contain a 'method' column.")

    metrics = [m for m in args.metrics if m in df.columns]
    if not metrics:
        raise ValueError(f"No requested metrics found in {args.csv}.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.2), dpi=300)
    if len(metrics) == 1:
        axes = [axes]

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]
    for ax, metric in zip(axes, metrics):
        err_col = f"{metric}_std"
        yerr = df[err_col] if err_col in df.columns else None
        ax.bar(df["method"], df[metric], yerr=yerr, color=colors[: len(df)], capsize=3, width=0.65)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"Saved metric comparison figure to: {args.out}")


if __name__ == "__main__":
    main()
