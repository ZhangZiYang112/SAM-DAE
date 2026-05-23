import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="results/la_ablation/summary_metrics.csv")
    parser.add_argument("--ensemble_only", type=int, default=1)
    args = parser.parse_args()

    paths = []
    for item in args.inputs:
        matched = glob.glob(item)
        paths.extend(matched if matched else [item])
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No input CSV files found.")

    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    if args.ensemble_only and "model" in df.columns:
        df = df[df["model"] == "Ensemble"].copy()

    # plot_metric_bars expects one row per method and metric columns.
    keep_cols = [c for c in ["method", "dice", "jaccard", "hd95", "asd"] if c in df.columns]
    df = df[keep_cols]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved merged metric CSV to: {args.out}")


if __name__ == "__main__":
    main()
