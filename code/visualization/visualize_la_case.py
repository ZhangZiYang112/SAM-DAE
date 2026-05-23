import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Compatibility for legacy code that still uses np.int.
np.int = int  # type: ignore[attr-defined]

from pancreas.Vnet import VNet
from networks.ResVNet import ResVNet
from utils.test_3d_patch import test_single_case, test_single_case_mean


def create_vnet():
    model = nn.DataParallel(VNet(n_channels=1, n_classes=2, normalization="instancenorm", has_dropout=True))
    return model.cuda()


def create_resvnet():
    model = nn.DataParallel(ResVNet(n_channels=1, n_classes=2, normalization="instancenorm", has_dropout=True))
    return model.cuda()


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_slice(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def pick_slice(label, requested=None):
    if requested is not None:
        return int(requested)
    areas = label.sum(axis=(0, 1))
    if areas.max() > 0:
        return int(np.argmax(areas))
    return label.shape[2] // 2


def overlay_contour(ax, mask, color, linewidth=1.4):
    ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=linewidth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="Path to an LA mri_norm2.h5 case")
    parser.add_argument("--vnet_ckpt", required=True)
    parser.add_argument("--resvnet_ckpt", default="", help="Optional ResVNet checkpoint for ensemble prediction")
    parser.add_argument("--out", default="figures/la_case.png")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--stride_xy", type=int, default=18)
    parser.add_argument("--stride_z", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with h5py.File(args.case, "r") as f:
        image = f["image"][:]
        label = f["label"][:].astype(np.uint8)

    vnet = load_checkpoint(create_vnet(), args.vnet_ckpt)
    if args.resvnet_ckpt:
        resvnet = load_checkpoint(create_resvnet(), args.resvnet_ckpt)
        pred, score_map = test_single_case_mean(
            vnet,
            resvnet,
            image,
            stride_xy=args.stride_xy,
            stride_z=args.stride_z,
            patch_size=(112, 112, 80),
            num_classes=2,
        )
    else:
        pred, score_map = test_single_case(
            vnet,
            image,
            stride_xy=args.stride_xy,
            stride_z=args.stride_z,
            patch_size=(112, 112, 80),
            num_classes=2,
        )

    prob = np.squeeze(score_map[0]).astype(np.float32)
    pred = (prob > args.threshold).astype(np.uint8)
    z = pick_slice(label, args.slice)

    img_s = normalize_slice(image[:, :, z])
    gt_s = label[:, :, z]
    pred_s = pred[:, :, z]
    prob_s = prob[:, :, z]
    fp = np.logical_and(pred_s == 1, gt_s == 0)
    fn = np.logical_and(pred_s == 0, gt_s == 1)
    entropy = -(prob_s * np.log(prob_s + 1e-8) + (1 - prob_s) * np.log(1 - prob_s + 1e-8))
    entropy = entropy / np.log(2.0)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=300)
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(img_s.T, cmap="gray", origin="lower")
    overlay_contour(axes[0], gt_s.T, "#00E5FF")
    axes[0].set_title("Image + GT")

    axes[1].imshow(img_s.T, cmap="gray", origin="lower")
    overlay_contour(axes[1], pred_s.T, "#FFB000")
    axes[1].set_title("Image + Prediction")

    error_rgb = np.zeros((*img_s.T.shape, 3), dtype=np.float32)
    base = img_s.T[..., None]
    error_rgb += base * 0.65
    error_rgb[fp.T] = [1.0, 0.15, 0.12]
    error_rgb[fn.T] = [0.1, 0.45, 1.0]
    axes[2].imshow(error_rgb, origin="lower")
    axes[2].set_title("Errors: FP red, FN blue")

    im = axes[3].imshow(entropy.T, cmap="magma", origin="lower", vmin=0, vmax=1)
    overlay_contour(axes[3], pred_s.T, "white", linewidth=0.9)
    axes[3].set_title("Uncertainty")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Case: {os.path.basename(os.path.dirname(args.case))}, slice={z}", y=0.98)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qualitative figure to: {args.out}")


if __name__ == "__main__":
    main()
