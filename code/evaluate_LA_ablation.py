import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from utils.test_3d_patch import test_all_case, test_all_case_average
from pancreas.Vnet import VNet
from networks.ResVNet import ResVNet


def create_vnet():
    net = VNet(n_channels=1, n_classes=2, normalization="instancenorm", has_dropout=True)
    return nn.DataParallel(net).cuda()


def create_resvnet():
    net = ResVNet(n_channels=1, n_classes=2, normalization="instancenorm", has_dropout=True)
    return nn.DataParallel(net).cuda()


def load_model(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    model.load_state_dict(state)
    model.eval()
    return model


def build_image_list(root_path):
    split_file = os.path.join(root_path, "test.txt")
    data_dir = os.path.join(root_path, "2018LA_Seg_Training Set")
    with open(split_file, "r", encoding="utf-8") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    return [os.path.join(data_dir, case_id, "mri_norm2.h5") for case_id in case_ids]


def write_metrics(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "model", "dice", "jaccard", "hd95", "asd"])
        writer.writeheader()
        writer.writerows(rows)


def metric_row(method, model_name, metric):
    metric = np.asarray(metric, dtype=np.float64)
    return {
        "method": method,
        "model": model_name,
        "dice": f"{metric[0]:.6f}",
        "jaccard": f"{metric[1]:.6f}",
        "hd95": f"{metric[2]:.6f}",
        "asd": f"{metric[3]:.6f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="code/Datasets/la/data_split")
    parser.add_argument("--method", required=True, help="Name written to the results CSV, e.g. SDCL+SAM+DAE")
    parser.add_argument("--snapshot", required=True, help="Path to self_train directory")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--detail", type=int, default=1)
    parser.add_argument("--nms", type=int, default=0)
    parser.add_argument("--save_result", type=int, default=0)
    parser.add_argument("--out_csv", default="results/la_ablation_metrics.csv")
    parser.add_argument("--pred_dir", default="", help="Optional prediction output directory")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    image_list = build_image_list(args.root_path)
    if not image_list:
        raise RuntimeError(f"No test cases found under {args.root_path}")

    vnet_ckpt = os.path.join(args.snapshot, "best_model.pth")
    resvnet_ckpt = os.path.join(args.snapshot, "best_model_res.pth")
    if not os.path.exists(resvnet_ckpt):
        alt = os.path.join(args.snapshot, "best_model_resnet.pth")
        if os.path.exists(alt):
            resvnet_ckpt = alt

    for ckpt in [vnet_ckpt, resvnet_ckpt]:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    pred_dir = args.pred_dir or os.path.join(args.snapshot, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    vnet_pred_dir = os.path.join(pred_dir, "vnet") + os.sep
    resvnet_pred_dir = os.path.join(pred_dir, "resvnet") + os.sep
    ensemble_pred_dir = os.path.join(pred_dir, "ensemble") + os.sep
    for path in [vnet_pred_dir, resvnet_pred_dir, ensemble_pred_dir]:
        os.makedirs(path, exist_ok=True)

    net1 = load_model(create_vnet(), vnet_ckpt)
    net2 = load_model(create_resvnet(), resvnet_ckpt)

    rows = []
    metric_vnet = test_all_case(
        net1,
        image_list,
        num_classes=2,
        patch_size=(112, 112, 80),
        stride_xy=18,
        stride_z=4,
        save_result=bool(args.save_result),
        test_save_path=vnet_pred_dir,
        metric_detail=args.detail,
        nms=args.nms,
    )
    rows.append(metric_row(args.method, "VNet", metric_vnet))

    metric_resvnet = test_all_case(
        net2,
        image_list,
        num_classes=2,
        patch_size=(112, 112, 80),
        stride_xy=18,
        stride_z=4,
        save_result=bool(args.save_result),
        test_save_path=resvnet_pred_dir,
        metric_detail=args.detail,
        nms=args.nms,
    )
    rows.append(metric_row(args.method, "ResVNet", metric_resvnet))

    metric_ensemble = test_all_case_average(
        net1,
        net2,
        image_list,
        num_classes=2,
        patch_size=(112, 112, 80),
        stride_xy=18,
        stride_z=4,
        save_result=bool(args.save_result),
        test_save_path=ensemble_pred_dir,
        metric_detail=args.detail,
        nms=args.nms,
    )
    rows.append(metric_row(args.method, "Ensemble", metric_ensemble))

    write_metrics(args.out_csv, rows)
    print(f"Saved metrics to: {args.out_csv}")


if __name__ == "__main__":
    main()
