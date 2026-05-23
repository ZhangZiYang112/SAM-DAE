# SAM-DAE for LA Semi-supervised Segmentation

This repository is prepared for reproducing the Left Atrium (LA) semi-supervised experiment with:

```bash
code/LA_train_smi_dae.py
```

It includes the SDCL training pipeline, a SAM-Med3D consistency branch, and DAE-based pseudo-label refinement. Large datasets and checkpoints are intentionally not committed to GitHub.

## Environment

Tested locally with:

- Python 3.8.20
- PyTorch 2.4.1+cu118
- torchvision 0.19.1+cu118
- CUDA available

Server setup:

```bash
git clone git@github.com:ZhangZiYang112/SAM-DAE.git
cd SAM-DAE

conda create -n sdcl python=3.8 -y
conda activate sdcl

# Install torch/torchvision matching your server CUDA driver.
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements-la-smi-dae.txt
```

## Required Local Files

Place these files manually on the server. They are ignored by git because they are too large for a normal GitHub repository.

```text
code/Datasets/la/data_split/2018LA_Seg_Training Set/*/mri_norm2.h5
code/segment_anything/ckpt/sam_med3d_turbo.pth
```

Optional pretrained checkpoints for skipping the pre-training stage:

```text
code/model/SDCL/LA_SDCL_8_labeled/pre_train/best_model.pth
code/model/SDCL/LA_SDCL_8_labeled/pre_train/best_model_resnet.pth
```

Use Git LFS, scp, rsync, Baidu Netdisk, or another storage service for these files.

## Quick Smoke Test

After placing the LA data and SAM checkpoint:

```bash
python -B code/LA_train_smi_dae.py \
  --skip_pretrain 1 \
  --use_sam 1 \
  --sam_skip_on_error 0 \
  --use_dae 0 \
  --self_max_iteration 1 \
  --gpu 0
```

Expected signs:

- `SAM-Med3D model loaded from code/segment_anything/ckpt/sam_med3d_turbo.pth`
- `epoch 1 iteration 1`
- `sam_loss` is printed and is not forced to zero

This local smoke test completed one self-training iteration and reported a non-zero `sam_loss`.

## Normal Training Configuration

Full training with VNet, ResVNet, SAM-Med3D, and DAE:

```bash
python -B code/LA_train_smi_dae.py \
  --root_path code/Datasets/la/data_split \
  --exp SDCL \
  --labelnum 8 \
  --gpu 0 \
  --pre_max_iteration 2000 \
  --self_max_iteration 15000 \
  --use_sam 1 \
  --sam_prompt unc \
  --sam_weight 0.1 \
  --sam_skip_on_error 0 \
  --use_dae 1 \
  --dae_pretrain_epochs 100 \
  --dae_weight 0.5
```

Train self-training only with existing SDCL pretrain checkpoints:

```bash
python -B code/LA_train_smi_dae.py \
  --root_path code/Datasets/la/data_split \
  --exp SDCL \
  --labelnum 8 \
  --gpu 0 \
  --skip_pretrain 1 \
  --use_sam 1 \
  --sam_prompt unc \
  --sam_weight 0.1 \
  --sam_skip_on_error 0 \
  --use_dae 0
```

Useful ablation settings:

```bash
# SDCL baseline without SAM and DAE
python -B code/LA_train_smi_dae.py --use_sam 0 --use_dae 0 --gpu 0

# SAM branch only
python -B code/LA_train_smi_dae.py --use_sam 1 --use_dae 0 --sam_weight 0.1 --gpu 0

# DAE branch only
python -B code/LA_train_smi_dae.py --use_sam 0 --use_dae 1 --dae_weight 0.5 --gpu 0

# SAM + DAE
python -B code/LA_train_smi_dae.py --use_sam 1 --use_dae 1 --sam_weight 0.1 --dae_weight 0.5 --gpu 0
```

## Outputs

Default outputs are written under:

```text
code/model/SDCL/LA_SDCL_8_labeled/
```

Important files:

```text
pre_train/log.txt
pre_train/best_model.pth
pre_train/best_model_resnet.pth
self_train/log.txt
self_train/best_model.pth
self_train/best_model_res.pth
dae_pretrain/best_dae_model.pth
```

## Paper Visualization

The scripts in `code/visualization/` generate common figures for semi-supervised segmentation papers. Use them only with real experimental logs, metrics, and predictions.

### 1. Training curves

```bash
python -B code/visualization/plot_training_curves.py \
  --log code/model/SDCL/LA_SDCL_8_labeled/self_train/log.txt \
  --out_dir figures/training_curves
```

Generated figures:

- `loss_curves.png`
- `sam_dae_curves.png`
- `dice_checkpoints.png`, if validation checkpoints are found in the log

### 2. Quantitative comparison bars

Create a CSV such as:

```csv
method,dice,jaccard,hd95,asd
UA-MT,87.79,78.39,13.25,3.41
SASSNet,89.27,80.82,9.45,2.83
SDCL,91.23,84.01,6.74,1.92
SAM-DAE,92.10,85.35,5.91,1.64
```

Then run:

```bash
python -B code/visualization/plot_metric_bars.py \
  --csv results/la_metrics.csv \
  --out figures/metric_bars.png
```

### 3. Qualitative LA prediction figure

```bash
python -B code/visualization/visualize_la_case.py \
  --case code/Datasets/la/data_split/2018LA_Seg_Training Set/06SR5RBREL16DQ6M8LWS/mri_norm2.h5 \
  --vnet_ckpt code/model/SDCL/LA_SDCL_8_labeled/self_train/best_model.pth \
  --resvnet_ckpt code/model/SDCL/LA_SDCL_8_labeled/self_train/best_model_res.pth \
  --out figures/la_case_06.png \
  --gpu 0
```

Generated panel:

- image with ground-truth contour
- image with predicted contour
- false-positive / false-negative error map
- model uncertainty map

## Original SDCL Paper

This code is based on SDCL: Students Discrepancy-Informed Correction Learning for Semi-supervised Medical Image Segmentation.

```bibtex
@inproceedings{song2024sdcl,
  title={SDCL: Students Discrepancy-Informed Correction Learning for Semi-supervised Medical Image Segmentation},
  author={Song, Bentao and Wang, Qingfeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={567--577},
  year={2024},
  organization={Springer}
}
```
