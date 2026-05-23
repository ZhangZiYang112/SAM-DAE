# LA semi-supervised training with SAM-Med3D

This project was smoke-tested from the repository root with:

```bash
python -B code/LA_train_smi_dae.py --skip_pretrain 1 --use_sam 1 --sam_skip_on_error 0 --use_dae 0 --self_max_iteration 1 --gpu 0
```

The smoke test loaded `code/segment_anything/ckpt/sam_med3d_turbo.pth`, completed one self-training iteration, and reported a non-zero `sam_loss`.

## Files to place manually on the server

These files are intentionally ignored by git because they are large:

- LA data: `code/Datasets/la/data_split/2018LA_Seg_Training Set/*/mri_norm2.h5`
- SAM checkpoint: `code/segment_anything/ckpt/sam_med3d_turbo.pth`
- Optional SDCL pretrain checkpoints:
  - `code/model/SDCL/LA_SDCL_8_labeled/pre_train/best_model.pth`
  - `code/model/SDCL/LA_SDCL_8_labeled/pre_train/best_model_resnet.pth`

Use Git LFS, scp, rsync, or your storage service for these large files.

## Server setup

```bash
conda create -n sdcl python=3.8 -y
conda activate sdcl

# Install torch/torchvision matching your CUDA driver first.
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements-la-smi-dae.txt
```

## Quick verification

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

## Full run

Train pretrain + self-training:

```bash
python -B code/LA_train_smi_dae.py --use_sam 1 --use_dae 1 --gpu 0
```

Use existing pretrain checkpoints and train self-training only:

```bash
python -B code/LA_train_smi_dae.py --skip_pretrain 1 --use_sam 1 --use_dae 0 --gpu 0
```
