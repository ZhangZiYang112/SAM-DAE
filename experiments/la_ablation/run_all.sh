#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
ROOT_PATH="${ROOT_PATH:-code/Datasets/la/data_split}"
LABELNUM="${LABELNUM:-8}"
PRE_MAX_ITERATION="${PRE_MAX_ITERATION:-2000}"
SELF_MAX_ITERATION="${SELF_MAX_ITERATION:-15000}"
DAE_PRETRAIN_EPOCHS="${DAE_PRETRAIN_EPOCHS:-100}"
SAM_WEIGHT="${SAM_WEIGHT:-0.1}"
DAE_WEIGHT="${DAE_WEIGHT:-0.5}"

PYTHON="${PYTHON:-python}"

echo "Running LA ablation experiments on GPU=${GPU}"
echo "ROOT_PATH=${ROOT_PATH}"

run_train() {
  local exp_name="$1"
  local use_sam="$2"
  local use_dae="$3"
  local extra_args=("${@:4}")

  echo "============================================================"
  echo "Training ${exp_name}: use_sam=${use_sam}, use_dae=${use_dae}"
  echo "============================================================"

  "${PYTHON}" -B code/LA_train_smi_dae.py \
    --root_path "${ROOT_PATH}" \
    --exp "${exp_name}" \
    --labelnum "${LABELNUM}" \
    --gpu "${GPU}" \
    --pre_max_iteration "${PRE_MAX_ITERATION}" \
    --self_max_iteration "${SELF_MAX_ITERATION}" \
    --use_sam "${use_sam}" \
    --sam_prompt unc \
    --sam_weight "${SAM_WEIGHT}" \
    --sam_skip_on_error 0 \
    --use_dae "${use_dae}" \
    --dae_pretrain_epochs "${DAE_PRETRAIN_EPOCHS}" \
    --dae_weight "${DAE_WEIGHT}" \
    "${extra_args[@]}"
}

# 1) Baseline SDCL. This run also produces the shared pretrain weights.
run_train "SDCL_BASE" 0 0

SHARED_PRETRAIN="code/model/SDCL/LA_SDCL_BASE_${LABELNUM}_labeled/pre_train"

# 2) Fair ablations reuse the same SDCL pretrain weights and only change self-training losses.
run_train "SDCL_SAM" 1 0 --skip_pretrain 1 --pretrain_path "${SHARED_PRETRAIN}"
run_train "SDCL_DAE" 0 1 --skip_pretrain 1 --pretrain_path "${SHARED_PRETRAIN}"
run_train "SDCL_SAM_DAE" 1 1 --skip_pretrain 1 --pretrain_path "${SHARED_PRETRAIN}"

echo "All LA ablation training jobs finished."
