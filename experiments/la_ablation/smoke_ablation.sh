#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
ROOT_PATH="${ROOT_PATH:-code/Datasets/la/data_split}"
PYTHON="${PYTHON:-python}"

run_smoke() {
  local exp_name="$1"
  local use_sam="$2"
  local use_dae="$3"

  echo "Smoke test ${exp_name}"
  "${PYTHON}" -B code/LA_train_smi_dae.py \
    --root_path "${ROOT_PATH}" \
    --exp "${exp_name}" \
    --gpu "${GPU}" \
    --skip_pretrain 1 \
    --pretrain_path code/model/SDCL/LA_SDCL_8_labeled/pre_train \
    --self_max_iteration 1 \
    --use_sam "${use_sam}" \
    --sam_skip_on_error 0 \
    --use_dae "${use_dae}" \
    --skip_dae_pretrain 1
}

run_smoke "SMOKE_SDCL_BASE" 0 0
run_smoke "SMOKE_SDCL_SAM" 1 0
run_smoke "SMOKE_SDCL_DAE" 0 1
run_smoke "SMOKE_SDCL_SAM_DAE" 1 1
