#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
ROOT_PATH="${ROOT_PATH:-code/Datasets/la/data_split}"
LABELNUM="${LABELNUM:-8}"
PYTHON="${PYTHON:-python}"

mkdir -p results/la_ablation

evaluate_one() {
  local method="$1"
  local exp_name="$2"
  local out_csv="results/la_ablation/${exp_name}_metrics.csv"

  "${PYTHON}" -B code/evaluate_LA_ablation.py \
    --root_path "${ROOT_PATH}" \
    --method "${method}" \
    --snapshot "code/model/SDCL/LA_${exp_name}_${LABELNUM}_labeled/self_train" \
    --gpu "${GPU}" \
    --detail 1 \
    --nms 0 \
    --save_result 0 \
    --out_csv "${out_csv}"
}

evaluate_one "SDCL" "SDCL_BASE"
evaluate_one "SDCL+SAM" "SDCL_SAM"
evaluate_one "SDCL+DAE" "SDCL_DAE"
evaluate_one "SDCL+SAM+DAE" "SDCL_SAM_DAE"

"${PYTHON}" -B code/visualization/merge_metric_csvs.py \
  --inputs results/la_ablation/*_metrics.csv \
  --out results/la_ablation/summary_metrics.csv \
  --ensemble_only 1

"${PYTHON}" -B code/visualization/plot_metric_bars.py \
  --csv results/la_ablation/summary_metrics.csv \
  --out figures/la_ablation_metric_bars.png
