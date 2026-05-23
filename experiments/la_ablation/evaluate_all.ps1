param(
    [string]$Gpu = "0",
    [string]$RootPath = "code/Datasets/la/data_split",
    [int]$LabelNum = 8,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "results/la_ablation" | Out-Null

function Evaluate-One {
    param(
        [string]$Method,
        [string]$ExpName
    )

    & $Python -B code/evaluate_LA_ablation.py `
        --root_path $RootPath `
        --method $Method `
        --snapshot "code/model/SDCL/LA_${ExpName}_${LabelNum}_labeled/self_train" `
        --gpu $Gpu `
        --detail 1 `
        --nms 0 `
        --save_result 0 `
        --out_csv "results/la_ablation/${ExpName}_metrics.csv"
}

Evaluate-One -Method "SDCL" -ExpName "SDCL_BASE"
Evaluate-One -Method "SDCL+SAM" -ExpName "SDCL_SAM"
Evaluate-One -Method "SDCL+DAE" -ExpName "SDCL_DAE"
Evaluate-One -Method "SDCL+SAM+DAE" -ExpName "SDCL_SAM_DAE"

& $Python -B code/visualization/merge_metric_csvs.py `
    --inputs results/la_ablation/*_metrics.csv `
    --out results/la_ablation/summary_metrics.csv `
    --ensemble_only 1

& $Python -B code/visualization/plot_metric_bars.py `
    --csv results/la_ablation/summary_metrics.csv `
    --out figures/la_ablation_metric_bars.png
