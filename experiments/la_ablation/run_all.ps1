param(
    [string]$Gpu = "0",
    [string]$RootPath = "code/Datasets/la/data_split",
    [int]$LabelNum = 8,
    [int]$PreMaxIteration = 2000,
    [int]$SelfMaxIteration = 15000,
    [int]$DaePretrainEpochs = 100,
    [double]$SamWeight = 0.1,
    [double]$DaeWeight = 0.5,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Run-Train {
    param(
        [string]$ExpName,
        [int]$UseSam,
        [int]$UseDae,
        [string[]]$ExtraArgs = @()
    )

    Write-Host "============================================================"
    Write-Host "Training $ExpName : use_sam=$UseSam, use_dae=$UseDae"
    Write-Host "============================================================"

    & $Python -B code/LA_train_smi_dae.py `
        --root_path $RootPath `
        --exp $ExpName `
        --labelnum $LabelNum `
        --gpu $Gpu `
        --pre_max_iteration $PreMaxIteration `
        --self_max_iteration $SelfMaxIteration `
        --use_sam $UseSam `
        --sam_prompt unc `
        --sam_weight $SamWeight `
        --sam_skip_on_error 0 `
        --use_dae $UseDae `
        --dae_pretrain_epochs $DaePretrainEpochs `
        --dae_weight $DaeWeight `
        @ExtraArgs
}

Run-Train -ExpName "SDCL_BASE" -UseSam 0 -UseDae 0

$SharedPretrain = "code/model/SDCL/LA_SDCL_BASE_${LabelNum}_labeled/pre_train"

Run-Train -ExpName "SDCL_SAM" -UseSam 1 -UseDae 0 -ExtraArgs @("--skip_pretrain", "1", "--pretrain_path", $SharedPretrain)
Run-Train -ExpName "SDCL_DAE" -UseSam 0 -UseDae 1 -ExtraArgs @("--skip_pretrain", "1", "--pretrain_path", $SharedPretrain)
Run-Train -ExpName "SDCL_SAM_DAE" -UseSam 1 -UseDae 1 -ExtraArgs @("--skip_pretrain", "1", "--pretrain_path", $SharedPretrain)

Write-Host "All LA ablation training jobs finished."
