param(
    [string]$Project = "src/PaddleOcr.Tools/PaddleOcr.Tools.csproj"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$tableCfg = "assets/configs/local/table_ci_fast.yml"
$kieCfg = "assets/configs/local/kie_ci_fast.yml"
$detTrainCfg = "assets/configs/local/train_bench_det_ci_fast.yml"

function Invoke-Checked([string[]]$CmdArgs) {
    dotnet run --project $Project -c Release -- @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "command failed: pocr $($CmdArgs -join ' ')"
    }
}

Write-Host "[smoke] config check (table)"
Invoke-Checked @("config", "check", "-c", $tableCfg)

Write-Host "[smoke] config check (kie)"
Invoke-Checked @("config", "check", "-c", $kieCfg)

Write-Host "[smoke] doctor check-models (table)"
Invoke-Checked @("doctor", "check-models", "-c", $tableCfg)

Write-Host "[smoke] doctor check-models (kie)"
Invoke-Checked @("doctor", "check-models", "-c", $kieCfg)

Write-Host "[smoke] doctor train-det-ready"
Invoke-Checked @("doctor", "train-det-ready", "-c", $detTrainCfg)

Write-Host "[smoke] done"
