param(
    [string]$Project = "src/PaddleOcr.Tools/PaddleOcr.Tools.csproj",
    [string]$ReportDir = "outputs/acceptance",
    [string]$ServiceUrl = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
$reportPath = Join-Path $ReportDir ("acceptance_replay_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".md")
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Acceptance Replay")
$lines.Add("")
$lines.Add("Generated: $(Get-Date -Format o)")
$lines.Add("")

function Invoke-Step([string]$Title, [string[]]$CmdArgs) {
    Write-Host "[acceptance] $Title"
    dotnet run --project $Project -c Release -- @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        $lines.Add("- FAIL: $Title")
        throw "step failed: $Title"
    }

    $lines.Add("- PASS: $Title")
}

try {
    Invoke-Step "config check table_ci_fast" @("config", "check", "-c", "assets/configs/local/table_ci_fast.yml")
    Invoke-Step "config check kie_ci_fast" @("config", "check", "-c", "assets/configs/local/kie_ci_fast.yml")
    Invoke-Step "doctor parity table" @("doctor", "parity-table-kie", "-c", "assets/configs/local/table_ci_fast.yml", "--mode", "table")
    Invoke-Step "doctor parity kie" @("doctor", "parity-table-kie", "-c", "assets/configs/local/kie_ci_fast.yml", "--mode", "kie")
    Invoke-Step "doctor train det ready" @("doctor", "train-det-ready", "-c", "assets/configs/local/train_bench_det_ci_fast.yml")

    $pluginDir = Join-Path $env:TEMP ("pocr_accept_plugin_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $pluginDir -Force | Out-Null
    $dllPath = Join-Path $pluginDir "demo.dll"
    Set-Content -Path $dllPath -Value "demo"
    $hash = (Get-FileHash $dllPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $pluginJson = @"
{
  "schema_version": "1.0",
  "name": "accept-alias-pre",
  "version": "1.0.0",
  "type": "preprocess",
  "runtime_name": "accept-rgb-chw-01",
  "alias_of": "rgb-chw-01",
  "trust": {
    "algorithm": "sha256",
    "files_sha256": { "demo.dll": "$hash" },
    "trust_level": "verified"
  }
}
"@
    Set-Content -Path (Join-Path $pluginDir "plugin.json") -Value $pluginJson

    Invoke-Step "plugin validate-package" @("plugin", "validate-package", "--package_dir", $pluginDir, "--require_trust", "true")
    Invoke-Step "plugin verify-trust" @("plugin", "verify-trust", "--package_dir", $pluginDir)
    Invoke-Step "plugin load-runtime" @("plugin", "load-runtime", "--package_dir", $pluginDir)

    Invoke-Step "benchmark e2e:eval" @("benchmark", "run", "--scenario", "e2e:eval", "--gt_dir", "assets/samples/tiny_det/images", "--pred_dir", "assets/samples/tiny_det/images", "--warmup", "0", "--iterations", "1")
    Invoke-Step "benchmark train:train" @("benchmark", "run", "--scenario", "train:train", "-c", "assets/configs/local/train_bench_rec_ci_fast.yml", "--warmup", "0", "--iterations", "1")
    Invoke-Step "benchmark train:train(det)" @("benchmark", "run", "--scenario", "train:train", "-c", "assets/configs/local/train_bench_det_ci_fast.yml", "--warmup", "0", "--iterations", "1")

    if (-not [string]::IsNullOrWhiteSpace($ServiceUrl)) {
        Invoke-Step "benchmark service:test" @("benchmark", "run", "--scenario", "service:test", "--profile", "smoke", "--server_url", $ServiceUrl, "--image_dir", "assets/samples/tiny_cls/images", "--warmup", "0", "--iterations", "1", "--continue_on_error", "true")
    } else {
        $lines.Add("- SKIP: benchmark service:test (ServiceUrl not provided)")
    }

    $lines.Add("")
    $lines.Add("Overall: PASS")
}
catch {
    $lines.Add("")
    $lines.Add("Overall: FAIL")
    $lines.Add("Reason: $($_.Exception.Message)")
    throw
}
finally {
    $lines | Set-Content -Path $reportPath
    Write-Host "[acceptance] report: $reportPath"
}
