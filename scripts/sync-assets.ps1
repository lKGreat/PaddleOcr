param(
    [string]$SourceRoot = "E:\codeding\AI\PaddleOCR-3.3.2",
    [string]$TargetRoot = "E:\codeding\PaddleOcr"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SourceRoot)) {
    throw "SourceRoot not found: $SourceRoot"
}

if (-not (Test-Path $TargetRoot)) {
    throw "TargetRoot not found: $TargetRoot"
}

$targetConfigs = Join-Path $TargetRoot "assets\configs"
$targetDicts = Join-Path $TargetRoot "assets\dicts"

New-Item -ItemType Directory -Force -Path $targetConfigs | Out-Null
New-Item -ItemType Directory -Force -Path $targetDicts | Out-Null

$sourceConfigs = Join-Path $SourceRoot "configs"
$sourceDictDir = Join-Path $SourceRoot "ppocr\utils\dict"

if (Test-Path $sourceConfigs) {
    Copy-Item -Path (Join-Path $sourceConfigs "*") -Destination $targetConfigs -Recurse -Force
}

if (Test-Path $sourceDictDir) {
    Copy-Item -Path (Join-Path $sourceDictDir "*") -Destination $targetDicts -Recurse -Force
}

$extraDicts = @(
    "ppocr\utils\ppocr_keys_v1.txt",
    "ppocr\utils\ic15_dict.txt"
)

foreach ($relative in $extraDicts) {
    $src = Join-Path $SourceRoot $relative
    if (Test-Path $src) {
        Copy-Item $src -Destination $targetDicts -Force
    }
}

Write-Host "Assets synced."
Write-Host "Configs: $targetConfigs"
Write-Host "Dicts:   $targetDicts"

