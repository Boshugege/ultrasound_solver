param(
    [int]$WaitForProcessId = 0,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$mvpProgress = Join-Path $repoRoot "outputs\hand_phase_surrogate_mvp_v1\progress.json"
$generator = Join-Path $PSScriptRoot "generate_mano_bem_dataset.py"

if ($WaitForProcessId -gt 0) {
    Wait-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
}

if (-not (Test-Path $mvpProgress)) {
    throw "MVP progress file does not exist: $mvpProgress"
}

$progress = Get-Content -Raw $mvpProgress | ConvertFrom-Json
if ([int]$progress.completed_geometries -ne 8000) {
    throw "MVP is incomplete: $($progress.completed_geometries) / 8000 geometries"
}

$freeBytes = (Get-PSDrive -Name C).Free
$minimumFreeBytes = 35GB
if ($freeBytes -lt $minimumFreeBytes) {
    throw "Full dataset requires at least 35 GB free on C:, found $([math]::Round($freeBytes / 1GB, 1)) GB"
}

& $PythonExe $generator `
    --output-dir (Join-Path $repoRoot "outputs\hand_phase_surrogate_full_v1") `
    --num-geometries 50000 `
    --targets-per-geometry 256 `
    --num-shapes 256 `
    --teacher-profile high `
    --stop-on-error

if ($LASTEXITCODE -ne 0) {
    throw "Full dataset generator exited with code $LASTEXITCODE"
}
