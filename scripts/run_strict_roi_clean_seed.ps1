param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(17, 42, 2026, 3407, 9103)]
    [int]$Seed,
    [switch]$ResumeIncompleteSeed
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectPath = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$pythonPath = Join-Path $projectPath ".venv-binary\Scripts\python.exe"
$absoluteConfig = [System.IO.Path]::GetFullPath(
    (Join-Path $projectPath "predicted_roi_study\config_strict_roi.json")
)

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python executable not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $absoluteConfig -PathType Leaf)) {
    throw "Locked clean-run protocol not found: $absoluteConfig"
}

$lockedConfig = Get-Content -LiteralPath $absoluteConfig -Raw -Encoding UTF8 | ConvertFrom-Json
if ($lockedConfig.protocol_version -ne "1.2.0") {
    throw "This launcher requires strict predicted-ROI protocol 1.2.0"
}
if ($lockedConfig.study_id -ne "strict_predicted_roi_binary_4model_v1_2_0_clean") {
    throw "This launcher refuses a non-clean study_id"
}
if ($lockedConfig.clean_run.import_upstream_artifacts -ne $false) {
    throw "Clean run must not import upstream checkpoints or ROI artifacts"
}
if ($lockedConfig.clean_run.test_evaluation_in_seed_launcher -ne $false) {
    throw "Per-seed clean launcher must not include test evaluation"
}
if ($lockedConfig.clean_run.launcher_stops_after -ne "validation_lock") {
    throw "Per-seed clean launcher must stop after the validation lock"
}
if ($lockedConfig.output -ne $lockedConfig.clean_run.output_namespace) {
    throw "Config output and clean-run namespace disagree"
}

$outputRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $projectPath ([string]$lockedConfig.output))
)
$priorOutput = [System.IO.Path]::GetFullPath(
    (Join-Path $projectPath ([string]$lockedConfig.clean_run.prior_failed_output))
)
$projectPrefix = $projectPath.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$priorPrefix = $priorOutput.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
if (-not $outputRoot.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Clean output must remain inside the project workspace: $outputRoot"
}
if (
    $outputRoot.Equals($priorOutput, [System.StringComparison]::OrdinalIgnoreCase) -or
    $outputRoot.StartsWith($priorPrefix, [System.StringComparison]::OrdinalIgnoreCase)
) {
    throw "Clean output must be isolated from prior failed output: $priorOutput"
}

$testAccessPath = Join-Path $outputRoot "state\test_access_opened.json"
if (Test-Path -LiteralPath $testAccessPath -PathType Leaf) {
    throw "Test access is already open; refusing to modify the clean training namespace"
}

$existingUnitRoots = @(
    foreach ($model in $lockedConfig.models) {
        $candidate = Join-Path $outputRoot ("runs\{0}\seed_{1}" -f $model, $Seed)
        if (Test-Path -LiteralPath $candidate) {
            $candidate
        }
    }
)
$receiptRoot = Join-Path $outputRoot "state\receipts"
$existingSeedReceipts = @()
if (Test-Path -LiteralPath $receiptRoot -PathType Container) {
    $existingSeedReceipts = @(
        Get-ChildItem -LiteralPath $receiptRoot -Recurse -File -Filter ("seed_{0}.json" -f $Seed)
    )
}
$hasExistingSeedState = (
    $existingUnitRoots.Count -gt 0 -or $existingSeedReceipts.Count -gt 0
)
if ($hasExistingSeedState -and -not $ResumeIncompleteSeed.IsPresent) {
    throw ((
            "Seed {0} already has clean-namespace artifacts. Refusing implicit resume; " +
            "re-run with -ResumeIncompleteSeed so code/config/split/input hash gates can verify them."
        ) -f $Seed)
}
if ($ResumeIncompleteSeed.IsPresent -and -not $hasExistingSeedState) {
    throw "-ResumeIncompleteSeed was requested, but the selected seed has no clean-namespace state."
}
if (
    $ResumeIncompleteSeed.IsPresent -and
    -not (Test-Path -LiteralPath (Join-Path $receiptRoot "prepare.json") -PathType Leaf)
) {
    throw "Resume requires an existing prepare receipt in the same clean namespace."
}

$orchestrationRoot = Join-Path $outputRoot "orchestration"
$statusPath = Join-Path $orchestrationRoot ("seed_{0}_status.json" -f $Seed)
$eventLogPath = Join-Path $orchestrationRoot ("seed_{0}_events.jsonl" -f $Seed)
[System.IO.Directory]::CreateDirectory($orchestrationRoot) | Out-Null
$studyLockPath = Join-Path $orchestrationRoot "clean_study_exclusive.lock"
$studyLockStream = $null
try {
    $studyLockStream = [System.IO.File]::Open(
        $studyLockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch [System.IO.IOException] {
    throw "Another clean strict-ROI seed launcher is active; seeds must run serially."
}
Set-Location -LiteralPath $projectPath
$currentStage = "initialization"

function Write-Status {
    param(
        [string]$State,
        [string]$Stage,
        [string]$Message
    )
    $record = [ordered]@{
        study_id = [string]$lockedConfig.study_id
        protocol_version = [string]$lockedConfig.protocol_version
        seed = $Seed
        state = $State
        stage = $Stage
        message = $Message
        process_id = $PID
        updated_at = [DateTimeOffset]::Now.ToString("o")
        project_root = $projectPath
        config = $absoluteConfig
        config_sha256 = (Get-FileHash -LiteralPath $absoluteConfig -Algorithm SHA256).Hash.ToLowerInvariant()
        output_root = $outputRoot
        prior_artifacts_imported = $false
        resume_incomplete_seed_requested = $ResumeIncompleteSeed.IsPresent
        test_evaluation_started = $false
    }
    $json = $record | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        $statusPath,
        $json + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
    [System.IO.File]::AppendAllText(
        $eventLogPath,
        ($record | ConvertTo-Json -Compress) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function Invoke-StudyStage {
    param(
        [string]$Stage,
        [string[]]$Arguments
    )
    $script:currentStage = $Stage
    Write-Status -State "running" -Stage $Stage -Message "Stage started"
    & $pythonPath -m predicted_roi_study @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Stage' failed with exit code $LASTEXITCODE"
    }
    Write-Status -State "running" -Stage $Stage -Message "Stage completed"
}

try {
    if (Test-Path -LiteralPath $testAccessPath -PathType Leaf) {
        throw "Test access opened while acquiring the clean-study lock; refusing seed work"
    }
    $currentStage = "prepare"
    Write-Status -State "running" -Stage "prepare" -Message (
        "Clean strict predicted-ROI seed {0} workflow started" -f $Seed
    )
    Invoke-StudyStage -Stage "prepare" -Arguments @(
        "prepare", "--config", $absoluteConfig
    )
    Invoke-StudyStage -Stage "preflight" -Arguments @(
        "preflight", "--config", $absoluteConfig, "--model", "all"
    )
    Invoke-StudyStage -Stage "train-segmenters" -Arguments @(
        "train-segmenters", "--config", $absoluteConfig,
        "--model", "all", "--seed", [string]$Seed
    )
    Invoke-StudyStage -Stage "build-rois" -Arguments @(
        "build-rois", "--config", $absoluteConfig,
        "--model", "all", "--seed", [string]$Seed
    )
    Invoke-StudyStage -Stage "train-classifiers" -Arguments @(
        "train-classifiers", "--config", $absoluteConfig,
        "--model", "all", "--seed", [string]$Seed
    )
    Invoke-StudyStage -Stage "lock" -Arguments @(
        "lock", "--config", $absoluteConfig,
        "--model", "all", "--seed", [string]$Seed
    )
    $currentStage = "validation-lock"
    if (Test-Path -LiteralPath $testAccessPath -PathType Leaf) {
        throw "Test access opened during the clean seed workflow; refusing completion status"
    }
    Write-Status -State "complete" -Stage "validation-lock" -Message (
        "All four seed {0} units are validation-locked; this launcher did not invoke test evaluation" -f $Seed
    )
}
catch {
    Write-Status -State "failed" -Stage $currentStage -Message $_.Exception.Message
    throw
}
finally {
    if ($null -ne $studyLockStream) {
        $studyLockStream.Dispose()
    }
}
