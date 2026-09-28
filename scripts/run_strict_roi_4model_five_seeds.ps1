param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectPath = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$pythonPath = Join-Path $projectPath ".venv-binary\Scripts\python.exe"
$seedLauncher = Join-Path $PSScriptRoot "run_strict_roi_clean_seed.ps1"
$configPath = Join-Path $projectPath "predicted_roi_study\config_strict_roi.json"

foreach ($requiredPath in @($pythonPath, $seedLauncher, $configPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($config.protocol_version -ne "1.2.0") {
    throw "Five-seed launcher requires protocol 1.2.0"
}
if ($config.study_id -ne "strict_predicted_roi_binary_4model_v1_2_0_clean") {
    throw "Five-seed launcher refuses a different study_id"
}
if (($config.models -join ",") -ne "yolo26,vit_method2,emcad,sam2_unet") {
    throw "Five-seed launcher requires the locked four-model order"
}
if (($config.split_seeds -join ",") -ne "17,42,2026,3407,9103") {
    throw "Five-seed launcher requires the five locked seeds"
}

$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $projectPath ([string]$config.output)))
$orchestrationRoot = Join-Path $outputRoot "orchestration"
[System.IO.Directory]::CreateDirectory($orchestrationRoot) | Out-Null
$statusPath = Join-Path $orchestrationRoot "five_seed_core_status.json"
$eventPath = Join-Path $orchestrationRoot "five_seed_core_events.jsonl"
$supervisorLockPath = Join-Path $orchestrationRoot "five_seed_core_exclusive.lock"
$supervisorLock = $null

try {
    $supervisorLock = [System.IO.File]::Open(
        $supervisorLockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch [System.IO.IOException] {
    throw "Another five-seed core launcher is active"
}

function Write-CoreStatus {
    param(
        [string]$State,
        [string]$Stage,
        [Nullable[int]]$Seed,
        [string]$Message
    )
    $record = [ordered]@{
        study_id = [string]$config.study_id
        protocol_version = [string]$config.protocol_version
        state = $State
        stage = $Stage
        seed = $Seed
        message = $Message
        process_id = $PID
        updated_at = [DateTimeOffset]::Now.ToString("o")
        output_root = $outputRoot
        config = $configPath
        config_sha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
        test_access_open = (Test-Path -LiteralPath (Join-Path $outputRoot "state\test_access_opened.json") -PathType Leaf)
        resume_requested = $Resume.IsPresent
    }
    $json = $record | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        $statusPath,
        $json + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
    [System.IO.File]::AppendAllText(
        $eventPath,
        ($record | ConvertTo-Json -Compress) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function Test-SeedHasState {
    param([int]$Seed)
    foreach ($model in $config.models) {
        if (Test-Path -LiteralPath (Join-Path $outputRoot ("runs\{0}\seed_{1}" -f $model, $Seed))) {
            return $true
        }
    }
    $receiptRoot = Join-Path $outputRoot "state\receipts"
    if (Test-Path -LiteralPath $receiptRoot -PathType Container) {
        $match = Get-ChildItem -LiteralPath $receiptRoot -Recurse -File -Filter ("seed_{0}.json" -f $Seed) |
            Select-Object -First 1
        if ($null -ne $match) {
            return $true
        }
    }
    return $false
}

function Test-SeedComplete {
    param([int]$Seed)
    foreach ($model in $config.models) {
        $lockReceipt = Join-Path $outputRoot ("state\receipts\lock\{0}\seed_{1}.json" -f $model, $Seed)
        if (-not (Test-Path -LiteralPath $lockReceipt -PathType Leaf)) {
            return $false
        }
    }
    return $true
}

Set-Location -LiteralPath $projectPath
$currentStage = "initialization"
$currentSeed = $null

try {
    Write-CoreStatus -State "running" -Stage "initialization" -Seed $null -Message (
        "Four-model five-seed core workflow started; test remains closed until 20/20 composite locks"
    )

    foreach ($seedValue in $config.split_seeds) {
        $currentSeed = [int]$seedValue
        $currentStage = "seed_validation_lock"
        if (Test-SeedComplete -Seed $currentSeed) {
            if (-not $Resume.IsPresent) {
                throw "Seed $currentSeed is already complete; explicit -Resume is required"
            }
            $testAccessPath = Join-Path $outputRoot "state\test_access_opened.json"
            if (Test-Path -LiteralPath $testAccessPath -PathType Leaf) {
                Write-CoreStatus -State "running" -Stage $currentStage -Seed $currentSeed -Message "Test gate already open; seed will be revalidated by evaluate without modifying the training namespace"
                continue
            }
            Write-CoreStatus -State "running" -Stage $currentStage -Seed $currentSeed -Message "Existing receipts will be fully revalidated by the seed launcher before test access"
        }

        $hasState = Test-SeedHasState -Seed $currentSeed
        if ($hasState -and -not $Resume.IsPresent) {
            throw "Seed $currentSeed has incomplete state; explicit -Resume is required"
        }
        Write-CoreStatus -State "running" -Stage $currentStage -Seed $currentSeed -Message "Seed workflow started"
        if ($hasState) {
            & $seedLauncher -Seed $currentSeed -ResumeIncompleteSeed
        }
        else {
            & $seedLauncher -Seed $currentSeed
        }
        if (-not (Test-SeedComplete -Seed $currentSeed)) {
            throw "Seed $currentSeed launcher returned without all four validation locks"
        }
        Write-CoreStatus -State "running" -Stage $currentStage -Seed $currentSeed -Message "All four composite validation locks complete"
    }

    $currentSeed = $null
    $currentStage = "evaluate"
    Write-CoreStatus -State "running" -Stage $currentStage -Seed $null -Message "Opening test only after the global 20-lock gate and evaluating 40 systems"
    & $pythonPath -m predicted_roi_study evaluate --config $configPath --model all
    if ($LASTEXITCODE -ne 0) {
        throw "Global evaluate failed with exit code $LASTEXITCODE"
    }

    $currentStage = "summarize"
    Write-CoreStatus -State "running" -Stage $currentStage -Seed $null -Message "Creating immutable five-seed primary and secondary summaries"
    & $pythonPath -m predicted_roi_study summarize --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Global summarize failed with exit code $LASTEXITCODE"
    }

    $currentStage = "audit"
    & $pythonPath -m predicted_roi_study audit --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Final audit failed with exit code $LASTEXITCODE"
    }
    Write-CoreStatus -State "complete" -Stage "complete" -Seed $null -Message "Five-seed core evaluation and summaries completed"
}
catch {
    Write-CoreStatus -State "failed" -Stage $currentStage -Seed $currentSeed -Message $_.Exception.Message
    throw
}
finally {
    if ($null -ne $supervisorLock) {
        $supervisorLock.Dispose()
    }
}
