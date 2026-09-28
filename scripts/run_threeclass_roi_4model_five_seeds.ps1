param(
    [switch]$Resume,
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# These values must exist before Python imports PyTorch / initializes CUDA.
# Individual model/seed RNGs are still set inside the locked Python engine.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONHASHSEED = "0"

$projectPath = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$pythonPath = Join-Path $projectPath ".venv-binary\Scripts\python.exe"
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $projectPath "threeclass_roi_study\config_threeclass_roi.json"
}
elseif (-not [System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath = Join-Path $projectPath $ConfigPath
}
$ConfigPath = [System.IO.Path]::GetFullPath($ConfigPath)

foreach ($requiredPath in @($pythonPath, $ConfigPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

$config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($config.protocol_version -ne "1.0.0") {
    throw "Three-class launcher requires protocol 1.0.0"
}
if ($config.study_id -ne "strict_predicted_roi_threeclass_4model_v1_0_0") {
    throw "Three-class launcher refuses a different study_id"
}
if (($config.models -join ",") -ne "yolo26,vit_method2,emcad,sam2_unet") {
    throw "Three-class launcher requires the locked four-model order"
}
if (($config.split_seeds -join ",") -ne "17,42,2026,3407,9103") {
    throw "Three-class launcher requires the five locked seeds"
}
if (($config.classifier.strategy_order -join ",") -ne "model_specific,standardized_resnet18") {
    throw "Three-class launcher requires both locked classifier strategies"
}

if ([System.IO.Path]::IsPathRooted([string]$config.output)) {
    $outputRoot = [System.IO.Path]::GetFullPath([string]$config.output)
}
else {
    $outputRoot = [System.IO.Path]::GetFullPath((Join-Path $projectPath ([string]$config.output)))
}
$orchestrationRoot = Join-Path $outputRoot "orchestration"
[System.IO.Directory]::CreateDirectory($orchestrationRoot) | Out-Null
$statusPath = Join-Path $orchestrationRoot "five_seed_core_status.json"
$eventPath = Join-Path $orchestrationRoot "five_seed_core_events.jsonl"
$supervisorLockPath = Join-Path $orchestrationRoot "five_seed_core_exclusive.lock"
$prepareReceipt = Join-Path $outputRoot "state\receipts\prepare.json"
$supervisorLock = $null

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)

    $fullPath = [System.IO.Path]::GetFullPath($LiteralPath)
    $stream = [System.IO.File]::Open(
        $fullPath,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read
    )
    try {
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha256.ComputeHash($stream)
        }
        finally {
            $sha256.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
    return ([System.BitConverter]::ToString($digest)).Replace("-", "").ToLowerInvariant()
}

try {
    $supervisorLock = [System.IO.File]::Open(
        $supervisorLockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch [System.IO.IOException] {
    throw "Another three-class five-seed launcher is active"
}

function Assert-CodeInventoryUnchanged {
    $inventoryPath = Join-Path $outputRoot "provenance\code_inventory.json"
    if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
        throw "Resume refused: provenance code inventory is missing"
    }
    $inventory = Get-Content -LiteralPath $inventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $expectedNames = @($inventory.PSObject.Properties.Name | Sort-Object)
    $packagePath = Join-Path $projectPath "threeclass_roi_study"
    $actualFiles = @(Get-ChildItem -LiteralPath $packagePath -File -Filter "*.py" | Sort-Object Name)
    $actualNames = @($actualFiles.Name)
    if (($expectedNames -join ",") -ne ($actualNames -join ",")) {
        throw "Resume refused: Python source inventory changed after prepare"
    }
    foreach ($file in $actualFiles) {
        $expected = [string]$inventory.($file.Name)
        $observed = Get-Sha256Hex -LiteralPath $file.FullName
        if ($observed -ne $expected.ToLowerInvariant()) {
            throw "Resume refused: source hash changed after prepare: $($file.Name)"
        }
    }
}

function Write-CoreStatus {
    param(
        [string]$State,
        [string]$Stage,
        [string]$Model,
        [Nullable[int]]$Seed,
        [string]$Strategy,
        [int]$Completed,
        [int]$Target,
        [string]$Message
    )
    $record = [ordered]@{
        study_id = [string]$config.study_id
        protocol_version = [string]$config.protocol_version
        state = $State
        stage = $Stage
        model = $Model
        seed = $Seed
        strategy = $Strategy
        completed_units = $Completed
        target_units = $Target
        message = $Message
        process_id = $PID
        updated_at = [DateTimeOffset]::Now.ToString("o")
        output_root = $outputRoot
        config = $ConfigPath
        config_sha256_file = Get-Sha256Hex -LiteralPath $ConfigPath
        test_access_open = (Test-Path -LiteralPath (Join-Path $outputRoot ([string]$config.test_access.sentinel)) -PathType Leaf)
        resume_requested = $Resume.IsPresent
    }
    $json = $record | ConvertTo-Json -Depth 8
    $temporaryStatus = $statusPath + ".tmp"
    [System.IO.File]::WriteAllText(
        $temporaryStatus,
        $json + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporaryStatus -Destination $statusPath -Force
    [System.IO.File]::AppendAllText(
        $eventPath,
        ($record | ConvertTo-Json -Compress) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function Invoke-StudyCli {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$ExtraArgs = @()
    )
    $cliArgs = @(
        "-m", "threeclass_roi_study", $Command,
        "--config", $ConfigPath,
        "--traceback"
    ) + $ExtraArgs
    & $pythonPath @cliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CLI stage '$Command' failed with exit code $LASTEXITCODE"
    }
}

function Test-Receipt {
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    $receiptPath = Join-Path $outputRoot $RelativePath
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        return $false
    }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($record in @($receipt.artifacts)) {
        $artifactPath = [string]$record.path
        if (-not [System.IO.Path]::IsPathRooted($artifactPath)) {
            $artifactPath = Join-Path $projectPath $artifactPath
        }
        $artifactPath = [System.IO.Path]::GetFullPath($artifactPath)
        if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
            throw "Receipt artifact is missing: $artifactPath"
        }
        $observedSize = (Get-Item -LiteralPath $artifactPath).Length
        if ($observedSize -ne [int64]$record.size_bytes) {
            throw "Receipt artifact size mismatch: $artifactPath"
        }
        $observedHash = Get-Sha256Hex -LiteralPath $artifactPath
        if ($observedHash -ne ([string]$record.sha256).ToLowerInvariant()) {
            throw "Receipt artifact hash mismatch: $artifactPath"
        }
    }
    return $true
}

$currentStage = "initialization"
$currentModel = $null
$currentSeed = $null
$currentStrategy = $null
$currentCompleted = 0
$currentTarget = 40

try {
    Set-Location -LiteralPath $projectPath

    if (Test-Receipt -RelativePath "state\receipts\prepare.json") {
        if (-not $Resume.IsPresent) {
            throw "Existing study state found; explicit -Resume is required"
        }
        Assert-CodeInventoryUnchanged
    }

    Write-CoreStatus -State "running" -Stage "initialization" -Model $null -Seed $null -Strategy $null -Completed 0 -Target 40 -Message (
        "Locked four-model/five-seed/two-strategy workflow started; test remains closed until 40/40 validation locks"
    )

    $currentStage = "prepare"
    $currentCompleted = 0
    $currentTarget = 1
    if (-not (Test-Receipt -RelativePath "state\receipts\prepare.json")) {
        Invoke-StudyCli -Command "prepare"
    }
    Write-CoreStatus -State "running" -Stage $currentStage -Model $null -Seed $null -Strategy $null -Completed 0 -Target 40 -Message "Provenance snapshot verified"

    $currentStage = "import_upstream_development"
    $currentCompleted = 0
    $currentTarget = 20
    $importCompleted = 0
    foreach ($modelValue in $config.models) {
        $currentModel = [string]$modelValue
        foreach ($seedValue in $config.split_seeds) {
            $currentSeed = [int]$seedValue
            $relative = "state\receipts\import-upstream-development\{0}\seed_{1}.json" -f $currentModel, $currentSeed
            $auditRelative = "state\receipts\audit-upstream-development\{0}\seed_{1}.json" -f $currentModel, $currentSeed
            $importReceiptComplete = Test-Receipt -RelativePath $relative
            $auditReceiptComplete = Test-Receipt -RelativePath $auditRelative
            if (-not ($importReceiptComplete -and $auditReceiptComplete)) {
                Write-CoreStatus -State "running" -Stage $currentStage -Model $currentModel -Seed $currentSeed -Strategy $null -Completed $importCompleted -Target 20 -Message "Auditing frozen train-OOF and validation ROI artifacts"
                Invoke-StudyCli -Command "import-upstream" -ExtraArgs @("--model", $currentModel, "--seed", [string]$currentSeed)
            }
            $importCompleted += 1
            $currentCompleted = $importCompleted
            Write-CoreStatus -State "running" -Stage $currentStage -Model $currentModel -Seed $currentSeed -Strategy $null -Completed $importCompleted -Target 20 -Message "Development-only upstream import verified"
        }
    }

    $currentStage = "preflight"
    $currentCompleted = 0
    $currentTarget = 4
    $preflightCompleted = 0
    $currentSeed = $null
    foreach ($modelValue in $config.models) {
        $currentModel = [string]$modelValue
        $relative = "state\receipts\preflight\{0}.json" -f $currentModel
        if (-not (Test-Receipt -RelativePath $relative)) {
            Write-CoreStatus -State "running" -Stage $currentStage -Model $currentModel -Seed $null -Strategy $null -Completed $preflightCompleted -Target 4 -Message "Running GPU/gradient/shape/outside-ROI preflight"
            Invoke-StudyCli -Command "preflight" -ExtraArgs @("--model", $currentModel)
        }
        $preflightCompleted += 1
        $currentCompleted = $preflightCompleted
        Write-CoreStatus -State "running" -Stage $currentStage -Model $currentModel -Seed $null -Strategy $null -Completed $preflightCompleted -Target 4 -Message "Model preflight verified"
    }

    $currentStage = "development_training_and_validation_lock"
    $currentCompleted = 0
    $currentTarget = 40
    $validationCompleted = 0
    foreach ($modelValue in $config.models) {
        $currentModel = [string]$modelValue
        foreach ($seedValue in $config.split_seeds) {
            $currentSeed = [int]$seedValue
            foreach ($strategyValue in $config.classifier.strategy_order) {
                $currentStrategy = [string]$strategyValue
                $trainingRelative = "state\receipts\train-classifier\{0}\seed_{1}\{2}.json" -f $currentModel, $currentSeed, $currentStrategy
                $lockPath = Join-Path $outputRoot ("runs\{0}\seed_{1}\locks\{2}_validation_lock.json" -f $currentModel, $currentSeed, $currentStrategy)
                if (-not (Test-Receipt -RelativePath $trainingRelative)) {
                    Write-CoreStatus -State "running" -Stage "classifier_training" -Model $currentModel -Seed $currentSeed -Strategy $currentStrategy -Completed $validationCompleted -Target 40 -Message "Training new three-logit classifier without test access"
                    Invoke-StudyCli -Command "train" -ExtraArgs @("--model", $currentModel, "--seed", [string]$currentSeed, "--strategy", $currentStrategy)
                }
                if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
                    Write-CoreStatus -State "running" -Stage "validation_lock" -Model $currentModel -Seed $currentSeed -Strategy $currentStrategy -Completed $validationCompleted -Target 40 -Message "Fitting validation-only calibration and writing immutable lock"
                    Invoke-StudyCli -Command "lock" -ExtraArgs @("--model", $currentModel, "--seed", [string]$currentSeed, "--strategy", $currentStrategy)
                }
                $validationCompleted += 1
                $currentCompleted = $validationCompleted
                Write-CoreStatus -State "running" -Stage $currentStage -Model $currentModel -Seed $currentSeed -Strategy $currentStrategy -Completed $validationCompleted -Target 40 -Message "Validation lock verified"
            }
        }
    }

    $currentStage = "global_test_gate"
    $currentCompleted = 40
    $currentTarget = 40
    $currentModel = $null
    $currentSeed = $null
    $currentStrategy = $null
    $testSentinel = Join-Path $outputRoot ([string]$config.test_access.sentinel)
    if (-not (Test-Path -LiteralPath $testSentinel -PathType Leaf)) {
        Write-CoreStatus -State "running" -Stage $currentStage -Model $null -Seed $null -Strategy $null -Completed 40 -Target 40 -Message "Verifying all 40 locks before opening test access"
        Invoke-StudyCli -Command "open-test"
    }
    Write-CoreStatus -State "running" -Stage "global_test_evaluation" -Model $null -Seed $null -Strategy $null -Completed 0 -Target 40 -Message "Global lock gate passed; test evaluation started"

    $currentStage = "global_test_evaluation"
    $currentCompleted = 0
    $currentTarget = 40
    $evaluationCompleted = 0
    foreach ($modelValue in $config.models) {
        $currentModel = [string]$modelValue
        foreach ($seedValue in $config.split_seeds) {
            $currentSeed = [int]$seedValue
            foreach ($strategyValue in $config.classifier.strategy_order) {
                $currentStrategy = [string]$strategyValue
                $relative = "state\receipts\evaluate\{0}\seed_{1}\{2}.json" -f $currentModel, $currentSeed, $currentStrategy
                if (-not (Test-Receipt -RelativePath $relative)) {
                    Write-CoreStatus -State "running" -Stage $currentStage -Model $currentModel -Seed $currentSeed -Strategy $currentStrategy -Completed $evaluationCompleted -Target 40 -Message "Evaluating locked classifier on the deferred test partition"
                    Invoke-StudyCli -Command "evaluate" -ExtraArgs @("--model", $currentModel, "--seed", [string]$currentSeed, "--strategy", $currentStrategy)
                }
                $evaluationCompleted += 1
                $currentCompleted = $evaluationCompleted
                Write-CoreStatus -State "running" -Stage $currentStage -Model $currentModel -Seed $currentSeed -Strategy $currentStrategy -Completed $evaluationCompleted -Target 40 -Message "Evaluation receipt verified"
            }
        }
    }

    $currentStage = "summarize"
    $currentCompleted = 40
    $currentTarget = 40
    $currentModel = $null
    $currentSeed = $null
    $currentStrategy = $null
    if (-not (Test-Receipt -RelativePath "state\receipts\summarize.json")) {
        Write-CoreStatus -State "running" -Stage $currentStage -Model $null -Seed $null -Strategy $null -Completed 40 -Target 40 -Message "Generating Q1 tables, figures, uncertainty, and claim limits"
        Invoke-StudyCli -Command "summarize"
    }

    $currentStage = "audit"
    # Always enter Python's full graph validator, including on resume.  The
    # audit command is idempotent, but unlike the launcher's fast artifact-only
    # check it revalidates receipt identities, config/code hashes, all locks,
    # the global test sentinel, publication schemas, and figure manifests.
    Write-CoreStatus -State "running" -Stage $currentStage -Model $null -Seed $null -Strategy $null -Completed 40 -Target 40 -Message "Verifying final receipt identities and complete artifact graph"
    Invoke-StudyCli -Command "audit"
    Write-CoreStatus -State "complete" -Stage "complete" -Model $null -Seed $null -Strategy $null -Completed 40 -Target 40 -Message "Three-class Q1 workflow completed and audited"
}
catch {
    Write-CoreStatus -State "failed" -Stage $currentStage -Model $currentModel -Seed $currentSeed -Strategy $currentStrategy -Completed $currentCompleted -Target $currentTarget -Message $_.Exception.Message
    throw
}
finally {
    if ($null -ne $supervisorLock) {
        $supervisorLock.Dispose()
    }
}
