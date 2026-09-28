param(
    [Parameter(Mandatory = $true)]
    [int]$MasterProcessId,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

$projectPath = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$configPath = Join-Path $projectPath "predicted_roi_study\config_strict_roi.json"
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $projectPath ([string]$config.output)))
$orchestrationRoot = Join-Path $outputRoot "orchestration"
[System.IO.Directory]::CreateDirectory($orchestrationRoot) | Out-Null
$telemetryPath = Join-Path $orchestrationRoot "gpu_telemetry.csv"
$alertPath = Join-Path $orchestrationRoot "monitor_alerts.jsonl"
$monitorStatusPath = Join-Path $orchestrationRoot "monitor_status.json"
$coreStatusPath = Join-Path $orchestrationRoot "five_seed_core_status.json"

$segmenterEpochWarningSeconds = @{
    yolo26 = 900.0
    vit_method2 = 450.0
    emcad = 450.0
    sam2_unet = 700.0
}
$classifierEpochWarningSeconds = 60.0
$lowGpuSamplesRequired = 15
$lowGpuUtilizationPercent = 15.0
$lowGpuPeakPercent = 30.0
$lowGpuWindow = [System.Collections.Generic.List[double]]::new()
$seenEpochs = @{}
$emittedAlerts = @{}

function Save-MonitorStatus {
    param([string]$State, [string]$Message)
    $record = [ordered]@{
        state = $State
        message = $Message
        monitor_process_id = $PID
        master_process_id = $MasterProcessId
        updated_at = [DateTimeOffset]::Now.ToString("o")
        telemetry = $telemetryPath
        alerts = $alertPath
    }
    [System.IO.File]::WriteAllText(
        $monitorStatusPath,
        ($record | ConvertTo-Json -Depth 6) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function Add-MonitorAlert {
    param(
        [string]$Id,
        [string]$Type,
        [string]$Severity,
        [string]$Message,
        [hashtable]$Details
    )
    if ($emittedAlerts.ContainsKey($Id)) {
        return
    }
    $emittedAlerts[$Id] = $true
    $record = [ordered]@{
        id = $Id
        type = $Type
        severity = $Severity
        message = $Message
        observed_at = [DateTimeOffset]::Now.ToString("o")
        details = $Details
    }
    [System.IO.File]::AppendAllText(
        $alertPath,
        ($record | ConvertTo-Json -Compress -Depth 8) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

if (-not (Test-Path -LiteralPath $telemetryPath -PathType Leaf)) {
    [System.IO.File]::WriteAllText(
        $telemetryPath,
        "observed_at,gpu_utilization_percent,memory_used_mib,memory_total_mib,temperature_c,power_w,training_process_active,phase,model,seed`r`n",
        [Text.UTF8Encoding]::new($false)
    )
}

Save-MonitorStatus -State "running" -Message "Monitoring epoch duration and sustained low GPU utilization"

while ($true) {
    $observedAt = [DateTimeOffset]::Now
    $masterProcess = Get-Process -Id $MasterProcessId -ErrorAction SilentlyContinue
    $progress = $null
    $progressPath = Join-Path $outputRoot "progress.json"
    if (Test-Path -LiteralPath $progressPath -PathType Leaf) {
        try {
            $progress = Get-Content -LiteralPath $progressPath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            $progress = $null
        }
    }

    $trainingProcesses = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -match '^python(.exe)?$' -and
                $_.CommandLine -match 'predicted_roi_study' -and
                $_.CommandLine -match '(train-segmenters|build-rois|train-classifiers)'
            }
    )
    $trainingActive = $trainingProcesses.Count -gt 0

    $gpuUtilization = $null
    $gpuMemoryUsed = $null
    $gpuMemoryTotal = $null
    $gpuTemperature = $null
    $gpuPower = $null
    try {
        $gpuLine = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1
        if ($null -ne $gpuLine) {
            $parts = @($gpuLine -split ',' | ForEach-Object { $_.Trim() })
            if ($parts.Count -ge 5) {
                $gpuUtilization = [double]::Parse($parts[0], [Globalization.CultureInfo]::InvariantCulture)
                $gpuMemoryUsed = [double]::Parse($parts[1], [Globalization.CultureInfo]::InvariantCulture)
                $gpuMemoryTotal = [double]::Parse($parts[2], [Globalization.CultureInfo]::InvariantCulture)
                $gpuTemperature = [double]::Parse($parts[3], [Globalization.CultureInfo]::InvariantCulture)
                $gpuPower = [double]::Parse($parts[4], [Globalization.CultureInfo]::InvariantCulture)
            }
        }
    }
    catch {
        Add-MonitorAlert -Id "nvidia-smi-unavailable" -Type "telemetry_failure" -Severity "warning" -Message "GPU telemetry could not be read" -Details @{}
    }

    $phase = if ($null -ne $progress) { [string]$progress.phase } else { "" }
    $model = if ($null -ne $progress) { [string]$progress.model } else { "" }
    $seed = if ($null -ne $progress) { [string]$progress.seed } else { "" }
    $telemetryValues = @(
        $observedAt.ToString("o"),
        $gpuUtilization,
        $gpuMemoryUsed,
        $gpuMemoryTotal,
        $gpuTemperature,
        $gpuPower,
        $trainingActive,
        $phase,
        $model,
        $seed
    ) | ForEach-Object { '"' + ([string]$_).Replace('"', '""') + '"' }
    [System.IO.File]::AppendAllText(
        $telemetryPath,
        ($telemetryValues -join ',') + "`r`n",
        [Text.UTF8Encoding]::new($false)
    )

    if ($trainingActive -and $null -ne $gpuUtilization) {
        $lowGpuWindow.Add($gpuUtilization)
        while ($lowGpuWindow.Count -gt $lowGpuSamplesRequired) {
            $lowGpuWindow.RemoveAt(0)
        }
        if ($lowGpuWindow.Count -eq $lowGpuSamplesRequired) {
            $averageUtilization = ($lowGpuWindow | Measure-Object -Average).Average
            $maximumUtilization = ($lowGpuWindow | Measure-Object -Maximum).Maximum
            if ($averageUtilization -lt $lowGpuUtilizationPercent -and $maximumUtilization -lt $lowGpuPeakPercent) {
                $bucket = [Math]::Floor($observedAt.ToUnixTimeSeconds() / 3600)
                Add-MonitorAlert -Id ("sustained-low-gpu-{0}" -f $bucket) -Type "sustained_low_gpu" -Severity "warning" -Message "GPU utilization stayed materially low during an active training command" -Details @{
                    samples = $lowGpuSamplesRequired
                    poll_seconds = $PollSeconds
                    average_utilization_percent = [Math]::Round([double]$averageUtilization, 2)
                    maximum_utilization_percent = [Math]::Round([double]$maximumUtilization, 2)
                    phase = $phase
                    model = $model
                    seed = $seed
                }
            }
        }
    }
    else {
        $lowGpuWindow.Clear()
    }

    $runsRoot = Join-Path $outputRoot "runs"
    if (Test-Path -LiteralPath $runsRoot -PathType Container) {
        foreach ($historyFile in Get-ChildItem -LiteralPath $runsRoot -Recurse -File -Filter "history.csv" -ErrorAction SilentlyContinue) {
            try {
                $lastRow = Import-Csv -LiteralPath $historyFile.FullName | Select-Object -Last 1
                if ($null -eq $lastRow -or $null -eq $lastRow.epoch -or $null -eq $lastRow.seconds) {
                    continue
                }
                $epochKey = "{0}|{1}" -f $historyFile.FullName, $lastRow.epoch
                if ($seenEpochs.ContainsKey($epochKey)) {
                    continue
                }
                $seenEpochs[$epochKey] = $true
                $seconds = [double]::Parse([string]$lastRow.seconds, [Globalization.CultureInfo]::InvariantCulture)
                $normalizedPath = $historyFile.FullName.Replace('/', '\')
                $isClassifier = $normalizedPath -match '\\classifiers\\'
                $historyModel = ""
                foreach ($candidate in $config.models) {
                    if ($normalizedPath -match ("\\runs\\{0}\\" -f [regex]::Escape([string]$candidate))) {
                        $historyModel = [string]$candidate
                        break
                    }
                }
                $threshold = if ($isClassifier) {
                    $classifierEpochWarningSeconds
                }
                elseif ($segmenterEpochWarningSeconds.ContainsKey($historyModel)) {
                    [double]$segmenterEpochWarningSeconds[$historyModel]
                }
                else {
                    900.0
                }
                if ($seconds -gt $threshold) {
                    Add-MonitorAlert -Id ("slow-epoch-{0}" -f ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($epochKey))).ToLowerInvariant())) -Type "slow_epoch" -Severity "warning" -Message "An epoch exceeded the Seed 17-derived duration warning threshold" -Details @{
                        model = $historyModel
                        epoch = [int]$lastRow.epoch
                        observed_seconds = [Math]::Round($seconds, 2)
                        warning_threshold_seconds = $threshold
                        classifier = $isClassifier
                        history = $historyFile.FullName
                    }
                }
            }
            catch {
                continue
            }
        }
    }

    if ($null -eq $masterProcess) {
        $terminalState = "stopped"
        $terminalMessage = "Master process exited"
        if (Test-Path -LiteralPath $coreStatusPath -PathType Leaf) {
            try {
                $coreStatus = Get-Content -LiteralPath $coreStatusPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $terminalState = [string]$coreStatus.state
                $terminalMessage = [string]$coreStatus.message
            }
            catch {
                $terminalState = "stopped"
            }
        }
        $severity = if ($terminalState -eq "complete") { "info" } else { "critical" }
        Add-MonitorAlert -Id ("master-terminal-{0}" -f $terminalState) -Type "master_terminal" -Severity $severity -Message $terminalMessage -Details @{
            state = $terminalState
            master_process_id = $MasterProcessId
        }
        Save-MonitorStatus -State $terminalState -Message $terminalMessage
        break
    }

    Save-MonitorStatus -State "running" -Message "Monitoring active"
    Start-Sleep -Seconds $PollSeconds
}
