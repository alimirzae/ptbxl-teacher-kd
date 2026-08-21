param(
    [string]$ProjectRoot = "C:\ptbxl",
    [int]$Epochs = 15,
    [int]$TeacherBatchSize = 32,
    [int]$StudentBatchSize = 32,
    [switch]$SkipSetup,
    [switch]$RestartTeacher
)

$ErrorActionPreference = "Stop"
$results = Join-Path $ProjectRoot "results"
$logs = Join-Path $results "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

if (-not $SkipSetup) {
    & (Join-Path $ProjectRoot "setup_environment.ps1") -ProjectRoot $ProjectRoot
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }

function Invoke-Stage {
    param([string]$Name, [string[]]$Arguments)
    $logPath = Join-Path $logs ("{0}.log" -f $Name)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Name. See $logPath" }
}

Invoke-Stage "01_audit_dataset" @((Join-Path $ProjectRoot "01_audit_dataset.py"))
Invoke-Stage "02_create_splits" @((Join-Path $ProjectRoot "02_create_splits.py"))

$teacherArgs = @(
    (Join-Path $ProjectRoot "03_train_teacher.py"), "--mode", "train",
    "--epochs", "$Epochs", "--batch-size", "$TeacherBatchSize",
    "--eval-batch-size", "16", "--num-workers", "0", "--patience", "5"
)
if (-not $RestartTeacher) { $teacherArgs += "--resume" }
Invoke-Stage "03_train_teacher" $teacherArgs

Invoke-Stage "04_student_baseline" @(
    (Join-Path $ProjectRoot "04_train_student.py"), "--mode", "baseline",
    "--epochs", "$Epochs", "--batch-size", "$StudentBatchSize",
    "--eval-batch-size", "64", "--num-workers", "0"
)
Invoke-Stage "04_student_kd" @(
    (Join-Path $ProjectRoot "04_train_student.py"), "--mode", "kd",
    "--epochs", "$Epochs", "--batch-size", "$StudentBatchSize",
    "--eval-batch-size", "64", "--num-workers", "0"
)
Invoke-Stage "05_evaluate_and_export" @((Join-Path $ProjectRoot "05_evaluate_and_export.py"))
Invoke-Stage "06_build_research_bundle" @((Join-Path $ProjectRoot "06_build_research_bundle.py"))

Write-Host "`nPipeline complete. Results: $results" -ForegroundColor Green

