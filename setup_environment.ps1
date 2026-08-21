param(
    [string]$ProjectRoot = "C:\ptbxl",
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"
$venvPath = Join-Path $ProjectRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

Write-Host "=== PTB-XL Windows environment setup ===" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "Project root does not exist: $ProjectRoot"
}

if ($ForceRecreate -and (Test-Path -LiteralPath $venvPath)) {
    $resolvedRoot = [IO.Path]::GetFullPath($ProjectRoot)
    $resolvedVenv = [IO.Path]::GetFullPath($venvPath)
    if (-not $resolvedVenv.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a virtual environment outside the project root."
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.12 -m venv $venvPath
}

& $pythonPath -m pip install --upgrade pip setuptools wheel --index-url https://pypi.org/simple
& $pythonPath -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 `
    --index-url https://download.pytorch.org/whl/cu121
& $pythonPath -m pip install -r (Join-Path $ProjectRoot "requirements.txt") `
    --index-url https://pypi.org/simple

& $pythonPath -c "import torch, numpy, pandas, wfdb, sklearn; print('Python environment OK'); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('NumPy:', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Environment validation failed." }

Write-Host "Environment ready: $venvPath" -ForegroundColor Green
Write-Host "Run the complete experiment with: .\run_pipeline.ps1" -ForegroundColor Green

