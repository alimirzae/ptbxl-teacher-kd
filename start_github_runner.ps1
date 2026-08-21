param([string]$RunnerDirectory = "C:\ptbxl\actions-runner2")

$run = Join-Path $RunnerDirectory 'run.cmd'
if (-not (Test-Path -LiteralPath $run)) { throw "Runner not configured: $run" }
Start-Process -FilePath $run -WorkingDirectory $RunnerDirectory -WindowStyle Hidden
Write-Host 'GitHub Actions runner started in the background.'
