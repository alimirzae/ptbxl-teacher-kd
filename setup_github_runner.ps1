param(
    [Parameter(Mandatory=$true)][string]$Repository,
    [string]$RunnerName = "$env:COMPUTERNAME-ptbxl",
    [string]$RunnerDirectory = "C:\ptbxl\actions-runner2"
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI (gh) is required.'
}

$repo = gh repo view $Repository --json nameWithOwner,url | ConvertFrom-Json
$downloads = gh api "repos/$($repo.nameWithOwner)/actions/runners/downloads" | ConvertFrom-Json
$download = $downloads |
    Where-Object { $_.os -eq 'win' -and $_.architecture -eq 'x64' } |
    Select-Object -First 1
if (-not $download) {
    $asset = gh api 'repos/actions/runner/releases/latest' |
        ConvertFrom-Json |
        Select-Object -ExpandProperty assets |
        Where-Object { $_.name -match '^actions-runner-win-x64-.*\.zip$' } |
        Select-Object -First 1
    if (-not $asset) { throw 'No official Windows x64 Actions runner package was found.' }
    $download = [pscustomobject]@{ filename = $asset.name; download_url = $asset.browser_download_url }
}

New-Item -ItemType Directory -Path $RunnerDirectory -Force | Out-Null
$archive = Join-Path $env:TEMP "actions-runner-$($download.filename)"
Invoke-WebRequest -Uri $download.download_url -OutFile $archive
Expand-Archive -LiteralPath $archive -DestinationPath $RunnerDirectory -Force

$token = gh api --method POST "repos/$($repo.nameWithOwner)/actions/runners/registration-token" --jq .token
Push-Location $RunnerDirectory
try {
    & .\config.cmd --unattended --replace --url $repo.url --token $token --name $RunnerName --labels ptbxl-local --work _work
    if ($LASTEXITCODE -ne 0) { throw 'Runner configuration failed.' }
    Write-Host "Runner configured at $RunnerDirectory"
    Write-Host 'Start it with: .\run.cmd'
} finally {
    Pop-Location
}
