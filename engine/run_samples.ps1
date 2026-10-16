# Run t1..t7 at native size and overwrite each sample's nr.png.
# Example: .\engine\run_samples.ps1 -Samples t1,t7
[CmdletBinding()]
param(
    [string]$Weights,
    [string]$SamplesDir,
    [string[]]$Samples = @('t1', 't2', 't3', 't4', 't5', 't6', 't7'),
    [string]$Engine,
    [double]$Style = 2,
    [double]$Tone = 1,
    [double]$Structure = 1,
    [double]$Skin = -1,
    [switch]$NoAutomask,
    [uint32]$Seed = 12345,
    [ValidateRange(1, 256)][int]$Threads = 8,
    [ValidateRange(0, 255)][int]$Device = 0
)
$ErrorActionPreference = 'Stop'
# Resolve script-relative defaults after binding (Windows PowerShell 5.1).
if (-not $Weights) { $Weights = Join-Path $PSScriptRoot '../weights/student_v43' }
if (-not $SamplesDir) { $SamplesDir = Join-Path $PSScriptRoot '../teacher/samples' }
if (-not $Engine) {
    foreach ($candidate in @('build-cuda/gpu_engine.exe', 'build-cuda/Release/gpu_engine.exe',
                            'build/Release/cpu_engine.exe', 'build/cpu_engine.exe')) {
        $path = Join-Path $PSScriptRoot $candidate
        if (Test-Path -LiteralPath $path -PathType Leaf) { $Engine = $path; break }
    }
}
if (-not $Engine -or -not (Test-Path -LiteralPath $Engine -PathType Leaf)) {
    throw 'No engine found. Build the C++ engine first or pass -Engine FILE.'
}
if (-not (Test-Path -LiteralPath (Join-Path $Weights 'manifest.json') -PathType Leaf)) {
    throw "Missing student package: $Weights"
}
$Engine = (Resolve-Path -LiteralPath $Engine).Path
$Weights = (Resolve-Path -LiteralPath $Weights).Path
$inputs = @()
foreach ($sample in $Samples) {
    if ($sample -notmatch '^t[1-9][0-9]*$') { throw "Invalid sample name: $sample" }
    $folder = Join-Path $SamplesDir $sample
    if (-not (Test-Path -LiteralPath $folder -PathType Container)) { throw "Missing sample directory: $folder" }
    $sources = @(Get-ChildItem -LiteralPath $folder -File | Where-Object {
        $_.BaseName -eq $sample -and $_.Extension.ToLowerInvariant() -in @('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    })
    if ($sources.Count -ne 1) { throw "Expected one original image in $folder; found $($sources.Count)." }
    $inputs += $sources[0]
}
$culture = [Globalization.CultureInfo]::InvariantCulture
$controls = @('--style', $Style.ToString('R', $culture), '--tone', $Tone.ToString('R', $culture),
              '--structure', $Structure.ToString('R', $culture), '--skin', $Skin.ToString('R', $culture),
              '--seed', "$Seed", '--threads', "$Threads", '--device', "$Device")
if ($NoAutomask) { $controls += '--no-automask' } else { $controls += '--automask' }
$savedPath = $env:PATH
$savedInt8 = $env:NR_INT8
$savedProfile = $env:NR_PROFILE
try {
    if ($env:CUDA_PATH -and (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin'))) {
        $env:PATH = (Join-Path $env:CUDA_PATH 'bin') + [IO.Path]::PathSeparator + $env:PATH
    }
    # These runs use FP32; presence of NR_INT8=0 would also enable INT8.
    $env:NR_INT8 = $null
    $env:NR_PROFILE = $null
    Write-Host "Engine: $Engine"
    foreach ($source in $inputs) {
        $output = Join-Path $source.DirectoryName 'nr.png'
        Write-Host "== $($source.Directory.Name): native size -> $output =="
        & $Engine $source.FullName --weights $Weights @controls -o $output
        if ($LASTEXITCODE -ne 0) { throw "Engine failed for $($source.FullName) (exit $LASTEXITCODE)." }
    }
    Write-Host "Done: $($inputs.Count) images; nr.png overwritten without backups."
} finally {
    $env:PATH = $savedPath
    $env:NR_INT8 = $savedInt8
    $env:NR_PROFILE = $savedProfile
}
