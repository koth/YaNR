# 一键基准(openspec 8.5,Windows win-x64):lane 构造 / 前向 / composite 分段计时,
# 30 次取中位。验收口径见 8.7(报告 CPU 型号/线程数)。
#
#   .\bench.ps1 -Idx student.idx -Feat features.bin -Proxy proxy.f32 -Size 320 [-Int8]
#
# 先构建(MSVC):cmake -B build -S . && cmake --build build --config Release
param(
    [Parameter(Mandatory = $true)][string]$Idx,
    [Parameter(Mandatory = $true)][string]$Feat,
    [Parameter(Mandatory = $true)][string]$Proxy,
    [Parameter(Mandatory = $true)][int]$Size,
    [switch]$Int8
)
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "8" }
if ($Int8) { $env:NR_INT8 = "1" }
$bin = Join-Path $PSScriptRoot "build\Release"
if (-not (Test-Path "$bin\cpu_engine.exe")) { $bin = Join-Path $PSScriptRoot "build" }
$tmp = $env:TEMP

Write-Host "== lane 构造 =="
& "$bin\cpu_lanes.exe" --proxy $Proxy --out "$tmp\bench_lanes.bin" --vw $Size --vh $Size `
    --seed 7 --bench 30
Write-Host "== 前向 =="
& "$bin\cpu_engine.exe" --idx $Idx --features $Feat --bench --repeats 30
Write-Host "== composite + PNG =="
& "$bin\cpu_engine.exe" --idx $Idx --features $Feat --out "$tmp\bench_head.bin" `
    --proxy $Proxy --png "$tmp\bench.png" --bench --repeats 30 | Select-String composite
Write-Host ""
Write-Host "CPU: $((Get-CimInstance Win32_Processor).Name)  OMP_NUM_THREADS=$($env:OMP_NUM_THREADS)"
