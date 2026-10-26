param(
    [ValidateSet('representative','tuning','ablation','proposal')][string]$Grid = 'representative',
    [string]$BuildDir = 'build-gpu',
    [ValidateSet('main','all','naive','tiled','parallel','optimized')][string]$Variants = 'main',
    [ValidateSet('default','study')][string]$Configs = 'default',
    [ValidateSet(8,16,32)][int]$Tile = 16,
    [ValidateSet(128,256)][int]$SoftmaxThreads = 256,
    [ValidateRange(5,10)][int]$Warmup = 5,
    [ValidateRange(10,30)][int]$Repeats = 20,
    [string]$Architecture = '120'
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path $PSScriptRoot -Parent
Push-Location $repoPath
try {
    if (-not $PSBoundParameters.ContainsKey('Variants')) {
        if ($Grid -eq 'tuning') { $Variants = 'optimized' }
        if ($Grid -eq 'ablation') { $Variants = 'all' }
    }
    $runId = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0,6)
    $resultPath = Join-Path $repoPath "results/gpu-benchmarks/$runId"
    New-Item -ItemType Directory -Path $resultPath | Out-Null
    function Run-Native([string]$Program, [string[]]$Arguments, [string]$Log) {
        $null = Get-Command $Program -ErrorAction Stop
        $savedPreference = $ErrorActionPreference
        try {
            # Capture native diagnostics
            $ErrorActionPreference = 'Continue'
            & $Program @Arguments 2>&1 | Tee-Object -FilePath $Log
            $code = $LASTEXITCODE
        } finally { $ErrorActionPreference = $savedPreference }
        Add-Content -LiteralPath (Join-Path $resultPath 'exit-codes.txt') -Value "$Program : $code"
        if ($code -ne 0) { throw "$Program failed with exit code $code; see $Log" }
    }
    Run-Native 'git' @('rev-parse','HEAD') (Join-Path $resultPath 'source-commit.txt')
    Run-Native 'git' @('diff','--no-ext-diff') (Join-Path $resultPath 'source-diff.txt')
    Run-Native 'git' @('status','--short') (Join-Path $resultPath 'worktree.txt')
    Run-Native 'nvidia-smi' @() (Join-Path $resultPath 'nvidia-smi.txt')
    Run-Native 'nvcc' @('--version') (Join-Path $resultPath 'nvcc.txt')
    Run-Native 'cmake' @('--version') (Join-Path $resultPath 'cmake-version.txt')
    Run-Native 'cmake' @('-S','.', '-B',$BuildDir,'-G','Visual Studio 17 2022','-A','x64',
        '-DPART_A_ENABLE_CUDA=ON',"-DCMAKE_CUDA_ARCHITECTURES=$Architecture",'-DPART_A_PROFILE_INFO=OFF') (Join-Path $resultPath 'configure.log')
    Run-Native 'cmake' @('--build',$BuildDir,'--config','Release') (Join-Path $resultPath 'build.log')
    Run-Native 'nvidia-smi' @('--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu','--format=csv') (Join-Path $resultPath 'gpu-state-before.csv')
    $exe = Join-Path $BuildDir 'benchmarks/Release/attention_benchmark.exe'
    $benchmarkArgs = @('--grid',$Grid,'--variants',$Variants,'--configs',$Configs,
        '--warmup',"$Warmup",'--repeats',"$Repeats",'--output',(Join-Path $resultPath 'benchmark'))
    if ($Configs -eq 'default') { $benchmarkArgs += @('--tile',"$Tile",'--softmax-threads',"$SoftmaxThreads") }
    Run-Native $exe $benchmarkArgs (Join-Path $resultPath 'benchmark.log')
    Run-Native 'nvidia-smi' @('--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu','--format=csv') (Join-Path $resultPath 'gpu-state-after.csv')
    Set-Content -LiteralPath (Join-Path $resultPath 'status.txt') -Value "PASS: build and benchmark completed"
    Write-Output "Results: $resultPath"
} catch {
    if ($resultPath) { Set-Content -LiteralPath (Join-Path $resultPath 'status.txt') -Value "INCOMPLETE: $($_.Exception.Message)" }
    throw
} finally { Pop-Location }
