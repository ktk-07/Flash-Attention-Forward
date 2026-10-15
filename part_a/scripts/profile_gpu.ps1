param(
    [ValidateRange(1,4096)][int]$N = 256,
    [ValidateSet('dense','causal')][string]$Mode = 'dense',
    [ValidateSet('naive','tiled','parallel','optimized')][string]$Variant = 'optimized',
    [ValidateSet(8,16,32)][int]$Tile = 16,
    [ValidateSet(128,256)][int]$SoftmaxThreads = 256,
    [string]$Architecture = '120',
    [string]$Ncu = 'ncu'
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path $PSScriptRoot -Parent
Push-Location $repoPath
try {
    $runId = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0,6)
    $resultPath = Join-Path $repoPath "results/profiles/$runId"
    New-Item -ItemType Directory -Path $resultPath | Out-Null
    function Run-Native([string]$Program,[string[]]$Arguments,[string]$Log) {
        $null = Get-Command $Program -ErrorAction Stop
        $savedPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $Program @Arguments 2>&1 | Tee-Object -FilePath $Log
            $code = $LASTEXITCODE
        } finally { $ErrorActionPreference = $savedPreference }
        Add-Content -LiteralPath (Join-Path $resultPath 'exit-codes.txt') -Value "$Program : $code"
        if ($code -ne 0) { throw "$Program failed with exit code $code" }
    }
    Run-Native $Ncu @('--version') (Join-Path $resultPath 'ncu-version.txt')
    Run-Native 'git' @('rev-parse','HEAD') (Join-Path $resultPath 'source-commit.txt')
    Run-Native 'git' @('diff','--no-ext-diff') (Join-Path $resultPath 'source-diff.txt')
    Run-Native 'cmake' @('-S','.', '-B','build-gpu-profile','-G','Visual Studio 17 2022','-A','x64',
        '-DPART_A_ENABLE_CUDA=ON',"-DCMAKE_CUDA_ARCHITECTURES=$Architecture",'-DPART_A_PROFILE_INFO=ON') (Join-Path $resultPath 'configure.log')
    Run-Native 'cmake' @('--build','build-gpu-profile','--config','Release') (Join-Path $resultPath 'build.log')
    $exe = Join-Path $repoPath 'build-gpu-profile/benchmarks/Release/attention_benchmark.exe'
    $reportBase = Join-Path $resultPath 'profile'
    # Skip validation and warmup launches
    $skip = 3 * (4 + 1 + 5)
    Run-Native $Ncu @('--set','full','--launch-skip',"$skip",'--launch-count','3','--export',$reportBase,
        $exe,'--shape','4','8',"$N",'64','--mode',$Mode,'--variants',$Variant,
        '--tile',"$Tile",'--softmax-threads',"$SoftmaxThreads",'--warmup','5','--profile-only',
        '--output',(Join-Path $resultPath 'validation')) (Join-Path $resultPath 'profile.log')
    $reports = @(Get-ChildItem -LiteralPath $resultPath -File | Where-Object { $_.Name -in @('profile.ncu-rep','profile.ncu-repz') })
    if ($reports.Count -ne 1) { throw 'Expected one Nsight report' }
    Run-Native $Ncu @('--import',$reports[0].FullName,'--csv','--page','raw') (Join-Path $resultPath 'metrics.csv')
    Set-Content -LiteralPath (Join-Path $resultPath 'status.txt') -Value 'PASS: profile captured; inspect three kernels and requested launch dimensions'
    Write-Output "Results: $resultPath"
} catch {
    if ($resultPath) { Set-Content -LiteralPath (Join-Path $resultPath 'status.txt') -Value "INCOMPLETE: $($_.Exception.Message)" }
    throw
} finally { Pop-Location }
