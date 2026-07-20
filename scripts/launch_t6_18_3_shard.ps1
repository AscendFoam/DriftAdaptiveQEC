param(
    [Parameter(Mandatory = $true)][int]$SeedIndexStart,
    [Parameter(Mandatory = $true)][int]$SeedIndexEnd,
    [Parameter(Mandatory = $true)][string]$Output
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$julia = (Resolve-Path -LiteralPath (Join-Path $repo '.tools\julia-1.10.11\bin\julia.exe')).Path
$project = (Resolve-Path -LiteralPath (Join-Path $repo 'configs\literature\t6_18_2_julia_env')).Path
$script = (Resolve-Path -LiteralPath (Join-Path $repo 'scripts\run_multimode_posterior_weighted_cpd.jl')).Path
$env:JULIA_DEPOT_PATH = (Resolve-Path -LiteralPath (Join-Path $repo '.tools\julia_depot_t6182')).Path

& $julia "--project=$project" '--compiled-modules=no' $script `
    '--mode' 'formal' `
    '--seed-index-start' "$SeedIndexStart" `
    '--seed-index-end' "$SeedIndexEnd" `
    '--output' $Output

exit $LASTEXITCODE
