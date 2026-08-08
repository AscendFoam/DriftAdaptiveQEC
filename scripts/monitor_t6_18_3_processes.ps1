param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$TargetProcessIdCsv = '',
    [int]$PollSeconds = 5
)

$ErrorActionPreference = 'Stop'
$resolvedRunDir = (Resolve-Path -LiteralPath $RunDir).Path
$output = Join-Path $resolvedRunDir 'process_memory_samples.csv'
'timestamp,pid,working_set_bytes,peak_working_set_bytes,cpu_seconds' | Set-Content -LiteralPath $output -Encoding ASCII
$emptyPolls = 0
$TargetProcessIds = if ($TargetProcessIdCsv) { @($TargetProcessIdCsv.Split(',') | ForEach-Object { [int]$_ }) } else { @() }

while ($emptyPolls -lt 3) {
    if ($TargetProcessIds.Count -gt 0) {
        $candidates = @($TargetProcessIds | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    } else {
        $candidates = @(Get-CimInstance Win32_Process -Filter "Name='julia.exe'" |
            Where-Object { $_.CommandLine -and $_.CommandLine.Contains($resolvedRunDir) } |
            ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue })
    }
    if ($candidates.Count -eq 0) {
        $emptyPolls += 1
    } else {
        $emptyPolls = 0
        $timestamp = [DateTimeOffset]::Now.ToString('o')
        foreach ($process in $candidates) {
            '{0},{1},{2},{3},{4}' -f $timestamp, $process.Id, $process.WorkingSet64, $process.PeakWorkingSet64, $process.CPU |
                Add-Content -LiteralPath $output -Encoding ASCII
        }
    }
    Start-Sleep -Seconds $PollSeconds
}
