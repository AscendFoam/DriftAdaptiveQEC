[CmdletBinding()]
param(
    [string]$RunDir = '',
    [string]$Python = "C:\ProgramData\anaconda3\envs\DLEnv\python.exe",
    [string]$RepoRoot = "D:\Codes\Quantum\DriftAdaptiveQEC",
    [string]$TargetGpuUuid = '',
    [switch]$ArtifactResume,
    [ValidateRange(1, 336)]
    [int]$TotalDeadlineHours = 36,
    [ValidateRange(1, 60)]
    [int]$PollSeconds = 5,
    [switch]$StaticSelfTest
)

# T9.1.3 production supervisor.  There is deliberately no attach-style
# -Resume mode: every supervisor invocation owns a brand-new RunDir.  The
# -ArtifactResume only authorizes reuse and re-audit of canonical retained
# agent artifacts.  It is not automatic finalization-lock recovery: any
# retained finalize.lock requires an explicit audited operator intervention.

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

# These controls are inherited by every Python probe and long-lived worker.
# They must exist before Python imports torch/CUDA; Python then enables and
# validates the matching deterministic backend flags.
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:NVIDIA_TF32_OVERRIDE = '0'
$env:TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = '0'
$env:PYTHONHASHSEED = '0'

function Get-UtcIso {
    return [DateTime]::UtcNow.ToString("o")
}

function Get-CanonicalPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$BasePath
    )
    $candidate = $Path
    if (-not [IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $BasePath $candidate
    }
    $full = [IO.Path]::GetFullPath($candidate)
    $root = [IO.Path]::GetPathRoot($full)
    if ($full.Length -gt $root.Length) {
        return $full.TrimEnd([char[]]@([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar))
    }
    return $full
}

function Test-PathsOverlap {
    param(
        [Parameter(Mandatory = $true)][string]$First,
        [Parameter(Mandatory = $true)][string]$Second
    )
    $comparison = [StringComparison]::OrdinalIgnoreCase
    if ([string]::Equals($First, $Second, $comparison)) {
        return $true
    }
    $separator = [IO.Path]::DirectorySeparatorChar
    $firstPrefix = $First.TrimEnd($separator) + $separator
    $secondPrefix = $Second.TrimEnd($separator) + $separator
    return $First.StartsWith($secondPrefix, $comparison) -or $Second.StartsWith($firstPrefix, $comparison)
}

function ConvertTo-WindowsCommandLineArgument {
    param([AllowEmptyString()][Parameter(Mandatory = $true)][string]$Argument)
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append([char]34)
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq [char]92) {
            $backslashes += 1
            continue
        }
        if ($character -eq [char]34) {
            if ($backslashes -gt 0) {
                [void]$builder.Append(('\' * ($backslashes * 2)))
            }
            [void]$builder.Append('\')
            [void]$builder.Append([char]34)
        }
        else {
            if ($backslashes -gt 0) {
                [void]$builder.Append(('\' * $backslashes))
            }
            [void]$builder.Append($character)
        }
        $backslashes = 0
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * ($backslashes * 2)))
    }
    [void]$builder.Append([char]34)
    return $builder.ToString()
}

function Join-NativeArgumentList {
    param([AllowEmptyString()][Parameter(Mandatory = $true)][string[]]$ArgumentList)
    $quoted = @(
        foreach ($argument in $ArgumentList) {
            ConvertTo-WindowsCommandLineArgument -Argument $argument
        }
    )
    return ($quoted -join ' ')
}

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [AllowEmptyString()][Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 300
    )
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = Join-NativeArgumentList -ArgumentList $ArgumentList
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "native command failed to start: $FilePath"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill()
        $process.WaitForExit()
        $timedOutStdout = $stdoutTask.Result
        $timedOutStderr = $stderrTask.Result
        throw "native command timed out after $TimeoutSeconds seconds: $FilePath $($startInfo.Arguments); stdout=$timedOutStdout stderr=$timedOutStderr"
    }
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    return [pscustomobject]@{
        FilePath = $FilePath
        ArgumentLine = $startInfo.Arguments
        ExitCode = [int]$process.ExitCode
        Stdout = [string]$stdout
        Stderr = [string]$stderr
    }
}

function Assert-NativeSuccess {
    param(
        [Parameter(Mandatory = $true)]$Result,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ([int]$Result.ExitCode -ne 0) {
        $stderr = ([string]$Result.Stderr).Trim()
        throw "$Label failed with native exit code $($Result.ExitCode): $stderr"
    }
}

function Initialize-KillOnCloseJob {
    if (-not ('T913.NativeJob' -as [type])) {
        $jobType = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace T913 {
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    public static class NativeJob {
        public const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
        public const int JobObjectExtendedLimitInformation = 9;

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr CreateJobObject(IntPtr securityAttributes, string name);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetInformationJobObject(
            IntPtr job,
            int informationClass,
            ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION information,
            uint informationLength
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool CloseHandle(IntPtr handle);

        public static IntPtr CreateKillOnCloseJob() {
            IntPtr job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateJobObject failed");
            }
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION information =
                new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            information.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            uint length = (uint)Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
            if (!SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    ref information,
                    length)) {
                int error = Marshal.GetLastWin32Error();
                CloseHandle(job);
                throw new Win32Exception(error, "SetInformationJobObject failed");
            }
            return job;
        }

        public static void AssignOrThrow(IntPtr job, IntPtr process) {
            if (job == IntPtr.Zero || process == IntPtr.Zero) {
                throw new ArgumentException("job and process handles must be nonzero");
            }
            if (!AssignProcessToJobObject(job, process)) {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "AssignProcessToJobObject failed"
                );
            }
        }
    }
}
'@
        Add-Type -TypeDefinition $jobType -Language CSharp -ErrorAction Stop
    }
    $handle = [T913.NativeJob]::CreateKillOnCloseJob()
    if ($handle -eq [IntPtr]::Zero) {
        throw 'kill-on-close Job Object returned a null handle'
    }
    return $handle
}

function Add-ProcessToKillOnCloseJob {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$Role
    )
    if ($script:KillOnCloseJobHandle -eq [IntPtr]::Zero) {
        try {
            if (-not $Process.HasExited) {
                $Process.Kill()
                $Process.WaitForExit()
            }
        }
        finally {
            Close-HiddenChildIo -Process $Process
        }
        throw "$Role child refused launch because the kill-on-close Job Object is absent"
    }
    try {
        [T913.NativeJob]::AssignOrThrow(
            $script:KillOnCloseJobHandle,
            $Process.Handle
        )
    }
    catch {
        try {
            if (-not $Process.HasExited) {
                $Process.Kill()
                $Process.WaitForExit()
            }
        }
        finally {
            Close-HiddenChildIo -Process $Process
        }
        throw "$Role child Job Object binding failed closed: $($_.Exception.Message)"
    }
}

function Close-KillOnCloseJob {
    if ($null -ne $script:KillOnCloseJobHandle -and $script:KillOnCloseJobHandle -ne [IntPtr]::Zero) {
        $handle = $script:KillOnCloseJobHandle
        $script:KillOnCloseJobHandle = [IntPtr]::Zero
        if (-not [T913.NativeJob]::CloseHandle($handle)) {
            throw "CloseHandle for kill-on-close Job Object failed with Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
        }
    }
}

function Assert-SyntheticJobBindingResult {
    param([Parameter(Mandatory = $true)][bool]$Succeeded)
    if (-not $Succeeded) {
        throw 'synthetic Job Object assignment failed closed'
    }
}

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = New-Object System.IO.FileStream(
        $Path,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::ReadWrite
    )
    try {
        $algorithm = [Security.Cryptography.SHA256]::Create()
        try {
            $digest = $algorithm.ComputeHash($stream)
        }
        finally {
            $algorithm.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
    return (($digest | ForEach-Object { $_.ToString('x2') }) -join '')
}

function Get-StringSha256Hex {
    param([AllowEmptyString()][Parameter(Mandatory = $true)][string]$Text)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $algorithm.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text))
    }
    finally {
        $algorithm.Dispose()
    }
    return (($digest | ForEach-Object { $_.ToString('x2') }) -join '')
}

function Write-Utf8NoBomAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        [IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $temporary = Join-Path $parent (".{0}.{1}.{2}.tmp" -f ([IO.Path]::GetFileName($Path)), $PID, [Guid]::NewGuid().ToString('N'))
    $backup = Join-Path $parent (".{0}.{1}.{2}.replace-backup" -f ([IO.Path]::GetFileName($Path)), $PID, [Guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllText($temporary, $Text, (New-Object System.Text.UTF8Encoding($false)))
        if ([IO.File]::Exists($Path)) {
            [IO.File]::Replace($temporary, $Path, $backup, $true)
            [IO.File]::Delete($backup)
        }
        else {
            [IO.File]::Move($temporary, $Path)
        }
    }
    finally {
        if ([IO.File]::Exists($temporary)) {
            [IO.File]::Delete($temporary)
        }
        if ([IO.File]::Exists($backup)) {
            [IO.File]::Delete($backup)
        }
    }
}

function Write-JsonAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    $json = $Value | ConvertTo-Json -Depth 24
    Write-Utf8NoBomAtomic -Path $Path -Text ($json + [Environment]::NewLine)
}

function Get-FileEvidence {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return [ordered]@{ path = $Path; exists = $false; bytes = $null; sha256 = $null; last_write_utc = $null }
    }
    $item = Get-Item -LiteralPath $Path
    return [ordered]@{
        path = $Path
        exists = $true
        bytes = [long]$item.Length
        sha256 = Get-Sha256Hex -Path $Path
        last_write_utc = $item.LastWriteTimeUtc.ToString('o')
    }
}

function Invoke-ProductionProbe {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    # The config path is passed through sys.argv.  It is never interpolated into
    # source code, so spaces, quotes, and backslashes retain their argv meaning.
    $probeCode = @'
import json
import socket
import sys
from pathlib import Path
from cnn_fpga.benchmark import puviani_paper_constrained_artifacts as m

config_path = Path(sys.argv[1]).resolve()
config = m._load_json(config_path)
m._validate_config(config, production=True)
m._configure_production_determinism()
runtime_signature, runtime_signature_sha256 = m._validate_runtime_for_config(
    config, production=True
)
m._verify_parent_protocol(config)
module_path = Path(m.__file__).resolve()
print(json.dumps({
    "config_path": str(config_path),
    "config_sha256": m._canonical_sha256(config),
    "implementation_sha256": m.implementation_sha256(),
    "production_config_sha256": m.PRODUCTION_CONFIG_SHA256,
    "runtime_signature_sha256": runtime_signature_sha256,
    "python_executable": str(Path(sys.executable).resolve()),
    "module_path": str(module_path),
    "repo_root": str(module_path.parents[2]),
    "hostname": socket.gethostname(),
}, sort_keys=True))
'@
    $result = Invoke-NativeCapture -FilePath $PythonPath -ArgumentList @('-c', $probeCode, $ConfigPath) -WorkingDirectory $WorkingDirectory
    Assert-NativeSuccess -Result $result -Label 'T9.1.3 production Python preflight'
    $text = ([string]$result.Stdout).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw 'T9.1.3 production Python preflight returned no JSON'
    }
    try {
        $probe = $text | ConvertFrom-Json
    }
    catch {
        throw "T9.1.3 production Python preflight returned invalid JSON: $text"
    }
    foreach ($name in @('config_path', 'config_sha256', 'implementation_sha256', 'production_config_sha256', 'runtime_signature_sha256', 'python_executable', 'module_path', 'repo_root', 'hostname')) {
        if ($null -eq $probe.PSObject.Properties[$name]) {
            throw "T9.1.3 production Python preflight omitted $name"
        }
    }
    if ([string]$probe.config_sha256 -notmatch '^[0-9a-f]{64}$' -or [string]$probe.implementation_sha256 -notmatch '^[0-9a-f]{64}$' -or [string]$probe.runtime_signature_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'T9.1.3 production Python preflight returned a malformed hash'
    }
    if ([string]$probe.config_sha256 -ne [string]$probe.production_config_sha256) {
        throw 'production config hash differs from the immutable in-module hash'
    }
    return $probe
}

function Assert-ProbeIdentity {
    param(
        [Parameter(Mandatory = $true)]$Expected,
        [Parameter(Mandatory = $true)]$Actual,
        [Parameter(Mandatory = $true)][string]$Stage
    )
    foreach ($field in @('config_sha256', 'implementation_sha256', 'production_config_sha256', 'runtime_signature_sha256')) {
        if ([string]$Expected.$field -ne [string]$Actual.$field) {
            throw "$Stage changed $field (expected $($Expected.$field), observed $($Actual.$field))"
        }
    }
}

function Test-StrictInvalidatedFinalizationCrashMarker {
    param(
        [Parameter(Mandatory = $true)]$Marker,
        [Parameter(Mandatory = $true)][string]$ExpectedConfigSha256
    )
    $required = @('schema_version', 'task_id', 'status', 'evidence_grade', 'started_at_utc', 'config_sha256', 'valid_pass_seal')
    $actual = @($Marker.PSObject.Properties.Name | Sort-Object)
    $expected = @($required | Sort-Object)
    if (($actual -join "`n") -ne ($expected -join "`n")) {
        return $false
    }
    $timestamp = [DateTime]::MinValue
    return (
        [string]$Marker.schema_version -eq 't9.1.3-puviani-paper-constrained-artifacts-v1' -and
        [string]$Marker.task_id -eq 'T9.1.3' -and
        [string]$Marker.status -eq 'INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL' -and
        [string]$Marker.evidence_grade -eq 'PAPER_CONSTRAINED_REIMPLEMENTATION' -and
        [string]$Marker.config_sha256 -eq $ExpectedConfigSha256 -and
        $Marker.valid_pass_seal -eq $false -and
        [DateTime]::TryParse([string]$Marker.started_at_utc, [ref]$timestamp)
    )
}

function Assert-ArtifactResumeHasNoFinalizeLock {
    param([AllowEmptyCollection()][Parameter(Mandatory = $true)][object[]]$LockAudit)
    $retained = @(
        $LockAudit | Where-Object {
            [IO.Path]::GetFileName([string]$_.lock_path) -eq 'finalize.lock'
        }
    )
    if ($retained.Count -gt 0) {
        throw 'MANUAL_OPERATOR_RECOVERY_REQUIRED: a retained finalize.lock is fail-closed; this supervisor never deletes, moves, or hands it to an automatic finalizer recovery path'
    }
}

function Get-NumericMedian {
    param([Parameter(Mandatory = $true)][double[]]$Values)
    if ($Values.Count -eq 0) {
        throw 'median requires at least one value'
    }
    $ordered = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($ordered.Count / 2.0)
    if (($ordered.Count % 2) -eq 1) {
        return [double]$ordered[$middle]
    }
    return ([double]$ordered[$middle - 1] + [double]$ordered[$middle]) / 2.0
}

function ConvertFrom-NvidiaSmiCsvSample {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][int]$Sequence,
        [Parameter(Mandatory = $true)][string]$CapturedAtUtc
    )
    if ([string]::IsNullOrWhiteSpace($Text)) {
        throw "nvidia-smi sample $Sequence returned no device rows"
    }
    $headers = @('index', 'uuid', 'name', 'memory_total_mib', 'memory_used_mib', 'memory_free_mib', 'utilization_percent')
    Add-Type -AssemblyName Microsoft.VisualBasic -ErrorAction Stop
    $reader = New-Object System.IO.StringReader($Text)
    $parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser($reader)
    $parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
    $parser.SetDelimiters(',')
    $parser.HasFieldsEnclosedInQuotes = $true
    $csvRows = @()
    try {
        while (-not $parser.EndOfData) {
            $fields = $parser.ReadFields()
            if ($null -eq $fields -or $fields.Count -ne $headers.Count) {
                throw "nvidia-smi sample $Sequence did not contain exactly seven CSV fields"
            }
            $row = [ordered]@{}
            for ($fieldIndex = 0; $fieldIndex -lt $headers.Count; $fieldIndex += 1) {
                $row[$headers[$fieldIndex]] = [string]$fields[$fieldIndex]
            }
            $csvRows += $row
        }
    }
    finally {
        $parser.Dispose()
        $reader.Dispose()
    }
    if ($csvRows.Count -eq 0) {
        throw "nvidia-smi sample $Sequence returned no parsed device rows"
    }
    $parsed = @()
    foreach ($csvRow in $csvRows) {
        $index = -1
        if (-not [int]::TryParse(([string]$csvRow.index).Trim(), [Globalization.NumberStyles]::Integer, [Globalization.CultureInfo]::InvariantCulture, [ref]$index) -or $index -lt 0) {
            throw "nvidia-smi sample $Sequence has an invalid device index"
        }
        $numeric = [ordered]@{}
        foreach ($field in @('memory_total_mib', 'memory_used_mib', 'memory_free_mib', 'utilization_percent')) {
            $value = 0.0
        if (-not [double]::TryParse(([string]$csvRow.$field).Trim(), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$value) -or [double]::IsNaN($value) -or [double]::IsInfinity($value)) {
                throw "nvidia-smi sample $Sequence has a non-numeric $field"
            }
            $numeric[$field] = [double]$value
        }
        $uuid = ([string]$csvRow.uuid).Trim()
        $name = ([string]$csvRow.name).Trim()
        if ([string]::IsNullOrWhiteSpace($uuid) -or [string]::IsNullOrWhiteSpace($name)) {
            throw "nvidia-smi sample $Sequence has an empty UUID or device name"
        }
        if ($numeric.memory_total_mib -le 0.0 -or $numeric.memory_used_mib -lt 0.0 -or $numeric.memory_free_mib -lt 0.0 -or $numeric.memory_used_mib -gt ($numeric.memory_total_mib + 1.0) -or $numeric.memory_free_mib -gt ($numeric.memory_total_mib + 1.0)) {
            throw "nvidia-smi sample $Sequence has impossible memory counters for $uuid"
        }
        if ($numeric.utilization_percent -lt 0.0 -or $numeric.utilization_percent -gt 100.0) {
            throw "nvidia-smi sample $Sequence has an impossible utilization for $uuid"
        }
        $parsed += [ordered]@{
            index = $index
            uuid = $uuid
            name = $name
            memory_total_mib = $numeric.memory_total_mib
            memory_used_mib = $numeric.memory_used_mib
            memory_free_mib = $numeric.memory_free_mib
            utilization_percent = $numeric.utilization_percent
        }
    }
    return [ordered]@{
        sequence = $Sequence
        captured_at_utc = $CapturedAtUtc
        rows = @($parsed)
    }
}

function Test-NvidiaLoadSamples {
    param(
        [AllowEmptyCollection()][Parameter(Mandatory = $true)][object[]]$Samples,
        [int]$ExpectedSampleCount = 5,
        [double]$MinimumFreeMemoryMiB = 4096.0,
        [double]$MaximumMedianUtilizationPercent = 15.0,
        [double]$MaximumPeakUtilizationPercent = 30.0,
        [string]$RequestedTargetUuid = '',
        [string]$CudaVisibleDevices = ''
    )
    $reasons = New-Object System.Collections.ArrayList
    if ($Samples.Count -ne $ExpectedSampleCount) {
        [void]$reasons.Add("expected $ExpectedSampleCount parsed samples, observed $($Samples.Count)")
    }
    $baselineRows = @()
    $baselineIdentity = $null
    $baselineDeviceDetails = $null
    for ($sampleIndex = 0; $sampleIndex -lt $Samples.Count; $sampleIndex += 1) {
        $sample = $Samples[$sampleIndex]
        if ($null -eq $sample -or $null -eq $sample.rows) {
            [void]$reasons.Add("sample $sampleIndex has no parsed device rows")
            continue
        }
        if ([int]$sample.sequence -ne $sampleIndex -or [string]::IsNullOrWhiteSpace([string]$sample.captured_at_utc)) {
            [void]$reasons.Add("sample $sampleIndex has a non-canonical sequence or missing capture timestamp")
        }
        $rows = @($sample.rows)
        if ($rows.Count -eq 0) {
            [void]$reasons.Add("sample $sampleIndex has zero devices")
            continue
        }
        $duplicateIndices = @($rows | Group-Object -Property index | Where-Object { $_.Count -ne 1 })
        $duplicateUuids = @($rows | Group-Object -Property uuid | Where-Object { $_.Count -ne 1 })
        if ($duplicateIndices.Count -gt 0 -or $duplicateUuids.Count -gt 0) {
            [void]$reasons.Add("sample $sampleIndex contains duplicate GPU indices or UUIDs")
        }
        $orderedRows = @($rows | Sort-Object { [int]$_.index })
        $identity = (($orderedRows | ForEach-Object { "$( [int]$_.index)|$([string]$_.uuid)" }) -join ';')
        $details = (($orderedRows | ForEach-Object { "$( [int]$_.index)|$([string]$_.uuid)|$([string]$_.name)|$([double]$_.memory_total_mib)" }) -join ';')
        if ($null -eq $baselineIdentity) {
            $baselineRows = @($orderedRows)
            $baselineIdentity = $identity
            $baselineDeviceDetails = $details
        }
        else {
            if ($identity -ne $baselineIdentity) {
                [void]$reasons.Add("sample $sampleIndex changed device count, index mapping, or UUID set")
            }
            if ($details -ne $baselineDeviceDetails) {
                [void]$reasons.Add("sample $sampleIndex changed device name or total-memory identity")
            }
        }
    }

    $targetRow = $null
    $selectionBasis = $null
    if ($baselineRows.Count -gt 0) {
        if (-not [string]::IsNullOrWhiteSpace($RequestedTargetUuid)) {
            $matches = @($baselineRows | Where-Object { [string]::Equals([string]$_.uuid, $RequestedTargetUuid.Trim(), [StringComparison]::OrdinalIgnoreCase) })
            if ($matches.Count -eq 1) {
                $targetRow = $matches[0]
                $selectionBasis = 'EXPLICIT_TARGET_GPU_UUID'
            }
            else {
                [void]$reasons.Add('explicit target GPU UUID did not identify exactly one sampled device')
            }
        }
        elseif (-not [string]::IsNullOrWhiteSpace($CudaVisibleDevices)) {
            $visibleToken = (($CudaVisibleDevices -split ',')[0]).Trim()
            if ($visibleToken -match '^GPU-') {
                $matches = @($baselineRows | Where-Object { [string]::Equals([string]$_.uuid, $visibleToken, [StringComparison]::OrdinalIgnoreCase) })
                if ($matches.Count -eq 1) {
                    $targetRow = $matches[0]
                    $selectionBasis = 'CUDA_VISIBLE_DEVICES_FIRST_UUID'
                }
                else {
                    [void]$reasons.Add('CUDA_VISIBLE_DEVICES UUID did not identify exactly one sampled device')
                }
            }
            elseif ($visibleToken -match '^\d+$') {
                $visibleIndex = [int]$visibleToken
                $matches = @($baselineRows | Where-Object { [int]$_.index -eq $visibleIndex })
                if ($matches.Count -eq 1) {
                    $targetRow = $matches[0]
                    $selectionBasis = 'CUDA_VISIBLE_DEVICES_FIRST_INDEX'
                }
                else {
                    [void]$reasons.Add('CUDA_VISIBLE_DEVICES index did not identify exactly one sampled device')
                }
            }
            else {
                [void]$reasons.Add('CUDA_VISIBLE_DEVICES target syntax is unsupported or disables CUDA')
            }
        }
        elseif ($baselineRows.Count -eq 1) {
            $targetRow = $baselineRows[0]
            $selectionBasis = 'SOLE_NVIDIA_DEVICE'
        }
        else {
            [void]$reasons.Add('multiple NVIDIA devices require -TargetGpuUuid or an unambiguous CUDA_VISIBLE_DEVICES first token')
        }
    }

    if ($null -ne $targetRow -and -not [string]::IsNullOrWhiteSpace($RequestedTargetUuid) -and -not [string]::IsNullOrWhiteSpace($CudaVisibleDevices)) {
        $visibleToken = (($CudaVisibleDevices -split ',')[0]).Trim()
        $visibleMatches = @()
        if ($visibleToken -match '^GPU-') {
            $visibleMatches = @($baselineRows | Where-Object { [string]::Equals([string]$_.uuid, $visibleToken, [StringComparison]::OrdinalIgnoreCase) })
        }
        elseif ($visibleToken -match '^\d+$') {
            $visibleIndex = [int]$visibleToken
            $visibleMatches = @($baselineRows | Where-Object { [int]$_.index -eq $visibleIndex })
        }
        if ($visibleMatches.Count -ne 1 -or -not [string]::Equals([string]$visibleMatches[0].uuid, [string]$targetRow.uuid, [StringComparison]::OrdinalIgnoreCase)) {
            [void]$reasons.Add('explicit target GPU UUID disagrees with the effective CUDA_VISIBLE_DEVICES first device')
        }
    }

    $targetFree = @()
    $targetUtilization = @()
    if ($null -ne $targetRow) {
        foreach ($sample in $Samples) {
            $matches = @($sample.rows | Where-Object { [string]::Equals([string]$_.uuid, [string]$targetRow.uuid, [StringComparison]::OrdinalIgnoreCase) })
            if ($matches.Count -ne 1) {
                [void]$reasons.Add("target GPU $($targetRow.uuid) was not present exactly once in every sample")
                continue
            }
            $targetFree += [double]$matches[0].memory_free_mib
            $targetUtilization += [double]$matches[0].utilization_percent
        }
    }

    $minimumObservedFree = $null
    $medianObservedUtilization = $null
    $maximumObservedUtilization = $null
    if ($targetFree.Count -gt 0) {
        $minimumObservedFree = [double](($targetFree | Measure-Object -Minimum).Minimum)
        if (@($targetFree | Where-Object { $_ -lt $MinimumFreeMemoryMiB }).Count -gt 0) {
            [void]$reasons.Add("target GPU free memory fell below $MinimumFreeMemoryMiB MiB")
        }
    }
    if ($targetUtilization.Count -gt 0) {
        $medianObservedUtilization = Get-NumericMedian -Values ([double[]]$targetUtilization)
        $maximumObservedUtilization = [double](($targetUtilization | Measure-Object -Maximum).Maximum)
        if ($medianObservedUtilization -gt $MaximumMedianUtilizationPercent) {
            [void]$reasons.Add("target GPU median utilization exceeded $MaximumMedianUtilizationPercent percent")
        }
        if ($maximumObservedUtilization -gt $MaximumPeakUtilizationPercent) {
            [void]$reasons.Add("target GPU peak utilization exceeded $MaximumPeakUtilizationPercent percent")
        }
    }
    if ($targetFree.Count -ne $ExpectedSampleCount -or $targetUtilization.Count -ne $ExpectedSampleCount) {
        [void]$reasons.Add('target GPU metric census does not match the required sample count')
    }

    $deviceSummaries = @()
    foreach ($device in $baselineRows) {
        $freeValues = @()
        $utilValues = @()
        foreach ($sample in $Samples) {
            $matches = @($sample.rows | Where-Object { [string]::Equals([string]$_.uuid, [string]$device.uuid, [StringComparison]::OrdinalIgnoreCase) })
            if ($matches.Count -eq 1) {
                $freeValues += [double]$matches[0].memory_free_mib
                $utilValues += [double]$matches[0].utilization_percent
            }
        }
        $deviceSummaries += [ordered]@{
            index = [int]$device.index
            uuid = [string]$device.uuid
            name = [string]$device.name
            metric_sample_count = $freeValues.Count
            minimum_free_memory_mib = $(if ($freeValues.Count -gt 0) { [double](($freeValues | Measure-Object -Minimum).Minimum) } else { $null })
            median_utilization_percent = $(if ($utilValues.Count -gt 0) { Get-NumericMedian -Values ([double[]]$utilValues) } else { $null })
            maximum_utilization_percent = $(if ($utilValues.Count -gt 0) { [double](($utilValues | Measure-Object -Maximum).Maximum) } else { $null })
        }
    }
    $uniqueReasons = @($reasons | Select-Object -Unique)
    return [ordered]@{
        passed = ($uniqueReasons.Count -eq 0)
        failure_reasons = @($uniqueReasons)
        thresholds = [ordered]@{
            expected_sample_count = $ExpectedSampleCount
            minimum_free_memory_mib_every_sample = $MinimumFreeMemoryMiB
            maximum_median_utilization_percent = $MaximumMedianUtilizationPercent
            maximum_peak_utilization_percent = $MaximumPeakUtilizationPercent
        }
        summary = [ordered]@{
            parsed_sample_count = $Samples.Count
            consistent_device_count = $baselineRows.Count
            device_identity_signature = $baselineIdentity
            target_selection_basis = $selectionBasis
            target_index = $(if ($null -ne $targetRow) { [int]$targetRow.index } else { $null })
            target_uuid = $(if ($null -ne $targetRow) { [string]$targetRow.uuid } else { $null })
            target_name = $(if ($null -ne $targetRow) { [string]$targetRow.name } else { $null })
            target_total_memory_mib = $(if ($null -ne $targetRow) { [double]$targetRow.memory_total_mib } else { $null })
            target_minimum_free_memory_mib = $minimumObservedFree
            target_median_utilization_percent = $medianObservedUtilization
            target_maximum_utilization_percent = $maximumObservedUtilization
            all_device_summaries = @($deviceSummaries)
        }
    }
}

function Invoke-NvidiaLoadGate {
    param(
        [string]$RequestedTargetUuid = '',
        [int]$SampleCount = 5,
        [int]$SampleIntervalSeconds = 2,
        [double]$MinimumFreeMemoryMiB = 4096.0,
        [double]$MaximumMedianUtilizationPercent = 15.0,
        [double]$MaximumPeakUtilizationPercent = 30.0,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    $queryArguments = @(
        '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'
    )
    $rawSamples = @()
    $parsedSamples = @()
    $commandError = $null
    for ($sequence = 0; $sequence -lt $SampleCount; $sequence += 1) {
        $capturedAt = Get-UtcIso
        try {
            $native = Invoke-NativeCapture -FilePath 'nvidia-smi.exe' -ArgumentList $queryArguments -WorkingDirectory $WorkingDirectory -TimeoutSeconds 30
            $raw = [ordered]@{
                sequence = $sequence
                captured_at_utc = $capturedAt
                completed_at_utc = Get-UtcIso
                command = 'nvidia-smi.exe'
                arguments = @($queryArguments)
                exit_code = [int]$native.ExitCode
                stdout = [string]$native.Stdout
                stderr = [string]$native.Stderr
                parse_error = $null
            }
            $rawSamples += $raw
            if ([int]$native.ExitCode -ne 0) {
                $commandError = "nvidia-smi sample $sequence exited $($native.ExitCode)"
                break
            }
            try {
                $parsedSamples += ConvertFrom-NvidiaSmiCsvSample -Text ([string]$native.Stdout) -Sequence $sequence -CapturedAtUtc $capturedAt
            }
            catch {
                $raw['parse_error'] = $_.Exception.Message
                $commandError = $_.Exception.Message
                break
            }
        }
        catch {
            $commandError = $_.Exception.Message
            $rawSamples += [ordered]@{
                sequence = $sequence
                captured_at_utc = $capturedAt
                completed_at_utc = Get-UtcIso
                command = 'nvidia-smi.exe'
                arguments = @($queryArguments)
                exit_code = $null
                stdout = $null
                stderr = $null
                parse_error = $commandError
            }
            break
        }
        if ($sequence -lt ($SampleCount - 1)) {
            Start-Sleep -Seconds $SampleIntervalSeconds
        }
    }
    $visibleDevices = [Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES', 'Process')
    $evaluation = Test-NvidiaLoadSamples -Samples ([object[]]$parsedSamples) -ExpectedSampleCount $SampleCount -MinimumFreeMemoryMiB $MinimumFreeMemoryMiB -MaximumMedianUtilizationPercent $MaximumMedianUtilizationPercent -MaximumPeakUtilizationPercent $MaximumPeakUtilizationPercent -RequestedTargetUuid $RequestedTargetUuid -CudaVisibleDevices ([string]$visibleDevices)
    $failureReasons = @($evaluation.failure_reasons)
    if (-not [string]::IsNullOrWhiteSpace($commandError)) {
        $failureReasons = @($failureReasons) + @("nvidia-smi acquisition or parsing failed: $commandError")
    }
    $failureReasons = @($failureReasons | Select-Object -Unique)
    return [ordered]@{
        schema_version = 't9.1.3-nvidia-load-gate-v1'
        passed = ($failureReasons.Count -eq 0 -and $evaluation.passed)
        failure_reasons = @($failureReasons)
        sampled_at_host = [Environment]::MachineName
        cuda_visible_devices = $(if ([string]::IsNullOrWhiteSpace($visibleDevices)) { $null } else { $visibleDevices })
        requested_target_gpu_uuid = $(if ([string]::IsNullOrWhiteSpace($RequestedTargetUuid)) { $null } else { $RequestedTargetUuid })
        sample_interval_seconds = $SampleIntervalSeconds
        thresholds = $evaluation.thresholds
        summary = $evaluation.summary
        parsed_samples = @($parsedSamples)
        raw_samples = @($rawSamples)
    }
}

function New-GpuLoadAttestation {
    param(
        [Parameter(Mandatory = $true)]$LoadGate,
        [ValidateSet('TRAINING_LAUNCH', 'FINALIZER_LAUNCH')]
        [Parameter(Mandatory = $true)][string]$Purpose,
        [Parameter(Mandatory = $true)]$Probe,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    if ($LoadGate.passed -ne $true -or @($LoadGate.parsed_samples).Count -ne 5 -or @($LoadGate.raw_samples).Count -ne 5) {
        throw "$Purpose cannot seal a non-PASS or incomplete NVIDIA load gate"
    }
    $target = $LoadGate.summary
    foreach ($field in @('target_index', 'target_uuid', 'target_name', 'target_total_memory_mib')) {
        if ($null -eq $target.$field) {
            throw "$Purpose load gate omitted target identity field $field"
        }
    }
    $issued = [DateTime]::UtcNow
    $body = [ordered]@{
        schema_version = 't9.1.3-gpu-load-attestation-v1'
        task_id = 'T9.1.3'
        purpose = $Purpose
        config_sha256 = [string]$Probe.config_sha256
        implementation_sha256 = [string]$Probe.implementation_sha256
        run_identity = [ordered]@{
            transaction_id = $script:TransactionId
            run_dir = $script:RunDirectory
            supervisor_pid = $PID
            supervisor_process_created_unix_ns = $script:SupervisorProcessCreatedUnixNs
            supervisor_hostname = [Environment]::MachineName
        }
        attestation_nonce = [Guid]::NewGuid().ToString('D')
        sampling_started_at_utc = [string]$LoadGate.raw_samples[0].captured_at_utc
        sampling_completed_at_utc = [string]$LoadGate.raw_samples[4].completed_at_utc
        issued_at_utc = $issued.ToString('o')
        expires_at_utc = $issued.AddSeconds(45).ToString('o')
        target_gpu = [ordered]@{
            index = [int]$target.target_index
            uuid = [string]$target.target_uuid
            name = [string]$target.target_name
            memory_total_mib = [double]$target.target_total_memory_mib
        }
        load_gate = $LoadGate
    }
    Write-JsonAtomic -Path $Path -Value $body
    $sealCode = @'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if "attestation_sha256" in payload:
    raise SystemExit("unsealed attestation unexpectedly contains a hash")
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
payload["attestation_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
'@
    $sealedResult = Invoke-NativeCapture -FilePath $PythonPath -ArgumentList @('-c', $sealCode, $Path) -WorkingDirectory $WorkingDirectory -TimeoutSeconds 30
    Assert-NativeSuccess -Result $sealedResult -Label "$Purpose canonical attestation seal"
    try {
        $sealed = (([string]$sealedResult.Stdout).Trim() | ConvertFrom-Json)
    }
    catch {
        throw "$Purpose canonical attestation seal returned invalid JSON"
    }
    if ([string]$sealed.attestation_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw "$Purpose canonical attestation seal returned a malformed hash"
    }
    Write-JsonAtomic -Path $Path -Value $sealed
    return $sealed
}

function Get-ProcessUnixNanoseconds {
    param([Parameter(Mandatory = $true)][DateTime]$StartTimeUtc)
    $epoch = [DateTime]::SpecifyKind([DateTime]'1970-01-01T00:00:00', [DateTimeKind]::Utc)
    return [long](($StartTimeUtc.Ticks - $epoch.Ticks) * 100)
}

function Get-LocalOwnerState {
    param(
        [Parameter(Mandatory = $true)]$Owner,
        [Parameter(Mandatory = $true)][string]$LocalHostname
    )
    if ([string]$Owner.hostname -ne $LocalHostname) {
        return 'CROSS_HOST_FAIL_CLOSED_DEFERRED_TO_PYTHON'
    }
    $ownerPid = -1
    if (-not [int]::TryParse([string]$Owner.pid, [ref]$ownerPid) -or $ownerPid -le 0) {
        return 'LOCAL_DEAD_OR_MALFORMED_DEFERRED_TO_PYTHON_RECOVERY'
    }
    try {
        $process = Get-Process -Id $ownerPid -ErrorAction Stop
        $startUtc = $process.StartTime.ToUniversalTime()
    }
    catch {
        return 'LOCAL_DEAD_DEFERRED_TO_PYTHON_RECOVERY'
    }
    if ($null -eq $Owner.process_created_unix_ns) {
        return 'ACTIVE_LOCAL_CANNOT_DISPROVE_IDENTITY'
    }
    $currentNs = Get-ProcessUnixNanoseconds -StartTimeUtc $startUtc
    $recordedNs = 0L
    if (-not [long]::TryParse([string]$Owner.process_created_unix_ns, [ref]$recordedNs)) {
        return 'LOCAL_PID_IDENTITY_MALFORMED_DEFERRED_TO_PYTHON_RECOVERY'
    }
    if ([Math]::Abs([double]$currentNs - [double]$recordedNs) -le 1000000.0) {
        return 'ACTIVE_LOCAL_MATCHED_PID_AND_CREATION_IDENTITY'
    }
    return 'LOCAL_PID_REUSED_DEFERRED_TO_PYTHON_RECOVERY'
}

function Get-OutputLockAudit {
    param(
        [Parameter(Mandatory = $true)][string]$OutputDirectory,
        [Parameter(Mandatory = $true)][string]$LocalHostname
    )
    $lockRoot = Join-Path $OutputDirectory '_locks'
    if (-not (Test-Path -LiteralPath $lockRoot -PathType Container)) {
        return @()
    }
    $rows = @()
    foreach ($lock in @(Get-ChildItem -LiteralPath $lockRoot -Directory -Filter '*.lock' -ErrorAction Stop | Sort-Object FullName)) {
        $ownerPath = Join-Path $lock.FullName 'owner.json'
        $owner = $null
        $parseError = $null
        try {
            $owner = Get-Content -LiteralPath $ownerPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $ownerState = Get-LocalOwnerState -Owner $owner -LocalHostname $LocalHostname
        }
        catch {
            $parseError = $_.Exception.Message
            $ownerState = 'UNREADABLE_OWNER_DEFERRED_TO_PYTHON_FAIL_CLOSED_OR_RECOVERY'
        }
        $row = [ordered]@{
            lock_path = $lock.FullName
            owner_path = $ownerPath
            owner_state = $ownerState
            hostname = $(if ($null -ne $owner) { [string]$owner.hostname } else { $null })
            pid = $(if ($null -ne $owner) { $owner.pid } else { $null })
            process_created_unix_ns = $(if ($null -ne $owner) { $owner.process_created_unix_ns } else { $null })
            parse_error = $parseError
        }
        $rows += $row
        if ($ownerState -like 'ACTIVE_LOCAL*') {
            throw "active local T9.1.3 lock detected at $($lock.FullName): pid=$($row.pid), state=$ownerState"
        }
        # Local dead locks and cross-host locks are never deleted here.  The
        # Python lock implementation alone decides local recovery and enforces
        # permanent fail-closed behavior for a remote owner.
    }
    return $rows
}

function Start-HiddenChild {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [AllowEmptyString()][Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath
    )
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = Join-NativeArgumentList -ArgumentList $ArgumentList
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    # A one-byte sink buffer is intentional: ready/consumed protocol records are
    # sub-kilobyte JSON lines that must be visible while the child is still alive.
    $stdoutStream = [IO.FileStream]::new($StdoutPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read, 1, $false)
    try {
        $stderrStream = [IO.FileStream]::new($StderrPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read, 1, $false)
    }
    catch {
        $stdoutStream.Dispose()
        throw
    }
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    $started = $false
    $stdoutTask = $null
    $stderrTask = $null
    try {
        if (-not $process.Start()) {
            throw "hidden child failed to start: $FilePath"
        }
        $started = $true
        $stdoutTask = $process.StandardOutput.BaseStream.CopyToAsync($stdoutStream)
        $stderrTask = $process.StandardError.BaseStream.CopyToAsync($stderrStream)
        Add-Member -InputObject $process -MemberType NoteProperty -Name T913StdoutTask -Value $stdoutTask
        Add-Member -InputObject $process -MemberType NoteProperty -Name T913StderrTask -Value $stderrTask
        Add-Member -InputObject $process -MemberType NoteProperty -Name T913StdoutStream -Value $stdoutStream
        Add-Member -InputObject $process -MemberType NoteProperty -Name T913StderrStream -Value $stderrStream
        Add-Member -InputObject $process -MemberType NoteProperty -Name T913IoClosed -Value $false
        return $process
    }
    catch {
        $launchError = $_
        if ($started -and -not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit()
        }
        foreach ($copyTask in @($stdoutTask, $stderrTask)) {
            if ($null -ne $copyTask) {
                try {
                    [void]$copyTask.GetAwaiter().GetResult()
                }
                catch {
                }
            }
        }
        $stdoutStream.Dispose()
        $stderrStream.Dispose()
        throw $launchError
    }
}

function Start-JobBoundPythonChild {
    param(
        [ValidateSet('mf', 'nmf', 'finalize')]
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [AllowEmptyString()][Parameter(Mandatory = $true)][string[]]$PayloadArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath,
        [Parameter(Mandatory = $true)][string]$BarrierRoot,
        [scriptblock]$BeforeReleaseCheck
    )
    if ($script:KillOnCloseJobHandle -eq [IntPtr]::Zero) {
        throw "$Role payload refused launch because the Job Object is absent"
    }
    if (-not (Test-Path -LiteralPath $BarrierRoot -PathType Container)) {
        [IO.Directory]::CreateDirectory($BarrierRoot) | Out-Null
    }
    $barrierId = [Guid]::NewGuid().ToString('N')
    $nonce = [Guid]::NewGuid().ToString('D') + [Guid]::NewGuid().ToString('D')
    $readyPath = Join-Path $BarrierRoot ("$Role.$barrierId.ready.json")
    $releasePath = Join-Path $BarrierRoot ("$Role.$barrierId.release.json")
    if ((Test-Path -LiteralPath $readyPath) -or (Test-Path -LiteralPath $releasePath)) {
        throw "$Role Job barrier paths unexpectedly pre-exist"
    }
    $payloadJson = ConvertTo-Json -InputObject ([object[]]$PayloadArgumentList) -Compress
    $payloadBase64 = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($payloadJson)
    )
    $wrapperCode = @'
import base64
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ready_path = Path(sys.argv[1])
release_path = Path(sys.argv[2])
role = sys.argv[3]
nonce = sys.argv[4]
payload = json.loads(base64.b64decode(sys.argv[5]).decode("utf-8"))
if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
    raise SystemExit("invalid gated payload argv")
if release_path.exists():
    raise SystemExit("gated payload release existed before wrapper readiness")
ready = {
    "schema_version": "t9.1.3-job-bound-python-ready-v1",
    "role": role,
    "pid": os.getpid(),
    "nonce_sha256": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
}
with ready_path.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(ready, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
deadline = time.monotonic() + 30.0
while not release_path.is_file():
    if time.monotonic() >= deadline:
        raise SystemExit("timed out waiting for Job Object payload release")
    time.sleep(0.01)
release = json.loads(release_path.read_text(encoding="utf-8"))
if set(release) != {"schema_version", "role", "pid", "nonce"}:
    raise SystemExit("gated payload release schema drifted")
if (
    release["schema_version"] != "t9.1.3-job-bound-python-release-v1"
    or release["role"] != role
    or int(release["pid"]) != os.getpid()
    or release["nonce"] != nonce
):
    raise SystemExit("gated payload release identity mismatch")
if len(payload) >= 2 and payload[0] == "-m":
    module = payload[1]
    sys.argv = [module, *payload[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=False)
elif len(payload) >= 2 and payload[0] == "-c":
    sys.argv = ["-c", *payload[2:]]
    namespace = {"__name__": "__main__", "__file__": "<job-bound-payload>"}
    exec(compile(payload[1], "<job-bound-payload>", "exec"), namespace, namespace)
else:
    raise SystemExit("gated payload supports only explicit -m or -c execution")
'@
    $process = $null
    try {
        $process = Start-HiddenChild -FilePath $PythonPath -ArgumentList @(
            '-c', $wrapperCode, $readyPath, $releasePath, $Role, $nonce, $payloadBase64
        ) -WorkingDirectory $WorkingDirectory -StdoutPath $StdoutPath -StderrPath $StderrPath

        # The wrapper cannot execute the payload without the nonce-bearing
        # release.  Bind first; a failed assignment kills the waiting wrapper.
        Add-ProcessToKillOnCloseJob -Process $process -Role $Role
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(10)
        while (-not (Test-Path -LiteralPath $readyPath -PathType Leaf)) {
            if ($process.HasExited) {
                Close-HiddenChildIo -Process $process
                throw "$Role Job barrier wrapper exited before readiness with code $($process.ExitCode)"
            }
            if ([DateTime]::UtcNow -ge $readyDeadline) {
                throw "$Role Job barrier wrapper did not become ready"
            }
            Start-Sleep -Milliseconds 10
        }
        $ready = Get-Content -LiteralPath $readyPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $expectedNonceHash = [BitConverter]::ToString(
            [Security.Cryptography.SHA256]::Create().ComputeHash(
                [Text.Encoding]::UTF8.GetBytes($nonce)
            )
        ).Replace('-', '').ToLowerInvariant()
        if (
            [string]$ready.schema_version -ne 't9.1.3-job-bound-python-ready-v1' -or
            [string]$ready.role -ne $Role -or
            [int]$ready.pid -ne [int]$process.Id -or
            [string]$ready.nonce_sha256 -ne $expectedNonceHash
        ) {
            throw "$Role Job barrier readiness identity mismatch"
        }
        if ($null -ne $BeforeReleaseCheck) {
            & $BeforeReleaseCheck
        }
        Write-JsonAtomic -Path $releasePath -Value ([ordered]@{
            schema_version = 't9.1.3-job-bound-python-release-v1'
            role = $Role
            pid = [int]$process.Id
            nonce = $nonce
        })
        Add-Member -InputObject $process -MemberType NoteProperty -Name T913JobBarrierReadyPath -Value $readyPath
        Add-Member -InputObject $process -MemberType NoteProperty -Name T913JobBarrierReleasePath -Value $releasePath
        Add-Member -InputObject $process -MemberType NoteProperty -Name T913JobBoundBeforePayloadRelease -Value $true
        return $process
    }
    catch {
        $launchError = $_
        if ($null -ne $process) {
            try {
                if (-not $process.HasExited) {
                    $process.Kill()
                    $process.WaitForExit()
                }
                Close-HiddenChildIo -Process $process
            }
            catch {
            }
        }
        throw $launchError
    }
}

function Close-HiddenChildIo {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process)
    if (-not $Process.HasExited) {
        throw "cannot close redirected child IO while PID $($Process.Id) is still running"
    }
    $Process.WaitForExit()
    if (-not [bool]$Process.T913IoClosed) {
        $copyErrors = @()
        try {
            try {
                [void]$Process.T913StdoutTask.GetAwaiter().GetResult()
            }
            catch {
                $copyErrors += "stdout: $($_.Exception.Message)"
            }
            try {
                [void]$Process.T913StderrTask.GetAwaiter().GetResult()
            }
            catch {
                $copyErrors += "stderr: $($_.Exception.Message)"
            }
            $Process.T913StdoutStream.Flush()
            $Process.T913StderrStream.Flush()
        }
        finally {
            $Process.T913StdoutStream.Dispose()
            $Process.T913StderrStream.Dispose()
            $Process.T913IoClosed = $true
        }
        if ($copyErrors.Count -gt 0) {
            throw "redirected child IO copy failed: $($copyErrors -join '; ')"
        }
    }
}

function New-ChildRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [AllowEmptyString()][Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath,
        [Parameter(Mandatory = $true)][string]$RunDirectory
    )
    $Process.Refresh()
    $startUtc = $Process.StartTime.ToUniversalTime()
    $record = [ordered]@{
        schema_version = 't9.1.3-supervised-child-v2'
        task_id = 'T9.1.3'
        role = $Role
        state = 'STARTED'
        pid = [int]$Process.Id
        process_created_utc = $startUtc.ToString('o')
        process_created_filetime_utc = [long]$startUtc.ToFileTimeUtc()
        process_created_unix_ns = Get-ProcessUnixNanoseconds -StartTimeUtc $startUtc
        recorded_at_utc = Get-UtcIso
        executable = $script:PythonPath
        arguments = @($ArgumentList)
        stdout = $StdoutPath
        stderr = $StderrPath
        job_object_bound_before_python_payload_release = [bool]$Process.T913JobBoundBeforePayloadRelease
        job_barrier_ready = [string]$Process.T913JobBarrierReadyPath
        job_barrier_release = [string]$Process.T913JobBarrierReleasePath
        exited_at_utc = $null
        exit_code = $null
        termination_reason = $null
        stdout_evidence = $null
        stderr_evidence = $null
    }
    Write-JsonAtomic -Path (Join-Path $RunDirectory ("$Role.process.json")) -Value $record
    Write-Utf8NoBomAtomic -Path (Join-Path $RunDirectory ("$Role.pid")) -Text (([string]$Process.Id) + [Environment]::NewLine)
    return $record
}

function Complete-ChildRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$RunDirectory,
        [string]$TerminationReason
    )
    if (-not $Process.HasExited) {
        return
    }
    Close-HiddenChildIo -Process $Process
    $record = $script:ChildRecords[$Role]
    $record['state'] = 'EXITED'
    $record['exited_at_utc'] = Get-UtcIso
    $record['exit_code'] = [int]$Process.ExitCode
    if (-not [string]::IsNullOrWhiteSpace($TerminationReason)) {
        $record['termination_reason'] = $TerminationReason
    }
    $record['stdout_evidence'] = Get-FileEvidence -Path ([string]$record.stdout)
    $record['stderr_evidence'] = Get-FileEvidence -Path ([string]$record.stderr)
    Write-JsonAtomic -Path (Join-Path $RunDirectory ("$Role.process.json")) -Value $record
}

function New-EmergencyChildRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process
    )
    $startUtc = $null
    try {
        $Process.Refresh()
        $startUtc = $Process.StartTime.ToUniversalTime()
    }
    catch {
    }
    return [ordered]@{
        schema_version = 't9.1.3-supervised-child-v2'
        task_id = 'T9.1.3'
        role = $Role
        state = 'STARTED_IDENTITY_RECORD_RECOVERY'
        pid = [int]$Process.Id
        process_created_utc = $(if ($null -ne $startUtc) { $startUtc.ToString('o') } else { $null })
        process_created_filetime_utc = $(if ($null -ne $startUtc) { [long]$startUtc.ToFileTimeUtc() } else { $null })
        process_created_unix_ns = $(if ($null -ne $startUtc) { Get-ProcessUnixNanoseconds -StartTimeUtc $startUtc } else { $null })
        recorded_at_utc = Get-UtcIso
        executable = $script:PythonPath
        arguments = @('UNAVAILABLE_BECAUSE_DURABLE_RECORDING_FAILED_DURING_LAUNCH')
        stdout = $(if ($script:LogPaths.Contains("${Role}_stdout")) { [string]$script:LogPaths["${Role}_stdout"] } else { $null })
        stderr = $(if ($script:LogPaths.Contains("${Role}_stderr")) { [string]$script:LogPaths["${Role}_stderr"] } else { $null })
        exited_at_utc = $null
        exit_code = $null
        termination_reason = $null
        stdout_evidence = $null
        stderr_evidence = $null
        emergency_record_reason = 'PRIMARY_DURABLE_IDENTITY_RECORD_FAILED'
    }
}

function Stop-OwnedChild {
    param(
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$RunDirectory,
        [Parameter(Mandatory = $true)][string]$Reason
    )
    if (-not $Process.HasExited) {
        # The Process object retains the original OS handle, so Kill cannot be
        # redirected to a later PID reuse.
        $Process.Kill()
        if (-not $Process.WaitForExit(30000)) {
            throw "owned child $Role did not exit after Kill()"
        }
    }
    Complete-ChildRecord -Role $Role -Process $Process -RunDirectory $RunDirectory -TerminationReason $Reason
}

function Get-LogEvidenceMap {
    $result = [ordered]@{}
    if ($null -eq $script:LogPaths) {
        return $result
    }
    foreach ($name in $script:LogPaths.Keys) {
        $result[$name] = Get-FileEvidence -Path ([string]$script:LogPaths[$name])
    }
    return $result
}

function Write-LaunchTransaction {
    param([Parameter(Mandatory = $true)][string]$State)
    $payload = [ordered]@{
        schema_version = 't9.1.3-launch-transaction-v3'
        task_id = 'T9.1.3'
        transaction_id = $script:TransactionId
        state = $State
        updated_at_utc = Get-UtcIso
        artifact_mode = $script:ArtifactMode
        deadline_utc = $script:DeadlineUtc.ToString('o')
        baseline_config_sha256 = $(if ($null -ne $script:BaselineProbe) { [string]$script:BaselineProbe.config_sha256 } else { $null })
        baseline_implementation_sha256 = $(if ($null -ne $script:BaselineProbe) { [string]$script:BaselineProbe.implementation_sha256 } else { $null })
        children = $script:ChildRecords
    }
    Write-JsonAtomic -Path (Join-Path $script:RunDirectory 'launch_transaction.json') -Value $payload
}

function Set-SupervisorPhase {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [string]$Detail
    )
    $event = [ordered]@{
        sequence = [int]$script:PhaseHistory.Count
        state = $State
        at_utc = Get-UtcIso
        detail = $(if ([string]::IsNullOrWhiteSpace($Detail)) { $null } else { $Detail })
    }
    [void]$script:PhaseHistory.Add($event)
    $script:Supervisor['state'] = $State
    $script:Supervisor['updated_at_utc'] = $event.at_utc
    $script:Supervisor['phase_history'] = @($script:PhaseHistory)
    Write-JsonAtomic -Path (Join-Path $script:RunDirectory 'phase_history.json') -Value @($script:PhaseHistory)
    Write-JsonAtomic -Path (Join-Path $script:RunDirectory 'supervisor_state.json') -Value $script:Supervisor
}

function Wait-MfWithBlockedNmf {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$MfProcess,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$NmfProcess,
        [Parameter(Mandatory = $true)][string]$ReleasePath
    )
    while ($true) {
        if (Test-Path -LiteralPath $ReleasePath) {
            if (-not $MfProcess.HasExited) {
                Stop-OwnedChild -Role 'mf' -Process $MfProcess -RunDirectory $script:RunDirectory -Reason 'PREMATURE_NMF_RELEASE_DETECTED'
            }
            if (-not $NmfProcess.HasExited) {
                Stop-OwnedChild -Role 'nmf' -Process $NmfProcess -RunDirectory $script:RunDirectory -Reason 'PREMATURE_NMF_RELEASE_DETECTED'
            }
            throw 'NMF release appeared before supervised MF success'
        }
        if ($NmfProcess.HasExited) {
            Complete-ChildRecord -Role 'nmf' -Process $NmfProcess -RunDirectory $script:RunDirectory
            $script:FailureExitCode = [Math]::Max(1, [int]$NmfProcess.ExitCode)
            if (-not $MfProcess.HasExited) {
                Stop-OwnedChild -Role 'mf' -Process $MfProcess -RunDirectory $script:RunDirectory -Reason 'NMF_WAITER_EXITED_BEFORE_RELEASE'
            }
            throw "NMF waiter exited before MF completed/released it (exit $($NmfProcess.ExitCode)); MF was stopped immediately"
        }
        if ($MfProcess.HasExited) {
            Complete-ChildRecord -Role 'mf' -Process $MfProcess -RunDirectory $script:RunDirectory
            return
        }
        $remaining = $script:DeadlineUtc - [DateTime]::UtcNow
        if ($remaining.TotalMilliseconds -le 0) {
            $script:FailureExitCode = 124
            if (-not $MfProcess.HasExited) {
                Stop-OwnedChild -Role 'mf' -Process $MfProcess -RunDirectory $script:RunDirectory -Reason 'TOTAL_DEADLINE_EXCEEDED'
            }
            if (-not $NmfProcess.HasExited) {
                Stop-OwnedChild -Role 'nmf' -Process $NmfProcess -RunDirectory $script:RunDirectory -Reason 'TOTAL_DEADLINE_EXCEEDED'
            }
            throw "total T9.1.3 deadline exceeded while MF ran and NMF remained blocked: $($script:DeadlineUtc.ToString('o'))"
        }
        $sleepMs = [int][Math]::Min([double]($script:PollIntervalSeconds * 1000), [Math]::Max(100.0, $remaining.TotalMilliseconds))
        Start-Sleep -Milliseconds $sleepMs
    }
}

function Wait-SingleChild {
    param(
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process
    )
    while (-not $Process.HasExited) {
        $remaining = $script:DeadlineUtc - [DateTime]::UtcNow
        if ($remaining.TotalMilliseconds -le 0) {
            $script:FailureExitCode = 124
            Stop-OwnedChild -Role $Role -Process $Process -RunDirectory $script:RunDirectory -Reason 'TOTAL_DEADLINE_EXCEEDED'
            throw "total T9.1.3 deadline exceeded during ${Role}: $($script:DeadlineUtc.ToString('o'))"
        }
        $sleepMs = [int][Math]::Min([double]($script:PollIntervalSeconds * 1000), [Math]::Max(100.0, $remaining.TotalMilliseconds))
        Start-Sleep -Milliseconds $sleepMs
    }
    Complete-ChildRecord -Role $Role -Process $Process -RunDirectory $script:RunDirectory
}

function Read-SharedUtf8Text {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return ''
    }
    $stream = New-Object System.IO.FileStream(
        $Path,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::ReadWrite
    )
    try {
        $length = [int]$stream.Length
        $bytes = New-Object byte[] $length
        $offset = 0
        while ($offset -lt $length) {
            $read = $stream.Read($bytes, $offset, $length - $offset)
            if ($read -le 0) { break }
            $offset += $read
        }
        return [Text.Encoding]::UTF8.GetString($bytes, 0, $offset)
    }
    finally {
        $stream.Dispose()
    }
}

function Wait-NmfSerialReleaseReady {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$ReleasePath,
        [Parameter(Mandatory = $true)][string]$ReleaseNonce,
        [Parameter(Mandatory = $true)]$TrainingAttestation,
        [Parameter(Mandatory = $true)]$Probe,
        [Parameter(Mandatory = $true)][string]$DeadlineUtc
    )
    $expectedKeys = @(
        'schema_version', 'event', 'task_id', 'family', 'waiter_pid',
        'release_path', 'release_nonce_sha256', 'transaction_id',
        'attestation_nonce', 'attestation_sha256', 'config_sha256',
        'implementation_sha256', 'deadline_utc'
    ) | Sort-Object
    $expectedNonceHash = Get-StringSha256Hex -Text $ReleaseNonce
    $attestationExpiry = [DateTimeOffset]::Parse([string]$TrainingAttestation.expires_at_utc).UtcDateTime
    $transactionDeadline = [DateTimeOffset]::Parse($DeadlineUtc).UtcDateTime
    $mfStartupReserveSeconds = 15.0
    $attestationReadyDeadline = $attestationExpiry.AddSeconds(-$mfStartupReserveSeconds)
    $readyDeadline = $(if ($attestationReadyDeadline -lt $transactionDeadline) { $attestationReadyDeadline } else { $transactionDeadline })
    if ([DateTime]::UtcNow -ge $readyDeadline) {
        throw 'NMF ready barrier has no fresh-attestation reserve left for the subsequent MF startup'
    }
    while ([DateTime]::UtcNow -lt $readyDeadline) {
        if (Test-Path -LiteralPath $ReleasePath) {
            throw 'NMF release existed before a fully bound ready record was observed'
        }
        if ($Process.HasExited) {
            throw "NMF child exited before serial-release readiness with code $($Process.ExitCode)"
        }
        $text = Read-SharedUtf8Text -Path $StdoutPath
        foreach ($line in @($text -split "`r?`n")) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try {
                $candidate = $line | ConvertFrom-Json
            }
            catch {
                continue
            }
            if ([string]$candidate.event -ne 't9_1_3_nmf_serial_release_wait_ready') {
                continue
            }
            $actualKeys = @($candidate.PSObject.Properties.Name | Sort-Object)
            if (($actualKeys -join "`n") -ne ($expectedKeys -join "`n")) {
                throw 'NMF serial-release ready record schema drifted'
            }
            $observedReleasePath = Get-CanonicalPath -Path ([string]$candidate.release_path) -BasePath $script:RunDirectory
            if (
                [string]$candidate.schema_version -ne 't9.1.3-nmf-release-wait-ready-v1' -or
                [string]$candidate.task_id -ne 'T9.1.3' -or
                [string]$candidate.family -ne 'nmf' -or
                [int]$candidate.waiter_pid -ne [int]$Process.Id -or
                -not [string]::Equals($observedReleasePath, $ReleasePath, [StringComparison]::OrdinalIgnoreCase) -or
                [string]$candidate.release_nonce_sha256 -ne $expectedNonceHash -or
                [string]$candidate.transaction_id -ne [string]$TrainingAttestation.run_identity.transaction_id -or
                [string]$candidate.attestation_nonce -ne [string]$TrainingAttestation.attestation_nonce -or
                [string]$candidate.attestation_sha256 -ne [string]$TrainingAttestation.attestation_sha256 -or
                [string]$candidate.config_sha256 -ne [string]$Probe.config_sha256 -or
                [string]$candidate.implementation_sha256 -ne [string]$Probe.implementation_sha256 -or
                [string]$candidate.deadline_utc -ne $DeadlineUtc
            ) {
                throw 'NMF serial-release ready record identity mismatch'
            }
            $observedReadyAt = [DateTime]::UtcNow
            if ($observedReadyAt -ge $readyDeadline) {
                throw 'NMF ready record became visible after the MF-startup reserve/deadline boundary'
            }
            $attestationSecondsRemaining = ($attestationExpiry - $observedReadyAt).TotalSeconds
            return [ordered]@{
                schema_version = 't9.1.3-supervisor-observed-nmf-ready-v1'
                observed_at_utc = $observedReadyAt.ToString('o')
                waiter_pid = [int]$Process.Id
                release_path = $ReleasePath
                release_nonce_sha256 = $expectedNonceHash
                transaction_id = [string]$candidate.transaction_id
                attestation_nonce = [string]$candidate.attestation_nonce
                attestation_sha256 = [string]$candidate.attestation_sha256
                config_sha256 = [string]$candidate.config_sha256
                implementation_sha256 = [string]$candidate.implementation_sha256
                deadline_utc = [string]$candidate.deadline_utc
                mf_startup_reserve_seconds = $mfStartupReserveSeconds
                attestation_seconds_remaining_when_observed = $attestationSecondsRemaining
            }
        }
        Start-Sleep -Milliseconds 100
    }
    throw 'NMF did not publish a fully bound ready record before the MF-startup reserve/deadline boundary'
}

function New-NmfSerialReleasePayload {
    param(
        [Parameter(Mandatory = $true)][bool]$MfExited,
        [Parameter(Mandatory = $true)][int]$MfExitCode,
        [Parameter(Mandatory = $true)][int]$MfPid,
        [Parameter(Mandatory = $true)][long]$MfProcessCreatedUnixNs,
        [Parameter(Mandatory = $true)][bool]$NmfExited,
        [Parameter(Mandatory = $true)][int]$NmfPid,
        [Parameter(Mandatory = $true)][string]$ReleasePath,
        [Parameter(Mandatory = $true)][string]$ReleaseNonce,
        [Parameter(Mandatory = $true)]$TrainingAttestation,
        [Parameter(Mandatory = $true)]$Probe,
        [Parameter(Mandatory = $true)][string]$DeadlineUtc
    )
    if (-not $MfExited -or $MfExitCode -ne 0) {
        throw 'NMF release refused because MF did not exit successfully'
    }
    if ($NmfExited) {
        throw 'NMF release refused because its waiting child already exited'
    }
    if ($MfPid -le 0 -or $MfProcessCreatedUnixNs -le 0 -or $NmfPid -le 0) {
        throw 'NMF release refused malformed process identities'
    }
    if (Test-Path -LiteralPath $ReleasePath) {
        throw 'NMF release path unexpectedly pre-existed publication'
    }
    $attestedRunDir = Get-CanonicalPath -Path ([string]$TrainingAttestation.run_identity.run_dir) -BasePath ([Environment]::CurrentDirectory)
    $expectedPath = Get-CanonicalPath -Path (Join-Path $attestedRunDir 'nmf_after_mf.release.json') -BasePath $attestedRunDir
    if (-not [string]::Equals($expectedPath, $ReleasePath, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'NMF release path is not bound to attested run identity'
    }
    $parsedNonce = [Guid]::Empty
    if (-not [Guid]::TryParse($ReleaseNonce, [ref]$parsedNonce) -or $parsedNonce.ToString('D') -ne $ReleaseNonce.ToLowerInvariant()) {
        throw 'NMF release nonce is not canonical'
    }
    $deadline = [DateTimeOffset]::Parse($DeadlineUtc)
    if ($deadline.Offset -ne [TimeSpan]::Zero -or $deadline.UtcDateTime -le [DateTime]::UtcNow) {
        throw 'NMF release transaction deadline is invalid or expired'
    }
    return [ordered]@{
        schema_version = 't9.1.3-nmf-after-mf-release-v1'
        task_id = 'T9.1.3'
        family = 'nmf'
        prerequisite_family = 'mf'
        prerequisite_exit_code = 0
        transaction_id = [string]$TrainingAttestation.run_identity.transaction_id
        run_dir = [string]$TrainingAttestation.run_identity.run_dir
        attestation_nonce = [string]$TrainingAttestation.attestation_nonce
        attestation_sha256 = [string]$TrainingAttestation.attestation_sha256
        release_nonce = $ReleaseNonce
        waiter_pid = $NmfPid
        config_sha256 = [string]$Probe.config_sha256
        implementation_sha256 = [string]$Probe.implementation_sha256
        deadline_utc = $DeadlineUtc
        mf_pid = $MfPid
        mf_process_created_unix_ns = $MfProcessCreatedUnixNs
        released_at_utc = Get-UtcIso
    }
}

function Publish-NmfSerialRelease {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$MfProcess,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$NmfProcess,
        [Parameter(Mandatory = $true)][string]$ReleasePath,
        [Parameter(Mandatory = $true)][string]$ReleaseNonce,
        [Parameter(Mandatory = $true)]$TrainingAttestation,
        [Parameter(Mandatory = $true)]$Probe,
        [Parameter(Mandatory = $true)][string]$DeadlineUtc
    )
    $MfProcess.Refresh()
    $NmfProcess.Refresh()
    $mfStartUtc = $MfProcess.StartTime.ToUniversalTime()
    $payload = New-NmfSerialReleasePayload -MfExited $MfProcess.HasExited -MfExitCode $(if ($MfProcess.HasExited) { [int]$MfProcess.ExitCode } else { -1 }) -MfPid $MfProcess.Id -MfProcessCreatedUnixNs (Get-ProcessUnixNanoseconds -StartTimeUtc $mfStartUtc) -NmfExited $NmfProcess.HasExited -NmfPid $NmfProcess.Id -ReleasePath $ReleasePath -ReleaseNonce $ReleaseNonce -TrainingAttestation $TrainingAttestation -Probe $Probe -DeadlineUtc $DeadlineUtc
    Write-JsonAtomic -Path $ReleasePath -Value $payload
    return [ordered]@{
        payload = $payload
        file_evidence = Get-FileEvidence -Path $ReleasePath
    }
}

function Wait-NmfSerialReleaseConsumed {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$ReleasePath,
        [Parameter(Mandatory = $true)][string]$ReleaseNonce,
        [Parameter(Mandatory = $true)]$ReleaseWitness,
        [Parameter(Mandatory = $true)]$TrainingAttestation,
        [Parameter(Mandatory = $true)][string]$DeadlineUtc
    )
    $expectedKeys = @(
        'schema_version', 'event', 'release_path', 'release_sha256',
        'transaction_id', 'attestation_sha256', 'release_nonce_sha256',
        'mf_pid', 'mf_process_created_unix_ns', 'waiter_pid',
        'released_at_utc', 'waited_seconds'
    ) | Sort-Object
    $expectedNonceHash = Get-StringSha256Hex -Text $ReleaseNonce
    $expectedReleaseHash = [string]$ReleaseWitness.file_evidence.sha256
    if (
        -not (Test-Path -LiteralPath $ReleasePath -PathType Leaf) -or
        [string]::IsNullOrWhiteSpace($expectedReleaseHash) -or
        (Get-Sha256Hex -Path $ReleasePath) -ne $expectedReleaseHash
    ) {
        throw 'NMF serial-release file changed before the child consumption witness'
    }
    $transactionDeadline = [DateTimeOffset]::Parse($DeadlineUtc).UtcDateTime
    $consumptionDeadline = [DateTime]::UtcNow.AddSeconds(30)
    if ($transactionDeadline -lt $consumptionDeadline) {
        $consumptionDeadline = $transactionDeadline
    }
    $ioDrainedAfterExit = $false
    while ($true) {
        $text = Read-SharedUtf8Text -Path $StdoutPath
        foreach ($line in @($text -split "`r?`n")) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try {
                $candidate = $line | ConvertFrom-Json
            }
            catch {
                continue
            }
            if ([string]$candidate.event -ne 't9_1_3_nmf_serial_release_consumed') {
                continue
            }
            $actualKeys = @($candidate.PSObject.Properties.Name | Sort-Object)
            if (($actualKeys -join "`n") -ne ($expectedKeys -join "`n")) {
                throw 'NMF serial-release consumption witness schema drifted'
            }
            $observedReleasePath = Get-CanonicalPath -Path ([string]$candidate.release_path) -BasePath $script:RunDirectory
            $waitedSeconds = [double]$candidate.waited_seconds
            if (
                [string]$candidate.schema_version -ne 't9.1.3-nmf-serial-release-witness-v1' -or
                -not [string]::Equals($observedReleasePath, $ReleasePath, [StringComparison]::OrdinalIgnoreCase) -or
                [string]$candidate.release_sha256 -ne $expectedReleaseHash -or
                [string]$candidate.transaction_id -ne [string]$TrainingAttestation.run_identity.transaction_id -or
                [string]$candidate.attestation_sha256 -ne [string]$TrainingAttestation.attestation_sha256 -or
                [string]$candidate.release_nonce_sha256 -ne $expectedNonceHash -or
                [long]$candidate.mf_pid -ne [long]$ReleaseWitness.payload.mf_pid -or
                [long]$candidate.mf_process_created_unix_ns -ne [long]$ReleaseWitness.payload.mf_process_created_unix_ns -or
                [int]$candidate.waiter_pid -ne [int]$Process.Id -or
                [string]$candidate.released_at_utc -ne [string]$ReleaseWitness.payload.released_at_utc -or
                [double]::IsNaN($waitedSeconds) -or
                [double]::IsInfinity($waitedSeconds) -or
                $waitedSeconds -lt 0.0 -or
                (Get-Sha256Hex -Path $ReleasePath) -ne $expectedReleaseHash
            ) {
                throw 'NMF serial-release consumption witness identity mismatch'
            }
            if ([DateTime]::UtcNow -ge $consumptionDeadline -or [DateTime]::UtcNow -ge $transactionDeadline) {
                throw 'NMF serial-release consumption witness was observed after its deadline'
            }
            return [ordered]@{
                schema_version = 't9.1.3-supervisor-observed-nmf-consumption-v1'
                observed_at_utc = Get-UtcIso
                child_witness = $candidate
            }
        }
        $Process.Refresh()
        if ($Process.HasExited) {
            if (-not $ioDrainedAfterExit) {
                Close-HiddenChildIo -Process $Process
                $ioDrainedAfterExit = $true
                continue
            }
            throw "NMF child exited before a valid serial-release consumption witness was observed (exit $($Process.ExitCode))"
        }
        if ([DateTime]::UtcNow -ge $consumptionDeadline -or [DateTime]::UtcNow -ge $transactionDeadline) {
            throw 'NMF did not publish a valid serial-release consumption witness within 30 seconds and before the total deadline'
        }
        Start-Sleep -Milliseconds 100
    }
}

function Invoke-StaticSelfTest {
    param([Parameter(Mandatory = $true)][string]$PythonPath)
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "static self-test Python is missing: $PythonPath"
    }
    $arguments = @('alpha beta', 'quote"inside', 'trailing\', '', 'ascii-safe-argument')
    $code = 'import json,sys; print(json.dumps(sys.argv[1:], ensure_ascii=False))'
    $result = Invoke-NativeCapture -FilePath $PythonPath -ArgumentList (@('-c', $code) + $arguments) -WorkingDirectory ([Environment]::CurrentDirectory)
    Assert-NativeSuccess -Result $result -Label 'native argv round-trip self-test'
    $observed = @((([string]$result.Stdout).Trim() | ConvertFrom-Json))
    if ($observed.Count -ne $arguments.Count) {
        throw 'native argv round-trip changed the argument count'
    }
    for ($index = 0; $index -lt $arguments.Count; $index += 1) {
        if ([string]$observed[$index] -ne [string]$arguments[$index]) {
            throw "native argv round-trip changed argument $index"
        }
    }
    $staticJobHandle = Initialize-KillOnCloseJob
    try {
        $jobPInvokeStructurePass = (
            $staticJobHandle -ne [IntPtr]::Zero -and
            $null -ne ([T913.NativeJob].GetMethod('AssignOrThrow')) -and
            $null -ne ([T913.NativeJob].GetMethod('CreateKillOnCloseJob'))
        )
        if (-not $jobPInvokeStructurePass) {
            throw 'Job Object P/Invoke structure self-test failed'
        }
    }
    finally {
        if ($staticJobHandle -ne [IntPtr]::Zero) {
            if (-not [T913.NativeJob]::CloseHandle($staticJobHandle)) {
                throw 'empty Job Object self-test handle could not be closed'
            }
        }
    }
    $syntheticJobBindingFailureRejected = $false
    try {
        Assert-SyntheticJobBindingResult -Succeeded $false
    }
    catch {
        $syntheticJobBindingFailureRejected = $true
    }
    Assert-SyntheticJobBindingResult -Succeeded $true
    if (-not $syntheticJobBindingFailureRejected) {
        throw 'synthetic Job Object assignment failure did not fail closed'
    }

    $expectedDeterministicEnvironment = [ordered]@{
        CUBLAS_WORKSPACE_CONFIG = ':4096:8'
        NVIDIA_TF32_OVERRIDE = '0'
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = '0'
        PYTHONHASHSEED = '0'
    }
    $environmentCode = 'import json,os; print(json.dumps({k:os.environ.get(k) for k in sys.argv[1:]}))'
    $environmentCode = 'import json,os,sys; print(json.dumps({k:os.environ.get(k) for k in sys.argv[1:]}, sort_keys=True))'
    $environmentResult = Invoke-NativeCapture -FilePath $PythonPath -ArgumentList (@('-c', $environmentCode) + @($expectedDeterministicEnvironment.Keys)) -WorkingDirectory ([Environment]::CurrentDirectory)
    Assert-NativeSuccess -Result $environmentResult -Label 'deterministic environment inheritance self-test'
    $environmentObserved = ([string]$environmentResult.Stdout).Trim() | ConvertFrom-Json
    foreach ($name in $expectedDeterministicEnvironment.Keys) {
        if ([string]$environmentObserved.$name -ne [string]$expectedDeterministicEnvironment[$name]) {
            throw "deterministic environment was not inherited exactly: $name"
        }
    }

    $barrierTemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('t913-job-barrier-' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($barrierTemporaryRoot) | Out-Null
    $payloadSentinel = Join-Path $barrierTemporaryRoot 'payload.executed'
    $barrierStdout = Join-Path $barrierTemporaryRoot 'payload.stdout.log'
    $barrierStderr = Join-Path $barrierTemporaryRoot 'payload.stderr.log'
    $barrierJobHandle = [IntPtr]::Zero
    $barrierProcess = $null
    $payloadAbsentBeforeRelease = $false
    try {
        $barrierJobHandle = Initialize-KillOnCloseJob
        $script:KillOnCloseJobHandle = $barrierJobHandle
        $payloadCode = 'import os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")'
        $barrierProcess = Start-JobBoundPythonChild -Role 'mf' -PythonPath $PythonPath -PayloadArgumentList @('-c', $payloadCode, $payloadSentinel) -WorkingDirectory ([Environment]::CurrentDirectory) -StdoutPath $barrierStdout -StderrPath $barrierStderr -BarrierRoot (Join-Path $barrierTemporaryRoot 'barriers') -BeforeReleaseCheck {
            $script:T913StaticPayloadAbsentBeforeRelease = -not (Test-Path -LiteralPath $payloadSentinel)
            if (-not $script:T913StaticPayloadAbsentBeforeRelease) {
                throw 'Python payload executed before Job Object binding/release'
            }
        }
        $payloadAbsentBeforeRelease = [bool]$script:T913StaticPayloadAbsentBeforeRelease
        if (-not $barrierProcess.WaitForExit(10000)) {
            throw 'Job-bound Python payload self-test timed out'
        }
        Close-HiddenChildIo -Process $barrierProcess
        if ([int]$barrierProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $payloadSentinel -PathType Leaf)) {
            throw 'Job-bound Python payload self-test did not execute after release'
        }
        if ([int](Get-Content -LiteralPath $payloadSentinel -Raw -Encoding UTF8) -ne [int]$barrierProcess.Id) {
            throw 'Job-bound Python wrapper changed PID before payload execution'
        }
    }
    finally {
        if ($null -ne $barrierProcess -and -not $barrierProcess.HasExited) {
            $barrierProcess.Kill()
            $barrierProcess.WaitForExit()
            Close-HiddenChildIo -Process $barrierProcess
        }
        if ($barrierJobHandle -ne [IntPtr]::Zero) {
            [void][T913.NativeJob]::CloseHandle($barrierJobHandle)
        }
        $script:KillOnCloseJobHandle = [IntPtr]::Zero
    }
    $missingJobRejectedBeforePayload = $false
    try {
        [void](Start-JobBoundPythonChild -Role 'nmf' -PythonPath $PythonPath -PayloadArgumentList @('-c', 'raise SystemExit("must not execute")') -WorkingDirectory ([Environment]::CurrentDirectory) -StdoutPath (Join-Path $barrierTemporaryRoot 'missing.stdout.log') -StderrPath (Join-Path $barrierTemporaryRoot 'missing.stderr.log') -BarrierRoot (Join-Path $barrierTemporaryRoot 'missing-barriers'))
    }
    catch {
        $missingJobRejectedBeforePayload = $true
    }
    if (-not $missingJobRejectedBeforePayload) {
        throw 'Job-bound Python launch did not reject a missing Job Object'
    }
    [IO.Directory]::Delete($barrierTemporaryRoot, $true)

    # Pure synthetic evidence tests.  These call only the sample evaluator and
    # never invoke nvidia-smi, CUDA, torch, or a GPU process.
    $goodGpuSamples = @()
    $syntheticUtilization = @(5.0, 10.0, 15.0, 10.0, 5.0)
    for ($sampleIndex = 0; $sampleIndex -lt 5; $sampleIndex += 1) {
        $goodGpuSamples += [ordered]@{
            sequence = $sampleIndex
            captured_at_utc = ('2026-01-01T00:00:{0:D2}Z' -f ($sampleIndex * 2))
            rows = @([ordered]@{
                index = 0
                uuid = 'GPU-STATIC-SELF-TEST'
                name = 'Synthetic GPU'
                memory_total_mib = 8192.0
                memory_used_mib = 1900.0
                memory_free_mib = 6292.0
                utilization_percent = $syntheticUtilization[$sampleIndex]
            })
        }
    }
    $gpuPass = Test-NvidiaLoadSamples -Samples ([object[]]$goodGpuSamples)
    if (-not $gpuPass.passed) {
        throw ('pure NVIDIA load-gate PASS sample was rejected: ' + (($gpuPass.failure_reasons) -join '; '))
    }
    $lowFreeSamples = [object[]](($goodGpuSamples | ConvertTo-Json -Depth 12) | ConvertFrom-Json)
    $lowFreeSamples[2].rows[0].memory_free_mib = 4095.0
    $lowFreeSamples[2].rows[0].memory_used_mib = 4097.0
    $lowFreeFail = Test-NvidiaLoadSamples -Samples ([object[]]$lowFreeSamples)
    $peakSamples = [object[]](($goodGpuSamples | ConvertTo-Json -Depth 12) | ConvertFrom-Json)
    $peakSamples[4].rows[0].utilization_percent = 31.0
    $peakFail = Test-NvidiaLoadSamples -Samples ([object[]]$peakSamples)
    $medianSamples = [object[]](($goodGpuSamples | ConvertTo-Json -Depth 12) | ConvertFrom-Json)
    foreach ($medianSample in $medianSamples) {
        $medianSample.rows[0].utilization_percent = 16.0
    }
    $medianFail = Test-NvidiaLoadSamples -Samples ([object[]]$medianSamples)
    $uuidSamples = [object[]](($goodGpuSamples | ConvertTo-Json -Depth 12) | ConvertFrom-Json)
    $uuidSamples[3].rows[0].uuid = 'GPU-STATIC-UUID-DRIFT'
    $uuidFail = Test-NvidiaLoadSamples -Samples ([object[]]$uuidSamples)
    $countSamples = [object[]](($goodGpuSamples | ConvertTo-Json -Depth 12) | ConvertFrom-Json)
    $countSamples[1].rows = @($countSamples[1].rows) + @([pscustomobject]@{
        index = 1
        uuid = 'GPU-STATIC-UNEXPECTED-SECOND-DEVICE'
        name = 'Unexpected synthetic GPU'
        memory_total_mib = 8192.0
        memory_used_mib = 0.0
        memory_free_mib = 8192.0
        utilization_percent = 0.0
    })
    $countFail = Test-NvidiaLoadSamples -Samples ([object[]]$countSamples)
    $missingSamplesFail = Test-NvidiaLoadSamples -Samples ([object[]]@())
    if ($lowFreeFail.passed -or $peakFail.passed -or $medianFail.passed -or $uuidFail.passed -or $countFail.passed -or $missingSamplesFail.passed) {
        throw 'pure NVIDIA load-gate FAIL samples were not all rejected'
    }
    $parsedCsvSample = ConvertFrom-NvidiaSmiCsvSample -Text '0, GPU-STATIC-PARSER, "Synthetic, Quoted GPU", 8192, 1000, 7192, 7' -Sequence 0 -CapturedAtUtc '2026-01-01T00:00:00Z'
    if ($parsedCsvSample.rows.Count -ne 1 -or [string]$parsedCsvSample.rows[0].name -ne 'Synthetic, Quoted GPU') {
        throw 'pure NVIDIA CSV parser did not preserve a quoted device name'
    }
    $badCsvRejected = $false
    try {
        [void](ConvertFrom-NvidiaSmiCsvSample -Text '0, GPU-STATIC-PARSER, Synthetic GPU, 8192, 1000, 7192, 7, EXTRA' -Sequence 0 -CapturedAtUtc '2026-01-01T00:00:00Z')
    }
    catch {
        $badCsvRejected = $true
    }
    if (-not $badCsvRejected) {
        throw 'pure NVIDIA CSV parser accepted an extra field'
    }
    $syntheticMarker = [pscustomobject][ordered]@{
        schema_version = 't9.1.3-puviani-paper-constrained-artifacts-v1'
        task_id = 'T9.1.3'
        status = 'INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL'
        evidence_grade = 'PAPER_CONSTRAINED_REIMPLEMENTATION'
        started_at_utc = '2026-01-01T00:00:00Z'
        config_sha256 = ('a' * 64)
        valid_pass_seal = $false
    }
    $strictCrashMarkerPass = Test-StrictInvalidatedFinalizationCrashMarker -Marker $syntheticMarker -ExpectedConfigSha256 ('a' * 64)
    $failedMarker = ($syntheticMarker | ConvertTo-Json -Depth 6) | ConvertFrom-Json
    $failedMarker.status = 'FINALIZATION_FAILED_NO_VALID_SEAL'
    $failedMarkerRejected = -not (Test-StrictInvalidatedFinalizationCrashMarker -Marker $failedMarker -ExpectedConfigSha256 ('a' * 64))
    $extraMarker = ($syntheticMarker | ConvertTo-Json -Depth 6) | ConvertFrom-Json
    $extraMarker | Add-Member -NotePropertyName failure_message -NotePropertyValue 'schema drift'
    $extraMarkerRejected = -not (Test-StrictInvalidatedFinalizationCrashMarker -Marker $extraMarker -ExpectedConfigSha256 ('a' * 64))
    if (-not $strictCrashMarkerPass -or -not $failedMarkerRejected -or -not $extraMarkerRejected) {
        throw 'strict invalidated finalization crash-marker semantics self-test failed'
    }
    $attestationSelfTestCode = @'
import json
from cnn_fpga.benchmark.t9_1_3_gpu_attestation import synthetic_attestation_self_test
print(json.dumps(synthetic_attestation_self_test(), sort_keys=True))
'@
    $attestationSelfTestNative = Invoke-NativeCapture -FilePath $PythonPath -ArgumentList @('-c', $attestationSelfTestCode) -WorkingDirectory ([Environment]::CurrentDirectory) -TimeoutSeconds 30
    Assert-NativeSuccess -Result $attestationSelfTestNative -Label 'pure GPU attestation static self-test'
    try {
        $attestationSelfTest = (([string]$attestationSelfTestNative.Stdout).Trim() | ConvertFrom-Json)
    }
    catch {
        throw 'pure GPU attestation static self-test returned invalid JSON'
    }
    if ([string]$attestationSelfTest.status -ne 'PASS' -or $attestationSelfTest.gpu_queried -ne $false -or $attestationSelfTest.production_started -ne $false) {
        throw 'pure GPU attestation static self-test did not pass fail-closed cases'
    }
    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('t9_1_3_supervisor_selftest_' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($temporaryRoot) | Out-Null
    $supervisorReleaseSuccessPass = $false
    $supervisorReleaseConsumedPass = $false
    $supervisorMfFailureNoReleasePass = $false
    $liveSmallJsonVisibleWhileAlive = $false
    try {
        $childStdout = Join-Path $temporaryRoot 'child stdout.log'
        $childStderr = Join-Path $temporaryRoot 'child stderr.log'
        $child = Start-HiddenChild -FilePath $PythonPath -ArgumentList (@('-c', $code) + $arguments) -WorkingDirectory ([Environment]::CurrentDirectory) -StdoutPath $childStdout -StderrPath $childStderr
        if (-not $child.WaitForExit(30000)) {
            $child.Kill()
            $child.WaitForExit()
            throw 'hidden ProcessStartInfo argv self-test timed out'
        }
        Close-HiddenChildIo -Process $child
        if ([int]$child.ExitCode -ne 0) {
            throw "hidden ProcessStartInfo argv self-test exited $($child.ExitCode): $(Get-Content -LiteralPath $childStderr -Raw)"
        }
        $hiddenObserved = @(((Get-Content -LiteralPath $childStdout -Raw -Encoding UTF8).Trim() | ConvertFrom-Json))
        if ($hiddenObserved.Count -ne $arguments.Count) {
            throw 'hidden ProcessStartInfo argv self-test changed the argument count'
        }
        for ($index = 0; $index -lt $arguments.Count; $index += 1) {
            if ([string]$hiddenObserved[$index] -ne [string]$arguments[$index]) {
                throw "hidden ProcessStartInfo argv self-test changed argument $index"
            }
        }

        $liveStdout = Join-Path $temporaryRoot 'live-small-json.stdout.log'
        $liveStderr = Join-Path $temporaryRoot 'live-small-json.stderr.log'
        $liveChild = $null
        try {
            $liveCode = 'import json,time; print(json.dumps({"event":"t9_1_3_live_small_json","payload":"x"*512},separators=(",",":")),flush=True); time.sleep(4)'
            $liveChild = Start-HiddenChild -FilePath $PythonPath -ArgumentList @('-c', $liveCode) -WorkingDirectory ([Environment]::CurrentDirectory) -StdoutPath $liveStdout -StderrPath $liveStderr
            $visibilityDeadline = [DateTime]::UtcNow.AddSeconds(3)
            while ([DateTime]::UtcNow -lt $visibilityDeadline -and -not $liveSmallJsonVisibleWhileAlive) {
                if ($liveChild.HasExited) {
                    throw 'live small-JSON child exited before the supervisor observed its record'
                }
                $liveText = Read-SharedUtf8Text -Path $liveStdout
                foreach ($liveLine in @($liveText -split "`r?`n")) {
                    if ([string]::IsNullOrWhiteSpace($liveLine)) { continue }
                    try {
                        $liveRecord = $liveLine | ConvertFrom-Json
                    }
                    catch {
                        continue
                    }
                    if ([string]$liveRecord.event -eq 't9_1_3_live_small_json' -and ([string]$liveRecord.payload).Length -eq 512) {
                        $liveSmallJsonVisibleWhileAlive = -not $liveChild.HasExited
                        break
                    }
                }
                if (-not $liveSmallJsonVisibleWhileAlive) {
                    Start-Sleep -Milliseconds 50
                }
            }
            if (-not $liveSmallJsonVisibleWhileAlive) {
                throw 'sub-kilobyte child JSON was not visible before child exit'
            }
            if (-not $liveChild.WaitForExit(10000)) {
                throw 'live small-JSON child did not exit after its visibility hold'
            }
            Close-HiddenChildIo -Process $liveChild
            if ([int]$liveChild.ExitCode -ne 0) {
                throw "live small-JSON child exited $($liveChild.ExitCode)"
            }
        }
        finally {
            if ($null -ne $liveChild) {
                if (-not $liveChild.HasExited) {
                    $liveChild.Kill()
                    $liveChild.WaitForExit()
                }
                if (-not [bool]$liveChild.T913IoClosed) {
                    Close-HiddenChildIo -Process $liveChild
                }
            }
        }

        $serialRunDir = Join-Path $temporaryRoot 'serial-run'
        [IO.Directory]::CreateDirectory($serialRunDir) | Out-Null
        $script:RunDirectory = $serialRunDir
        $serialReleasePath = Join-Path $serialRunDir 'nmf_after_mf.release.json'
        $serialNonce = [Guid]::NewGuid().ToString('D')
        $serialDeadline = [DateTime]::UtcNow.AddMinutes(1).ToString('o')
        $serialAttestation = [pscustomobject][ordered]@{
            run_identity = [pscustomobject][ordered]@{
                transaction_id = [Guid]::NewGuid().ToString('D')
                run_dir = $serialRunDir
            }
            attestation_nonce = [Guid]::NewGuid().ToString('D')
            attestation_sha256 = ('3' * 64)
        }
        $serialProbe = [pscustomobject][ordered]@{
            config_sha256 = ('1' * 64)
            implementation_sha256 = ('2' * 64)
        }
        $successRelease = New-NmfSerialReleasePayload -MfExited $true -MfExitCode 0 -MfPid 101 -MfProcessCreatedUnixNs 102 -NmfExited $false -NmfPid $PID -ReleasePath $serialReleasePath -ReleaseNonce $serialNonce -TrainingAttestation $serialAttestation -Probe $serialProbe -DeadlineUtc $serialDeadline
        Write-JsonAtomic -Path $serialReleasePath -Value $successRelease
        $successWitness = [ordered]@{
            payload = $successRelease
            file_evidence = Get-FileEvidence -Path $serialReleasePath
        }
        $supervisorReleaseSuccessPass = (
            (Test-Path -LiteralPath $serialReleasePath -PathType Leaf) -and
            [string]$successRelease.prerequisite_family -eq 'mf' -and
            [int]$successRelease.prerequisite_exit_code -eq 0 -and
            (Get-Sha256Hex -Path $serialReleasePath) -match '^[0-9a-f]{64}$'
        )
        $serialConsumptionLog = Join-Path $serialRunDir 'nmf-consumption.stdout.log'
        $serialConsumption = [ordered]@{
            schema_version = 't9.1.3-nmf-serial-release-witness-v1'
            event = 't9_1_3_nmf_serial_release_consumed'
            release_path = $serialReleasePath
            release_sha256 = [string]$successWitness.file_evidence.sha256
            transaction_id = [string]$serialAttestation.run_identity.transaction_id
            attestation_sha256 = [string]$serialAttestation.attestation_sha256
            release_nonce_sha256 = Get-StringSha256Hex -Text $serialNonce
            mf_pid = 101
            mf_process_created_unix_ns = 102
            waiter_pid = $PID
            released_at_utc = [string]$successRelease.released_at_utc
            waited_seconds = 0.1
        }
        Write-Utf8NoBomAtomic -Path $serialConsumptionLog -Text (($serialConsumption | ConvertTo-Json -Compress) + [Environment]::NewLine)
        $observedConsumption = Wait-NmfSerialReleaseConsumed -Process (Get-Process -Id $PID -ErrorAction Stop) -StdoutPath $serialConsumptionLog -ReleasePath $serialReleasePath -ReleaseNonce $serialNonce -ReleaseWitness $successWitness -TrainingAttestation $serialAttestation -DeadlineUtc $serialDeadline
        $supervisorReleaseConsumedPass = [string]$observedConsumption.schema_version -eq 't9.1.3-supervisor-observed-nmf-consumption-v1'
        Remove-Item -LiteralPath $serialReleasePath -Force
        $mfFailureRejected = $false
        try {
            [void](New-NmfSerialReleasePayload -MfExited $true -MfExitCode 7 -MfPid 101 -MfProcessCreatedUnixNs 102 -NmfExited $false -NmfPid 103 -ReleasePath $serialReleasePath -ReleaseNonce $serialNonce -TrainingAttestation $serialAttestation -Probe $serialProbe -DeadlineUtc $serialDeadline)
        }
        catch {
            $mfFailureRejected = $true
        }
        $supervisorMfFailureNoReleasePass = $mfFailureRejected -and -not (Test-Path -LiteralPath $serialReleasePath)
        if (-not $supervisorReleaseSuccessPass -or -not $supervisorReleaseConsumedPass -or -not $supervisorMfFailureNoReleasePass) {
            throw 'supervisor MF-success/MF-failure serial release self-test failed'
        }

        $lockOutput = Join-Path $temporaryRoot 'lock-audit-output'
        $finalizeLock = Join-Path $lockOutput '_locks\finalize.lock'
        [IO.Directory]::CreateDirectory($finalizeLock) | Out-Null
        $ownerPath = Join-Path $finalizeLock 'owner.json'
        $localHostname = [Environment]::MachineName
        Write-JsonAtomic -Path $ownerPath -Value ([ordered]@{
            hostname = $localHostname
            pid = -1
            process_created_unix_ns = $null
        })
        $deadAudit = @(Get-OutputLockAudit -OutputDirectory $lockOutput -LocalHostname $localHostname)
        if ($deadAudit.Count -ne 1 -or [string]$deadAudit[0].owner_state -notlike 'LOCAL_DEAD*') {
            throw 'dead local finalize-lock self-test was prematurely blocked or misclassified'
        }
        $deadCrashStateRequiresManualRecovery = $false
        try {
            if (-not (Test-StrictInvalidatedFinalizationCrashMarker -Marker $syntheticMarker -ExpectedConfigSha256 ('a' * 64))) {
                throw 'synthetic crash marker unexpectedly invalid'
            }
            Assert-ArtifactResumeHasNoFinalizeLock -LockAudit $deadAudit
        }
        catch {
            $deadCrashStateRequiresManualRecovery = $_.Exception.Message -like 'MANUAL_OPERATOR_RECOVERY_REQUIRED*'
        }
        if (-not $deadCrashStateRequiresManualRecovery) {
            throw 'crash marker plus dead finalize.lock was not routed to manual recovery'
        }
        Write-JsonAtomic -Path $ownerPath -Value ([ordered]@{
            hostname = 'definitely-a-different-host.example.invalid'
            pid = 42
            process_created_unix_ns = 1
        })
        $remoteAudit = @(Get-OutputLockAudit -OutputDirectory $lockOutput -LocalHostname $localHostname)
        if ($remoteAudit.Count -ne 1 -or [string]$remoteAudit[0].owner_state -ne 'CROSS_HOST_FAIL_CLOSED_DEFERRED_TO_PYTHON') {
            throw 'cross-host finalize-lock self-test did not defer fail-closed handling to Python'
        }
        $selfProcess = Get-Process -Id $PID -ErrorAction Stop
        $selfStart = $selfProcess.StartTime.ToUniversalTime()
        Write-JsonAtomic -Path $ownerPath -Value ([ordered]@{
            hostname = $localHostname
            pid = $PID
            process_created_unix_ns = Get-ProcessUnixNanoseconds -StartTimeUtc $selfStart
        })
        $activeRejected = $false
        try {
            [void]@(Get-OutputLockAudit -OutputDirectory $lockOutput -LocalHostname $localHostname)
        }
        catch {
            $activeRejected = $true
        }
        if (-not $activeRejected) {
            throw 'active local finalize-lock self-test was not rejected'
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporaryRoot -PathType Container) {
            $resolvedTemporary = Get-CanonicalPath -Path $temporaryRoot -BasePath ([IO.Path]::GetTempPath())
            $resolvedSystemTemp = Get-CanonicalPath -Path ([IO.Path]::GetTempPath()) -BasePath ([IO.Path]::GetTempPath())
            if (-not $resolvedTemporary.StartsWith(($resolvedSystemTemp.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) {
                throw "self-test cleanup target escaped the system temp root: $resolvedTemporary"
            }
            [IO.Directory]::Delete($resolvedTemporary, $true)
        }
    }
    $failure = Invoke-NativeCapture -FilePath $PythonPath -ArgumentList @('-c', 'raise SystemExit(7)') -WorkingDirectory ([Environment]::CurrentDirectory)
    if ([int]$failure.ExitCode -ne 7) {
        throw "native exit-code self-test expected 7 and observed $($failure.ExitCode)"
    }
    $base = Get-CanonicalPath -Path (Join-Path ([Environment]::CurrentDirectory) 'path overlap root') -BasePath ([Environment]::CurrentDirectory)
    $child = Get-CanonicalPath -Path (Join-Path $base 'child') -BasePath ([Environment]::CurrentDirectory)
    $sibling = Get-CanonicalPath -Path ($base + '_sibling') -BasePath ([Environment]::CurrentDirectory)
    if (-not (Test-PathsOverlap -First $base -Second $child) -or (Test-PathsOverlap -First $base -Second $sibling)) {
        throw 'path-overlap self-test failed'
    }
    [pscustomobject]@{
        schema_version = 't9.1.3-supervisor-static-self-test-v1'
        status = 'PASS'
        argv_roundtrip_cases = $arguments.Count
        hidden_process_start_info_argv_roundtrip = $true
        dead_local_finalize_lock_reported_without_script_recovery = $true
        cross_host_finalize_lock_deferred_fail_closed_to_python = $true
        active_local_lock_rejected = $true
        job_object_pinvoke_structure_pass = $jobPInvokeStructurePass
        job_object_synthetic_binding_failure_rejected = $syntheticJobBindingFailureRejected
        job_object_policy = 'JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE'
        job_bound_wrapper_payload_absent_before_release = $payloadAbsentBeforeRelease
        job_bound_wrapper_same_pid_payload_pass = $true
        job_bound_wrapper_missing_job_rejected_before_payload = $missingJobRejectedBeforePayload
        deterministic_environment_exact = $true
        deterministic_environment_inherited_by_child = $true
        job_object_claim_scope = @('mf', 'nmf', 'finalize')
        nvidia_load_gate_pure_pass_case = $true
        nvidia_load_gate_low_free_fail_case = $true
        nvidia_load_gate_peak_utilization_fail_case = $true
        nvidia_load_gate_median_utilization_fail_case = $true
        nvidia_load_gate_uuid_drift_fail_case = $true
        nvidia_load_gate_device_count_drift_fail_case = $true
        nvidia_load_gate_missing_samples_fail_case = $true
        nvidia_csv_quoted_field_pass_case = $true
        nvidia_csv_extra_field_fail_case = $true
        gpu_attestation_training_gate_pass_case = [bool]$attestationSelfTest.training_gate_pass
        gpu_attestation_finalizer_gate_pass_case = [bool]$attestationSelfTest.finalizer_gate_pass
        gpu_attestation_missing_fail_case = [bool]$attestationSelfTest.missing_rejected
        gpu_attestation_stale_fail_case = [bool]$attestationSelfTest.stale_rejected
        gpu_attestation_tamper_fail_case = [bool]$attestationSelfTest.tamper_rejected
        gpu_attestation_uuid_mismatch_fail_case = [bool]$attestationSelfTest.uuid_mismatch_rejected
        gpu_attestation_purpose_swap_fail_case = [bool]$attestationSelfTest.purpose_swap_rejected
        gpu_attestation_real_gpu_queried = [bool]$attestationSelfTest.gpu_queried
        nmf_serial_release_success_published = $supervisorReleaseSuccessPass
        nmf_serial_release_consumption_verified = $supervisorReleaseConsumedPass
        nmf_serial_release_mf_failure_absent = $supervisorMfFailureNoReleasePass
        sub_kib_json_visible_before_child_exit = $liveSmallJsonVisibleWhileAlive
        strict_invalidated_crash_marker_pass_case = $strictCrashMarkerPass
        finalization_failed_marker_rejected = $failedMarkerRejected
        invalidated_crash_marker_extra_field_rejected = $extraMarkerRejected
        crash_marker_plus_dead_finalize_lock_requires_manual_recovery = $deadCrashStateRequiresManualRecovery
        native_nonzero_exit_observed = [int]$failure.ExitCode
        production_started = $false
    } | ConvertTo-Json -Depth 4
}

if ($StaticSelfTest) {
    Invoke-StaticSelfTest -PythonPath $Python
    exit 0
}

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    throw "RunDir is required for production execution; it is optional only with -StaticSelfTest."
}

# Bootstrap only computes and atomically creates the supervisor namespace.  A
# pre-existing RunDir is never reused, even for -ArtifactResume.
$RepoRoot = Get-CanonicalPath -Path $RepoRoot -BasePath ([Environment]::CurrentDirectory)
if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "repository root is missing: $RepoRoot"
}
$Python = Get-CanonicalPath -Path $Python -BasePath $RepoRoot
$Config = Get-CanonicalPath -Path (Join-Path $RepoRoot 'configs\phase9\t9_1_3_puviani_paper_constrained.json') -BasePath $RepoRoot
$OutputDir = Get-CanonicalPath -Path (Join-Path $RepoRoot 'docs\t9_1_3_puviani_agents') -BasePath $RepoRoot
$Report = Get-CanonicalPath -Path (Join-Path $RepoRoot 'docs\t9_1_3_puviani_paper_constrained.json') -BasePath $RepoRoot
$AgentRegistry = Get-CanonicalPath -Path (Join-Path $RepoRoot 'docs\t9_1_3_puviani_agent_registry.csv') -BasePath $RepoRoot
$SelectionLedger = Get-CanonicalPath -Path (Join-Path $RepoRoot 'docs\t9_1_3_puviani_selection_ledger.csv') -BasePath $RepoRoot
$TrainingLedger = Get-CanonicalPath -Path (Join-Path $RepoRoot 'docs\t9_1_3_puviani_training_ledger.parquet') -BasePath $RepoRoot
$Trajectories = Get-CanonicalPath -Path (Join-Path $RepoRoot 'docs\t9_1_3_puviani_six_state_trajectories.parquet') -BasePath $RepoRoot
$Events = Get-CanonicalPath -Path (Join-Path $RepoRoot 'docs\t9_1_3_puviani_six_state_events.parquet') -BasePath $RepoRoot
$RunDir = Get-CanonicalPath -Path $RunDir -BasePath $RepoRoot

$finalTargets = @($Report, $AgentRegistry, $SelectionLedger, $TrainingLedger, $Trajectories, $Events)
foreach ($protectedPath in @($OutputDir) + $finalTargets) {
    if (Test-PathsOverlap -First $RunDir -Second $protectedPath) {
        throw "RunDir overlaps a production output/final artifact path: RunDir=$RunDir protected=$protectedPath"
    }
}
if (Test-Path -LiteralPath $RunDir) {
    throw "RunDir must be brand-new and cannot be attached or reused: $RunDir"
}
New-Item -ItemType Directory -Path $RunDir -ErrorAction Stop | Out-Null
if (-not (Test-Path -LiteralPath $RunDir -PathType Container)) {
    throw "failed to create the brand-new RunDir: $RunDir"
}

$script:RunDirectory = $RunDir
$script:JobBarrierRoot = Join-Path $RunDir 'job_barriers'
$script:PythonPath = $Python
$script:PollIntervalSeconds = $PollSeconds
$script:DeadlineUtc = [DateTime]::UtcNow.AddHours($TotalDeadlineHours)
$script:ArtifactMode = $(if ($ArtifactResume) { 'ARTIFACT_RESUME_IN_NEW_SUPERVISOR' } else { 'FRESH' })
$script:OriginalCudaVisibleDevices = [Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES', 'Process')
$script:TransactionId = [Guid]::NewGuid().ToString('D')
$script:SupervisorProcessCreatedUnixNs = Get-ProcessUnixNanoseconds -StartTimeUtc ((Get-Process -Id $PID -ErrorAction Stop).StartTime.ToUniversalTime())
$script:BaselineProbe = $null
$script:ChildRecords = [ordered]@{}
$script:PhaseHistory = New-Object System.Collections.ArrayList
$script:FailureExitCode = 1
$script:Processes = [ordered]@{}
$script:KillOnCloseJobHandle = [IntPtr]::Zero
$script:LogPaths = [ordered]@{
    mf_stdout = Join-Path $RunDir 'mf.stdout.log'
    mf_stderr = Join-Path $RunDir 'mf.stderr.log'
    nmf_stdout = Join-Path $RunDir 'nmf.stdout.log'
    nmf_stderr = Join-Path $RunDir 'nmf.stderr.log'
    finalize_stdout = Join-Path $RunDir 'finalize.stdout.log'
    finalize_stderr = Join-Path $RunDir 'finalize.stderr.log'
}
$script:Supervisor = [ordered]@{
    schema_version = 't9.1.3-production-supervisor-v3'
    task_id = 'T9.1.3'
    transaction_id = $script:TransactionId
    state = 'CREATED'
    status = 'IN_PROGRESS_NOT_A_VALID_RESULT'
    started_at_utc = Get-UtcIso
    updated_at_utc = Get-UtcIso
    deadline_utc = $script:DeadlineUtc.ToString('o')
    total_deadline_hours = $TotalDeadlineHours
    poll_seconds = $PollSeconds
    artifact_mode = $script:ArtifactMode
    requested_target_gpu_uuid = $(if ([string]::IsNullOrWhiteSpace($TargetGpuUuid)) { $null } else { $TargetGpuUuid })
    original_cuda_visible_devices = $(if ([string]::IsNullOrWhiteSpace($script:OriginalCudaVisibleDevices)) { $null } else { $script:OriginalCudaVisibleDevices })
    effective_cuda_visible_devices = $null
    repo_root = $RepoRoot
    run_dir = $RunDir
    output_dir = $OutputDir
    final_targets = @($finalTargets)
    preflight = $null
    children = $script:ChildRecords
    probes = [ordered]@{}
    gpu_load_attestations = [ordered]@{
        training_launch = $null
        finalizer_launch = $null
    }
    serial_training_gate = [ordered]@{
        execution_order = 'MF_THEN_NMF'
        release_path = Join-Path $RunDir 'nmf_after_mf.release.json'
        ready_witness = $null
        release_witness = $null
        consumption_witness = $null
        mf_failure_release_absent = $null
    }
    child_job_object = [ordered]@{
        policy = 'JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE'
        claim_scope = @('mf', 'nmf', 'finalize')
        short_lived_git_nvidia_and_python_probes_in_scope = $false
        initialized = $false
        every_long_lived_worker_bound_before_python_payload_release = $true
        closed = $false
    }
    log_evidence = [ordered]@{}
    cleanup_errors = @()
    exception = $null
    phase_history = @()
}

$env:PYTHONUNBUFFERED = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
Set-Location -LiteralPath $RepoRoot

try {
    Set-SupervisorPhase -State 'PREFLIGHT_RUNNING' -Detail 'No training/finalization child has been started.'

    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "DLEnv Python is missing: $Python"
    }
    if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        throw "production config is missing: $Config"
    }
    $script:KillOnCloseJobHandle = Initialize-KillOnCloseJob
    $script:Supervisor['child_job_object']['initialized'] = $true
    if (-not [string]::IsNullOrWhiteSpace($TargetGpuUuid)) {
        $TargetGpuUuid = $TargetGpuUuid.Trim()
        if ($TargetGpuUuid -cnotmatch '^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$') {
            throw '-TargetGpuUuid must be a full NVIDIA GPU-* UUID'
        }
        # An explicit target also pins CUDA logical device zero for every later
        # Python workload/probe in this supervisor process.
        [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', $TargetGpuUuid, 'Process')
    }
    $script:Supervisor['requested_target_gpu_uuid'] = $(if ([string]::IsNullOrWhiteSpace($TargetGpuUuid)) { $null } else { $TargetGpuUuid })
    $script:Supervisor['effective_cuda_visible_devices'] = $(if ([string]::IsNullOrWhiteSpace($env:CUDA_VISIBLE_DEVICES)) { $null } else { $env:CUDA_VISIBLE_DEVICES })

    # Every git command is a checked native invocation.  A shell, pipeline, or
    # implicit $LASTEXITCODE is never used for provenance capture.
    $gitRootResult = Invoke-NativeCapture -FilePath 'git.exe' -ArgumentList @('-C', $RepoRoot, 'rev-parse', '--show-toplevel') -WorkingDirectory $RepoRoot
    Assert-NativeSuccess -Result $gitRootResult -Label 'git rev-parse --show-toplevel'
    $gitRoot = Get-CanonicalPath -Path (([string]$gitRootResult.Stdout).Trim()) -BasePath $RepoRoot
    if (-not [string]::Equals($gitRoot, $RepoRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "RepoRoot differs from git toplevel: configured=$RepoRoot git=$gitRoot"
    }
    $gitHeadResult = Invoke-NativeCapture -FilePath 'git.exe' -ArgumentList @('-C', $RepoRoot, 'rev-parse', 'HEAD') -WorkingDirectory $RepoRoot
    Assert-NativeSuccess -Result $gitHeadResult -Label 'git rev-parse HEAD'
    $gitBranchResult = Invoke-NativeCapture -FilePath 'git.exe' -ArgumentList @('-C', $RepoRoot, 'branch', '--show-current') -WorkingDirectory $RepoRoot
    Assert-NativeSuccess -Result $gitBranchResult -Label 'git branch --show-current'
    $gitStatusResult = Invoke-NativeCapture -FilePath 'git.exe' -ArgumentList @('-C', $RepoRoot, 'status', '--short', '--untracked-files=all') -WorkingDirectory $RepoRoot
    Assert-NativeSuccess -Result $gitStatusResult -Label 'git status --short'
    $gitHead = (([string]$gitHeadResult.Stdout).Trim())
    $gitBranch = (([string]$gitBranchResult.Stdout).Trim())
    $gitStatusText = (([string]$gitStatusResult.Stdout).TrimEnd())
    $gitStatusRows = $(if ([string]::IsNullOrWhiteSpace($gitStatusText)) { @() } else { @($gitStatusText -split "`r?`n") })

    # Gate the production GPU before importing the torch-backed production
    # module.  A rejected gate therefore cannot create a CUDA context through
    # either the training/finalization workers or the Python runtime probe.
    $gpuLoadGate = Invoke-NvidiaLoadGate -RequestedTargetUuid $TargetGpuUuid -WorkingDirectory $RepoRoot
    if (-not $gpuLoadGate.passed) {
        $preflight = [ordered]@{
            completed_at_utc = Get-UtcIso
            git_head = $gitHead
            git_branch = $gitBranch
            git_status_short = @($gitStatusRows)
            config_sha256 = $null
            implementation_sha256 = $null
            python_executable = $Python
            module_path = $null
            final_target_states = @()
            lock_audit = @()
            nvidia_load_gate = $gpuLoadGate
            execution_plan = 'REJECTED_BEFORE_PYTHON_RUNTIME_PROBE'
            no_child_started = ($script:Processes.Count -eq 0)
        }
        $script:Supervisor['preflight'] = $preflight
        Write-JsonAtomic -Path (Join-Path $RunDir 'preflight.json') -Value $preflight
        Set-SupervisorPhase -State 'PREFLIGHT_GPU_LOAD_REJECTED' -Detail (($gpuLoadGate.failure_reasons) -join '; ')
        throw ('NVIDIA production load gate failed closed: ' + (($gpuLoadGate.failure_reasons) -join '; '))
    }

    $probe1 = Invoke-ProductionProbe -PythonPath $Python -ConfigPath $Config -WorkingDirectory $RepoRoot
    $probe2 = Invoke-ProductionProbe -PythonPath $Python -ConfigPath $Config -WorkingDirectory $RepoRoot
    Assert-ProbeIdentity -Expected $probe1 -Actual $probe2 -Stage 'repeated baseline preflight'
    $script:BaselineProbe = $probe2
    if (-not [string]::Equals((Get-CanonicalPath -Path ([string]$probe2.python_executable) -BasePath $RepoRoot), $Python, [StringComparison]::OrdinalIgnoreCase)) {
        throw "probe executed a different Python: requested=$Python observed=$($probe2.python_executable)"
    }
    if (-not [string]::Equals((Get-CanonicalPath -Path ([string]$probe2.repo_root) -BasePath $RepoRoot), $RepoRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "probe imported the module from a different repository: $($probe2.repo_root)"
    }

    $targetStates = @(
        foreach ($target in $finalTargets) {
            [ordered]@{
                path = $target
                exists = Test-Path -LiteralPath $target
                is_leaf = Test-Path -LiteralPath $target -PathType Leaf
                atomic_temporaries = @(
                    Get-ChildItem -LiteralPath (Split-Path -Parent $target) -File -Filter ('.' + [IO.Path]::GetFileName($target) + '.*.tmp') -ErrorAction SilentlyContinue |
                        ForEach-Object { $_.FullName }
                )
            }
        }
    )
    if (-not $ArtifactResume) {
        if (Test-Path -LiteralPath $OutputDir) {
            throw "fresh production requires an absent output directory: $OutputDir"
        }
        $staleTargets = @($targetStates | Where-Object { $_.exists -or $_.atomic_temporaries.Count -gt 0 })
        if ($staleTargets.Count -gt 0) {
            throw 'fresh production found an old report, ledger, or atomic finalization temporary; use a new audited artifact plan rather than merging evidence'
        }
    }
    else {
        if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
            throw "-ArtifactResume requires a retained canonical output directory: $OutputDir"
        }
        $existingTargets = @($targetStates | Where-Object { $_.exists })
        if (Test-Path -LiteralPath $Report -PathType Leaf) {
            try {
                $oldReport = Get-Content -LiteralPath $Report -Raw -Encoding UTF8 | ConvertFrom-Json
            }
            catch {
                throw "artifact resume refuses an unreadable old report: $Report"
            }
            if (-not (Test-StrictInvalidatedFinalizationCrashMarker -Marker $oldReport -ExpectedConfigSha256 ([string]$probe2.config_sha256))) {
                throw "artifact resume refuses a report that is not the exact strict INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL crash marker: status=$($oldReport.status)"
            }
        }
        elseif ($existingTargets.Count -gt 0) {
            throw 'artifact resume found final ledgers without the exact strict invalidated-before-finalization crash marker'
        }
    }

    $lockAudit = Get-OutputLockAudit -OutputDirectory $OutputDir -LocalHostname ([string]$probe2.hostname)
    Assert-ArtifactResumeHasNoFinalizeLock -LockAudit @($lockAudit)
    $preflight = [ordered]@{
        completed_at_utc = Get-UtcIso
        git_head = $gitHead
        git_branch = $gitBranch
        git_status_short = @($gitStatusRows)
        config_sha256 = [string]$probe2.config_sha256
        implementation_sha256 = [string]$probe2.implementation_sha256
        python_executable = [string]$probe2.python_executable
        module_path = [string]$probe2.module_path
        final_target_states = @($targetStates)
        lock_audit = @($lockAudit)
        nvidia_load_gate = $gpuLoadGate
        execution_plan = 'TRAIN_MF_THEN_ATOMIC_RELEASE_THEN_NMF_THEN_FINALIZE'
        no_child_started = ($script:Processes.Count -eq 0)
    }
    if (-not $preflight.no_child_started) {
        throw 'internal ordering violation: a child existed before preflight completion'
    }
    $script:Supervisor['preflight'] = $preflight
    $script:Supervisor['probes']['baseline'] = $probe2
    Write-JsonAtomic -Path (Join-Path $RunDir 'preflight.json') -Value $preflight
    Set-SupervisorPhase -State 'PREFLIGHT_COMPLETE' -Detail 'Git, runtime, hashes, artifact namespace, locks, and final targets were checked before child launch.'
    if ([DateTime]::UtcNow -ge $script:DeadlineUtc) {
        $script:FailureExitCode = 124
        throw 'total T9.1.3 deadline expired during preflight; no child was started'
    }

    $module = 'cnn_fpga.benchmark.puviani_paper_constrained_artifacts'
    $common = @(
        '-m', $module,
        '--config', $Config,
        '--output-dir', $OutputDir,
        '--report', $Report,
        '--agent-registry', $AgentRegistry,
        '--selection-ledger', $SelectionLedger,
        '--training-ledger', $TrainingLedger,
        '--trajectories', $Trajectories,
        '--events', $Events
    )
    $mfArgs = $common + @('--family', 'mf', '--train-only')
    $nmfArgs = $common + @('--family', 'nmf', '--train-only')
    $nmfReleasePath = Get-CanonicalPath -Path (Join-Path $RunDir 'nmf_after_mf.release.json') -BasePath $RunDir
    $nmfReleaseNonce = [Guid]::NewGuid().ToString('D')
    $trainingDeadlineUtc = $script:DeadlineUtc.ToString('o')
    if (Test-Path -LiteralPath $nmfReleasePath) {
        throw 'fresh supervisor unexpectedly found a pre-existing NMF serial-release gate'
    }
    $nmfArgs = $nmfArgs + @(
        '--supervisor-nmf-release-gate', $nmfReleasePath,
        '--supervisor-nmf-release-nonce', $nmfReleaseNonce,
        '--supervisor-training-deadline-utc', $trainingDeadlineUtc
    )

    Write-LaunchTransaction -State 'PREPARED_NO_CHILDREN'
    # Artifact resume still executes both family commands.  Valid retained
    # bundles are re-audited/reused by Python; no finalize-lock handoff exists.
        Set-SupervisorPhase -State 'TRAINING_GPU_LOAD_ATTESTATION_RUNNING' -Detail 'Five new samples are acquired after all other preflight work and immediately before both training children.'
        $trainingGpuGate = Invoke-NvidiaLoadGate -RequestedTargetUuid $TargetGpuUuid -WorkingDirectory $RepoRoot
        if (-not $trainingGpuGate.passed) {
            throw ('fresh training-launch NVIDIA load gate failed closed: ' + (($trainingGpuGate.failure_reasons) -join '; '))
        }
        $trainingAttestationPath = Join-Path $RunDir 'training_launch_gpu_attestation.json'
        $trainingAttestation = New-GpuLoadAttestation -LoadGate $trainingGpuGate -Purpose 'TRAINING_LAUNCH' -Probe $script:BaselineProbe -Path $trainingAttestationPath -PythonPath $Python -WorkingDirectory $RepoRoot
        $script:Supervisor['gpu_load_attestations']['training_launch'] = [ordered]@{
            path = $trainingAttestationPath
            attestation_sha256 = [string]$trainingAttestation.attestation_sha256
            purpose = [string]$trainingAttestation.purpose
            target_gpu = $trainingAttestation.target_gpu
            sampling_completed_at_utc = [string]$trainingAttestation.sampling_completed_at_utc
            expires_at_utc = [string]$trainingAttestation.expires_at_utc
        }
        $mfArgs = $mfArgs + @('--gpu-attestation', $trainingAttestationPath)
        $nmfArgs = $nmfArgs + @('--gpu-attestation', $trainingAttestationPath)
        Set-SupervisorPhase -State 'LAUNCHING_SERIAL_TRAINING_TRANSACTION' -Detail 'The release-gated NMF waiter launches first so it can validate the fresh attestation without competing with MF CUDA training.'
        $nmf = Start-JobBoundPythonChild -Role 'nmf' -PythonPath $Python -PayloadArgumentList $nmfArgs -WorkingDirectory $RepoRoot -StdoutPath $script:LogPaths.nmf_stdout -StderrPath $script:LogPaths.nmf_stderr -BarrierRoot $script:JobBarrierRoot
        $script:Processes['nmf'] = $nmf
        $script:ChildRecords['nmf'] = New-ChildRecord -Role 'nmf' -Process $nmf -ArgumentList $nmfArgs -StdoutPath $script:LogPaths.nmf_stdout -StderrPath $script:LogPaths.nmf_stderr -RunDirectory $RunDir
        Write-LaunchTransaction -State 'NMF_RECORDED_MF_NOT_STARTED'
        Set-SupervisorPhase -State 'NMF_POST_ATTESTATION_RELEASE_WAIT_STARTING' -Detail 'The Job-bound NMF child must prove it reached the pre-output release barrier while the MF child is still absent.'
        $nmfReady = Wait-NmfSerialReleaseReady -Process $nmf -StdoutPath $script:LogPaths.nmf_stdout -ReleasePath $nmfReleasePath -ReleaseNonce $nmfReleaseNonce -TrainingAttestation $trainingAttestation -Probe $script:BaselineProbe -DeadlineUtc $trainingDeadlineUtc
        $script:Supervisor['serial_training_gate']['ready_witness'] = $nmfReady
        Write-LaunchTransaction -State 'NMF_READY_BLOCKED_MF_NOT_STARTED'
        Set-SupervisorPhase -State 'NMF_READY_AND_BLOCKED_BEFORE_OUTPUT' -Detail 'The fully bound NMF ready JSON was observed with startup reserve remaining; no release file or MF process exists yet.'

        $mf = Start-JobBoundPythonChild -Role 'mf' -PythonPath $Python -PayloadArgumentList $mfArgs -WorkingDirectory $RepoRoot -StdoutPath $script:LogPaths.mf_stdout -StderrPath $script:LogPaths.mf_stderr -BarrierRoot $script:JobBarrierRoot
        $script:Processes['mf'] = $mf
        $script:ChildRecords['mf'] = New-ChildRecord -Role 'mf' -Process $mf -ArgumentList $mfArgs -StdoutPath $script:LogPaths.mf_stdout -StderrPath $script:LogPaths.mf_stderr -RunDirectory $RunDir
        Write-LaunchTransaction -State 'SERIAL_CHILDREN_COMMITTED_MF_RUNNING_NMF_BLOCKED'
        Set-SupervisorPhase -State 'MF_STARTED_AFTER_NMF_READY' -Detail 'MF and the blocked NMF waiter share the same training attestation; only MF may touch the production output namespace now.'

        Wait-MfWithBlockedNmf -MfProcess $mf -NmfProcess $nmf -ReleasePath $nmfReleasePath
        if ([int]$mf.ExitCode -ne 0) {
            $script:FailureExitCode = [Math]::Max(1, [int]$mf.ExitCode)
            $script:Supervisor['serial_training_gate']['mf_failure_release_absent'] = -not (Test-Path -LiteralPath $nmfReleasePath)
            if (-not $nmf.HasExited) {
                Stop-OwnedChild -Role 'nmf' -Process $nmf -RunDirectory $RunDir -Reason 'MF_FAILED_NO_SERIAL_RELEASE'
            }
            throw "MF training child failed with exit code $($mf.ExitCode); NMF was terminated without release"
        }
        Set-SupervisorPhase -State 'MF_EXITED_ZERO_NMF_RELEASE_PENDING' -Detail 'MF completed successfully; NMF remains blocked and the atomic release has not yet been written.'
        $nmfRelease = Publish-NmfSerialRelease -MfProcess $mf -NmfProcess $nmf -ReleasePath $nmfReleasePath -ReleaseNonce $nmfReleaseNonce -TrainingAttestation $trainingAttestation -Probe $script:BaselineProbe -DeadlineUtc $trainingDeadlineUtc
        $script:Supervisor['serial_training_gate']['release_witness'] = $nmfRelease
        Write-LaunchTransaction -State 'MF_EXITED_ZERO_NMF_ATOMICALLY_RELEASED'
        Set-SupervisorPhase -State 'NMF_ATOMICALLY_RELEASED_AFTER_MF_SUCCESS' -Detail 'The release payload binds MF exit zero, both PIDs, run identity, attestation nonce/hash, config, implementation, and total deadline.'

        $nmfConsumed = Wait-NmfSerialReleaseConsumed -Process $nmf -StdoutPath $script:LogPaths.nmf_stdout -ReleasePath $nmfReleasePath -ReleaseNonce $nmfReleaseNonce -ReleaseWitness $nmfRelease -TrainingAttestation $trainingAttestation -DeadlineUtc $trainingDeadlineUtc
        $script:Supervisor['serial_training_gate']['consumption_witness'] = $nmfConsumed
        Set-SupervisorPhase -State 'NMF_RELEASE_CONSUMPTION_VERIFIED' -Detail 'The NMF child emitted a fully bound release-consumption witness before its output/training path continued.'

        Wait-SingleChild -Role 'nmf' -Process $nmf
        if ([int]$nmf.ExitCode -ne 0) {
            $script:FailureExitCode = [Math]::Max(1, [int]$nmf.ExitCode)
            throw "NMF training child failed after MF-success release with exit code $($nmf.ExitCode)"
        }
        Write-LaunchTransaction -State 'SERIAL_TRAINING_CHILDREN_EXITED_ZERO'
        $trainingOutcome = [ordered]@{
            schema_version = 't9.1.3-production-serialized-training-outcome-v3'
            task_id = 'T9.1.3'
            transaction_id = $script:TransactionId
            completed_at_utc = Get-UtcIso
            execution_plan = 'TRAIN_MF_THEN_ATOMIC_RELEASE_THEN_NMF_THEN_FINALIZE'
            mf = $script:ChildRecords['mf']
            nmf = $script:ChildRecords['nmf']
            nmf_serial_ready_witness = $nmfReady
            nmf_serial_release_witness = $nmfRelease
            nmf_serial_consumption_witness = $nmfConsumed
            shared_training_attestation_sha256 = [string]$trainingAttestation.attestation_sha256
            log_evidence = Get-LogEvidenceMap
        }
        Write-JsonAtomic -Path (Join-Path $RunDir 'training_outcome.json') -Value $trainingOutcome
        Set-SupervisorPhase -State 'TRAINING_COMPLETE'

    $preFinalizeProbe = Invoke-ProductionProbe -PythonPath $Python -ConfigPath $Config -WorkingDirectory $RepoRoot
    Assert-ProbeIdentity -Expected $script:BaselineProbe -Actual $preFinalizeProbe -Stage 'pre-finalize freeze check'
    $gitPreFinalizeResult = Invoke-NativeCapture -FilePath 'git.exe' -ArgumentList @('-C', $RepoRoot, 'rev-parse', 'HEAD') -WorkingDirectory $RepoRoot
    Assert-NativeSuccess -Result $gitPreFinalizeResult -Label 'pre-finalize git rev-parse HEAD'
    if (([string]$gitPreFinalizeResult.Stdout).Trim() -ne $gitHead) {
        throw 'git HEAD changed between preflight and finalization'
    }
    $script:Supervisor['probes']['pre_finalize'] = $preFinalizeProbe
    Set-SupervisorPhase -State 'PRE_FINALIZE_FREEZE_VERIFIED' -Detail 'Config, implementation, and git HEAD still match preflight.'
    if ([DateTime]::UtcNow -ge $script:DeadlineUtc) {
        $script:FailureExitCode = 124
        throw 'total T9.1.3 deadline expired before finalizer launch'
    }

    Set-SupervisorPhase -State 'FINALIZER_GPU_LOAD_ATTESTATION_RUNNING' -Detail 'The finalizer receives an independent five-sample launch attestation.'
    $finalizerGpuGate = Invoke-NvidiaLoadGate -RequestedTargetUuid $TargetGpuUuid -WorkingDirectory $RepoRoot
    if (-not $finalizerGpuGate.passed) {
        throw ('fresh finalizer-launch NVIDIA load gate failed closed: ' + (($finalizerGpuGate.failure_reasons) -join '; '))
    }
    $finalizerAttestationPath = Join-Path $RunDir 'finalizer_launch_gpu_attestation.json'
    $finalizerAttestation = New-GpuLoadAttestation -LoadGate $finalizerGpuGate -Purpose 'FINALIZER_LAUNCH' -Probe $script:BaselineProbe -Path $finalizerAttestationPath -PythonPath $Python -WorkingDirectory $RepoRoot
    $script:Supervisor['gpu_load_attestations']['finalizer_launch'] = [ordered]@{
        path = $finalizerAttestationPath
        attestation_sha256 = [string]$finalizerAttestation.attestation_sha256
        purpose = [string]$finalizerAttestation.purpose
        target_gpu = $finalizerAttestation.target_gpu
        sampling_completed_at_utc = [string]$finalizerAttestation.sampling_completed_at_utc
        expires_at_utc = [string]$finalizerAttestation.expires_at_utc
    }
    $finalizeArgs = $common + @('--finalize-only', '--gpu-attestation', $finalizerAttestationPath)
    Set-SupervisorPhase -State 'FINALIZER_GPU_LOAD_ATTESTATION_SEALED' -Detail 'Finalizer launch follows the independent attestation without another probe or discretionary step.'
    $finalize = Start-JobBoundPythonChild -Role 'finalize' -PythonPath $Python -PayloadArgumentList $finalizeArgs -WorkingDirectory $RepoRoot -StdoutPath $script:LogPaths.finalize_stdout -StderrPath $script:LogPaths.finalize_stderr -BarrierRoot $script:JobBarrierRoot
    $script:Processes['finalize'] = $finalize
    $script:ChildRecords['finalize'] = New-ChildRecord -Role 'finalize' -Process $finalize -ArgumentList $finalizeArgs -StdoutPath $script:LogPaths.finalize_stdout -StderrPath $script:LogPaths.finalize_stderr -RunDirectory $RunDir
    Write-LaunchTransaction -State 'FINALIZER_RECORDED'
    Set-SupervisorPhase -State 'FINALIZATION_RUNNING' -Detail 'Finalizer PID plus creation identity is durably recorded.'

    Wait-SingleChild -Role 'finalize' -Process $finalize
    $postFinalizeProbe = Invoke-ProductionProbe -PythonPath $Python -ConfigPath $Config -WorkingDirectory $RepoRoot
    Assert-ProbeIdentity -Expected $script:BaselineProbe -Actual $postFinalizeProbe -Stage 'post-finalize freeze check'
    if ([DateTime]::UtcNow -gt $script:DeadlineUtc) {
        $script:FailureExitCode = 124
        throw 'total T9.1.3 deadline expired during the post-finalize freeze probe'
    }
    $script:Supervisor['probes']['post_finalize'] = $postFinalizeProbe
    $gitPostFinalizeResult = Invoke-NativeCapture -FilePath 'git.exe' -ArgumentList @('-C', $RepoRoot, 'rev-parse', 'HEAD') -WorkingDirectory $RepoRoot
    Assert-NativeSuccess -Result $gitPostFinalizeResult -Label 'post-finalize git rev-parse HEAD'
    if (([string]$gitPostFinalizeResult.Stdout).Trim() -ne $gitHead) {
        throw 'git HEAD changed during finalization'
    }
    if ([int]$finalize.ExitCode -ne 0) {
        $script:FailureExitCode = [Math]::Max(1, [int]$finalize.ExitCode)
        throw "finalization failed with exit code $($finalize.ExitCode)"
    }
    if (-not (Test-Path -LiteralPath $Report -PathType Leaf)) {
        throw "zero-exit finalizer did not publish the report: $Report"
    }
    $sealedReport = Get-Content -LiteralPath $Report -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$sealedReport.status -ne 'PASS_ARTIFACT_LANE_AND_EXECUTABLE_REIMPLEMENTATION') {
        throw "zero-exit finalizer published an unexpected report status: $($sealedReport.status)"
    }
    if ([string]$sealedReport.config_sha256 -ne [string]$script:BaselineProbe.config_sha256 -or [string]$sealedReport.implementation_sha256 -ne [string]$script:BaselineProbe.implementation_sha256) {
        throw 'sealed report hashes differ from the supervisor freeze'
    }
    Write-LaunchTransaction -State 'COMPLETED'
    Set-SupervisorPhase -State 'POST_FINALIZE_FREEZE_VERIFIED' -Detail 'Config, implementation, git HEAD, and sealed report binding remained unchanged.'

    Close-KillOnCloseJob
    $script:Supervisor['child_job_object']['closed'] = $true

    $script:Supervisor['children'] = $script:ChildRecords
    $script:Supervisor['log_evidence'] = Get-LogEvidenceMap
    $script:Supervisor['status'] = 'PASS_CANDIDATE_REQUIRES_INDEPENDENT_LIVE_VALIDATION'
    $script:Supervisor['report'] = [ordered]@{
        path = $Report
        status = [string]$sealedReport.status
        analysis_sha256 = [string]$sealedReport.analysis_sha256
        config_sha256 = [string]$sealedReport.config_sha256
        implementation_sha256 = [string]$sealedReport.implementation_sha256
        file_evidence = Get-FileEvidence -Path $Report
    }
    Set-SupervisorPhase -State 'COMPLETED'
    $script:Supervisor['completed_at_utc'] = Get-UtcIso
    $script:Supervisor['phase_history_sha256'] = Get-Sha256Hex -Path (Join-Path $RunDir 'phase_history.json')
    Write-JsonAtomic -Path (Join-Path $RunDir 'supervisor_outcome.json') -Value $script:Supervisor
    Write-JsonAtomic -Path (Join-Path $RunDir 'final_outcome.json') -Value $script:Supervisor
    exit 0
}
catch {
    $caught = $_
    try {
        Set-SupervisorPhase -State 'ABORTING_FAIL_CLOSED' -Detail $caught.Exception.Message
    }
    catch {
        # The unified outcome write below remains the final best-effort seal.
    }

    try {
        Close-KillOnCloseJob
        $script:Supervisor['child_job_object']['closed'] = $true
    }
    catch {
        $cleanup = [ordered]@{ role = 'job_object'; message = $_.Exception.Message; at_utc = Get-UtcIso }
        $script:Supervisor['cleanup_errors'] = @($script:Supervisor['cleanup_errors']) + @($cleanup)
    }

    foreach ($role in @('mf', 'nmf', 'finalize')) {
        if ($script:Processes.Contains($role)) {
            $process = $script:Processes[$role]
            try {
                if (-not $script:ChildRecords.Contains($role)) {
                    $script:ChildRecords[$role] = New-EmergencyChildRecord -Role $role -Process $process
                }
                if (-not $process.HasExited) {
                    Stop-OwnedChild -Role $role -Process $process -RunDirectory $RunDir -Reason 'SUPERVISOR_FAIL_CLOSED_ABORT'
                }
                elseif ($script:ChildRecords.Contains($role) -and $script:ChildRecords[$role].state -ne 'EXITED') {
                    Complete-ChildRecord -Role $role -Process $process -RunDirectory $RunDir
                }
            }
            catch {
                # Retain the original failure and record the cleanup failure.
                $cleanup = [ordered]@{ role = $role; message = $_.Exception.Message; at_utc = Get-UtcIso }
                $script:Supervisor['cleanup_errors'] = @($script:Supervisor['cleanup_errors']) + @($cleanup)
            }
        }
    }
    try {
        Write-LaunchTransaction -State 'ABORTED_FAIL_CLOSED'
    }
    catch {
    }

    $failureProbe = $null
    $gpuGateRejected = $false
    if ($null -ne $script:Supervisor['preflight'] -and $null -ne $script:Supervisor['preflight']['nvidia_load_gate']) {
        $gpuGateRejected = -not [bool]$script:Supervisor['preflight']['nvidia_load_gate']['passed']
    }
    if ($gpuGateRejected) {
        $script:Supervisor['probes']['failure_time_error'] = 'SKIPPED_BECAUSE_NVIDIA_LOAD_GATE_FAILED_BEFORE_PYTHON_IMPORT'
    }
    else {
        try {
            $failureProbe = Invoke-ProductionProbe -PythonPath $Python -ConfigPath $Config -WorkingDirectory $RepoRoot
            $script:Supervisor['probes']['failure_time'] = $failureProbe
        }
        catch {
            $script:Supervisor['probes']['failure_time_error'] = $_.Exception.Message
        }
    }
    $script:Supervisor['children'] = $script:ChildRecords
    $script:Supervisor['log_evidence'] = Get-LogEvidenceMap
    $script:Supervisor['status'] = 'FAILED_FAIL_CLOSED'
    $script:Supervisor['exception'] = [ordered]@{
        type = $caught.Exception.GetType().FullName
        message = $caught.Exception.Message
        fully_qualified_error_id = $caught.FullyQualifiedErrorId
        script_stack_trace = $caught.ScriptStackTrace
        sealed_at_utc = Get-UtcIso
        exit_code = $script:FailureExitCode
    }
    try {
        Set-SupervisorPhase -State 'FAILED_FAIL_CLOSED' -Detail $caught.Exception.Message
    }
    catch {
        $script:Supervisor['state'] = 'FAILED_FAIL_CLOSED'
    }
    $script:Supervisor['completed_at_utc'] = Get-UtcIso
    try {
        if (Test-Path -LiteralPath (Join-Path $RunDir 'phase_history.json') -PathType Leaf) {
            $script:Supervisor['phase_history_sha256'] = Get-Sha256Hex -Path (Join-Path $RunDir 'phase_history.json')
        }
        Write-JsonAtomic -Path (Join-Path $RunDir 'supervisor_outcome.json') -Value $script:Supervisor
        Write-JsonAtomic -Path (Join-Path $RunDir 'final_outcome.json') -Value $script:Supervisor
    }
    catch {
        [Console]::Error.WriteLine("failed to write the unified supervisor outcome: $($_.Exception.Message)")
    }
    [Console]::Error.WriteLine([string]$caught)
    exit $script:FailureExitCode
}
