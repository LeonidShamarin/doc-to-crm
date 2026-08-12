<#
.SYNOPSIS
    Послідовний запуск pytest із жорстким kernel-лімітом пам'яті (Job Object).

.DESCRIPTION
    Навіщо це існує. Звичайний `pytest` у цьому проекті вмів з'їсти десятки
    гігабайт і покласти VS Code ("The window terminated unexpectedly
    (reason: 'oom')"). Двох речей вистачає, щоб такого більше не сталося:

    1. **Windows Job Object з JOB_OBJECT_LIMIT_JOB_MEMORY.** Ліміт тримає ядро,
       а не скрипт. Коли процес переходить межу, чергове виділення памʼяті
       просто провалюється (Python дістає MemoryError) — падає тест, а не машина.
       Ліміт діє на все дерево нащадків, і це важливо: `.venv\Scripts\python.exe`
       — це лаунчер-заглушка, справжній інтерпретатор працює в дочірньому
       процесі. Сторож, який дивиться лише на запущений PID, бачить 1 МБ і
       спокійно спостерігає, як дочірній процес виїдає 27 ГБ.

    2. **Вивід іде у файл, не в термінал.** У термінал — один рядок на файл.
       Renderer VS Code тримає буфер xterm.js і має ліміт heap ~4 ГБ незалежно
       від того, скільки RAM у машині; потік тексту з тестів валить саме його.

    Кожен тестовий файл — окремий процес, послідовно, з таймаутом.

.PARAMETER MemoryLimitMB
    Ліміт памʼяті на дерево процесів одного тестового файлу. Дефолт 1024 МБ.

.PARAMETER TimeoutSec
    Ліміт часу на один тестовий файл. Дефолт 120 с.

.PARAMETER MinFreeSystemMB
    Нижче цього рівня вільної фізичної памʼяті прогін зупиняється повністю.

.PARAMETER Path
    Конкретний файл або тека замість повного набору.

.PARAMETER PerTest
    Дрібніша гранулярність: кожна тест-функція окремим процесом. Повільніше,
    але точно показує, який саме тест з'їдає памʼять.

.EXAMPLE
    .\scripts\run_tests_safe.ps1
    .\scripts\run_tests_safe.ps1 -Path tests\test_backoff_llm.py -PerTest
    .\scripts\run_tests_safe.ps1 -MemoryLimitMB 512 -TimeoutSec 60
#>
[CmdletBinding()]
param(
    [int]$MemoryLimitMB   = 1024,
    [int]$TimeoutSec      = 120,
    [int]$MinFreeSystemMB = 2048,
    [string]$Path         = "",
    [switch]$PerTest
)

$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root '.test-logs'

if (-not (Test-Path $Python)) {
    Write-Host "python не знайдено: $Python" -ForegroundColor Red
    exit 2
}
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# --- Job Object: ліміт памʼяті на рівні ядра --------------------------------
if (-not ('JobMem' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class JobMem {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateJobObject(IntPtr a, string lpName);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetInformationJobObject(IntPtr hJob, int infoClass, IntPtr info, uint cb);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool QueryInformationJobObject(IntPtr hJob, int infoClass, IntPtr info, uint cb, IntPtr ret);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool TerminateJobObject(IntPtr hJob, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr h);

    [StructLayout(LayoutKind.Sequential)]
    public struct BASIC_LIMIT {
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
        public ulong r1, r2, r3, r4, r5, r6;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct EXT_LIMIT {
        public BASIC_LIMIT BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    const int ExtendedLimitInformation = 9;
    const uint LIMIT_JOB_MEMORY        = 0x00000200;
    const uint LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

    public static IntPtr Create(ulong memBytes) {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) return IntPtr.Zero;
        EXT_LIMIT info = new EXT_LIMIT();
        info.BasicLimitInformation.LimitFlags = LIMIT_JOB_MEMORY | LIMIT_KILL_ON_JOB_CLOSE;
        info.JobMemoryLimit = new UIntPtr(memBytes);
        int len = Marshal.SizeOf(typeof(EXT_LIMIT));
        IntPtr p = Marshal.AllocHGlobal(len);
        Marshal.StructureToPtr(info, p, false);
        bool ok = SetInformationJobObject(job, ExtendedLimitInformation, p, (uint)len);
        Marshal.FreeHGlobal(p);
        if (!ok) { CloseHandle(job); return IntPtr.Zero; }
        return job;
    }

    public static ulong PeakBytes(IntPtr job) {
        int len = Marshal.SizeOf(typeof(EXT_LIMIT));
        IntPtr p = Marshal.AllocHGlobal(len);
        ulong v = 0;
        if (QueryInformationJobObject(job, ExtendedLimitInformation, p, (uint)len, IntPtr.Zero)) {
            EXT_LIMIT info = (EXT_LIMIT)Marshal.PtrToStructure(p, typeof(EXT_LIMIT));
            v = info.PeakJobMemoryUsed.ToUInt64();
        }
        Marshal.FreeHGlobal(p);
        return v;
    }
}
'@
}

# --- що ганяємо ------------------------------------------------------------
$pytestBase = @('-q', '--no-header', '--tb=line', '-p', 'no:cacheprovider')

function Get-TestFiles {
    if ($Path) {
        $item = Get-Item (Join-Path $Root $Path)
        if ($item.PSIsContainer) {
            return @(Get-ChildItem -Path $item.FullName -Filter 'test_*.py' -File | Sort-Object Name |
                     ForEach-Object { $_.FullName.Substring($Root.Length + 1) })
        }
        return @($item.FullName.Substring($Root.Length + 1))
    }
    return @(Get-ChildItem -Path (Join-Path $Root 'tests') -Filter 'test_*.py' -File | Sort-Object Name |
             ForEach-Object { $_.FullName.Substring($Root.Length + 1) })
}

$files = Get-TestFiles
if (-not $files) { Write-Host "тестових файлів не знайдено" -ForegroundColor Yellow; exit 0 }

# у -PerTest режимі одиниця роботи — окремий node id
$units = @()
if ($PerTest) {
    foreach ($f in $files) {
        $collect = Join-Path $LogDir 'collect.txt'
        Start-Process -FilePath $Python -ArgumentList (@('-m','pytest',$f,'--collect-only','-q','-p','no:cacheprovider')) `
                      -WorkingDirectory $Root -NoNewWindow -Wait `
                      -RedirectStandardOutput $collect -RedirectStandardError (Join-Path $LogDir 'collect.err') | Out-Null
        $units += @(Get-Content $collect | Where-Object { $_ -match '::' } | ForEach-Object { $_.Trim() })
    }
} else {
    $units = $files
}

# --- перевірка стану системи ДО прогону ------------------------------------
$freeMB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB)
Write-Host ("вільно {0} МБ RAM | ліміт на процес {1} МБ | таймаут {2} с | одиниць {3}" -f `
    $freeMB, $MemoryLimitMB, $TimeoutSec, $units.Count) -ForegroundColor DarkGray
if ($freeMB -lt $MinFreeSystemMB) {
    Write-Host "замало вільної памʼяті ($freeMB МБ) — прогін скасовано" -ForegroundColor Red
    exit 3
}
Write-Host ("логи: {0}" -f $LogDir) -ForegroundColor DarkGray
Write-Host ""

# --- прогін ----------------------------------------------------------------
$results = @()
$aborted = $false
$limitBytes = [uint64]$MemoryLimitMB * 1MB

foreach ($unit in $units) {
    $name   = ($unit -replace '[\\/:]', '_') -replace '\.py', ''
    if ($name.Length -gt 45) { $name = $name.Substring($name.Length - 45) }
    $outLog = Join-Path $LogDir "$name.out.log"
    $errLog = Join-Path $LogDir "$name.err.log"

    $job = [JobMem]::Create($limitBytes)
    if ($job -eq [IntPtr]::Zero) {
        Write-Host "не вдалося створити Job Object — прогін скасовано" -ForegroundColor Red
        exit 4
    }

    $sw   = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $Python -ArgumentList (@('-m','pytest',$unit) + $pytestBase) `
                          -WorkingDirectory $Root -PassThru -NoNewWindow `
                          -RedirectStandardOutput $outLog -RedirectStandardError $errLog

    [void][JobMem]::AssignProcessToJobObject($job, $proc.Handle)

    $verdict = ''
    $tick    = 0
    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 300
        $tick++
        if ($sw.Elapsed.TotalSeconds -gt $TimeoutSec) {
            $verdict = "TIMEOUT (> $TimeoutSec с)"
            [void][JobMem]::TerminateJobObject($job, 1)
            break
        }
        # системний запобіжник раз на ~3 с: щось поза job теж може їсти памʼять
        if (($tick % 10) -eq 0) {
            if (((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB) -lt $MinFreeSystemMB) {
                $verdict = 'SYSTEM-LOW-MEM'
                [void][JobMem]::TerminateJobObject($job, 1)
                $aborted = $true
                break
            }
        }
    }

    $peakMB = [math]::Round([JobMem]::PeakBytes($job) / 1MB)
    $proc.WaitForExit()
    $sw.Stop()
    $code = $proc.ExitCode
    [void][JobMem]::CloseHandle($job)

    if (-not $verdict) {
        if ($code -eq 0) { $verdict = 'PASS' } else { $verdict = "FAIL (exit $code)" }
    }
    # уперлись у ліміт памʼяті — видно по peak біля межі
    if ($verdict -like 'FAIL*' -and $peakMB -ge ($MemoryLimitMB * 0.9)) {
        $verdict = "MEMLIMIT ($peakMB МБ)"
    }

    $summary = ''
    if (Test-Path $outLog) {
        $hits = @(Get-Content $outLog -Tail 40 -ErrorAction SilentlyContinue |
                  Where-Object { $_ -match '(passed|failed|error|no tests ran)' })
        if ($hits.Count -gt 0) { $summary = $hits[-1].Trim() }
    }

    $color = 'Green'
    if ($verdict -ne 'PASS') { $color = 'Red' }
    Write-Host ("{0,-46} {1,-22} peak {2,5} МБ {3,6:N1} с  {4}" -f `
        $name, $verdict, $peakMB, $sw.Elapsed.TotalSeconds, $summary) -ForegroundColor $color

    $results += [pscustomobject]@{
        Unit = $unit; Verdict = $verdict; PeakMB = $peakMB
        Sec = [math]::Round($sw.Elapsed.TotalSeconds, 1); Summary = $summary
    }

    if ($aborted) {
        Write-Host "прогін зупинено: у системи закінчується памʼять" -ForegroundColor Red
        break
    }
}

# --- підсумок --------------------------------------------------------------
Write-Host ""
$bad = @($results | Where-Object { $_.Verdict -ne 'PASS' })
$maxPeak = 0
if ($results.Count -gt 0) { $maxPeak = ($results | Measure-Object PeakMB -Maximum).Maximum }
Write-Host ("одиниць: {0} | не-PASS: {1} | максимальний peak: {2} МБ" -f $results.Count, $bad.Count, $maxPeak)
if ($bad.Count -gt 0) {
    Write-Host "деталі — у $LogDir\<назва>.out.log" -ForegroundColor Yellow
    exit 1
}
exit 0
