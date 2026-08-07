<#
.SYNOPSIS
    F216 - make an INSTALLED Sorta do work, and fail loudly when it cannot.

.DESCRIPTION
    The installer is the one part of the product that cannot be judged from the sources:
    a payload that stages, an .iss that compiles and a wizard whose commands are unit
    tested still say nothing about whether the thing installs and runs. This script is
    what turns "it installed" into "it worked", and it is a file of its own rather than
    a block of workflow YAML for two reasons: the workflow runs it TWICE - once against
    the real installation and once against an empty directory, where it has to go red -
    and somebody installing by hand can run the same checks the machine runs.

    Every assertion here reads OUTPUT, not just an exit code. A stage that found nothing
    exits zero as cheerfully as one that found everything, so `sorta index` on an empty
    index and `sorta index` on fourteen frames are indistinguishable by return code and
    completely distinguishable by the line they print.

    What it proves:
      * the payload landed whole - interpreter, lib, uv, the install manifest;
      * `config.yaml` reached the per-user directory the shortcuts name as their
        working directory (the [Files] entry of the .iss, which nothing else checks);
      * the interpreter finds `lib` through its one .pth file - it imports Pillow and
        writes the frames itself, so the frames are evidence rather than fixtures;
      * `index` reads them, counts them, and marks the duplicates among them;
      * `stats` reports the same count back out of SQLite;
      * `doctor` names the tiers of this machine.

.PARAMETER InstallDir
    Where the installer put the program ({app}) - the directory holding python\, lib\
    and sorta-install.json.

.PARAMETER WorkDir
    The per-user directory the installer created and the shortcuts start in; config.yaml
    and the index live here. Defaults to what the .iss writes.

.PARAMETER Frames
    How many distinct frames to synthesise. Two exact copies are added on top, so the
    index is expected to report Frames + 2 files and 2 duplicates.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [string]$WorkDir = (Join-Path $env:APPDATA 'sorta'),
    [int]$Frames = 12
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$script:Failures = @()

function Assert-True {
    param([bool]$Condition, [string]$What)
    if ($Condition) {
        Write-Host "  ok    $What"
    } else {
        Write-Host "  FAIL  $What"
        $script:Failures += $What
    }
}

function Assert-Match {
    <# The assertion that matters: a claim about what the command SAID. #>
    param([string]$Text, [string]$Pattern, [string]$What)
    Assert-True ($Text -match $Pattern) "$What  (expected /$Pattern/)"
}

Write-Host "== layout =="
$python = Join-Path $InstallDir 'python\python.exe'
Assert-True (Test-Path -LiteralPath $python) 'the interpreter is installed'
Assert-True (Test-Path -LiteralPath (Join-Path $InstallDir 'lib')) 'lib\ is installed'
Assert-True (Test-Path -LiteralPath (Join-Path $InstallDir 'uv.exe')) 'uv.exe is installed'
Assert-True (Test-Path -LiteralPath (Join-Path $InstallDir 'sorta-install.json')) `
    'the install manifest is installed'
# Not the payload's doing but the .iss's, and the only check it has: an installed
# program whose working directory has no config.yaml writes its index into system32.
Assert-True (Test-Path -LiteralPath (Join-Path $WorkDir 'config.yaml')) `
    'config.yaml reached the per-user directory'

if ($script:Failures.Count -gt 0) {
    # Nothing below can mean anything without an interpreter to run it.
    Write-Host "`nverify: $($script:Failures.Count) failure(s) before anything ran"
    exit 1
}

# --- frames the installed program writes for itself ---------------------------------
# `python -m sorta.cli` and not a console script: the payload is a `uv pip install
# --target` tree, so the entry points live beside it rather than on PATH, and this is
# the same invocation the Start-menu shortcuts use.
$photos = Join-Path $WorkDir 'verify-frames'
if (Test-Path -LiteralPath $photos) { Remove-Item -LiteralPath $photos -Recurse -Force }
New-Item -ItemType Directory -Path $photos -Force | Out-Null

Write-Host "`n== the installed interpreter writes $Frames frames + 2 copies =="
$maker = @"
import shutil, sys
from pathlib import Path
from PIL import Image

out, count = Path(sys.argv[1]), int(sys.argv[2])
for i in range(count):
    Image.new('RGB', (640, 480), (i * 17 % 256, i * 53 % 256, 90)).save(out / f'frame{i:02d}.jpg', quality=90)
# Two byte-for-byte copies: the duplicate counter of `index` is an assertion below, and
# a run that silently stopped deduplicating would otherwise still look healthy.
for i in (0, 1):
    shutil.copyfile(out / f'frame{i:02d}.jpg', out / f'copy{i:02d}.jpg')
print('frames written:', len(list(out.glob('*.jpg'))))
"@
$makerFile = Join-Path $WorkDir 'verify_make_frames.py'
Set-Content -LiteralPath $makerFile -Value $maker -Encoding UTF8
$made = & $python -X utf8 $makerFile $photos $Frames 2>&1 | Out-String
Write-Host $made.Trim()
$expected = $Frames + 2
Assert-Match $made "frames written: $expected" 'the payload imports Pillow out of lib\'

# --- the work itself ------------------------------------------------------------------
Push-Location $WorkDir
try {
    Write-Host "`n== sorta index =="
    $index = & $python -X utf8 -m sorta.cli index $photos 2>&1 | Out-String
    Write-Host $index.Trim()
    Assert-Match $index "\+$expected new" "index added all $expected files"
    Assert-Match $index '0 errors' 'index reported no errors'
    Assert-Match $index '2 duplicates marked' 'index marked the two copies as duplicates'

    Write-Host "`n== sorta stats =="
    $stats = & $python -X utf8 -m sorta.cli stats 2>&1 | Out-String
    Write-Host $stats.Trim()
    Assert-Match $stats "Files in the index: $expected" 'stats reads the same count back'
    Assert-True ($stats -notmatch 'The index is empty') 'stats did not find an empty index'

    Write-Host "`n== sorta doctor =="
    $doctor = & $python -X utf8 -m sorta.cli doctor 2>&1 | Out-String
    Write-Host $doctor.Trim()
    Assert-Match $doctor 'Installed tiers:' 'doctor names the tiers of this machine'
} finally {
    Pop-Location
}

Write-Host ''
if ($script:Failures.Count -gt 0) {
    Write-Host "verify: $($script:Failures.Count) failure(s)"
    $script:Failures | ForEach-Object { Write-Host "  - $_" }
    exit 1
}
Write-Host 'verify: the installed program did the work'
exit 0
