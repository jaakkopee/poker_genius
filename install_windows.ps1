# ────────────────────────────────────────────────────────
#  Poker Genius – Windows 10/11 PowerShell installer
#  Usage (in PowerShell):  .\install_windows.ps1
#
#  If execution policy blocks the script run:
#    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# ────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir   = Join-Path $ScriptDir ".venv"

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║      Poker Genius – Windows Setup        ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check Python 3 ──────────────────────────────────
Write-Host "→ Checking for Python 3…" -ForegroundColor Yellow

$PythonCmd = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $PythonCmd = $cmd
            Write-Host "  Found: $ver" -ForegroundColor Green
            break
        }
    } catch { }
}

if (-not $PythonCmd) {
    Write-Host ""
    Write-Host "  ERROR: Python 3 not found." -ForegroundColor Red
    Write-Host "  Download from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  Ensure 'Add Python to PATH' is checked during installation." -ForegroundColor Red
    exit 1
}

# Require Python >= 3.8
$VerDetails = & $PythonCmd -c "import sys; print(sys.version_info.major, sys.version_info.minor)"
$parts = $VerDetails -split " "
$major = [int]$parts[0]
$minor = [int]$parts[1]
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
    Write-Host "  ERROR: Python 3.8+ required." -ForegroundColor Red
    exit 1
}

# ── 2. Check Tesseract ─────────────────────────────────
Write-Host ""
Write-Host "→ Checking for Tesseract OCR…" -ForegroundColor Yellow

$TessFound = $false
$TessPaths = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
)

if (Get-Command tesseract -ErrorAction SilentlyContinue) {
    $TessFound = $true
    $tv = & tesseract --version 2>&1 | Select-Object -First 1
    Write-Host "  Found in PATH: $tv" -ForegroundColor Green
} else {
    foreach ($p in $TessPaths) {
        if (Test-Path $p) {
            $TessFound = $true
            Write-Host "  Found at: $p" -ForegroundColor Green
            break
        }
    }
}

if (-not $TessFound) {
    Write-Host ""
    Write-Host "  WARNING: Tesseract OCR not found." -ForegroundColor Yellow
    Write-Host "  Download the Windows installer from:" -ForegroundColor Yellow
    Write-Host "  https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    Write-Host "  After installation, ensure its folder is added to PATH" -ForegroundColor Yellow
    Write-Host "  (e.g. C:\Program Files\Tesseract-OCR)." -ForegroundColor Yellow
    Write-Host ""
    $ans = Read-Host "  Continue without Tesseract? [y/N]"
    if ($ans -notmatch "^[Yy]$") { exit 1 }
}

# ── 3. Create virtual environment ─────────────────────
Write-Host ""
Write-Host "→ Creating virtual environment at .venv…" -ForegroundColor Yellow
if (Test-Path $VenvDir) {
    Write-Host "  .venv already exists, skipping creation." -ForegroundColor Gray
} else {
    & $PythonCmd -m venv $VenvDir
    Write-Host "  Created." -ForegroundColor Green
}

# ── 4. Upgrade pip ────────────────────────────────────
Write-Host ""
Write-Host "→ Upgrading pip…" -ForegroundColor Yellow
$PipExe = Join-Path $VenvDir "Scripts\pip.exe"
& $PipExe install --quiet --upgrade pip

# ── 5. Install requirements ───────────────────────────
Write-Host ""
Write-Host "→ Installing Python dependencies from requirements.txt…" -ForegroundColor Yellow
$ReqFile = Join-Path $ScriptDir "requirements.txt"
& $PipExe install --quiet -r $ReqFile
Write-Host "  Done." -ForegroundColor Green

# ── 6. Verify tkinter ────────────────────────────────
Write-Host ""
Write-Host "→ Verifying tkinter availability…" -ForegroundColor Yellow
$PyExe = Join-Path $VenvDir "Scripts\python.exe"
$tkCheck = & $PyExe -c "import tkinter; print('OK')" 2>&1
if ($tkCheck -match "OK") {
    Write-Host "  tkinter: OK" -ForegroundColor Green
} else {
    Write-Host "  WARNING: tkinter not available. Re-install Python with 'tcl/tk' option." -ForegroundColor Yellow
}

# ── 7. Create launch batch file ───────────────────────
$BatchFile = Join-Path $ScriptDir "run.bat"
@"
@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0poker_genius.py" %*
"@ | Set-Content -Path $BatchFile -Encoding ASCII

# ── 8. Create desktop shortcut (optional) ─────────────
Write-Host ""
$createShortcut = Read-Host "→ Create a Desktop shortcut? [y/N]"
if ($createShortcut -match "^[Yy]$") {
    $Desktop = [Environment]::GetFolderPath("Desktop")
    $ShortcutPath = Join-Path $Desktop "Poker Genius.lnk"
    $WScriptShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath  = $PyExe
    $Shortcut.Arguments   = "`"$(Join-Path $ScriptDir 'poker_genius.py')`""
    $Shortcut.WorkingDirectory = $ScriptDir
    $Shortcut.Description = "Poker Genius GTO Advisor"
    $Shortcut.Save()
    Write-Host "  Shortcut created on Desktop." -ForegroundColor Green
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          Installation complete!          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Run the app:  .\run.bat" -ForegroundColor White
Write-Host "  Or manually:  .venv\Scripts\python.exe poker_genius.py" -ForegroundColor White
Write-Host ""
