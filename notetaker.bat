@echo off
REM ==========================================================================
REM notetaker.bat -- installs dependencies and runs Notetaker via uv (Windows).
REM
REM This script prepares everything Notetaker needs and runs it:
REM   1. Installs 'uv' (Python package manager/venv) if absent.
REM   2. Checks 'ffmpeg' (necessary for audio capture) and attempts to install it.
REM   3. Creates/syncs the virtual environment with 'uv sync'.
REM   4. Runs Notetaker via 'uv run', passing through received arguments.
REM
REM Usage:
REM   notetaker.bat <command> [options]
REM   notetaker.bat --setup     (only installs/updates dependencies)
REM   notetaker.bat --help
REM ==========================================================================
setlocal EnableDelayedExpansion

REM Project directory = directory of this script (works from anywhere).
cd /d "%~dp0"

REM --- Show help when explicitly requested or without arguments ----------
if "%~1"=="" goto :usage
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="help" goto :usage

call :ensure_uv || exit /b 1

REM --- Setup mode: only prepares environment and exits ----------------------
if /i "%~1"=="--setup" (
    call :ensure_ffmpeg || exit /b 1
    call :sync_env || exit /b 1
    echo ==^> ready. Use: notetaker.bat start "my meeting"
    exit /b 0
)

REM ffmpeg is only essential for recording (start); for other commands
REM we avoid blocking if it is not yet installed.
if /i "%~1"=="start" (
    call :ensure_ffmpeg || exit /b 1
)

REM Ensure environment is synced (fast when already up to date).
call :sync_env || exit /b 1

echo ==^> running: notetaker %*
uv run notetaker %*
exit /b %ERRORLEVEL%


REM ==========================================================================
REM Ensure uv is installed
REM ==========================================================================
:ensure_uv
where uv >nul 2>nul
if %ERRORLEVEL%==0 exit /b 0

REM Common location where the installer places uv.
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    exit /b 0
)

echo ==^> uv not found; installing...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if %ERRORLEVEL%==0 (
    for /f "delims=" %%v in ('uv --version') do echo ==^> uv installed: %%v
    exit /b 0
)
echo [error] uv installed but not found in PATH. Open a new terminal and try again.
exit /b 1


REM ==========================================================================
REM Check/install ffmpeg (system dependency for audio capture)
REM ==========================================================================
:ensure_ffmpeg
where ffmpeg >nul 2>nul
if %ERRORLEVEL%==0 exit /b 0

echo [warning] ffmpeg not found (required for audio recording).
where winget >nul 2>nul
if %ERRORLEVEL%==0 (
    echo ==^> attempting installation via winget...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    where ffmpeg >nul 2>nul
    if !ERRORLEVEL!==0 exit /b 0
    echo [warning] install ffmpeg and open a new terminal to update PATH.
    exit /b 1
)
where choco >nul 2>nul
if %ERRORLEVEL%==0 (
    echo ==^> attempting installation via choco...
    choco install ffmpeg -y
    where ffmpeg >nul 2>nul
    if !ERRORLEVEL!==0 exit /b 0
    echo [warning] install ffmpeg and open a new terminal to update PATH.
    exit /b 1
)
echo [error] could not install ffmpeg automatically (winget/choco not found).
echo         Download from https://ffmpeg.org/download.html and add to PATH.
exit /b 1


REM ==========================================================================
REM Sync virtual environment (creates .venv and installs dependencies)
REM ==========================================================================
:sync_env
echo ==^> syncing environment (uv sync)...
uv sync
exit /b %ERRORLEVEL%


REM ==========================================================================
REM Help
REM ==========================================================================
:usage
echo notetaker.bat -- records meetings, transcribes locally, and generates AI summary.
echo.
echo USAGE
echo     notetaker.bat ^<command^> [options]
echo     notetaker.bat --setup        (only installs/updates dependencies)
echo     notetaker.bat --help         (this help)
echo.
echo On first run, the script installs 'uv' (if necessary), checks
echo 'ffmpeg', and creates the virtual environment automatically. Then, all
echo arguments are passed to Notetaker.
echo.
echo NOTETAKER COMMANDS
echo     start "^<title^>"      Starts recording a meeting.
echo                          Live monitoring (time + size); Ctrl+C stops
echo                          and generates summary.
echo     stop                 Stops the current meeting and generates summary.
echo     status               Shows the status of the most recent meeting.
echo     list                 Lists all recorded meetings.
echo     devices              Shows detected audio devices.
echo     summarize ^<folder^>   Regenerates summary from existing transcription.
echo.
echo 'start' OPTIONS
echo     --mode online^|in-person^|listener
echo                                  online = mic + system audio (default)
echo                                  in-person = microphone only
echo                                  listener = system audio only
echo     --lang auto^|pt^|es^|en         Language spoken in meeting (default: config)
echo     --output-lang meeting^|pt^|es^|en
echo                                  Summary language (default: meeting language)
echo     --diarization level1^|level2  level1 = you vs. participants (default)
echo     --no-watch                   Do not monitor live; use 'stop' after.
echo.
echo SYSTEM AUDIO (ONLINE MODE) ON WINDOWS
echo     DirectShow does not expose output natively. Enable "Stereo Mix"/
echo     "Stereo Mixing" in Sound Settings ^> Recording (if your sound card
echo     offers it), or install VB-CABLE (https://vb-audio.com/Cable/) or
echo     VoiceMeeter and route output to it. Without this, use --mode in-person.
echo.
echo EXAMPLES
echo     notetaker.bat --setup
echo     notetaker.bat start "sprint planning"
echo     notetaker.bat start "client meeting" --mode in-person
echo     notetaker.bat start "weekly" --lang en --output-lang pt
echo     notetaker.bat status
echo     notetaker.bat list
echo.
echo CONFIGURATION
echo     Editable at: %%USERPROFILE%%\.config\notetaker\config.toml
echo     (Whisper model, default language, LLM command, audio devices)
exit /b 0
