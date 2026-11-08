@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 13 - REBOOT 16 mm cylinder = DEV set (has official failure
REM  timestamps in meta\phase.json). The 4 plug parts stay untouched.
REM  Downloads about 4.6 GB: README + meta + color video of 3 cameras
REM  (top cam_high, left wrist, right wrist) for
REM    recovery: REBOOT26/sample_recovery-demonstration (60 episodes)
REM    expert:   REBOOT26/16mm-cylinder_install (about 62 episodes)
REM  NOT downloaded: depth/joint data (about 34 GB each), low front camera.
REM  Then: index + one check picture per episode (3 cameras x 8 frames).
REM  Safe to close; double-click again to continue (finished files are skipped).
REM  1 self-test  2 size check  3 download  4 check pictures + report
REM  Output: data\raw\reboot\   experiments\p13_cyl_check\report.md
REM ==================================================================
title REASSEMBLE_VLM package 13 - REBOOT 16 mm cylinder (dev) download + check
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
set "HF_HUB_OFFLINE="
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
  if not defined CONDA_BAT if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%C"
if not defined CONDA_BAT goto :no_conda
call "%CONDA_BAT%" activate %ENV_NAME%
python -c "import os,sys; sys.exit(0 if os.environ.get('CONDA_DEFAULT_ENV')=='%ENV_NAME%' else 1)"
if errorlevel 1 goto :no_env
if not exist "logs" mkdir "logs"

echo.
echo [1/4] Self-test ...
python -m pytest -q tests\test_p13_cyl.py tests\test_p13_meta.py >"logs\pytest_p13cyl.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p13cyl.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/4] Size check, nothing downloaded yet ...
python -m scripts.reboot_cyl plan --log "logs\reboot_cyl_plan.log"
if errorlevel 3 goto :too_big
if errorlevel 1 goto :failed

echo [3/4] Downloading about 4.6 GB, this can take a while ...
python -m scripts.reboot_cyl download --log "logs\reboot_cyl_download.log"
if errorlevel 3 goto :too_big
if errorlevel 1 goto :failed

echo [4/4] Index + check pictures ...
python -m scripts.reboot_cyl check --log "logs\reboot_cyl_check.log"
set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%). Report: experiments\p13_cyl_check\report.md
echo   Pictures: experiments\p13_cyl_check\sheets\   Tell Claude when done.
echo ==================================================================
if exist "experiments\p13_cyl_check\report.md" start "" notepad "experiments\p13_cyl_check\report.md"
pause
exit /b %RUN_RC%

:too_big
echo.
echo   [STOP] Download larger than the 6 GB limit or not enough disk space.
echo   Nothing more was downloaded. See logs\reboot_cyl_plan.log and tell Claude.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see the newest reboot_cyl log in logs\ and tell Claude.
echo   Double-click again to retry; finished files are skipped.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p13cyl.txt and tell Claude.
pause
exit /b 1

:no_conda
echo conda.bat not found - run setup_windows.bat first.
pause
exit /b 1

:no_env
echo Could not activate the conda env "%ENV_NAME%" - run setup_windows.bat first.
pause
exit /b 1
