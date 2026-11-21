@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 18 - which step failed? (bolt 4 main, BNC as the
REM  development part). Answers = the first failed step in the
REM  dataset's own step labels (Approach / Align / Twist).
REM  Same frozen F2S2 recipe as package 14, only the answers change.
REM  Reads cached DINOv2 features - no GPU, no download, no API cost.
REM  Includes a shuffle test (100 shuffles), so it takes about
REM  5-15 minutes. Pre-registered in docs\package18_failstep.md.
REM ==================================================================
title REASSEMBLE_VLM package 18 - which step failed (bolt 4)
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
set "HF_HUB_OFFLINE=1"
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
echo [1/2] Self-test (about 1 minute) ...
python -m pytest -q tests\test_p18_failstep.py tests\test_perpart_final.py >"logs\pytest_p18.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p18.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/2] Which step failed - the PC will not sleep (about 5-15 minutes) ...
python -m scripts.failstep --log "logs\failstep.log"
set "RUN_RC=%ERRORLEVEL%"
if "%RUN_RC%"=="3" goto :stopped
if not "%RUN_RC%"=="0" goto :failed
echo.
echo ==================================================================
echo   Finished. Report: experiments\p18_failstep\report.md
echo   Frames by step: experiments\p18_failstep\bolt_4_frames_by_step.jpg
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p18_failstep\report.md" start "" notepad "experiments\p18_failstep\report.md"
pause
exit /b 0

:stopped
echo.
echo   [STOP] Features or frames are missing - see logs\failstep.log and tell Claude.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see logs\failstep.log and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p18.txt and tell Claude.
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
