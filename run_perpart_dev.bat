@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 14 (dev stage) - one small classifier PER PART.
REM  5 dev parts: BNC, square peg 2, round peg 3, D-SUB, medium gear.
REM  Each part: trained and scored only on its own segments with nested
REM  cross-validation grouped by recording (outer 5 folds x 5 repeats,
REM  inner 4 folds), at most 10 features, 6 candidate recipes.
REM  The recipe is chosen by the rule written in docs\package14_perpart.md
REM  and frozen in configs\perpart_selected.yaml - the other 10 parts run later.
REM  round peg 4 and small gear stay untouched.
REM  1 self-test  2 frames + DINOv2 features, GPU, about 10-20 minutes
REM  3 cross-validation, CPU, about 5-10 minutes  4 open report
REM  No API cost. Safe to close; double-click again to continue.
REM ==================================================================
title REASSEMBLE_VLM package 14 dev - one classifier per part
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
echo [1/3] Self-test ...
python -m pytest -q tests\test_p14_perpart.py >"logs\pytest_p14.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p14.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/3] Frames and DINOv2 features for 15 parts - the PC will not sleep ...
python -m scripts.perpart prepare --log "logs\perpart_prepare.log"
if errorlevel 1 goto :failed

echo [3/3] Nested cross-validation on the 5 dev parts ...
python -m scripts.perpart dev --log "logs\perpart_dev.log"
if errorlevel 1 goto :failed
echo.
echo ==================================================================
echo   Finished. Report: experiments\p14_perpart_dev\report.md
echo   Frozen recipe:    configs\perpart_selected.yaml
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p14_perpart_dev\report.md" start "" notepad "experiments\p14_perpart_dev\report.md"
pause
exit /b 0

:failed
echo.
echo   [STOP] A step failed - see logs\perpart_prepare.log or logs\perpart_dev.log and tell Claude.
echo   Double-click again to continue; finished frames and features are kept.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p14.txt and tell Claude.
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
