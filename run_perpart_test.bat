@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 14 (test stage) - the FROZEN recipe F2S2 from the dev stage
REM  (configs\perpart_selected.yaml), unchanged, on the other 10 parts:
REM  USB, bolt 4, Ethernet, square peg 1/3/4, round peg 1/2, waterproof,
REM  large gear. One small classifier per part, nested cross-validation.
REM  Also a stricter check: group by RECORDING DAY (train on other days,
REM  test on the unseen day) for the 5 dev parts and the 10 test parts.
REM  round peg 4 and small gear stay untouched. No API cost.
REM  1 self-test  2 wrist-crop features (GPU, few minutes) + cross-validation
REM  (CPU, about 10 minutes)  3 open report
REM ==================================================================
title REASSEMBLE_VLM package 14 test - frozen per-part classifier on 10 parts
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
echo [1/2] Self-test ...
python -m pytest -q tests\test_p14_perpart.py >"logs\pytest_p14t.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p14t.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/2] Frozen recipe on 10 parts + day-grouped check - the PC will not sleep ...
python -m scripts.perpart test --log "logs\perpart_test.log"
if errorlevel 3 goto :not_frozen
if errorlevel 1 goto :failed
echo.
echo ==================================================================
echo   Finished. Report: experiments\p14_perpart_test\report.md
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p14_perpart_test\report.md" start "" notepad "experiments\p14_perpart_test\report.md"
pause
exit /b 0

:not_frozen
echo.
echo   [STOP] No frozen recipe - run run_perpart_dev.bat first. Nothing was run on the test parts.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see logs\perpart_test.log and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p14t.txt and tell Claude.
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
