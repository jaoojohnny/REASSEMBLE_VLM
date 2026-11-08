@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 9: NEW test set = bolt 4 inserts (83 segments, 30 failures).
REM  Run it once (configs\final_test_bolt4.yaml is frozen).
REM  Main question: does Skill v3 (checklist + fixed wrist crop, decided on
REM  the BNC dev set) beat the local Qwen3-VL-8B without a Skill on a new part?
REM  Methods: majority, Qwen + Skill v3 (main), Qwen (no Skill), Qwen + crop,
REM  Qwen + checklist, DINOv2 small classifier, GPT-6.1 Sol, Claude Sonnet 5.5.
REM  1 self-test  2 frozen plan  3 bolt 4 split (list only)
REM  4 frames of the small classifier's training pool (other parts)
REM  5 bolt 4 frames, run every method once, score + time + cost
REM     -> experiments\final_test_bolt4\report.md  (about 1-1.5 hours)
REM  The first run writes experiments\final_test_bolt4\lock.json.
REM  Safe to close; double-click again to continue (finished segments are skipped).
REM ==================================================================
title REASSEMBLE_VLM package 9 - bolt 4 test run (run once)
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
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

set "CFG=configs\final_test_bolt4.yaml"

echo.
echo [1/5] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_bolt4.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_bolt4.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/5] Frozen plan ...
python -m scripts.final_test plan --config "%CFG%" --log "logs\final_test_bolt4_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed

echo [3/5] bolt 4 split ...
if not exist "data\splits\bolt4_test100\manifest.json" (
  python -m scripts.make_split --part "bolt 4" --ratios test=1 --name bolt4_test100 --no-hash --log "logs\split_bolt4.log"
  if errorlevel 2 goto :failed
)

echo [4/5] Training-pool frames for the small classifier (other parts; cached ones are skipped) ...
python -m scripts.probe extract --config configs\probe_bolt4.yaml --log "logs\probe_bolt4_extract.log"
if errorlevel 2 goto :failed

set "HF_HUB_OFFLINE=1"
echo.
echo [5/5] bolt 4 test set: frames, every method once, score (about 1-1.5 hours; the PC will not sleep) ...
python -m scripts.final_test all --config "%CFG%" --log "logs\final_test_bolt4.log"
set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%). Report: experiments\final_test_bolt4\report.md
echo   If a method failed (for example a missing API key), fix it and double-click again.
echo ==================================================================
if exist "experiments\final_test_bolt4\report.md" start "" notepad "experiments\final_test_bolt4\report.md"
pause
exit /b %RUN_RC%

:not_frozen
echo.
echo   [STOP] configs\final_test_bolt4.yaml is not frozen. The test set was NOT opened.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see the newest file in logs\ and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_bolt4.txt and tell Claude.
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
