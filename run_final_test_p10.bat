@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 10: TWO new test sets = Ethernet (72 segments, 20 failures) and
REM  square peg 1 (73 segments, 20 failures). Run once (both configs frozen).
REM  Main question: does the hybrid (local Qwen + Skill v3 first; only when it
REM  does not answer "success", use the GPT-6.1 Sol answer) beat Qwen + Skill v3?
REM  Methods per part: majority, hybrid (main), Qwen + Skill v3, GPT-6.1 Sol,
REM  Qwen (no Skill), DINOv2 small classifier.
REM  1 self-test  2 frozen plans  3 splits (list only)
REM  4 frames of the small classifier's training pools (other parts)
REM  5 Ethernet: frames, every method once, score   (about 45 minutes)
REM  6 square peg 1: the same                       (about 45 minutes)
REM     -> experiments\final_test_eth\report.md, experiments\final_test_sqpeg1\report.md
REM  API: about 1.2 US dollars (GPT on all segments). Uses the key in .env.
REM  Safe to close; double-click again to continue (finished segments are skipped).
REM ==================================================================
title REASSEMBLE_VLM package 10 - Ethernet + square peg 1 test runs (run once)
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

set "CFG1=configs\final_test_eth.yaml"
set "CFG2=configs\final_test_sqpeg1.yaml"

echo.
echo [1/6] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_p10.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p10.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/6] Frozen plans ...
python -m scripts.final_test plan --config "%CFG1%" --log "logs\final_test_eth_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed
python -m scripts.final_test plan --config "%CFG2%" --log "logs\final_test_sqpeg1_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed

echo [3/6] Splits ...
if not exist "data\splits\ethernet_test100\manifest.json" (
  python -m scripts.make_split --part Ethernet --ratios test=1 --name ethernet_test100 --no-hash --log "logs\split_ethernet.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\sqpeg1_test100\manifest.json" (
  python -m scripts.make_split --part "square peg 1" --ratios test=1 --name sqpeg1_test100 --no-hash --log "logs\split_sqpeg1.log"
  if errorlevel 2 goto :failed
)

echo [4/6] Training-pool frames for the small classifiers (other parts; cached ones are skipped) ...
python -m scripts.probe extract --config configs\probe_eth.yaml --log "logs\probe_eth_extract.log"
if errorlevel 2 goto :failed
python -m scripts.probe extract --config configs\probe_sqpeg1.yaml --log "logs\probe_sqpeg1_extract.log"
if errorlevel 2 goto :failed

set "HF_HUB_OFFLINE=1"
echo.
echo [5/6] Ethernet test set: frames, every method once, score (the PC will not sleep) ...
python -m scripts.final_test all --config "%CFG1%" --log "logs\final_test_eth.log"
set "RUN_RC=%ERRORLEVEL%"
echo.
echo [6/6] square peg 1 test set: frames, every method once, score ...
python -m scripts.final_test all --config "%CFG2%" --log "logs\final_test_sqpeg1.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%). Reports:
echo     experiments\final_test_eth\report.md
echo     experiments\final_test_sqpeg1\report.md
echo   If a method failed, fix it and double-click again. Tell Claude when done.
echo ==================================================================
if exist "experiments\final_test_eth\report.md" start "" notepad "experiments\final_test_eth\report.md"
if exist "experiments\final_test_sqpeg1\report.md" start "" notepad "experiments\final_test_sqpeg1\report.md"
pause
exit /b %RUN_RC%

:not_frozen
echo.
echo   [STOP] A package 10 config is not frozen. The test sets were NOT opened.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see the newest file in logs\ and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p10.txt and tell Claude.
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
