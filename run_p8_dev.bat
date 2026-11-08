@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 8 dev run on BNC (all 86 BNC segments are now the dev set).
REM  1 self-test
REM  2 make the splits (BNC all-dev; USB all-test - only the list, nothing is opened)
REM  3 execution time of every method in the package 6/7 test runs -> experiments\timing\timing.md
REM  4 execution time of the small classifier (DINOv2) on 20 dev segments
REM  5 BNC frames (+0.5 s, 4 frames; already extracted ones are skipped)
REM  6 same prompt and frames for: local Qwen3-VL-8B, Qwen + zoom Skill,
REM    Claude Sonnet 5.5 and GPT-6.1 Sol (APIs; needs setup_api.bat first)
REM    (about 1 hour; models without a key are skipped)
REM  7 dev scores + time + cost -> experiments\p8_dev_evaluation\report.md
REM  Safe to close; double-click again to continue (finished steps are skipped).
REM ==================================================================
title REASSEMBLE_VLM package 8 - BNC dev run
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

echo.
echo [1/7] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_p8.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p8.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/7] Splits ...
if not exist "data\splits\bnc_alldev\manifest.json" (
  python -m scripts.make_split --part BNC --ratios dev=1 --name bnc_alldev --no-hash --log "logs\split_bnc_alldev.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\usb_test100\manifest.json" (
  python -m scripts.make_split --part USB --ratios test=1 --name usb_test100 --no-hash --log "logs\split_usb.log"
  if errorlevel 2 goto :failed
)

echo [3/7] Execution time of the package 6/7 test runs (reads results only) ...
python -m scripts.timing --log "logs\timing.log"

set "HF_HUB_OFFLINE=1"
echo [4/7] Execution time of the small classifier (DINOv2, 20 dev segments) ...
python -m scripts.probe time --config configs\probe_eb1.yaml --n-segments 20 --log "logs\probe_time.log"

echo [5/7] BNC frames (+0.5 s, 4 frames) ...
python -m scripts.extract_frames --set dev --split-dir data\splits\bnc_alldev --part BNC --n-frames 4 --end-buffer 0.5 --log "logs\extract_bnc_alldev.log"
if errorlevel 2 goto :failed

echo.
echo [6/7] BNC 86 segments: Qwen3-VL-8B, Qwen + zoom Skill, Claude, GPT (about 1 hour; the PC will not sleep) ...
python -m scripts.run_vlm --experiment configs\experiment_bnc_p8.yaml --models configs\model.yaml configs\model_p8.yaml --log "logs\run_p8_dev.log"
set "RUN_RC=%ERRORLEVEL%"

echo [7/7] Dev scores, time and cost ...
python -m scripts.evaluate --experiments configs\experiment_bnc_p8.yaml --out experiments\p8_dev_evaluation --log "logs\p8_dev_eval.log"
echo.
echo ==================================================================
echo   Finished (run rc=%RUN_RC%).
echo   Scores + time + cost : experiments\p8_dev_evaluation\report.md
echo   Old test runs - time : experiments\timing\timing.md
echo   Next: double-click run_final_test_usb.bat (new USB test set, run once)
echo ==================================================================
if exist "experiments\p8_dev_evaluation\report.md" start "" notepad "experiments\p8_dev_evaluation\report.md"
pause
exit /b %RUN_RC%

:failed
echo.
echo   [STOP] A step failed - see the newest file in logs\ and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p8.txt and tell Claude.
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
