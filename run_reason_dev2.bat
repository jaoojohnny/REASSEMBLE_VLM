@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 11, round 2 - Skill + prompt combined with the Codex evidence review (DEV only).
REM  Same 4 dev parts as round 1 (square peg 2, round peg 3, D-SUB, medium gear).
REM  One new run per part (Qwen3-VL-8B 4-bit, same frames):
REM    new checklist (Codex rules: gripper close / fingers hiding / flush part /
REM    wrist camera moving are NOT failure evidence) + crops of the last TWO wrist images,
REM    then a second, independent wrist-only look (first-vs-last view + crops; it is NOT
REM    told the first answer). Probabilities of both looks are saved.
REM  1 self-test  2 quick check (2 segments)  3 the 4 runs (about 15 minutes)
REM  4 dev report (round 1 + round 2) -> experiments\p11_reason_dev\report.md
REM  No test set is opened. No API. Safe to close; double-click to continue.
REM ==================================================================
title REASSEMBLE_VLM package 11 round 2 - Codex rules + two looks (dev)
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
echo [1/4] Self-test (about 40 seconds) ...
python -m pytest -q >"logs\pytest_p11b.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p11b.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

set "HF_HUB_OFFLINE=1"
for %%K in (sqpeg2 rpeg3 dsub mgear) do if not exist "data\splits\%%K_ref10_dev90\manifest.json" goto :no_round1
echo [2/4] Quick check: 2 segments of square peg 2 ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_sqpeg2_v4.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --limit 2 --log "logs\run_rdev_check_v4.log"
python -m scripts.reason_dev --check-run experiments\rdev_sqpeg2_v4\qwen3vl_8b_rv4\f4 --log "logs\reason_check_v4.log"
if errorlevel 1 goto :failed

echo [3/4] Runs on the dev sets (about 15 minutes; the PC will not sleep) ...
set "RUN_RC=0"
echo        square peg 2 ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_sqpeg2_v4.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_sqpeg2_v4.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        round peg 3 ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_rpeg3_v4.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_rpeg3_v4.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        D-SUB ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_dsub_v4.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_dsub_v4.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        medium gear ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_mgear_v4.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_mgear_v4.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"

echo [4/4] Dev report (round 1 + round 2) ...
python -m scripts.reason_dev --log "logs\reason_dev2.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%).
echo   Dev report : experiments\p11_reason_dev\report.md
echo   Tell Claude when it is done.
echo ==================================================================
if exist "experiments\p11_reason_dev\report.md" start "" notepad "experiments\p11_reason_dev\report.md"
pause
exit /b %RUN_RC%

:no_round1
echo.
echo   [STOP] Run run_reason_dev.bat (round 1) first - its splits and frames are needed.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see the newest file in logs\ and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p11b.txt and tell Claude.
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
