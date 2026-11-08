@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 13 step 4 - REBOOT 16 mm cylinder (DEV set) x the frozen
REM  package 12 methods (same prompts, models, sampling; part = round peg).
REM  Success segments in two versions: A = ends when the gripper starts to
REM  let go, B = ends 1 s after letting go. Failure segments: end at the
REM  official recovery start (shared by A and B).
REM  Cameras: hama1 = top cam_high, hama2 = low front cam_low, hand = wrist.
REM  1 self-test  2 download low front camera (about 0.93 GB)
REM  3 frames + split + input pictures  4 GPT (about 147 calls, about
REM  US$1 to 1.5, uses the key in .env)  5 local Qwen (about 1 hour)
REM  6 scores -> experiments\p13_reboot_cyl\report.md
REM  Safe to close; double-click again to continue (finished parts are skipped).
REM ==================================================================
title REASSEMBLE_VLM package 13 step 4 - cylinder dev x frozen methods
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
set "HF_HUB_OFFLINE="
set "MODELS=configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml"
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
echo [1/6] Self-test ...
python -m pytest -q tests\test_p13_run.py tests\test_p13_seg.py tests\test_p13_cyl.py tests\test_p13_meta.py >"logs\pytest_p13run.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p13run.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/6] Low front camera cam_low, about 0.93 GB ...
python -m scripts.reboot_run cam-low --log "logs\reboot_cam_low.log"
if errorlevel 1 goto :failed

echo [3/6] Frames, split, input pictures ...
python -m scripts.reboot_run prepare --log "logs\reboot_prepare.log"
if errorlevel 1 goto :failed

set "RUN_RC=0"
echo [4/6] GPT-6.1 Sol on every segment ...
python -m scripts.run_vlm --experiment configs\experiment_reboot_cyl_gpt.yaml --models %MODELS% --log "logs\reboot_cyl_gpt.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"

echo [5/6] Local Qwen: Skill v3 and probability threshold (the PC will not sleep) ...
set "HF_HUB_OFFLINE=1"
python -m scripts.run_vlm --experiment configs\experiment_reboot_cyl_v3.yaml --models %MODELS% --log "logs\reboot_cyl_v3.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"

echo [6/6] Scores ...
python -m scripts.reboot_run evaluate --log "logs\reboot_evaluate.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%). Report: experiments\p13_reboot_cyl\report.md
echo   If a model step failed, double-click again. Tell Claude when done.
echo ==================================================================
if exist "experiments\p13_reboot_cyl\report.md" start "" notepad "experiments\p13_reboot_cyl\report.md"
pause
exit /b %RUN_RC%

:failed
echo.
echo   [STOP] A step failed - see the newest reboot log in logs\ and tell Claude.
echo   Double-click again to retry; finished parts are skipped.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p13run.txt and tell Claude.
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
