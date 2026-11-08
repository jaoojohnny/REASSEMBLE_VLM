@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 11 - new ways for the local Qwen to reach its answer (DEV only).
REM  4 new parts = dev set: square peg 2, round peg 3, D-SUB, medium gear
REM  (each split: ref 10 percent = example source, not scored; dev 90 percent = scored).
REM  Same Qwen3-VL-8B 4-bit, same frames as Skill v3. Per part 4 runs:
REM    Skill v3 (written answer)            - the method to beat
REM    probability + double-check on failure (methods 1 and 3)
REM    look at examples (1 success + 1 failure of the same part) (method 4)
REM    first-vs-last wrist view (method 2)
REM  1 self-test  2 splits  3 frames  4 the runs (about 1 hour)
REM  5 dev report -> experiments\p11_reason_dev\report.md
REM  No test set is opened. No API. Safe to close; double-click to continue.
REM ==================================================================
title REASSEMBLE_VLM package 11 - reasoning methods (dev, 4 new parts)
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
echo [1/5] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_p11.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p11.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/5] Splits (ref 10%% / dev 90%%) ...
if not exist "data\splits\sqpeg2_ref10_dev90\manifest.json" (
  python -m scripts.make_split --part "square peg 2" --ratios ref=0.1,dev=0.9 --name sqpeg2_ref10_dev90 --no-hash --log "logs\split_sqpeg2.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\rpeg3_ref10_dev90\manifest.json" (
  python -m scripts.make_split --part "round peg 3" --ratios ref=0.1,dev=0.9 --name rpeg3_ref10_dev90 --no-hash --log "logs\split_rpeg3.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\dsub_ref10_dev90\manifest.json" (
  python -m scripts.make_split --part "D-SUB" --ratios ref=0.1,dev=0.9 --name dsub_ref10_dev90 --no-hash --log "logs\split_dsub.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\mgear_ref10_dev90\manifest.json" (
  python -m scripts.make_split --part "medium gear" --ratios ref=0.1,dev=0.9 --name mgear_ref10_dev90 --no-hash --log "logs\split_mgear.log"
  if errorlevel 2 goto :failed
)

set "HF_HUB_OFFLINE=1"
echo [3/5] Frames (+0.5 s, 4 frames, all segments of the 4 parts) ...
python -m scripts.extract_frames --set all --part "square peg 2" --n-frames 4 --end-buffer 0.5 --log "logs\extract_sqpeg2.log"
if errorlevel 2 goto :failed
python -m scripts.extract_frames --set all --part "round peg 3" --n-frames 4 --end-buffer 0.5 --log "logs\extract_rpeg3.log"
if errorlevel 2 goto :failed
python -m scripts.extract_frames --set all --part "D-SUB" --n-frames 4 --end-buffer 0.5 --log "logs\extract_dsub.log"
if errorlevel 2 goto :failed
python -m scripts.extract_frames --set all --part "medium gear" --n-frames 4 --end-buffer 0.5 --log "logs\extract_mgear.log"
if errorlevel 2 goto :failed

echo.
echo        Quick check: the 3 new runs on 2 segments of square peg 2 ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_sqpeg2_v3.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --use qwen3vl_8b_rverify,qwen3vl_8b_rfewshot --limit 2 --log "logs\run_rdev_check.log"
python -m scripts.run_vlm --experiment configs\experiment_rdev_sqpeg2_contrast.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --limit 2 --log "logs\run_rdev_check2.log"
python -m scripts.reason_dev --check-run experiments\rdev_sqpeg2_v3\qwen3vl_8b_rverify\f4 experiments\rdev_sqpeg2_v3\qwen3vl_8b_rfewshot\f4 experiments\rdev_sqpeg2_contrast\qwen3vl_8b_rcontrast\f4 --log "logs\reason_check.log"
if errorlevel 1 goto :failed
echo [4/5] Runs on the dev sets (about 1 hour; the PC will not sleep) ...
set "RUN_RC=0"
echo        square peg 2: Skill v3, probability + double-check, examples ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_sqpeg2_v3.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_sqpeg2_v3.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        square peg 2: first-vs-last wrist view ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_sqpeg2_contrast.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_sqpeg2_contrast.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        round peg 3: Skill v3, probability + double-check, examples ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_rpeg3_v3.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_rpeg3_v3.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        round peg 3: first-vs-last wrist view ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_rpeg3_contrast.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_rpeg3_contrast.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        D-SUB: Skill v3, probability + double-check, examples ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_dsub_v3.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_dsub_v3.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        D-SUB: first-vs-last wrist view ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_dsub_contrast.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_dsub_contrast.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        medium gear: Skill v3, probability + double-check, examples ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_mgear_v3.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_mgear_v3.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        medium gear: first-vs-last wrist view ...
python -m scripts.run_vlm --experiment configs\experiment_rdev_mgear_contrast.yaml --models configs\model.yaml configs\model_p8.yaml configs\model_skill2.yaml configs\model_reason.yaml --log "logs\run_rdev_mgear_contrast.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"

echo [5/5] Dev report ...
python -m scripts.reason_dev --log "logs\reason_dev.log"
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

:failed
echo.
echo   [STOP] A step failed - see the newest file in logs\ and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p11.txt and tell Claude.
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
