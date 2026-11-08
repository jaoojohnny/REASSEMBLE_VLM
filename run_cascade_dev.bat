@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 12 - two stages: the local Qwen (probability threshold) decides
REM  first; only when it does not answer "success" is GPT-6.1 Sol asked.
REM  DEV only. This run gives the GPT answers on the 241 dev segments of the
REM  4 package-11 dev parts (square peg 2, round peg 3, D-SUB, medium gear),
REM  so every threshold can be scored afterwards. Qwen is NOT re-run (its
REM  probabilities are already saved from package 11).
REM  1 self-test  2 frames (cached ones are skipped)
REM  3 API check on 2 segments  4 GPT on all dev segments (about 15 minutes)
REM  5 dev report -> experiments\p12_cascade_dev\report.md
REM  API: about 1.5 to 2 US dollars. Uses the key already in .env.
REM  No test set is opened. Safe to close; double-click to continue
REM  (finished segments are skipped).
REM ==================================================================
title REASSEMBLE_VLM package 12 - two stages (dev, GPT answers)
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
set "MODELS=configs\model.yaml configs\model_p8.yaml"

echo.
echo [1/5] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_p12.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p12.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/5] Frames (same as package 11; cached ones are skipped) ...
python -m scripts.extract_frames --set all --part "square peg 2" --n-frames 4 --end-buffer 0.5 --log "logs\extract_sqpeg2.log"
if errorlevel 2 goto :failed
python -m scripts.extract_frames --set all --part "round peg 3" --n-frames 4 --end-buffer 0.5 --log "logs\extract_rpeg3.log"
if errorlevel 2 goto :failed
python -m scripts.extract_frames --set all --part "D-SUB" --n-frames 4 --end-buffer 0.5 --log "logs\extract_dsub.log"
if errorlevel 2 goto :failed
python -m scripts.extract_frames --set all --part "medium gear" --n-frames 4 --end-buffer 0.5 --log "logs\extract_mgear.log"
if errorlevel 2 goto :failed

echo [3/5] API check: GPT on 2 segments of square peg 2 ...
python -m scripts.run_vlm --experiment configs\experiment_cdev_sqpeg2_gpt.yaml --models %MODELS% --limit 2 --log "logs\run_cdev_check.log"
python -m scripts.cascade_dev --check-run experiments\cdev_sqpeg2_gpt\gpt61_sol\f4 --log "logs\cascade_check.log"
if errorlevel 1 goto :api_failed

echo [4/5] GPT on all dev segments (about 15 minutes) ...
set "RUN_RC=0"
echo        square peg 2 ...
python -m scripts.run_vlm --experiment configs\experiment_cdev_sqpeg2_gpt.yaml --models %MODELS% --log "logs\run_cdev_sqpeg2.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        round peg 3 ...
python -m scripts.run_vlm --experiment configs\experiment_cdev_rpeg3_gpt.yaml --models %MODELS% --log "logs\run_cdev_rpeg3.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        D-SUB ...
python -m scripts.run_vlm --experiment configs\experiment_cdev_dsub_gpt.yaml --models %MODELS% --log "logs\run_cdev_dsub.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        medium gear ...
python -m scripts.run_vlm --experiment configs\experiment_cdev_mgear_gpt.yaml --models %MODELS% --log "logs\run_cdev_mgear.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"

echo [5/5] Dev report ...
python -m scripts.cascade_dev --log "logs\cascade_dev.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%).
echo   Dev report : experiments\p12_cascade_dev\report.md
echo   If some segments failed, double-click again (finished ones are skipped).
echo   Tell Claude when it is done.
echo ==================================================================
if exist "experiments\p12_cascade_dev\report.md" start "" notepad "experiments\p12_cascade_dev\report.md"
pause
exit /b %RUN_RC%

:api_failed
echo.
echo   [STOP] GPT did not answer correctly on the 2 check segments.
echo   Check that .env has the OpenAI key (setup_api.bat), then see
echo   logs\run_cdev_check.log and logs\cascade_check.log and tell Claude.
pause
exit /b 2

:failed
echo.
echo   [STOP] A step failed - see the newest file in logs\ and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p12.txt and tell Claude.
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
