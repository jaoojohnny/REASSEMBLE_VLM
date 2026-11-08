@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Retraining (pre-registered 2026-10-16): VLM features + small classifier.
REM  Qwen3-VL-8B is NOT changed. It looks at each segment with the generic
REM  prompt; we read its internal features at the point where it is about to
REM  answer success/failure, then train a logistic regression on OTHER parts.
REM  BNC dev segments are only shown. The test set is never read.
REM  1 self-test  2 list the data  3 extract features (about 25 minutes, GPU,
REM  safe to close and resume)  4 cross-validation + training + dev scores
REM  Result: experiments\vlm_probe_selection\selection.md and report.md
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM retraining - VLM features + small classifier (dev only)
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
set "HF_HUB_OFFLINE=1"

set "STATUS=%CD%\logs\run_vlm_probe_status.txt"
>"%STATUS%" echo started %DATE% %TIME%

echo.
echo [1/4] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_vlm_probe.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_vlm_probe.txt"
>>"%STATUS%" echo pytest rc=%PYTEST_RC%
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/4] Training data (other parts) and BNC dev ...
python -m scripts.vlm_probe plan --log "logs\vlm_probe_plan.log"
if errorlevel 1 goto :plan_failed

echo.
echo [3/4] Extracting Qwen3-VL-8B features (about 25 minutes; the PC will not sleep) ...
python -m scripts.vlm_probe extract --log "logs\vlm_probe_extract.log"
set "EX_RC=%ERRORLEVEL%"
>>"%STATUS%" echo extract rc=%EX_RC% %DATE% %TIME%
if not "%EX_RC%"=="0" goto :extract_failed

echo [4/4] Cross-validation, training, dev scores ...
python -m scripts.vlm_probe fit --log "logs\vlm_probe_fit.log"
set "FIT_RC=%ERRORLEVEL%"
>>"%STATUS%" echo fit rc=%FIT_RC% %DATE% %TIME%
echo.
echo ==================================================================
echo   Finished (rc=%FIT_RC%).
echo   Candidates by cross-validation: experiments\vlm_probe_selection\selection.md
echo   Dev scores with 95%% ranges:     experiments\vlm_probe_selection\report.md
echo   Tell Claude when it is done.
echo ==================================================================
if exist "experiments\vlm_probe_selection\selection.md" start "" notepad "experiments\vlm_probe_selection\selection.md"
pause
exit /b %FIT_RC%

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_vlm_probe.txt and tell Claude.
pause
exit /b 1

:plan_failed
echo.
echo   [STOP] Could not list the data - see logs\vlm_probe_plan.log and tell Claude.
pause
exit /b 1

:extract_failed
echo.
echo   [STOP] Feature extraction failed - see logs\vlm_probe_extract.log. Double-click again to resume, or tell Claude.
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
