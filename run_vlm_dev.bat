@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 5 dev trial: 20 dev segments only (the test set stays locked)
REM  1 self-test   2 extract frames for the dev segments (1/4/8 frames)
REM  3 quick check: every model on 1 segment with 8 frames (first time only)
REM  4 prompt v1 (configs\experiment.yaml): all candidate models x 1/4/8 frames
REM  5 later prompt versions (configs\experiment_p*.yaml), each with its own models
REM  6 trained baseline (approach A): small classifier trained on OTHER parts (first time only)
REM  7 comparison table of the prompt versions: experiments\prompt_versions.md
REM  Finished combinations are skipped, so only new work runs.
REM  Safe to close; double-click again to continue.
REM  Run setup_vlm.bat first. Extra arguments go to step 4, e.g. --use qwen3vl_8b
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM package 5 dev trial
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
REM Models must already be downloaded by setup_vlm.bat (no surprise downloads here)
set "HF_HUB_OFFLINE=1"
set "STATUS=%CD%\logs\run_vlm_status.txt"
>"%STATUS%" echo started %DATE% %TIME%

echo.
echo [1/7] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_vlm.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_vlm.txt"
>>"%STATUS%" echo pytest rc=%PYTEST_RC%
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/7] Extracting frames for the 20 dev segments (a few minutes) ...
python -m scripts.extract_frames --set dev --log "logs\extract_dev.log"
set "EX_RC=%ERRORLEVEL%"
>>"%STATUS%" echo extract rc=%EX_RC%
if not "%EX_RC%"=="0" goto :extract_failed

if exist "experiments\dev_quick\summary.md" goto :quick_done
echo [3/7] Quick check: each model on 1 segment, 8 frames ...
python -m scripts.run_vlm --name dev_quick --frames 8 --limit 1 --log "logs\run_vlm_quick.log"
set "Q_RC=%ERRORLEVEL%"
>>"%STATUS%" echo quick rc=%Q_RC%
if exist "experiments\dev_quick\summary.md" type "experiments\dev_quick\summary.md" | findstr /B /C:"|"
:quick_done

echo.
echo [4/7] Prompt v1, all candidate models (finished ones are skipped; the PC will not sleep) ...
python -m scripts.run_vlm --log "logs\run_vlm_dev.log" %*
set "RUN_RC=%ERRORLEVEL%"
>>"%STATUS%" echo full rc=%RUN_RC% %DATE% %TIME%

echo.
echo [5/7] Later prompt versions (configs\experiment_p*.yaml) ...
for %%E in (configs\experiment_p*.yaml) do call :run_exp "%%E" "%%~nE"

echo.
if exist "experiments\dev_smoke\probe_dinov2\f8\predictions.jsonl" goto :probe_done
echo [6/7] Trained baseline: frames for other parts, DINOv2 features, classifier (about 20-40 minutes) ...
python -m scripts.probe all --log "logs\probe_dev.log"
set "PROBE_RC=%ERRORLEVEL%"
>>"%STATUS%" echo probe rc=%PROBE_RC% %DATE% %TIME%
if not "%PROBE_RC%"=="0" set "RUN_RC=%PROBE_RC%"
:probe_done

echo [7/7] Comparing prompt versions ...
python -m scripts.run_vlm --compare "configs/experiment*.yaml" --log "logs\prompt_versions.log"
python -m scripts.evaluate --log "logs\dev_evaluation.log"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%).
echo   Prompt versions side by side: experiments\prompt_versions.md
echo   Per version: experiments\dev_smoke\summary.md, experiments\dev_p2\summary.md
echo   Scores with 95%% ranges: experiments\dev_evaluation\report.md
echo   Raw answers: experiments\^<name^>\^<model^>\f^<frames^>\raw.jsonl
echo   If something failed, double-click again to retry, or tell Claude.
echo ==================================================================
if exist "experiments\prompt_versions.md" start "" notepad "experiments\prompt_versions.md"
pause
exit /b %RUN_RC%

:run_exp
echo       %~2 ...
python -m scripts.run_vlm --experiment "%~1" --log "logs\run_vlm_%~2.log"
set "E_RC=%ERRORLEVEL%"
>>"%STATUS%" echo %~2 rc=%E_RC% %DATE% %TIME%
if not "%E_RC%"=="0" set "RUN_RC=%E_RC%"
exit /b 0

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_vlm.txt and tell Claude.
pause
exit /b 1

:extract_failed
echo.
echo   [STOP] Frame extraction failed - see logs\extract_dev.log and tell Claude.
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
