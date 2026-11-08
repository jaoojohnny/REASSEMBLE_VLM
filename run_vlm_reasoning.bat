@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Next-week experiments, pre-registered 2026-10-07 (dev segments only; test stays locked)
REM  No training - only the prompt or the input changes:
REM    E1 / E0.5  last image 1 s / 0.5 s after the action ends (prompt v1 was 2 s)
REM    R1         step-by-step written reasoning before the answer
REM    R3         prompt v1 at +0.5 s plus the alignment cue (connector drops onto the socket)
REM    R2         the models' built-in thinking mode (slow)
REM    trained baseline with the 1 s / 0.5 s frames (checks the 2 s shortcut)
REM  1 self-test  2 extract dev frames at +1 s and +0.5 s  3 VLM runs  4 trained baseline
REM  5 comparison table: experiments\prompt_versions.md
REM  Takes several hours. Safe to close; double-click again to continue.
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM reasoning and end-buffer experiments (dev only)
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
set "STATUS=%CD%\logs\run_vlm_reasoning_status.txt"
>"%STATUS%" echo started %DATE% %TIME%
set "RUN_RC=0"

echo.
echo [1/5] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_vlm.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_vlm.txt"
>>"%STATUS%" echo pytest rc=%PYTEST_RC%
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/5] Extracting dev frames ending 1 s and 0.5 s after each action (a few minutes) ...
python -m scripts.extract_frames --set dev --end-buffer 1 --log "logs\extract_dev_eb1.log"
set "EX_RC=%ERRORLEVEL%"
python -m scripts.extract_frames --set dev --end-buffer 0.5 --log "logs\extract_dev_eb05.log"
if not "%ERRORLEVEL%"=="0" set "EX_RC=%ERRORLEVEL%"
>>"%STATUS%" echo extract rc=%EX_RC%
if not "%EX_RC%"=="0" goto :extract_failed

echo.
echo [3/5] VLM runs (finished ones are skipped; the PC will not sleep) ...
for %%E in (configs\experiment_e1.yaml configs\experiment_e05.yaml configs\experiment_r1.yaml configs\experiment_r3.yaml configs\experiment_r2.yaml) do call :run_exp "%%E" "%%~nE"

echo.
echo [4/5] Trained baseline with the +1 s and +0.5 s frames (frames for other parts first; 20-40 minutes each) ...
for %%P in (configs\probe_eb1.yaml configs\probe_eb05.yaml) do call :run_probe "%%P" "%%~nP"

echo [5/5] Comparing ...
python -m scripts.run_vlm --compare "configs/experiment*.yaml" --log "logs\prompt_versions.log"
python -m scripts.evaluate --log "logs\dev_evaluation.log"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%).
echo   Side by side: experiments\prompt_versions.md
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

:run_probe
echo       %~2 ...
python -m scripts.probe all --config "%~1" --log "logs\%~2.log"
set "P_RC=%ERRORLEVEL%"
>>"%STATUS%" echo %~2 rc=%P_RC% %DATE% %TIME%
if not "%P_RC%"=="0" set "RUN_RC=%P_RC%"
exit /b 0

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_vlm.txt and tell Claude.
pause
exit /b 1

:extract_failed
echo.
echo   [STOP] Frame extraction failed - see logs\extract_dev_eb1.log / extract_dev_eb05.log and tell Claude.
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
