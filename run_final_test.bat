@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 6: FINAL evaluation on the TEST set - run it only after the methods
REM  in configs\final_test.yaml were chosen on the dev set and set to frozen: true.
REM  1 self-test  2 show the frozen plan (stops if not frozen)
REM  3 extract test frames, run every method, score -> experiments\final_test\report.md
REM  The first run writes experiments\final_test\lock.json. If the settings change
REM  afterwards it refuses to run. Safe to close; double-click again to continue.
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM final test (test set, run once)
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

echo.
echo [1/3] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_final.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_final.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/3] Frozen plan ...
python -m scripts.final_test plan --log "logs\final_test_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed

echo.
echo [3/3] Test set: extract frames, run every method, score (can take a few hours) ...
python -m scripts.final_test all --log "logs\final_test.log"
set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%). Report: experiments\final_test\report.md
echo   If something failed, double-click again to continue, or tell Claude.
echo ==================================================================
if exist "experiments\final_test\report.md" start "" notepad "experiments\final_test\report.md"
pause
exit /b %RUN_RC%

:not_frozen
echo.
echo   [STOP] configs\final_test.yaml is not frozen yet. Choose the methods on the dev set
echo   with Claude first. The test set was NOT opened.
pause
exit /b 3

:failed
echo.
echo   [STOP] Could not read configs\final_test.yaml - see logs\final_test_plan.log and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_final.txt and tell Claude.
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
