@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 19 (test stage) - apply the frozen step-classifier recipe
REM  (configs\stepcls_selected.yaml: F2S2, wrist frame 1 s after the
REM  step ends) to bolt 4, ONCE:
REM    - "did this step succeed?" for bolt 4 Twist and bolt 4 Align
REM    - "which step failed?" for the 30 failed bolt 4 segments
REM    - shuffle test (100 shuffles per question)
REM  Extracts bolt 4 step frames, DINOv2 on the GPU. No download, no
REM  API cost. About 10-20 minutes. It will not run a second time.
REM  Details were written in docs\package19_stepcls.md before running.
REM ==================================================================
title REASSEMBLE_VLM package 19 - step classifiers on bolt 4 (test)
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
set "HF_HUB_OFFLINE=1"
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
echo [1/2] Self-test (about 1-2 minutes) ...
python -m pytest -q tests\test_p19_stepcls.py >"logs\pytest_p19_test.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p19_test.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/2] bolt 4 step frames, DINOv2, step classifiers, shuffle test - the PC will not sleep (about 10-20 minutes) ...
python -m scripts.stepcls test --log "logs\stepcls_test.log"
set "RUN_RC=%ERRORLEVEL%"
if "%RUN_RC%"=="3" goto :stopped
if "%RUN_RC%"=="2" goto :extract_errors
if not "%RUN_RC%"=="0" goto :failed
echo.
echo ==================================================================
echo   Finished. Report: experiments\p19_stepcls_test\report.md
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p19_stepcls_test\report.md" start "" notepad "experiments\p19_stepcls_test\report.md"
pause
exit /b 0

:extract_errors
echo.
echo   Finished, but some frames could not be extracted - the report was
echo   still written. See logs\stepcls_test.log and tell Claude.
if exist "experiments\p19_stepcls_test\report.md" start "" notepad "experiments\p19_stepcls_test\report.md"
pause
exit /b 2

:stopped
echo.
echo   [STOP] Not run: already has a result, or the frozen recipe is missing.
echo   See logs\stepcls_test.log and tell Claude.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see logs\stepcls_test.log and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p19_test.txt and tell Claude.
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
