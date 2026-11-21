@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 20 (development stage) - try to make the step classifiers
REM  better. Development parts only; bolt 4 is NOT touched here.
REM    C0 = package 19 (wrist, 1 s after the step ends)
REM    C1 = look at 3 frames (0 / 0.5 / 1 s after the step ends)
REM    C2 = keep spatial detail (3 x 3 regions of the wrist image)
REM    C3 = add the two external cameras
REM  Extracts the external-camera frames, DINOv2 on the GPU, then the
REM  frozen F2S2 recipe for every option. Pre-registered rule in
REM  docs\package20_stepboost.md. No download, no API cost.
REM  About 15-30 minutes.
REM ==================================================================
title REASSEMBLE_VLM package 20 - better step classifiers (development parts)
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
echo [1/2] Self-test (about 2 minutes) ...
python -m pytest -q tests\test_p20_stepcls2.py >"logs\pytest_p20.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p20.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/2] External-camera frames, DINOv2, 4 options x 5 questions - the PC will not sleep (about 15-30 minutes) ...
python -m scripts.stepcls2 dev --log "logs\stepcls2_dev.log"
set "RUN_RC=%ERRORLEVEL%"
if "%RUN_RC%"=="3" goto :stopped
if "%RUN_RC%"=="2" goto :extract_errors
if not "%RUN_RC%"=="0" goto :failed
echo.
echo ==================================================================
echo   Finished. Report: experiments\p20_stepboost_dev\report.md
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p20_stepboost_dev\report.md" start "" notepad "experiments\p20_stepboost_dev\report.md"
pause
exit /b 0

:extract_errors
echo.
echo   Finished, but some frames could not be extracted - the report was
echo   still written. See logs\stepcls2_dev.log and tell Claude.
if exist "experiments\p20_stepboost_dev\report.md" start "" notepad "experiments\p20_stepboost_dev\report.md"
pause
exit /b 2

:stopped
echo.
echo   [STOP] Not enough data to score - see logs\stepcls2_dev.log and tell Claude.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see logs\stepcls2_dev.log and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p20.txt and tell Claude.
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
