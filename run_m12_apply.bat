@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 16 step 2 - apply the bolt 4 model (models\perpart_final\
REM  bolt_4) to REBOOT M12, WITHOUT retraining, threshold 0.5 unchanged.
REM  Needs experiments\p16_m12\failure_annotations.json written by Claude
REM  first (failure moments marked before any model answer was seen).
REM  Success frame = wrist camera 1 s after the expert's last release;
REM  failure frame = the moment Claude marked. DINOv2 on GPU, a few minutes.
REM  No API cost.
REM ==================================================================
title REASSEMBLE_VLM package 16 step 2 - bolt 4 model on REBOOT M12
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
echo [1/2] Self-test ...
python -m pytest -q tests\test_p16_m12.py tests\test_perpart_final.py >"logs\pytest_p16.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p16.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/2] Apply the bolt 4 model - the PC will not sleep ...
python -m scripts.reboot_m12 apply --log "logs\m12_apply.log"
set "RUN_RC=%ERRORLEVEL%"
if "%RUN_RC%"=="3" goto :stopped
if not "%RUN_RC%"=="0" goto :failed
echo.
echo ==================================================================
echo   Finished. Report: experiments\p16_m12\report.md
echo   Model inputs: experiments\p16_m12\model_inputs.jpg
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p16_m12\report.md" start "" notepad "experiments\p16_m12\report.md"
pause
exit /b 0

:stopped
echo.
echo   [STOP] Failure annotations missing or incomplete - the model was not scored.
echo   See logs\m12_apply.log and tell Claude.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see logs\m12_apply.log and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p16.txt and tell Claude.
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
