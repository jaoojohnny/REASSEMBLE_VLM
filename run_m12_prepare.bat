@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 16 step 1 - REBOOT M12 fastener (external data, Hugging Face):
REM  the bolt 4 model will be applied to it WITHOUT retraining.
REM    REBOOT26/M12-fastener_install  = expert demos (success)
REM    REBOOT26/m12_recovery_install  = induced failure + recovery
REM  Downloads README, meta, and color video of the top camera (cam_high)
REM  plus the wrist camera of the working arm. Limit 5 GB (expected about
REM  3-4 GB); stops before any video if the total is larger.
REM  Reads ONLY the gripper columns of the data files (no depth images).
REM  Then: last release of each expert demo, and one picture sheet per
REM  failure episode (every 2 s) for Claude to mark the failure moment.
REM  No API cost. The model is NOT run in this step.
REM ==================================================================
title REASSEMBLE_VLM package 16 step 1 - REBOOT M12 download + sheets
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
set "HF_HUB_OFFLINE="
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

echo [2/2] Download (up to 5 GB), gripper data, picture sheets - the PC will not sleep ...
python -m scripts.reboot_m12 prepare --log "logs\m12_prepare.log"
set "RUN_RC=%ERRORLEVEL%"
if "%RUN_RC%"=="3" goto :stopped
if not "%RUN_RC%"=="0" goto :failed
echo.
echo ==================================================================
echo   Finished. Report: experiments\p16_m12\prepare_report.md
echo   Sheets: experiments\p16_m12\sheets\
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p16_m12\prepare_report.md" start "" notepad "experiments\p16_m12\prepare_report.md"
pause
exit /b 0

:stopped
echo.
echo   [STOP] Over the 5 GB limit, disk full, or a download failed - videos may be incomplete.
echo   See logs\m12_prepare.log and tell Claude.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see logs\m12_prepare.log and tell Claude.
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
