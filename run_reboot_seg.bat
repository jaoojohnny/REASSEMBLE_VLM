@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 13 step 3 - REBOOT 16 mm cylinder (dev): find the moment the
REM  expert lets go of the cylinder (gripper opening), then cut segments.
REM  Reads ONLY the joint columns (gripper) of the expert data files on
REM  Hugging Face - the depth images in the same files are NOT read.
REM  Expected under 100 MB; stops if a file would need more than 20 MB
REM  or the total would pass 300 MB.
REM    success segment = 3 s ending where the expert starts to let go
REM    failure segment = 3 s ending where the official recovery starts
REM  1 self-test  2 gripper data + segments + check pictures
REM  Output: experiments\p13_cyl_segments\report.md
REM ==================================================================
title REASSEMBLE_VLM package 13 step 3 - cylinder gripper data + segments
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
python -m pytest -q tests\test_p13_seg.py tests\test_p13_cyl.py tests\test_p13_meta.py >"logs\pytest_p13seg.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p13seg.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/2] Gripper data, release moments, segments, check pictures ...
python -m scripts.reboot_seg --log "logs\reboot_seg.log"
set "RUN_RC=%ERRORLEVEL%"
if "%RUN_RC%"=="3" goto :too_big
if not "%RUN_RC%"=="0" goto :failed
echo.
echo ==================================================================
echo   Finished. Report: experiments\p13_cyl_segments\report.md
echo   Pictures: experiments\p13_cyl_segments\gripper_plots.jpg and sheets\
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p13_cyl_segments\report.md" start "" notepad "experiments\p13_cyl_segments\report.md"
pause
exit /b 0

:too_big
echo.
echo   [STOP] Reading would pass the size limit - nothing more was read.
echo   See logs\reboot_seg.log and tell Claude.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see logs\reboot_seg.log and tell Claude.
echo   Double-click again to retry; gripper data already read is kept.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p13seg.txt and tell Claude.
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
