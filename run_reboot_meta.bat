@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 13 step 1 - external plug data REBOOT (NIST task board).
REM  Downloads ONLY the small description files (README + meta, about 5 MB)
REM  of 4 plug parts: USB-A, RJ45, USB-C, NEMA power plug
REM  (expert = success demos, recovery = induced failure then fixed),
REM  plus the official sample. NO videos, NO depth data.
REM  Checks: are failure timestamps (t_f, t_r) included? Can this PC decode AV1?
REM  If pyarrow is missing it is installed first (reads LeRobot tables, ~30 MB).
REM  1 pyarrow  2 self-test  3 download + check  4 open report
REM  Output: data\raw\reboot_meta\  experiments\p13_reboot_meta\report.md
REM ==================================================================
title REASSEMBLE_VLM package 13 step 1 - REBOOT description files only
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
echo [1/4] pyarrow ...
python -c "import pyarrow" 1>nul 2>nul
if not errorlevel 1 goto :have_pyarrow
echo        not installed - installing pyarrow, about 30 MB ...
python -m pip install "pyarrow>=15" >"logs\pip_pyarrow.txt" 2>&1
if errorlevel 1 echo        [note] install failed - see logs\pip_pyarrow.txt. Continuing; Claude will read the tables.
:have_pyarrow

echo [2/4] Self-test ...
python -m pytest -q tests\test_p13_meta.py >"logs\pytest_p13meta.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p13meta.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [3/4] Listing 10 datasets and downloading description files only ...
python -m scripts.reboot_meta_check --log "logs\reboot_meta_check.log"
set "RUN_RC=%ERRORLEVEL%"
if "%RUN_RC%"=="2" goto :net_failed
if not "%RUN_RC%"=="0" goto :failed

echo [4/4] Report ...
echo.
echo ==================================================================
echo   Finished. Report: experiments\p13_reboot_meta\report.md
echo   Downloaded files: data\raw\reboot_meta\   Tell Claude when done.
echo ==================================================================
if exist "experiments\p13_reboot_meta\report.md" start "" notepad "experiments\p13_reboot_meta\report.md"
pause
exit /b 0

:net_failed
echo.
echo   [STOP] Could not reach Hugging Face for any dataset - check the internet
echo   connection and double-click again. Details: logs\reboot_meta_check.log
pause
exit /b 2

:failed
echo.
echo   [STOP] A step failed - see logs\reboot_meta_check.log and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p13meta.txt and tell Claude.
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
