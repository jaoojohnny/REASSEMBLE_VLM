@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Final model for ONE part (default: bolt 4) with the FROZEN recipe
REM  F2S2 (configs\perpart_selected.yaml, unchanged).
REM  Package 14 only scored the recipe with cross-validation and threw
REM  the temporary models away. This trains ONE model on ALL segments
REM  of the part and saves it, so new segments can be judged later.
REM  1 self-test  2 train + save + re-encode 3 frames as a check
REM  (DINOv2 on GPU, about 1-2 minutes)  3 open report
REM  Output: models\perpart_final\bolt_4.npz / .json
REM          experiments\perpart_final\bolt_4.md
REM  round peg 4 and small gear stay untouched. No API cost.
REM ==================================================================
title REASSEMBLE_VLM final per-part model - bolt 4
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
set "PART=bolt 4"
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
python -m pytest -q tests\test_perpart_final.py >"logs\pytest_final.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_final.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/2] Train the final model for "%PART%" on all its segments ...
python -m scripts.perpart_final --part "%PART%" --check 3 --log "logs\perpart_final.log"
if errorlevel 3 goto :stopped
if errorlevel 2 goto :check_failed
if errorlevel 1 goto :failed
echo.
echo ==================================================================
echo   Finished. Model: models\perpart_final\bolt_4.npz / .json
echo   Report: experiments\perpart_final\bolt_4.md
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\perpart_final\bolt_4.md" start "" notepad "experiments\perpart_final\bolt_4.md"
pause
exit /b 0

:stopped
echo.
echo   [STOP] See logs\perpart_final.log (frozen recipe or features missing) and tell Claude.
pause
exit /b 3

:check_failed
echo.
echo   [CHECK] The model was saved but a check did not pass - see logs\perpart_final.log and tell Claude.
pause
exit /b 2

:failed
echo.
echo   [STOP] A step failed - see logs\perpart_final.log and tell Claude.
pause
exit /b 1

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
