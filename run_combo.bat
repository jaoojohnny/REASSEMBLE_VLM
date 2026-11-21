@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 15 - per-part classifier (package 14, F2S2) + local Qwen
REM  probability threshold (package 11, tau 0.999089). No GPT, no API cost.
REM  1 self-test
REM  2 run the frozen Qwen threshold model on segments that never had it:
REM    BNC, USB, bolt 4, Ethernet, square peg 1 and the ref segments of
REM    the 4 dev parts - 412 segments, GPU, about 35 minutes
REM  3 dev stage: 6 methods on the 5 dev parts, the rule in
REM    docs\package15_combo.md picks one and freezes it
REM  4 test stage: the frozen method on the 10 test parts + day check
REM  round peg 4 and small gear stay untouched.
REM  Safe to close; double-click again to continue (finished segments are skipped).
REM ==================================================================
title REASSEMBLE_VLM package 15 - per-part classifier + Qwen threshold
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
echo [1/4] Self-test ...
python -m pytest -q tests\test_p15_combo.py tests\test_p14_perpart.py >"logs\pytest_p15.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p15.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/4] Qwen probability threshold on 412 segments - GPU, the PC will not sleep ...
python -m scripts.combo qwen --log "logs\combo_qwen.log"
if errorlevel 1 goto :failed

echo [3/4] Dev stage: 6 methods on 5 dev parts, pick and freeze ...
python -m scripts.combo dev --log "logs\combo_dev.log"
if errorlevel 3 goto :frozen_problem
if errorlevel 1 goto :failed

echo [4/4] Test stage: frozen method on 10 test parts ...
python -m scripts.combo test --log "logs\combo_test.log"
if errorlevel 3 goto :frozen_problem
if errorlevel 1 goto :failed
echo.
echo ==================================================================
echo   Finished. Reports: experiments\p15_combo_dev\report.md
echo                      experiments\p15_combo_test\report.md
echo   Tell Claude when done.
echo ==================================================================
if exist "experiments\p15_combo_test\report.md" start "" notepad "experiments\p15_combo_test\report.md"
pause
exit /b 0

:frozen_problem
echo.
echo   [STOP] Frozen setting missing or different - see logs\combo_dev.log / combo_test.log and tell Claude.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see the newest combo log in logs\ and tell Claude.
echo   Double-click again to continue; finished Qwen segments are skipped.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p15.txt and tell Claude.
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
