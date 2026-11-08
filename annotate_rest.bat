@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 4, remaining segments: the annotation page for the 56 BNC segments
REM  that are not in pilot30 (41 test + 15 dev). Same v2 rules and page as the pilot.
REM  1 self-test   2 dense frames + annotation page (a few minutes; reuses what exists)
REM  3 dataset reference labels   4 open the page
REM  Safe to run again: frames already made are reused, your drafts stay in the browser.
REM  When finished, click "export JSONL" on the page, then run compare_rest.bat
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM annotation page (remaining 56 segments)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
set "TASK=rest56"
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
echo [1/4] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_rest.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_rest.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/4] Building the %TASK% annotation page (dense frames, a few minutes) ...
python -m scripts.annotate build --task %TASK% --rest-of pilot30 --log "logs\annotate_build_%TASK%.log"
set "BUILD_RC=%ERRORLEVEL%"
echo [3/4] Dataset reference labels ...
python -m scripts.annotate reference --task %TASK% --log "logs\annotate_reference_%TASK%.log"
>"logs\annotate_%TASK%_status.txt" echo build=%BUILD_RC% %DATE% %TIME%
echo [4/4] Opening the annotation page ...
if exist "data\derived\annotation\%TASK%\index.html" start "" "data\derived\annotation\%TASK%\index.html"
echo.
echo ==================================================================
echo   Done (build=%BUILD_RC%).
echo   Annotation page: data\derived\annotation\%TASK%\index.html
echo   Rules: docs\annotation_manual.md (v2). 56 segments; export any time,
echo   the page keeps your drafts. Do not use what you see in these videos
echo   to change the VLM prompts (41 of them are test segments).
echo   When finished, click "export JSONL" on the page, then run compare_rest.bat
echo ==================================================================
pause
exit /b %BUILD_RC%

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_rest.txt and tell Claude.
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
