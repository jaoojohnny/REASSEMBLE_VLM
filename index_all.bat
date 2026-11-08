@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 2: index ALL 149 REASSEMBLE recordings (run overnight)
REM  1 quick self-test   2 sanity index of data\raw   3 confirm
REM  4 download each recording -> index -> keep slim copy -> delete
REM  Safe to close any time; double-click again to resume.
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM full index
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
  if not defined CONDA_BAT if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%C"
if not defined CONDA_BAT goto :no_conda
call "%CONDA_BAT%" activate %ENV_NAME%
if not exist "logs" mkdir "logs"
set "STATUS=%CD%\logs\index_status.txt"
>"%STATUS%" echo started %DATE% %TIME%

echo.
echo [1/3] Self-test (about 20 seconds) ...
python -m pytest -q >"logs\pytest_index.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_index.txt"
>>"%STATUS%" echo pytest rc=%PYTEST_RC%
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/3] Sanity check on data\raw (a few seconds) ...
python -m scripts.index_segments data\raw --out data\index\_precheck --force --log "logs\index_precheck.log"
set "PRE_RC=%ERRORLEVEL%"
>>"%STATUS%" echo precheck rc=%PRE_RC%
if not "%PRE_RC%"=="0" goto :precheck_failed

echo.
echo [3/3] Full index of all 149 recordings
echo       - downloads about 54.8 GB in total, one recording at a time
echo       - keeps a slim copy (videos + timestamps + labels, about 8 GB) in data\slim
echo       - deletes each full file after indexing
echo       - expected time: several hours; the PC will not sleep while this runs
echo       - you can close this window any time and double-click again to resume
echo.
choice /C YN /M "Start the full index now"
if errorlevel 2 goto :cancelled
>>"%STATUS%" echo full index started %DATE% %TIME%
python -m scripts.index_segments --source remote --log "logs\index_all.log"
set "IDX_RC=%ERRORLEVEL%"
>>"%STATUS%" echo full index rc=%IDX_RC% %DATE% %TIME%
echo.
echo ==================================================================
echo   Finished (rc=%IDX_RC%). Report: data\index\action_summary.md
echo   If some recordings failed, just double-click again to retry them.
echo ==================================================================
if exist "data\index\action_summary.md" start "" "data\index\action_summary.md"
pause
exit /b 0

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_index.txt and tell Claude.
>>"%STATUS%" echo stopped: tests failed
pause
exit /b 1

:precheck_failed
echo.
echo   [STOP] Sanity check failed - see logs\index_precheck.log and tell Claude.
>>"%STATUS%" echo stopped: precheck failed
pause
exit /b 1

:cancelled
echo   Cancelled. Nothing was downloaded.
>>"%STATUS%" echo cancelled by user
pause
exit /b 0

:no_conda
echo conda.bat not found - run setup_windows.bat first.
pause
exit /b 1
