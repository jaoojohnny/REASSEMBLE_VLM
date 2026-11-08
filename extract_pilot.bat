@echo off
setlocal EnableExtensions
REM Package 3 pilot: extract frames (1 / 4 / 8 per segment, by real timestamps) for 40 segments
REM of the chosen Insert part (20 failure + 20 success), then open the preview page.
REM Extra arguments are passed through, e.g.:  extract_pilot.bat --end-buffer 3
cd /d "%~dp0"
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
echo.
echo [1/2] Self-test (about 20 seconds) ...
python -m pytest -q >"logs\pytest_extract.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_extract.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed
echo [2/2] Extracting pilot frames (40 segments x 3 cameras, a few minutes) ...
python -m scripts.extract_frames --set pilot --log "logs\extract_pilot.log" %*
set "RC=%ERRORLEVEL%"
>"logs\extract_pilot_status.txt" echo rc=%RC% %DATE% %TIME%
if exist "data\derived\frames\pilot_index.html" start "" "data\derived\frames\pilot_index.html"
echo.
echo ==================================================================
echo   Done (rc=%RC%). The preview page should open in your browser:
echo   data\derived\frames\pilot_index.html
echo ==================================================================
pause
exit /b 0

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_extract.txt and tell Claude.
>"logs\extract_pilot_status.txt" echo stopped: tests failed %DATE% %TIME%
pause
exit /b 1

:no_conda
echo conda.bat not found - run setup_windows.bat first.
pause
exit /b 1
