@echo off
setlocal EnableExtensions
REM Make short multi-view clips of 5 segments for the manual check (package 1 acceptance).
REM Green border = annotated segment, grey border = padding before/after.
REM Extra arguments are passed through, e.g.:  check_segments.bat --n 8 --pad 2
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
echo Making clips for the manual check (about 1-3 minutes) ...
echo.
python -m scripts.export_check_clips --log "logs\check_clips.log" %*
set "RC=%ERRORLEVEL%"
>"logs\check_clips_status.txt" echo rc=%RC% %DATE% %TIME%
if exist "data\derived\manual_check\checklist.md" start "" "data\derived\manual_check"
echo.
echo ==================================================================
echo   Done. A folder window should open: data\derived\manual_check
echo   Play the .mp4 files (green border = annotated segment).
echo   If a clip does not play, open it with VLC.
echo ==================================================================
pause
exit /b 0

:no_conda
echo conda.bat not found - run setup_windows.bat first.
pause
exit /b 1
