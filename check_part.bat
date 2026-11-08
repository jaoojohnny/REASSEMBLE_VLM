@echo off
setlocal EnableExtensions
REM Package 2 manual check: 5 clips of the chosen Insert part (target_part in configs\field_mapping.yaml),
REM sampled from data\index\segments.csv and read from the slim copies in data\slim.
REM Extra arguments are passed through, e.g.:  check_part.bat --part USB --n 8
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
echo Making 5 clips of the chosen Insert part (about 1 minute) ...
echo.
python -m scripts.export_check_clips --from-index --out data\derived\manual_check_part --log "logs\check_part.log" %*
set "RC=%ERRORLEVEL%"
>"logs\check_part_status.txt" echo rc=%RC% %DATE% %TIME%
if exist "data\derived\manual_check_part\checklist.md" start "" "data\derived\manual_check_part"
echo.
echo ==================================================================
echo   Done. A folder window should open: data\derived\manual_check_part
echo   Play the .mp4 files (green border = annotated segment).
echo ==================================================================
pause
exit /b 0

:no_conda
echo conda.bat not found - run setup_windows.bat first.
pause
exit /b 1
