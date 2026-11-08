@echo off
setlocal EnableExtensions
REM Re-run HDF5 inspection on data\raw (after setup_windows.bat has created the env).
REM Extra arguments are passed through, e.g.:  run_inspect.bat --expect-cameras hama1,hama2,hand
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
  if not defined CONDA_BAT if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%C"
if not defined CONDA_BAT (
  echo conda.bat not found - run setup_windows.bat first.
  pause
  exit /b 1
)
call "%CONDA_BAT%" activate %ENV_NAME%
if not exist "logs" mkdir "logs"
python -m scripts.inspect_h5 data\raw --log "logs\inspect.log" %*
echo.
echo Results: data\index\inspect\   Log: logs\inspect.log
pause
