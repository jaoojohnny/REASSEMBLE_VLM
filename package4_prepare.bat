@echo off
setlocal EnableExtensions
REM Package 4: install new requirement -> self-test -> build pilot30 annotation page (dense frames)
REM -> dataset reference labels -> split (dev 25% / test 75%) + leakage audit -> open the annotation page.
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
set "TESTLOG=pytest_package4.txt"
set "STATUSFILE=package4_status.txt"
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
  if not defined CONDA_BAT if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%C"
if not defined CONDA_BAT goto :no_conda
call "%CONDA_BAT%" activate %ENV_NAME%
if not exist "logs" mkdir "logs"
echo.
echo [1/5] Installing new requirement (jsonschema) ...
python -m pip install -q -r requirements.txt >"logs\pip_package4.txt" 2>&1
if errorlevel 1 echo   (pip failed - see logs\pip_package4.txt; continuing)
echo [2/5] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\%TESTLOG%" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\%TESTLOG%"
if not "%PYTEST_RC%"=="0" goto :tests_failed
echo [3/5] Building the pilot30 annotation page (30 segments, dense frames, 1-3 minutes) ...
python -m scripts.annotate build --log "logs\annotate_build.log"
set "BUILD_RC=%ERRORLEVEL%"
python -m scripts.annotate reference --log "logs\annotate_reference.log"
python -m scripts.annotate schema --log "logs\annotate_schema.log"
echo [4/5] Split (dev 25%% / test 75%%, grouped by recording) and leakage audit ...
python -m scripts.make_split --log "logs\make_split.log"
set "SPLIT_RC=%ERRORLEVEL%"
>"logs\%STATUSFILE%" echo build=%BUILD_RC% split=%SPLIT_RC% %DATE% %TIME%
echo [5/5] Opening the annotation page ...
if exist "data\derived\annotation\pilot30\index.html" start "" "data\derived\annotation\pilot30\index.html"
echo.
echo ==================================================================
echo   Done (build=%BUILD_RC%, split=%SPLIT_RC%).
echo   Annotation page: data\derived\annotation\pilot30\index.html
echo   Leakage report : data\splits\bnc_dev25_test75_s0\leakage_report.md
echo   When finished, click "export JSONL" on the page, then run compare_pilot.bat
echo ==================================================================
pause
exit /b 0

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\%TESTLOG% and tell Claude.
>"logs\%STATUSFILE%" echo stopped: tests failed %DATE% %TIME%
pause
exit /b 1

:no_conda
echo conda.bat not found - run setup_windows.bat first.
pause
exit /b 1
