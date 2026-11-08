@echo off
setlocal EnableExtensions
REM Package 4: collect the exported annotation files from Downloads -> compare with Claude and the
REM dataset reference -> merge gold (adjudicated segments) -> refresh split summary -> open the report.
REM Run it again after exporting from the adjudication page.
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
set "TESTLOG=pytest_package4.txt"
set "STATUSFILE=compare_status.txt"
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
  if not defined CONDA_BAT if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%C"
if not defined CONDA_BAT goto :no_conda
call "%CONDA_BAT%" activate %ENV_NAME%
if not exist "logs" mkdir "logs"
set "ANN=data\annotations\pilot30"
echo.
echo [1/5] Collecting exported files (pilot30__*.jsonl / .csv) from Downloads ...
python -m scripts.annotate collect --log "logs\annotate_collect.log"
if not exist "%ANN%\human__*.jsonl" goto :no_human
if not exist "%ANN%\ai__claude.jsonl" echo   NOTE: Claude's file ai__claude.jsonl is not there yet - comparing with the dataset reference only.
echo [2/5] Updating the annotation page settings (configs\annotation.yaml) ...
python -m scripts.annotate refresh --log "logs\annotate_refresh.log"
echo [3/5] Comparing ...
python -m scripts.annotate compare --log "logs\annotate_compare.log"
set "CMP_RC=%ERRORLEVEL%"
echo [4/5] Merging gold ...
python -m scripts.annotate merge --log "logs\annotate_merge.log"
echo [5/5] Refreshing split summary and leakage audit ...
python -m scripts.make_split --log "logs\make_split.log"
>"logs\%STATUSFILE%" echo compare=%CMP_RC% %DATE% %TIME%
if exist "%ANN%\reports\agreement.html" start "" "%ANN%\reports\agreement.html"
if exist "%ANN%\reports\needs_adjudication.flag" start "" "data\derived\annotation\pilot30\adjudicate.html"
echo.
echo ==================================================================
echo   Report : %ANN%\reports\agreement.html
if exist "%ANN%\reports\needs_adjudication.flag" echo   Some segments need your decision: the adjudication page has opened.
if exist "%ANN%\reports\needs_adjudication.flag" echo   After deciding, click "export JSONL" there and run this file again.
if not exist "%ANN%\reports\needs_adjudication.flag" echo   All disputes are resolved: gold is in %ANN%\gold.jsonl
echo ==================================================================
pause
exit /b 0

:no_human
echo.
echo   No annotation file found. On the annotation page click "export JSONL"
echo   (it goes to your Downloads folder), then run this file again.
pause
exit /b 1

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
