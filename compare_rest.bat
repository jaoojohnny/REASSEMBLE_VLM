@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 4, remaining segments (rest56): collect your exported file from Downloads
REM  -> compare with Claude's independent labels and the dataset reference -> merge gold
REM  -> refresh split summary -> open the report (and the adjudication page if needed).
REM  Run it again after exporting from the adjudication page.
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM compare (remaining 56 segments)
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
set "ANN=data\annotations\%TASK%"
echo.
echo [1/5] Collecting exported files (%TASK%__*.jsonl / .csv) from Downloads ...
python -m scripts.annotate collect --task %TASK% --log "logs\annotate_collect_%TASK%.log"
if not exist "%ANN%\human__*.jsonl" goto :no_human
if not exist "%ANN%\ai__claude.jsonl" echo   NOTE: Claude's file ai__claude.jsonl is not there yet - comparing with the dataset reference only.
echo [2/5] Updating the annotation page settings (configs\annotation.yaml) ...
python -m scripts.annotate refresh --task %TASK% --log "logs\annotate_refresh_%TASK%.log"
echo [3/5] Comparing ...
python -m scripts.annotate compare --task %TASK% --log "logs\annotate_compare_%TASK%.log"
set "CMP_RC=%ERRORLEVEL%"
echo [4/5] Merging gold ...
python -m scripts.annotate merge --task %TASK% --log "logs\annotate_merge_%TASK%.log"
echo [5/5] Refreshing split summary and leakage audit ...
python -m scripts.make_split --log "logs\make_split.log"
>"logs\compare_%TASK%_status.txt" echo compare=%CMP_RC% %DATE% %TIME%
if exist "%ANN%\reports\agreement.html" start "" "%ANN%\reports\agreement.html"
if exist "%ANN%\reports\needs_adjudication.flag" start "" "data\derived\annotation\%TASK%\adjudicate.html"
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

:no_conda
echo conda.bat not found - run setup_windows.bat first.
pause
exit /b 1

:no_env
echo Could not activate the conda env "%ENV_NAME%" - run setup_windows.bat first.
pause
exit /b 1
