@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 12 - TWO new test sets (run once; both configs frozen):
REM    round peg 1 and large gear (round peg 4 and small gear stay unused).
REM  Main question (chosen on the clean dev set by the rule written before
REM  the GPT dev results): does GPT-6.1 Sol alone beat the local Qwen with the
REM  probability threshold (tau 0.999089, package 11)? Both parts pooled.
REM  Also: two stages (Qwen threshold -> GPT; Qwen Skill v3 -> GPT),
REM  Qwen Skill v3, always-success. Two-stage results reuse the same answers.
REM  1 self-test  2 frozen plans  3 splits
REM  4 each part: frames, every method once, score  (about 40 minutes total)
REM  5 pooled report -> experiments\p12_final_pooled\report.md
REM  API: about 1 US dollar (GPT on all test segments). Uses the key in .env.
REM  Safe to close; double-click again to continue (finished segments are skipped).
REM ==================================================================
title REASSEMBLE_VLM package 12 - round peg 1 + large gear test runs (run once)
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ENV_NAME=reassemble-vlm"
if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
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
echo [1/5] Self-test (about 40 seconds) ...
python -m pytest -q >"logs\pytest_p12t.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p12t.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/5] Frozen plans ...
python -m scripts.final_test plan --config configs\final_test_rpeg1.yaml --log "logs\final_test_rpeg1_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed
python -m scripts.final_test plan --config configs\final_test_lgear.yaml --log "logs\final_test_lgear_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed

echo [3/5] Splits (every segment = test) ...
if not exist "data\splits\rpeg1_test100\manifest.json" (
  python -m scripts.make_split --part "round peg 1" --ratios test=1 --name rpeg1_test100 --no-hash --log "logs\split_rpeg1.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\lgear_test100\manifest.json" (
  python -m scripts.make_split --part "large gear" --ratios test=1 --name lgear_test100 --no-hash --log "logs\split_lgear.log"
  if errorlevel 2 goto :failed
)

set "HF_HUB_OFFLINE=1"
set "RUN_RC=0"
echo.
echo [4/5] Test sets: frames, every method once, score (the PC will not sleep) ...
echo        round peg 1 ...
python -m scripts.final_test all --config configs\final_test_rpeg1.yaml --log "logs\final_test_rpeg1.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        large gear ...
python -m scripts.final_test all --config configs\final_test_lgear.yaml --log "logs\final_test_lgear.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"

echo [5/5] Pooled report (2 parts) ...
python -m scripts.cascade_test --log "logs\cascade_test.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%). Reports:
echo     experiments\p12_final_pooled\report.md   (main: 2 parts pooled)
echo     experiments\final_test_rpeg1\report.md
echo     experiments\final_test_lgear\report.md
echo   If a method failed, double-click again. Tell Claude when done.
echo ==================================================================
if exist "experiments\p12_final_pooled\report.md" start "" notepad "experiments\p12_final_pooled\report.md"
pause
exit /b %RUN_RC%

:not_frozen
echo.
echo   [STOP] A package 12 config is not frozen. The test sets were NOT opened.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see the newest file in logs\ and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p12t.txt and tell Claude.
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
