@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 11 - FOUR new test sets, run once (all configs frozen 2026-10-31):
REM  square peg 4, square peg 3, round peg 2, waterproof (every segment = test).
REM  Main question (pooled over the 4 parts): does the probability threshold
REM  (read Qwen's yes-probabilities, failure only if the score > 0.999089)
REM  beat Qwen + Skill v3 (written answer)?
REM  Methods per part: majority, probability threshold (main),
REM  wrist second look (Codex ideas, secondary), Qwen + Skill v3.
REM  1 self-test  2 frozen plans  3 splits  4 each part: frames, methods, score
REM  5 pooled report -> experiments\p11_final_pooled\report.md   (about 1 hour)
REM  No API. Safe to close; double-click again to continue (finished segments are skipped).
REM ==================================================================
title REASSEMBLE_VLM package 11 - 4 new test sets (run once)
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
python -m pytest -q >"logs\pytest_p11t.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_p11t.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/5] Frozen plans ...
python -m scripts.final_test plan --config configs\final_test_sqpeg4.yaml --log "logs\final_test_sqpeg4_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed
python -m scripts.final_test plan --config configs\final_test_sqpeg3.yaml --log "logs\final_test_sqpeg3_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed
python -m scripts.final_test plan --config configs\final_test_rpeg2.yaml --log "logs\final_test_rpeg2_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed
python -m scripts.final_test plan --config configs\final_test_wproof.yaml --log "logs\final_test_wproof_plan.log"
set "PLAN_RC=%ERRORLEVEL%"
if "%PLAN_RC%"=="3" goto :not_frozen
if not "%PLAN_RC%"=="0" goto :failed

echo [3/5] Splits (every segment = test) ...
if not exist "data\splits\sqpeg4_test100\manifest.json" (
  python -m scripts.make_split --part "square peg 4" --ratios test=1 --name sqpeg4_test100 --no-hash --log "logs\split_sqpeg4.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\sqpeg3_test100\manifest.json" (
  python -m scripts.make_split --part "square peg 3" --ratios test=1 --name sqpeg3_test100 --no-hash --log "logs\split_sqpeg3.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\rpeg2_test100\manifest.json" (
  python -m scripts.make_split --part "round peg 2" --ratios test=1 --name rpeg2_test100 --no-hash --log "logs\split_rpeg2.log"
  if errorlevel 2 goto :failed
)
if not exist "data\splits\wproof_test100\manifest.json" (
  python -m scripts.make_split --part "waterproof" --ratios test=1 --name wproof_test100 --no-hash --log "logs\split_wproof.log"
  if errorlevel 2 goto :failed
)

set "HF_HUB_OFFLINE=1"
set "RUN_RC=0"
echo.
echo [4/5] Test sets: frames, every method once, score (the PC will not sleep) ...
echo        square peg 4 ...
python -m scripts.final_test all --config configs\final_test_sqpeg4.yaml --log "logs\final_test_sqpeg4.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        square peg 3 ...
python -m scripts.final_test all --config configs\final_test_sqpeg3.yaml --log "logs\final_test_sqpeg3.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        round peg 2 ...
python -m scripts.final_test all --config configs\final_test_rpeg2.yaml --log "logs\final_test_rpeg2.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo        waterproof ...
python -m scripts.final_test all --config configs\final_test_wproof.yaml --log "logs\final_test_wproof.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"

echo [5/5] Pooled report (4 parts) ...
python -m scripts.reason_test --log "logs\reason_test.log"
if errorlevel 1 set "RUN_RC=%ERRORLEVEL%"
echo.
echo ==================================================================
echo   Finished (rc=%RUN_RC%).
echo   Pooled report : experiments\p11_final_pooled\report.md
echo   Per part      : experiments\final_test_sqpeg4\report.md (and sqpeg3, rpeg2, wproof)
echo   If a method failed, fix it and double-click again. Tell Claude when done.
echo ==================================================================
if exist "experiments\p11_final_pooled\report.md" start "" notepad "experiments\p11_final_pooled\report.md"
pause
exit /b %RUN_RC%

:not_frozen
echo.
echo   [STOP] A package 11 test config is not frozen. The test sets were NOT opened.
pause
exit /b 3

:failed
echo.
echo   [STOP] A step failed - see the newest file in logs\ and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_p11t.txt and tell Claude.
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
