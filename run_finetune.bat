@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Training phase, main track: LoRA fine-tune of Qwen3-VL-8B (pre-registered 2026-10-14)
REM  Trains only on OTHER parts' insert segments (never BNC; never BNC test recordings).
REM  Settings: configs\finetune_qwen3vl8b.yaml   prompt: configs\prompts\ft_generic.txt
REM  1 self-test  2 install peft (small)  3 list the training data
REM  4 smoke run on 6 segments + check the adapter answers  -> time estimate
REM  5 full training (asks first; about 1-2 hours, the PC will not sleep)
REM  6 BNC dev: fine-tuned model vs the same model without fine-tuning (same prompt)
REM  7 dev scores: experiments\dev_evaluation\report.md
REM  Safe to close; double-click again to continue (finished steps are skipped).
REM ==================================================================
title REASSEMBLE_VLM training phase - LoRA fine-tune (dev only)
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
set "STATUS=%CD%\logs\run_finetune_status.txt"
>"%STATUS%" echo started %DATE% %TIME%
set "FT_FINAL=models\ft\qwen3vl8b_lora_r16\final\train_report.json"

echo.
echo [1/7] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_train.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_train.txt"
>>"%STATUS%" echo pytest rc=%PYTEST_RC%
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/7] Checking peft (LoRA library) ...
python -c "import peft" >nul 2>&1
if not errorlevel 1 goto :have_peft
echo       Installing peft (small download; torch and transformers stay as they are) ...
python -c "import torch, transformers; print('torch==' + torch.__version__); print('transformers==' + transformers.__version__)" >"logs\train_constraint.txt"
if errorlevel 1 goto :pip_failed
python -m pip install -r requirements-train.txt -c "logs\train_constraint.txt" >"logs\setup_train_pip.txt" 2>&1
if errorlevel 1 goto :pip_failed
:have_peft
python -c "import peft; print('      peft ' + peft.__version__)"
>>"%STATUS%" echo peft ok

REM Model weights are already downloaded (Qwen3-VL-8B); no surprise downloads from here on
set "HF_HUB_OFFLINE=1"
echo [3/7] Training data (other parts only) ...
python -m scripts.finetune plan --log "logs\finetune_plan.log"
if errorlevel 1 goto :plan_failed

if exist "%FT_FINAL%" goto :trained

echo.
echo [4/7] Smoke run: 6 segments, 1 short epoch, then the adapter answers 2 dev segments (5-10 minutes) ...
python -m scripts.finetune smoke --log "logs\finetune_smoke.log"
set "SM_RC=%ERRORLEVEL%"
>>"%STATUS%" echo smoke rc=%SM_RC% %DATE% %TIME%
if not "%SM_RC%"=="0" goto :smoke_failed
python -m scripts.run_vlm --experiment configs\experiment_ft.yaml --models configs\model.yaml configs\model_train.yaml --use qwen3vl_8b_ft_smoke --name dev_ft_smoke --limit 2 --force --log "logs\finetune_smoke_infer.log"
set "SI_RC=%ERRORLEVEL%"
>>"%STATUS%" echo smoke inference rc=%SI_RC%
if not "%SI_RC%"=="0" goto :smoke_failed
python -c "import json; r=json.load(open('models/ft/qwen3vl8b_lora_r16_smoke/train_report.json', encoding='utf-8')); print('      %%.1f s per segment, peak GPU memory %%s MB' %% (r['sec_per_sample'], r['peak_vram_mb'])); print('      Full training estimate: about %%s minutes' %% r.get('estimated_full_minutes'))"
echo.
choice /C YN /M "Start the full training now"
if errorlevel 2 goto :not_now

echo.
echo [5/7] Full training (the PC will not sleep; progress in logs\finetune_train.log) ...
:trained
python -m scripts.finetune train --log "logs\finetune_train.log"
set "TR_RC=%ERRORLEVEL%"
>>"%STATUS%" echo train rc=%TR_RC% %DATE% %TIME%
if not "%TR_RC%"=="0" goto :train_failed

echo.
echo [6/7] BNC dev: fine-tuned vs not fine-tuned, same prompt (about 10 minutes) ...
python -m scripts.run_vlm --experiment configs\experiment_ft.yaml --models configs\model.yaml configs\model_train.yaml --log "logs\run_vlm_experiment_ft.log"
set "DV_RC=%ERRORLEVEL%"
>>"%STATUS%" echo dev rc=%DV_RC% %DATE% %TIME%

echo [7/7] Scoring ...
python -m scripts.run_vlm --compare "configs/experiment*.yaml" --log "logs\prompt_versions.log"
python -m scripts.evaluate --log "logs\dev_evaluation.log"
echo.
echo ==================================================================
echo   Finished (dev rc=%DV_RC%).
echo   Dev scores: experiments\dev_evaluation\report.md  (rows qwen3vl_8b_ft / ft-generic)
echo   Training log and per-epoch dev check: models\ft\qwen3vl8b_lora_r16\train_report.json
echo   Tell Claude when it is done.
echo ==================================================================
if exist "experiments\dev_evaluation\report.md" start "" notepad "experiments\dev_evaluation\report.md"
pause
exit /b %DV_RC%

:not_now
echo   OK - double-click run_finetune.bat again when you want to train.
pause
exit /b 0

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_train.txt and tell Claude.
pause
exit /b 1

:pip_failed
echo.
echo   [STOP] Installing peft failed - see logs\setup_train_pip.txt and tell Claude.
pause
exit /b 1

:plan_failed
echo.
echo   [STOP] Could not list the training data - see logs\finetune_plan.log and tell Claude.
pause
exit /b 1

:smoke_failed
echo.
echo   [STOP] Smoke run failed - see logs\finetune_smoke.log / logs\finetune_smoke_infer.log and tell Claude.
pause
exit /b 1

:train_failed
echo.
echo   [STOP] Training failed - see logs\finetune_train.log and tell Claude.
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
