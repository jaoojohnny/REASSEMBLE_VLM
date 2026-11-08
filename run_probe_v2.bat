@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Training phase, side track: better trained classifier (pre-registered 2026-10-14)
REM  Same training pool as before (other parts, never BNC test recordings).
REM  Candidates: 3 image encoders x last image +0.5 s / +1 s x 1/4/8 frames
REM              x 2 ways to combine frames (configs\probe_sweep_v2.yaml).
REM  The winner is picked by cross-validation on the training pool only;
REM  the 20 BNC dev segments are shown but not used to pick. Test stays locked.
REM  1 self-test  2 download 2 new encoders (asks first, about 2.7 GB)  3 run all candidates
REM  Result: experiments\probe_selection\selection.md   (about 20-40 minutes)
REM ==================================================================
title REASSEMBLE_VLM training phase - better trained classifier (dev only)
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
set "STATUS=%CD%\logs\run_probe_v2_status.txt"
>"%STATUS%" echo started %DATE% %TIME%

echo.
echo [1/3] Self-test (about 30 seconds) ...
python -m pytest -q >"logs\pytest_train.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_train.txt"
>>"%STATUS%" echo pytest rc=%PYTEST_RC%
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo.
echo [2/3] New image encoders (DINOv2-large, SigLIP 2) from Hugging Face
python -m scripts.download_models --models configs\model.yaml configs\model_train.yaml --use dinov2_large,siglip2_b384 --plan --log "logs\download_plan_train.log"
if errorlevel 1 goto :no_plan
choice /C YN /M "Download the encoders that are not downloaded yet"
if errorlevel 2 goto :after_download
python -m scripts.download_models --models configs\model.yaml configs\model_train.yaml --use dinov2_large,siglip2_b384 --log "logs\download_models_train.log"
>>"%STATUS%" echo download rc=%ERRORLEVEL%
goto :after_download
:no_plan
echo       Could not reach Hugging Face - continuing with the encoders already downloaded.
:after_download

echo.
echo [3/3] Running all candidates (features are cached; the PC will not sleep) ...
set "HF_HUB_OFFLINE=1"
python -m scripts.probe sweep --config configs\probe_sweep_v2.yaml --log "logs\probe_sweep_v2.log"
set "SW_RC=%ERRORLEVEL%"
>>"%STATUS%" echo sweep rc=%SW_RC% %DATE% %TIME%
echo.
echo ==================================================================
echo   Finished (rc=%SW_RC%).
echo   Candidates ranked by cross-validation: experiments\probe_selection\selection.md
echo   Chosen setting: configs\probe_v2_selected.yaml
echo   Log: logs\probe_sweep_v2.log   - tell Claude when it is done.
echo ==================================================================
if exist "experiments\probe_selection\selection.md" start "" notepad "experiments\probe_selection\selection.md"
pause
exit /b %SW_RC%

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_train.txt and tell Claude.
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
