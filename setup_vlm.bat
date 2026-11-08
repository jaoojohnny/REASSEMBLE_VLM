@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 5 setup: GPU packages + model weights
REM  1 show GPU driver   2 install PyTorch (CUDA)   3 install transformers etc.
REM  4 check the GPU      5 list model sizes, then download after you confirm
REM  Model weights go to models\hf in this folder (set HF_HOME to change).
REM  To use another CUDA build:  set TORCH_INDEX=https://download.pytorch.org/whl/cu128
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM package 5 setup
if not defined TORCH_INDEX set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
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
echo [1/5] GPU driver ...
nvidia-smi >"logs\nvidia_smi.txt" 2>&1
if errorlevel 1 (
  echo       nvidia-smi not found - is the NVIDIA driver installed? See logs\nvidia_smi.txt
) else (
  findstr /C:"Driver Version" /C:"MiB" "logs\nvidia_smi.txt"
)

echo [2/5] Installing PyTorch with CUDA from %TORCH_INDEX%
echo       (about 3 GB download, several minutes) ...
python -m pip install --upgrade torch torchvision --index-url %TORCH_INDEX% >"logs\setup_vlm_torch.txt" 2>&1
if errorlevel 1 goto :pip_failed

echo [3/5] Installing transformers, accelerate, bitsandbytes ...
REM keep the CUDA build of torch: pin the installed version while installing the rest
python -c "import torch; print('torch==' + torch.__version__)" >"logs\torch_constraint.txt"
if errorlevel 1 goto :pip_failed
python -m pip install -r requirements-vlm.txt -c "logs\torch_constraint.txt" >"logs\setup_vlm_pip.txt" 2>&1
if errorlevel 1 goto :pip_failed

echo [4/5] Checking the GPU ...
python -m scripts.run_vlm --check --log "logs\vlm_check.log"
if errorlevel 1 goto :check_failed

echo.
echo [5/5] Model weights (from Hugging Face)
python -m scripts.download_models --plan --log "logs\download_plan.log"
if errorlevel 1 goto :plan_failed
echo.
echo       Needs this much free disk space on this drive, plus some spare.
echo       You can close the window during the download and run this file again to resume.
choice /C YN /M "Download these models now"
if errorlevel 2 goto :skip_download
python -m scripts.download_models --log "logs\download_models.log"
if errorlevel 1 goto :download_failed
echo.
echo ==================================================================
echo   Setup finished. Next: double-click run_vlm_dev.bat
echo ==================================================================
pause
exit /b 0

:skip_download
echo   Skipped the download. Run setup_vlm.bat again when you are ready.
pause
exit /b 0

:pip_failed
echo.
echo   [STOP] pip install failed - see logs\setup_vlm_torch.txt / logs\setup_vlm_pip.txt and tell Claude.
pause
exit /b 1

:check_failed
echo.
echo   [STOP] GPU check failed - see logs\vlm_check.log and logs\nvidia_smi.txt and tell Claude.
pause
exit /b 1

:plan_failed
echo.
echo   [STOP] Could not reach Hugging Face - see logs\download_plan.log and tell Claude.
pause
exit /b 1

:download_failed
echo.
echo   [STOP] Download failed or was interrupted - run setup_vlm.bat again to resume (see logs\download_models.log).
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
