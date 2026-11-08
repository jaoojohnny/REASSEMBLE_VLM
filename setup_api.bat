@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Package 8: commercial models (Claude, GPT) through their APIs.
REM  1 install the official SDKs (anthropic, openai) - torch/transformers stay as they are
REM  2 create .env from .env.example and open it in Notepad: paste your API keys
REM    (ANTHROPIC_API_KEY, OPENAI_API_KEY), save, close Notepad
REM  3 send one very short test message to each model (costs far less than 1 cent)
REM  The keys stay in .env on this PC only (not in Git, never printed).
REM  ChatGPT Plus / Claude Pro subscriptions do NOT include API access:
REM  create keys at platform.openai.com and platform.claude.com (with billing).
REM ==================================================================
title REASSEMBLE_VLM package 8 - API setup
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
echo [1/3] Installing the API SDKs (small download) ...
python -c "import torch, transformers; print('torch==' + torch.__version__); print('transformers==' + transformers.__version__)" >"logs\api_constraint.txt"
if errorlevel 1 goto :pip_failed
python -m pip install -r requirements-api.txt -c "logs\api_constraint.txt" >"logs\setup_api_pip.txt" 2>&1
if errorlevel 1 goto :pip_failed
python -c "import anthropic, openai; print('      anthropic ' + anthropic.__version__ + ', openai ' + openai.__version__)"
if errorlevel 1 goto :pip_failed

echo.
echo [2/3] API keys ...
if not exist ".env" copy ".env.example" ".env" >nul
echo       Notepad opens .env - paste each key after the = sign, save (Ctrl+S) and close Notepad.
echo       Leave a line empty if you do not use that company.
start /wait "" notepad ".env"

echo.
echo [3/3] Test message to each model ...
python -m scripts.check_api --log "logs\check_api.log"
set "RC=%ERRORLEVEL%"
echo.
echo ==================================================================
if "%RC%"=="0" echo   Both APIs work. Next: double-click run_p8_dev.bat
if not "%RC%"=="0" echo   Some API did not work - see the lines above / logs\check_api.log and tell Claude.
echo ==================================================================
pause
exit /b %RC%

:pip_failed
echo.
echo   [STOP] Could not install the SDKs - see logs\setup_api_pip.txt and tell Claude.
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
