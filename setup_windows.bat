@echo off
setlocal EnableExtensions
REM ==================================================================
REM  REASSEMBLE_VLM one-click setup for Windows (Anaconda / Miniconda)
REM  1 find conda  2 create env  3 activate  4 pip install
REM  5 env info (GPU)  6 tests + synthetic smoke test
REM  7 sample download (asks y/N) + HDF5 inspection
REM  Safe to run again: existing env and downloaded files are reused.
REM ==================================================================
cd /d "%~dp0"
title REASSEMBLE_VLM setup
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "ENV_NAME=reassemble-vlm"
set "PY_VER=3.11"
if not exist "logs" mkdir "logs"
set "LOG=%CD%\logs\setup_windows.log"
set "STATUS=%CD%\logs\setup_status.txt"
>"%LOG%" echo ==== REASSEMBLE_VLM setup %DATE% %TIME% ====
>"%STATUS%" echo started %DATE% %TIME%

echo.
echo ==================================================================
echo   REASSEMBLE_VLM setup   (details: logs\setup_windows.log)
echo ==================================================================

echo [1/7] Looking for Anaconda / Miniconda ...
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
  if not defined CONDA_BAT if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%C"
if not defined CONDA_BAT goto :no_conda
echo       found: %CONDA_BAT%
>>"%LOG%" echo conda: %CONDA_BAT%
call "%CONDA_BAT%" --version >>"%LOG%" 2>&1
>>"%STATUS%" echo STEP1 find_conda OK

echo [2/7] Conda env "%ENV_NAME%" (Python %PY_VER%, conda-forge) ...
call "%CONDA_BAT%" env list >"logs\conda_envs.txt" 2>&1
findstr /B /C:"%ENV_NAME% " "logs\conda_envs.txt" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :env_ready
echo       creating env, first time takes a few minutes ...
call "%CONDA_BAT%" create -y -n %ENV_NAME% -c conda-forge --override-channels python=%PY_VER% pip >>"%LOG%" 2>&1
if errorlevel 1 goto :fail_env
:env_ready
>>"%STATUS%" echo STEP2 conda_env OK

echo [3/7] Activating env ...
call "%CONDA_BAT%" activate %ENV_NAME%
python -c "import os,sys; sys.exit(0 if os.environ.get('CONDA_DEFAULT_ENV')=='%ENV_NAME%' else 1)"
if errorlevel 1 goto :fail_activate
python -c "import sys; print(sys.executable); print(sys.version)" >>"%LOG%" 2>&1
>>"%STATUS%" echo STEP3 activate OK

echo [4/7] Installing Python packages (pip, a few minutes the first time) ...
python -m pip install --upgrade pip >>"%LOG%" 2>&1
python -m pip install -r requirements.txt >>"%LOG%" 2>&1
if errorlevel 1 goto :fail_pip
python -m pip install -e . >>"%LOG%" 2>&1
if errorlevel 1 goto :fail_pip
>>"%STATUS%" echo STEP4 pip_install OK

echo [5/7] Recording environment info (Python, packages, GPU) ...
python -m scripts.envinfo --out logs >"logs\envinfo_console.txt" 2>&1
if exist "logs\envinfo.txt" type "logs\envinfo.txt"
>>"%STATUS%" echo STEP5 envinfo done

echo [6/7] Running tests ...
python -m pytest -q >"logs\pytest.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest.txt"
>>"%STATUS%" echo STEP6 pytest rc=%PYTEST_RC%
if not "%PYTEST_RC%"=="0" echo       [WARN] some tests failed - see logs\pytest.txt
echo       synthetic smoke test (real h5py) ...
python -m scripts.make_synthetic --out data\derived\synthetic >"logs\synthetic.txt" 2>&1
python -m scripts.inspect_h5 data\derived\synthetic\synthetic_normal.h5 --out docs\examples --config none.yaml --log "logs\inspect_synthetic.log" >nul 2>&1
set "SMOKE_RC=%ERRORLEVEL%"
>>"%STATUS%" echo STEP6b synthetic_smoke rc=%SMOKE_RC%

echo [7/7] REASSEMBLE samples + HDF5 inspection
if exist "data\raw\*.h5" goto :inspect
echo.
echo       data\raw has no .h5 yet. Next: download 1-3 REASSEMBLE files only
echo       (not the whole 54.8 GB zip). The file list and sizes are shown first;
echo       type y and press Enter to start, or just Enter to skip.
echo.
python -m scripts.fetch_samples --log "logs\fetch_samples.log"
set "FETCH_RC=%ERRORLEVEL%"
>>"%STATUS%" echo STEP7a fetch rc=%FETCH_RC%

:inspect
if not exist "data\raw\*.h5" goto :no_samples
echo.
echo       inspecting data\raw ...
python -m scripts.inspect_h5 data\raw --log "logs\inspect.log"
set "INSPECT_RC=%ERRORLEVEL%"
>>"%STATUS%" echo STEP7b inspect rc=%INSPECT_RC%
goto :done

:no_samples
echo       no samples in data\raw - inspection skipped.
>>"%STATUS%" echo STEP7b inspect SKIPPED

:done
>>"%STATUS%" echo finished %DATE% %TIME%
echo.
echo ==================================================================
echo   Finished.  Results: data\index\inspect\    Logs: logs\
echo   You can close this window and tell Claude it is done.
echo ==================================================================
pause
exit /b 0

:no_conda
echo       [FAIL] conda.bat not found. Please install Anaconda or Miniconda.
>>"%STATUS%" echo STEP1 find_conda FAIL
goto :fail
:fail_env
echo       [FAIL] conda create failed. See logs\setup_windows.log
>>"%STATUS%" echo STEP2 conda_env FAIL
goto :fail
:fail_activate
echo       [FAIL] could not activate env %ENV_NAME%. See logs\setup_windows.log
>>"%STATUS%" echo STEP3 activate FAIL
goto :fail
:fail_pip
echo       [FAIL] pip install failed (network?). See logs\setup_windows.log
>>"%STATUS%" echo STEP4 pip_install FAIL
goto :fail

:fail
>>"%STATUS%" echo stopped %DATE% %TIME%
echo.
echo   Setup stopped. Please tell Claude - the log is logs\setup_windows.log
pause
exit /b 1
