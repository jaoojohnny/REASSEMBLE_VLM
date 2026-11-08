@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Optional: build the Docker image on this PC and run the unit tests
REM  inside it. Needs Docker Desktop running. The first build downloads
REM  about 8-10 GB - Python, CUDA torch and the other packages.
REM  Tests run on the CPU; at the end it checks whether Docker sees the GPU.
REM  Data, model weights, results and .env are NOT put into the image.
REM  1 build  2 unit tests in the container  3 GPU check
REM ==================================================================
title REASSEMBLE_VLM - Docker build and test
cd /d "%~dp0"
where docker >nul 2>nul
if errorlevel 1 goto :no_docker
docker info >nul 2>nul
if errorlevel 1 goto :not_running

echo.
echo [1/3] Building image reassemble-vlm:latest - first time 20 to 60 minutes ...
docker build -t reassemble-vlm:latest .
if errorlevel 1 goto :build_failed

echo [2/3] Unit tests inside the container ...
docker run --rm reassemble-vlm:latest python -m pytest -q
if errorlevel 1 goto :tests_failed

echo [3/3] GPU check ...
docker run --rm --gpus all reassemble-vlm:latest python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.__version__)"
if errorlevel 1 echo        Docker cannot see the GPU - the tests still passed. See docs\docker.md.
echo.
echo ==================================================================
echo   Done. Image: reassemble-vlm:latest. Usage: docs\docker.md
echo   Tell Claude when done.
echo ==================================================================
pause
exit /b 0

:build_failed
echo.
echo   [STOP] docker build failed - see the messages above and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Unit tests failed inside the container - tell Claude.
pause
exit /b 1

:not_running
echo.
echo   Docker is installed but not running. Start Docker Desktop, wait until it
echo   says it is running, then double-click this file again.
pause
exit /b 1

:no_docker
echo.
echo   Docker is not installed. Install Docker Desktop, then double-click again.
pause
exit /b 1
