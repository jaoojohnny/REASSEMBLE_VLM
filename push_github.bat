@echo off
setlocal EnableExtensions
REM ==================================================================
REM  Upload this project to YOUR GitHub as a PRIVATE repo: REASSEMBLE_VLM
REM  Uses Git on this PC and your own GitHub login - a browser window may
REM  open the first time. Claude never sees your password or token.
REM  NOT uploaded (.gitignore): .env with API keys, data\raw, data\derived,
REM  models, experiments, logs, docs\*.pptx, docs\*.pdf
REM  Before committing, scripts\repo_check.py checks the files again and
REM  stops if it finds an API key, raw data, model weights or a file over 50 MB.
REM  Run it again later to upload only what changed.
REM  1 self-test  2 git repo  3 check files  4 commit  5 upload
REM ==================================================================
title REASSEMBLE_VLM - upload to GitHub, private
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "REPO_NAME=REASSEMBLE_VLM"
set "ENV_NAME=reassemble-vlm"

where git >nul 2>nul
if errorlevel 1 goto :no_git
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
echo [1/5] Self-test ...
python -m pytest -q tests\test_repo_upload.py >"logs\pytest_upload.txt" 2>&1
set "PYTEST_RC=%ERRORLEVEL%"
findstr /R /C:" passed" /C:" failed" /C:" error" "logs\pytest_upload.txt"
if not "%PYTEST_RC%"=="0" goto :tests_failed

echo [2/5] Git repository ...
if exist ".git" goto :have_repo
git init -q
if errorlevel 1 goto :failed
:have_repo
git check-ignore -q .env
if errorlevel 1 goto :env_not_ignored
set "GEMAIL="
for /f "delims=" %%A in ('git config user.email 2^>nul') do set "GEMAIL=%%A"
if defined GEMAIL goto :have_identity
echo        Git needs a name and e-mail for the commit - saved only for this folder.
set "GNAME="
set /p "GNAME=        Your name: "
set /p "GEMAIL=        Your GitHub e-mail: "
if not defined GEMAIL goto :no_identity
git config user.name "%GNAME%"
git config user.email "%GEMAIL%"
:have_identity

echo [3/5] Checking the files that will be uploaded ...
git add -A
if errorlevel 1 goto :failed
python -m scripts.repo_check --write-msg "logs\commit_msg.txt"
if errorlevel 1 goto :check_failed

echo [4/5] Commit ...
git diff --cached --quiet
if not errorlevel 1 goto :nothing_new
git commit -q -F "logs\commit_msg.txt"
if errorlevel 1 goto :failed
:nothing_new
git branch -M main

echo [5/5] Upload ...
git remote get-url origin >nul 2>nul
if not errorlevel 1 goto :push
where gh >nul 2>nul
if errorlevel 1 goto :manual_remote
gh auth status >nul 2>nul
if errorlevel 1 goto :manual_remote
gh repo create %REPO_NAME% --private --source . --remote origin --push
if errorlevel 1 goto :failed
goto :done

:manual_remote
echo.
echo   Create the empty PRIVATE repo once on github.com - the page opens now:
echo     1. Repository name: %REPO_NAME%
echo     2. Choose Private
echo     3. Do NOT add a README, .gitignore or license
echo     4. Click Create repository and copy the HTTPS address,
echo        for example https://github.com/your-name/%REPO_NAME%.git
start "" "https://github.com/new"
echo.
set "REPO_URL="
set /p "REPO_URL=  Paste the address here and press Enter: "
if not defined REPO_URL goto :no_url
git remote add origin "%REPO_URL%"
if errorlevel 1 goto :failed

:push
git push -u origin main
if errorlevel 1 goto :push_failed

:done
set "ORIGIN="
for /f "delims=" %%U in ('git remote get-url origin') do set "ORIGIN=%%U"
echo.
echo ==================================================================
echo   Uploaded to %ORIGIN%
echo   Private repo. Run this file again later to upload new changes.
echo   Tell Claude when done.
echo ==================================================================
pause
exit /b 0

:check_failed
git reset -q
echo.
echo   [STOP] Some files must not be uploaded - see the list above.
echo   Nothing was committed or uploaded. Tell Claude.
pause
exit /b 3

:env_not_ignored
echo.
echo   [STOP] .env is not listed in .gitignore - nothing was uploaded. Tell Claude.
pause
exit /b 3

:push_failed
echo.
echo   [STOP] Upload failed. Check that the repo exists on GitHub, is empty or yours,
echo   and that you logged in with the right account. The commit is kept on this PC;
echo   double-click again to retry. Tell Claude if it keeps failing.
pause
exit /b 2

:no_url
echo.
echo   [STOP] No address entered - the commit is kept on this PC. Double-click again.
pause
exit /b 2

:no_identity
echo.
echo   [STOP] No e-mail entered - nothing was committed. Double-click again.
pause
exit /b 2

:failed
echo.
echo   [STOP] A git step failed - see the messages above and tell Claude.
pause
exit /b 2

:tests_failed
echo.
echo   [STOP] Self-test failed - see logs\pytest_upload.txt and tell Claude.
pause
exit /b 1

:no_git
echo.
echo   Git is not installed on this PC. Install Git for Windows from
echo   https://git-scm.com/download/win with the default options - it includes
echo   the GitHub login helper - then double-click this file again.
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
