@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
set "HF_HUB_OFFLINE=1"
set "REVIEW_PYTHON=%USERPROFILE%\anaconda3\envs\reassemble-vlm\python.exe"
if not exist "%REVIEW_PYTHON%" (
  echo Cannot find reassemble-vlm Python. Activate your environment and use the command in docs/evidence_review.md.
  pause
  exit /b 1
)
"%REVIEW_PYTHON%" -m pytest tests/test_evidence_review.py -q
if errorlevel 1 goto :failed
"%REVIEW_PYTHON%" -m scripts.run_vlm --experiment configs/experiment_evidence_review.yaml --models configs/model_evidence_review.yaml --log logs/evidence_review.log
if errorlevel 1 goto :failed
echo Results: experiments\dev_evidence_review_v1\summary.md
pause
exit /b 0
:failed
echo Failed. Inspect the error above and logs/evidence_review.log.
pause
exit /b 1
