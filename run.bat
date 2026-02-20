@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating venv and installing dependencies...
  python -m venv .venv
  call .venv\Scripts\pip.exe install -r test_eeg_code\requirements.txt
)
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py
.venv\Scripts\python.exe test_eeg_code\eeg_to_time_series.py
pause
