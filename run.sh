#!/bin/sh
cd "$(dirname "$0")"
if [ ! -f .venv/bin/python ]; then
  echo "Creating venv and installing dependencies..."
  python3 -m venv .venv
  .venv/bin/pip install -r test_eeg_code/requirements.txt
fi
.venv/bin/python test_eeg_code/eeg_to_time_series.py
