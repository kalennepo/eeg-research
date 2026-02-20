@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating venv and installing dependencies...
  python -m venv .venv
  call .venv\Scripts\pip.exe install -r test_eeg_code\requirements.txt
)

echo P01 Recording 1:
echo Task 1: Default...

.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream
echo Task 2: No reference...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --no-reference
echo Task 3: No quality check...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --no-quality-check
echo Task 4: Forced reference...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --force-reference




echo ========================================================
 

echo P01 Recording 2:
echo Task 1: Default...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task02
echo Task 2: No reference...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task02 --no-reference
echo Task 3: No quality check...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task02 --no-quality-check
echo Task 4: Forced reference...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task02 --force-reference


echo ========================================================


echo P01 Recording 3:
echo Task 1: Default...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task03
echo Task 2: No reference...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task03 --no-reference
echo Task 3: No quality check...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task03 --no-quality-check
echo Task 4: Forced reference...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task03 --force-reference


echo ========================================================


echo P01 Recording 4:
echo Task 1: Default...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task04
echo Task 2: No reference...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task04 --no-reference
echo Task 3: No quality check...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task04 --no-quality-check
echo Task 4: Forced reference...
.venv\Scripts\python.exe test_eeg_code\eeg_pipeline.py --stream --recording Task04 --force-reference

echo ========================================================



pause
