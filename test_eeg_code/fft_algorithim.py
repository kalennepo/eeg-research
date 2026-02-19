import numpy as np
import pandas as pd
import time
from scipy.signal import welch, butter, filtfilt

# load data (need to change file path depending on location on computer)
df = pd.read_csv('/Users/ansh/Desktop/eeg-research/test_eeg_code/ALAS_Recording01_P01_Dyad01_Task01.csv')

for ch in ["MV1","MV2","MV3","MV4"]:
    df[ch] = pd.to_numeric(df[ch], errors="coerce")
df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
df = df.dropna(subset=["Time","MV1","MV2","MV3","MV4"]).reset_index(drop=True)

channels = ["MV1","MV2","MV3","MV4"]

# estimate sampling rate
time_vals = df["Time"].values
dt = np.diff(time_vals)
dt = dt[(dt > 0) & (~np.isnan(dt))]
fs = int(round(1/np.mean(dt)))

print("Sampling rate:", fs)

# band definitions
bands = {
    "delta": (0.5,4),
    "theta": (4,8),
    "alpha": (8,12),
    "beta": (12,30),
    "gamma": (30, min(40, fs/2 - 1))
}


def bandpass(sig):
    nyq = fs / 2
    low = 0.5
    high = min(45, nyq - 1)   # ensure valid

    b, a = butter(4, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)


# compute band powers
def compute_powers(sig):
    freqs, psd = welch(sig, fs=fs, nperseg=fs)
    out = {}
    for name,(low,high) in bands.items():
        idx = (freqs>=low)&(freqs<=high)
        out[name] = float(np.trapz(psd[idx], freqs[idx]))
    return out

# streaming parameters
window_seconds = 2
window_size = fs * window_seconds
step_size = fs          # update every 1 second

buffers = {ch: [] for ch in channels}

for i in range(len(df)):
    row = df.iloc[i]

    # append new sample
    for ch in channels:
        buffers[ch].append(float(row[ch]))
        if len(buffers[ch]) > window_size:
            buffers[ch].pop(0)

    # every 1 second, compute output
    if i % step_size == 0 and len(buffers["MV1"]) == window_size:
        totals = {"alpha": 0, "beta": 0}

        for ch in channels:
            sig = bandpass(np.array(buffers[ch]))
            powers = compute_powers(sig)
            totals["alpha"] += powers["alpha"]
            totals["beta"] += powers["beta"]

        # compute beta/alpha ratio
        beta_alpha_ratio = totals["beta"] / totals["alpha"] if totals["alpha"] != 0 else np.nan
        print(f"Beta/Alpha ratio: {beta_alpha_ratio:.2f}")

        time.sleep(1)  # simulate live feed delay