"""
EEG Time Series Starter Code
Uses the ALAS Muse recording to produce band-power time series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ---- Load data ----
df = pd.read_csv("ALAS_Recording01_P01_Dyad01_Task01.csv")
df = df[df["Time"] != "Time"].apply(pd.to_numeric)  # file has duplicate header rows, remove them

# Columns: Time (unix timestamp), MV1-MV4
# MV1/MV3 are slowly drifting reference values — ignore these
# MV2/MV4 are the actual EEG signals (microvolts)
time = df["Time"].values - df["Time"].values[0]  # seconds starting at 0
eeg1 = df["MV2"].values
eeg2 = df["MV4"].values

# ---- Resample to a uniform grid ----
# The Muse sends data in irregular bursts (~250 samples/sec average).
# Filtering requires evenly spaced samples, so we interpolate to a clean 250 Hz grid.
fs = 250
t_uniform = np.arange(0, time[-1], 1/fs)
eeg1 = np.interp(t_uniform, time, eeg1)
eeg2 = np.interp(t_uniform, time, eeg2)

# ---- Bandpass filter (1–50 Hz) ----
# Removes slow electrode drift (below 1 Hz) and high-frequency noise (above 50 Hz),
# keeping only the brain-wave frequencies we care about.
b, a = signal.butter(4, [1/(fs/2), 50/(fs/2)], btype="band")
eeg1_filt = signal.filtfilt(b, a, eeg1)
eeg2_filt = signal.filtfilt(b, a, eeg2)

# ---- Plot a 5-second window of the filtered signal ----
plt.figure(figsize=(12, 4))
mask = (t_uniform >= 10) & (t_uniform <= 15)  # skip first 10 sec (headband settling artifact)
plt.plot(t_uniform[mask], eeg1_filt[mask], label="EEG 1")
plt.plot(t_uniform[mask], eeg2_filt[mask], label="EEG 2", alpha=0.8)
plt.xlabel("Time (sec)")
plt.ylabel("uV")
plt.title("Filtered EEG (1-50 Hz)")
plt.legend()
plt.tight_layout()
plt.savefig("filtered_eeg.png", dpi=150)
plt.close()

# ---- Compute band power over time ----
# For each 2-second window, use Welch's method to estimate the power spectrum,
# then average the power within each standard EEG frequency band.
bands = {"Delta": (1,4), "Theta": (4,8), "Alpha": (8,13), "Beta": (13,30), "Gamma": (30,50)}
win = 2 * fs     # 2-second window (500 samples)
step = win // 2  # slide forward 1 second each time (50% overlap)

rows = []
for i in range((len(eeg1_filt) - win) // step):
    seg = eeg1_filt[i*step : i*step+win]
    freqs, psd = signal.welch(seg, fs=fs, nperseg=256)
    row = {"time_sec": (i*step + win/2) / fs}  # center of the window
    for name, (lo, hi) in bands.items():
        row[name] = np.mean(psd[(freqs >= lo) & (freqs <= hi)])
    rows.append(row)

band_df = pd.DataFrame(rows)
band_df.to_csv("band_power_timeseries.csv", index=False)

# ---- Plot band power ----
fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
for ax, name in zip(axes, bands):
    ax.plot(band_df["time_sec"], band_df[name])
    ax.set_ylabel(name)
axes[-1].set_xlabel("Time (sec)")
axes[0].set_title("Band Power Over Time")
plt.tight_layout()
plt.savefig("band_power.png", dpi=150)
plt.close()

print("Saved: filtered_eeg.png, band_power.png, band_power_timeseries.csv")