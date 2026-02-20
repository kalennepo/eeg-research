"""
Combined EEG pipeline: band-power time series + Beta/Alpha ratio.
Uses the ALAS Muse recording: load → resample → filter → band powers → CSV + plots.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ---------------------------------------------------------------------------
# Config (shared by batch and streaming-style use)
# ---------------------------------------------------------------------------
FS = 250
BANDS = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 50)}
WINDOW_SEC = 2
STEP_SEC = 1  # 50% overlap when step = window/2


def get_data_path():
    """Portable path to the ALAS CSV (works from any CWD)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "TestEEGData", "Dyad01", "P01", "ALAS_Recording01_P01_Dyad01_Task01.csv")


def load_and_prepare(csv_path=None):
    """
    Load CSV, keep MV2/MV4 only, resample to uniform FS grid, bandpass 1–50 Hz.
    Returns: t_uniform, eeg1_filt, eeg2_filt, fs
    """
    csv_path = csv_path or get_data_path()
    df = pd.read_csv(csv_path)
    df = df[df["Time"] != "Time"].apply(pd.to_numeric)

    time = df["Time"].values - df["Time"].values[0]
    eeg1 = df["MV2"].values
    eeg2 = df["MV4"].values

    t_uniform = np.arange(0, time[-1], 1 / FS)
    eeg1 = np.interp(t_uniform, time, eeg1)
    eeg2 = np.interp(t_uniform, time, eeg2)

    b, a = signal.butter(4, [1 / (FS / 2), 50 / (FS / 2)], btype="band")
    eeg1_filt = signal.filtfilt(b, a, eeg1)
    eeg2_filt = signal.filtfilt(b, a, eeg2)

    return t_uniform, eeg1_filt, eeg2_filt, FS


def compute_band_power_timeseries(eeg_filt, fs=FS, bands=BANDS, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    """
    Sliding-window band power and Beta/Alpha ratio over time.
    Returns: DataFrame with time_sec, Delta, Theta, Alpha, Beta, Gamma, beta_alpha_ratio.
    """
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    nperseg = min(256, win)

    rows = []
    for i in range((len(eeg_filt) - win) // step):
        seg = eeg_filt[i * step : i * step + win]
        freqs, psd = signal.welch(seg, fs=fs, nperseg=nperseg)
        row = {"time_sec": (i * step + win / 2) / fs}
        band_power = {}
        for name, (lo, hi) in bands.items():
            p = np.mean(psd[(freqs >= lo) & (freqs <= hi)])
            band_power[name] = p
            row[name] = p
        # Beta/Alpha ratio (from fft_algorithim.py)
        alpha = band_power["Alpha"]
        beta = band_power["Beta"]
        row["beta_alpha_ratio"] = beta / alpha if alpha > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def run_batch(output_dir=None):
    """Full offline pipeline: load, filter, band power + ratio, save CSV and plots."""
    output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    t_uniform, eeg1_filt, eeg2_filt, fs = load_and_prepare()

    # Band power time series from channel 1 (MV2); add channel 2 if you want
    band_df = compute_band_power_timeseries(eeg1_filt, fs=fs)
    csv_path = os.path.join(output_dir, "band_power_timeseries.csv")
    band_df.to_csv(csv_path, index=False)

    # Filtered EEG plot (5 s window)
    fig, ax = plt.subplots(figsize=(12, 4))
    mask = (t_uniform >= 10) & (t_uniform <= 15)
    ax.plot(t_uniform[mask], eeg1_filt[mask], label="EEG 1 (MV2)")
    ax.plot(t_uniform[mask], eeg2_filt[mask], label="EEG 2 (MV4)", alpha=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("uV")
    ax.set_title("Filtered EEG (1–50 Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "filtered_eeg.png"), dpi=150)
    plt.close(fig)

    # Band power + Beta/Alpha ratio plots
    fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True)
    for ax, name in zip(axes, list(BANDS) + ["Beta/Alpha ratio"]):
        if name == "Beta/Alpha ratio":
            axes[-1].plot(band_df["time_sec"], band_df["beta_alpha_ratio"], color="C5")
            axes[-1].set_ylabel("Beta/Alpha")
            axes[-1].axhline(y=1, color="gray", linestyle="--", alpha=0.7)
        else:
            ax.plot(band_df["time_sec"], band_df[name])
            ax.set_ylabel(name)
    axes[-1].set_xlabel("Time (sec)")
    axes[0].set_title("Band Power and Beta/Alpha Ratio Over Time")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "band_power.png"), dpi=150)
    plt.close(fig)

    # Summary stats for ratio (like streaming output, but from full run)
    ratio = band_df["beta_alpha_ratio"].dropna()
    print(f"Saved: {csv_path}, filtered_eeg.png, band_power.png")
    print(f"Beta/Alpha ratio: mean={ratio.mean():.2f}, std={ratio.std():.2f}, last={band_df['beta_alpha_ratio'].iloc[-1]:.2f}")
    return band_df


def run_streaming_style(csv_path=None, out_dir=None):
    """
    Simulate streaming: step through the same pipeline in 1 s steps and print
    Beta/Alpha ratio each second (no real I/O, uses precomputed data).
    """
    t_uniform, eeg1_filt, eeg2_filt, fs = load_and_prepare(csv_path)
    band_df = compute_band_power_timeseries(eeg1_filt, fs=fs)

    print("Streaming-style Beta/Alpha ratio (every 1 s):")
    for _, row in band_df.iterrows():
        r = row["beta_alpha_ratio"]
        print(f"  t={row['time_sec']:.1f}s  Beta/Alpha = {r:.2f}")

    if out_dir:
        band_df.to_csv(os.path.join(out_dir, "band_power_timeseries.csv"), index=False)
        print(f"Also saved band_power_timeseries.csv to {out_dir}")
    return band_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG pipeline: band power time series + Beta/Alpha ratio")
    parser.add_argument("--stream", action="store_true", help="Print Beta/Alpha ratio every 1 s (streaming-style)")
    parser.add_argument("--out", default=None, help="Output directory for CSV and plots (default: script dir)")
    args = parser.parse_args()

    if args.stream:
        run_streaming_style(out_dir=args.out)
    else:
        run_batch(output_dir=args.out)
