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

# Reference quality thresholds (for auto-detection)
REF_VARIANCE_THRESHOLD = 1000.0  # uV^2 - if reference variance exceeds this, skip re-referencing
REF_AMPLITUDE_THRESHOLD = 200.0  # uV - if reference peak-to-peak exceeds this, skip re-referencing
REF_SNR_THRESHOLD = 0.1  # ratio - if ref_variance / measurement_variance > this, skip

# Pipeline states (for output naming and reporting)
PIPELINE_STATE_DEFAULT = "default"           # auto, quality check on
PIPELINE_STATE_NO_REFERENCE = "no_reference"
PIPELINE_STATE_FORCED_REFERENCE = "forced_reference"
PIPELINE_STATE_NO_QUALITY_CHECK = "no_quality_check"

PIPELINE_STATE_LABELS = {
    PIPELINE_STATE_DEFAULT: "default (auto, quality check on)",
    PIPELINE_STATE_NO_REFERENCE: "no reference (raw MV2/MV4)",
    PIPELINE_STATE_FORCED_REFERENCE: "forced reference (MV2-MV1, MV4-MV3)",
    PIPELINE_STATE_NO_QUALITY_CHECK: "no quality check (auto ref, no check)",
}


def get_project_root():
    """Project root (parent of test_eeg_code)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_p01_dir():
    """Path to P01 data folder (TestEEGData/Dyad01/P01)."""
    return os.path.join(get_project_root(), "TestEEGData", "Dyad01", "P01")


def get_p01_results_base():
    """Path to P01Results folder (TestEEGData/Dyad01/P01Results)."""
    return os.path.join(get_project_root(), "TestEEGData", "Dyad01", "P01Results")


def get_recording_id_from_csv_path(csv_path):
    """
    Extract recording id from CSV filename for use as folder name.
    E.g. ALAS_Recording01_P01_Dyad01_Task01.csv -> Task01
    """
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    # Assume last token is TaskNN
    parts = basename.split("_")
    return parts[-1] if parts else "unknown"


def get_output_dir_for_csv(csv_path):
    """
    Output directory for a given P01 recording CSV: P01Results/<TaskNN>/.
    Creates base P01Results and the recording subfolder.
    """
    base = get_p01_results_base()
    recording_id = get_recording_id_from_csv_path(csv_path)
    out = os.path.join(base, recording_id)
    os.makedirs(out, exist_ok=True)
    return out


def get_data_path(recording_id=None):
    """
    Portable path to an ALAS CSV in P01 (works from any CWD).
    recording_id: e.g. 'Task01' (default), 'Task02', 'Task03', 'Task04'.
    """
    p01 = get_p01_dir()
    if recording_id is None:
        recording_id = "Task01"
    # Filename pattern: ALAS_Recording01_P01_Dyad01_<TaskNN>.csv
    return os.path.join(p01, f"ALAS_Recording01_P01_Dyad01_{recording_id}.csv")


def check_reference_quality(mv_ref, mv_meas, variance_threshold=REF_VARIANCE_THRESHOLD,
                            amplitude_threshold=REF_AMPLITUDE_THRESHOLD, snr_threshold=REF_SNR_THRESHOLD):
    """
    Check if reference channels (MV1/MV3) are suitable for re-referencing.
    
    Returns: (is_good, reason)
        is_good: True if reference is clean enough to use
        reason: string explaining why it passed/failed
    """
    ref_var = np.var(mv_ref)
    ref_pp = np.ptp(mv_ref)  # peak-to-peak amplitude
    meas_var = np.var(mv_meas)
    
    # Check variance threshold
    if ref_var > variance_threshold:
        return False, f"reference variance too high ({ref_var:.1f} > {variance_threshold:.1f} uV²)"
    
    # Check amplitude threshold
    if ref_pp > amplitude_threshold:
        return False, f"reference amplitude too high ({ref_pp:.1f} > {amplitude_threshold:.1f} uV)"
    
    # Check SNR: if reference is much noisier than measurement, skip
    if meas_var > 0 and ref_var / meas_var > snr_threshold:
        return False, f"reference too noisy relative to measurement (SNR ratio {ref_var/meas_var:.3f} > {snr_threshold:.3f})"
    
    return True, f"reference quality OK (var={ref_var:.1f} uV², pp={ref_pp:.1f} uV)"


def load_and_prepare(csv_path=None, referenced=True, auto_quality_check=True, verbose=True):
    """
    Load CSV, resample to uniform FS grid, optionally re-reference, then bandpass 1–50 Hz.

    Re-referencing: MV1 and MV3 are reference channels. If referenced=True we form
    differential signals (measurement minus reference) to remove common drift/noise:
      EEG1 = MV2 - MV1,  EEG2 = MV4 - MV3
    If referenced=False, we use raw MV2 and MV4 only.

    Args:
        csv_path: Path to CSV file (default: auto-detect)
        referenced: True=always re-reference, False=never, 'auto'=check quality first
        auto_quality_check: If True and referenced='auto', check reference quality before re-referencing
        verbose: Print quality check results

    Returns: t_uniform, eeg1_filt, eeg2_filt, fs, (actual_referenced, quality_info)
        actual_referenced: Whether re-referencing was actually applied
        quality_info: Dict with quality check results for both channels
    """
    csv_path = csv_path or get_data_path()
    df = pd.read_csv(csv_path)
    df = df[df["Time"] != "Time"].apply(pd.to_numeric)

    time = df["Time"].values - df["Time"].values[0]
    # Load all four channels so we can re-reference
    mv1 = df["MV1"].values
    mv2 = df["MV2"].values
    mv3 = df["MV3"].values
    mv4 = df["MV4"].values

    t_uniform = np.arange(0, time[-1], 1 / FS)
    mv1 = np.interp(t_uniform, time, mv1)
    mv2 = np.interp(t_uniform, time, mv2)
    mv3 = np.interp(t_uniform, time, mv3)
    mv4 = np.interp(t_uniform, time, mv4)

    # Determine if we should re-reference
    actual_referenced = False
    quality_info = {}
    
    if referenced == 'auto':
        # Auto mode: check quality if enabled, otherwise default to re-referencing
        if auto_quality_check:
            # Check quality of both reference pairs
            ref1_good, ref1_reason = check_reference_quality(mv1, mv2)
            ref2_good, ref2_reason = check_reference_quality(mv3, mv4)
            
            quality_info = {
                'MV1': {'good': ref1_good, 'reason': ref1_reason},
                'MV3': {'good': ref2_good, 'reason': ref2_reason}
            }
            
            # Only re-reference if both references are good
            if ref1_good and ref2_good:
                actual_referenced = True
                if verbose:
                    print(f"✓ Re-referencing enabled: {ref1_reason}, {ref2_reason}")
            else:
                actual_referenced = False
                if verbose:
                    print(f"✗ Re-referencing skipped:")
                    if not ref1_good:
                        print(f"  MV1: {ref1_reason}")
                    if not ref2_good:
                        print(f"  MV3: {ref2_reason}")
        else:
            # Auto mode but quality check disabled: default to re-referencing
            actual_referenced = True
            if verbose:
                print("✓ Re-referencing enabled (auto mode, quality check disabled)")
    elif referenced:
        # Forced re-referencing (no quality check)
        actual_referenced = True
        if verbose:
            print("✓ Re-referencing enabled (forced)")
    else:
        # Explicitly disabled
        actual_referenced = False
        if verbose:
            print("✗ Re-referencing disabled")

    # Apply re-referencing if enabled
    if actual_referenced:
        eeg1 = mv2 - mv1  # measurement minus reference
        eeg2 = mv4 - mv3
    else:
        eeg1 = mv2
        eeg2 = mv4

    b, a = signal.butter(4, [1 / (FS / 2), 50 / (FS / 2)], btype="band")
    eeg1_filt = signal.filtfilt(b, a, eeg1)
    eeg2_filt = signal.filtfilt(b, a, eeg2)

    return t_uniform, eeg1_filt, eeg2_filt, FS, actual_referenced, quality_info


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


def run_batch(output_dir=None, csv_path=None, referenced=True, auto_quality_check=True, pipeline_state=None):
    """Full offline pipeline: load, filter, band power + ratio, save CSV and plots."""
    csv_path = csv_path or get_data_path()
    output_dir = output_dir or get_output_dir_for_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    pipeline_state = pipeline_state or PIPELINE_STATE_DEFAULT
    suffix = f"_{pipeline_state}"
    print(f"Pipeline state: {PIPELINE_STATE_LABELS.get(pipeline_state, pipeline_state)}")
    print(f"Input: {csv_path}")
    print(f"Output dir: {output_dir}")

    t_uniform, eeg1_filt, eeg2_filt, fs, actual_referenced, quality_info = load_and_prepare(
        csv_path=csv_path, referenced=referenced, auto_quality_check=auto_quality_check
    )

    # Band power time series from channel 1 (MV2); add channel 2 if you want
    band_df = compute_band_power_timeseries(eeg1_filt, fs=fs)
    csv_name = f"band_power_timeseries{suffix}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    # Write pipeline state as first comment line so output is self-describing
    with open(csv_path, "w") as f:
        f.write(f"# pipeline_state={pipeline_state}\n")
    band_df.to_csv(csv_path, mode="a", index=False)

    # Filtered EEG plot (5 s window)
    fig, ax = plt.subplots(figsize=(12, 4))
    mask = (t_uniform >= 10) & (t_uniform <= 15)
    ax.plot(t_uniform[mask], eeg1_filt[mask], label="EEG 1 (MV2)")
    ax.plot(t_uniform[mask], eeg2_filt[mask], label="EEG 2 (MV4)", alpha=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("uV")
    ax.set_title(f"Filtered EEG (1–50 Hz) — {PIPELINE_STATE_LABELS.get(pipeline_state, pipeline_state)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"filtered_eeg{suffix}.png"), dpi=150)
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
    axes[0].set_title(f"Band Power and Beta/Alpha Ratio — {PIPELINE_STATE_LABELS.get(pipeline_state, pipeline_state)}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"band_power{suffix}.png"), dpi=150)
    plt.close(fig)

    # Summary stats for ratio (like streaming output, but from full run)
    ratio = band_df["beta_alpha_ratio"].dropna()
    print(f"Saved: {csv_name}, filtered_eeg{suffix}.png, band_power{suffix}.png")
    print(f"Beta/Alpha ratio: mean={ratio.mean():.2f}, std={ratio.std():.2f}, last={band_df['beta_alpha_ratio'].iloc[-1]:.2f}")
    return band_df


def run_streaming_style(csv_path=None, out_dir=None, referenced=True, auto_quality_check=True, pipeline_state=None):
    """
    Simulate streaming: step through the same pipeline in 1 s steps and print
    Beta/Alpha ratio each second (no real I/O, uses precomputed data).
    """
    csv_path = csv_path or get_data_path()
    out_dir = out_dir or get_output_dir_for_csv(csv_path)

    pipeline_state = pipeline_state or PIPELINE_STATE_DEFAULT
    print(f"Pipeline state: {PIPELINE_STATE_LABELS.get(pipeline_state, pipeline_state)}")
    print(f"Input: {csv_path}")
    print(f"Output dir: {out_dir}")

    t_uniform, eeg1_filt, eeg2_filt, fs, actual_referenced, quality_info = load_and_prepare(
        csv_path=csv_path, referenced=referenced, auto_quality_check=auto_quality_check
    )
    band_df = compute_band_power_timeseries(eeg1_filt, fs=fs)

    print("Streaming-style Beta/Alpha ratio (every 1 s):")
    for _, row in band_df.iterrows():
        r = row["beta_alpha_ratio"]
        print(f"  t={row['time_sec']:.1f}s  Beta/Alpha = {r:.2f}")

    suffix = f"_{pipeline_state}"
    path = os.path.join(out_dir, f"band_power_timeseries{suffix}.csv")
    with open(path, "w") as f:
        f.write(f"# pipeline_state={pipeline_state}\n")
    band_df.to_csv(path, mode="a", index=False)
    print(f"Saved band_power_timeseries{suffix}.csv to {out_dir}")
    return band_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG pipeline: band power time series + Beta/Alpha ratio")
    parser.add_argument("--stream", action="store_true", help="Print Beta/Alpha ratio every 1 s (streaming-style)")
    parser.add_argument("--recording", default=None, help="P01 recording id: Task01, Task02, Task03, or Task04 (default: Task01)")
    parser.add_argument("--input", default=None, help="Path to input CSV (overrides --recording)")
    parser.add_argument("--out", default=None, help="Output directory (default: P01Results/<TaskNN>/)")
    parser.add_argument("--no-reference", action="store_true", help="Use raw MV2/MV4 only; do not re-reference with MV1/MV3")
    parser.add_argument("--force-reference", "--forced-reference", dest="force_reference", action="store_true", help="Always re-reference (ignore quality checks)")
    parser.add_argument("--no-quality-check", action="store_true", help="Disable automatic quality checking (use with --force-reference)")
    args = parser.parse_args()

    csv_path = None
    if args.input:
        csv_path = os.path.abspath(args.input)
    elif args.recording:
        csv_path = get_data_path(args.recording)

    # Determine referencing mode and pipeline state for output naming
    if args.no_reference:
        referenced = False
        auto_quality_check = False
        pipeline_state = PIPELINE_STATE_NO_REFERENCE
    elif args.force_reference:
        referenced = True
        auto_quality_check = False
        pipeline_state = PIPELINE_STATE_FORCED_REFERENCE
    elif args.no_quality_check:
        referenced = "auto"
        auto_quality_check = False
        pipeline_state = PIPELINE_STATE_NO_QUALITY_CHECK
    else:
        referenced = "auto"
        auto_quality_check = True
        pipeline_state = PIPELINE_STATE_DEFAULT

    if args.stream:
        run_streaming_style(
            csv_path=csv_path,
            out_dir=args.out,
            referenced=referenced,
            auto_quality_check=auto_quality_check,
            pipeline_state=pipeline_state,
        )
    else:
        run_batch(
            output_dir=args.out,
            csv_path=csv_path,
            referenced=referenced,
            auto_quality_check=auto_quality_check,
            pipeline_state=pipeline_state,
        )
