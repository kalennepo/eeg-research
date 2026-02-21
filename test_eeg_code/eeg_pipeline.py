"""
Combined EEG pipeline: band-power time series + Beta/Alpha and Theta/Beta ratios.
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


def format_reference_report(quality_info, actual_referenced):
    """
    Build lines for reference channel stats (uV^2, uV) and re-referencing status.
    Returns list of strings for console or text file.
    """
    lines = []
    if not quality_info:
        return lines
    for ch in ("MV1", "MV3"):
        info = quality_info.get(ch, {})
        var = info.get("var")
        pp = info.get("pp")
        if var is not None and pp is not None:
            lines.append(f"  {ch}: var={var:.1f} uV^2, pp={pp:.1f} uV")
    if actual_referenced:
        lines.append("Re-referencing: applied")
    else:
        reasons = []
        for ch in ("MV1", "MV3"):
            info = quality_info.get(ch, {})
            if not info.get("good", True):
                reasons.append(f"{ch}: {info.get('reason', '?')}")
        reason_str = "; ".join(reasons) if reasons else "quality check"
        lines.append(f"Re-referencing: skipped ({reason_str})")
    return lines


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

    Returns: (is_good, reason, ref_var, ref_pp)
        is_good: True if reference is clean enough to use
        reason: string explaining why it passed/failed
        ref_var: reference variance (uV^2)
        ref_pp: reference peak-to-peak amplitude (uV)
    """
    ref_var = np.var(mv_ref)
    ref_pp = np.ptp(mv_ref)  # peak-to-peak amplitude
    meas_var = np.var(mv_meas)

    # Check variance threshold
    if ref_var > variance_threshold:
        return False, f"reference variance too high ({ref_var:.1f} > {variance_threshold:.1f} uV^2)", ref_var, ref_pp

    # Check amplitude threshold
    if ref_pp > amplitude_threshold:
        return False, f"reference amplitude too high ({ref_pp:.1f} > {amplitude_threshold:.1f} uV)", ref_var, ref_pp

    # Check SNR: if reference is much noisier than measurement, skip
    if meas_var > 0 and ref_var / meas_var > snr_threshold:
        return False, f"reference too noisy relative to measurement (SNR ratio {ref_var/meas_var:.3f} > {snr_threshold:.3f})", ref_var, ref_pp

    return True, f"reference quality OK (var={ref_var:.1f} uV^2, pp={ref_pp:.1f} uV)", ref_var, ref_pp


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
    
    # Always compute reference stats (uV², uV) for reporting
    ref1_good, ref1_reason, ref1_var, ref1_pp = check_reference_quality(mv1, mv2)
    ref2_good, ref2_reason, ref2_var, ref2_pp = check_reference_quality(mv3, mv4)
    quality_info = {
        'MV1': {'good': ref1_good, 'reason': ref1_reason, 'var': ref1_var, 'pp': ref1_pp},
        'MV3': {'good': ref2_good, 'reason': ref2_reason, 'var': ref2_var, 'pp': ref2_pp}
    }

    if referenced == 'auto':
        # Auto mode: check quality if enabled, otherwise default to re-referencing
        if auto_quality_check:
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
    Sliding-window band power and Beta/Alpha + Theta/Beta ratios over time.
    Returns: DataFrame with time_sec, Delta, Theta, Alpha, Beta, Gamma, beta_alpha_ratio, theta_beta_ratio.
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
        # Beta/Alpha ratio: engagement vs. relaxation (higher = more focused/alert)
        alpha = band_power["Alpha"]
        beta = band_power["Beta"]
        row["beta_alpha_ratio"] = beta / alpha if alpha > 0 else np.nan
        # Theta/Beta ratio: alertness/attention (higher = more drowsy/inattentive)
        theta = band_power["Theta"]
        row["theta_beta_ratio"] = theta / beta if beta > 0 else np.nan
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
    for line in format_reference_report(quality_info, actual_referenced):
        print(line)

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

    # Band power + ratio plots (Beta/Alpha and Theta/Beta with purpose labels)
    fig, axes = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
    ratio_names = ["Beta/Alpha ratio", "Theta/Beta ratio"]
    ratio_cols = ["beta_alpha_ratio", "theta_beta_ratio"]
    ratio_labels = [
        "Engagement vs. relaxation (higher = more focused/alert)",
        "Alertness/attention (higher = more drowsy/inattentive)",
    ]
    ratio_colors = ["C5", "C6"]
    ratio_box_colors = ["wheat", "lightblue"]
    for ax, name in zip(axes, list(BANDS) + ratio_names):
        if name in ratio_names:
            idx = ratio_names.index(name)
            a = axes[5 + idx]
            a.plot(band_df["time_sec"], band_df[ratio_cols[idx]], color=ratio_colors[idx], label=name)
            a.set_ylabel(f"{name.replace(' ratio', '')}\n(ratio)")
            a.text(0.02, 0.95, ratio_labels[idx],
                   transform=a.transAxes, fontsize=8, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor=ratio_box_colors[idx], alpha=0.5))
            if idx == 0:
                a.axhline(y=1, color="gray", linestyle="--", alpha=0.7)
        else:
            ax.plot(band_df["time_sec"], band_df[name])
            ax.set_ylabel(name)
    axes[-1].set_xlabel("Time (sec)")
    axes[0].set_title(f"Band Power and Ratios — {PIPELINE_STATE_LABELS.get(pipeline_state, pipeline_state)}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"band_power{suffix}.png"), dpi=150)
    plt.close(fig)

    # Summary stats for ratios (like streaming output, but from full run)
    beta_alpha = band_df["beta_alpha_ratio"].dropna()
    theta_beta = band_df["theta_beta_ratio"].dropna()
    print(f"Saved: {csv_name}, filtered_eeg{suffix}.png, band_power{suffix}.png")
    print(f"Beta/Alpha ratio: mean={beta_alpha.mean():.2f}, std={beta_alpha.std():.2f}, last={band_df['beta_alpha_ratio'].iloc[-1]:.2f}")
    print(f"Theta/Beta ratio: mean={theta_beta.mean():.2f}, std={theta_beta.std():.2f}, last={band_df['theta_beta_ratio'].iloc[-1]:.2f}")
    return band_df


def run_streaming_style(csv_path=None, out_dir=None, referenced=True, auto_quality_check=True, pipeline_state=None):
    """
    Simulate streaming: step through the same pipeline in 1 s steps and write
    Beta/Alpha and Theta/Beta ratios each second to a text file in the task
    results folder (no console output).
    """
    csv_path = csv_path or get_data_path()
    out_dir = out_dir or get_output_dir_for_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    pipeline_state = pipeline_state or PIPELINE_STATE_DEFAULT
    print(f"Pipeline state: {PIPELINE_STATE_LABELS.get(pipeline_state, pipeline_state)}")
    lines = [
        f"Pipeline state: {PIPELINE_STATE_LABELS.get(pipeline_state, pipeline_state)}",
        f"Input: {csv_path}",
        f"Output dir: {out_dir}",
    ]

    t_uniform, eeg1_filt, eeg2_filt, fs, actual_referenced, quality_info = load_and_prepare(
        csv_path=csv_path, referenced=referenced, auto_quality_check=auto_quality_check, verbose=False
    )
    ref_lines = format_reference_report(quality_info, actual_referenced)
    for line in ref_lines:
        print(line)
    lines.extend(ref_lines)
    lines.append("")
    lines.append("Streaming-style ratios (every 1 s):")

    band_df = compute_band_power_timeseries(eeg1_filt, fs=fs)

    for _, row in band_df.iterrows():
        ba = row["beta_alpha_ratio"]
        tb = row["theta_beta_ratio"]
        line = f"  t={row['time_sec']:.1f}s  Beta/Alpha = {ba:.2f}, Theta/Beta = {tb:.2f}"
        lines.append(line)

    # Summary stats for ratios (match batch-mode summary)
    beta_alpha = band_df["beta_alpha_ratio"].dropna()
    theta_beta = band_df["theta_beta_ratio"].dropna()
    summary_lines = [
        f"Beta/Alpha ratio: mean={beta_alpha.mean():.2f}, std={beta_alpha.std():.2f}, last={band_df['beta_alpha_ratio'].iloc[-1]:.2f}",
        f"Theta/Beta ratio: mean={theta_beta.mean():.2f}, std={theta_beta.std():.2f}, last={band_df['theta_beta_ratio'].iloc[-1]:.2f}",
    ]
    # Print summary to console
    for s in summary_lines:
        print(s)
    # And append to text output
    lines.append("")
    lines.extend(summary_lines)

    suffix = f"_{pipeline_state}"
    txt_path = os.path.join(out_dir, f"stream_ratios{suffix}.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    csv_path_out = os.path.join(out_dir, f"band_power_timeseries{suffix}.csv")
    with open(csv_path_out, "w") as f:
        f.write(f"# pipeline_state={pipeline_state}\n")
    band_df.to_csv(csv_path_out, mode="a", index=False)
    return band_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG pipeline: band power time series + Beta/Alpha and Theta/Beta ratios")
    parser.add_argument("--stream", action="store_true", help="Print Beta/Alpha ratio every 1 s (streaming-style)")
    parser.add_argument("--recording", default=None, help="P01 recording id: Task01, Task02, Task03, or Task04 (default: Task01)")
    parser.add_argument("--input", default=None, help="Path to input CSV (overrides --recording)")
    parser.add_argument("--out", default=None, help="Output directory (default: P01Results/<TaskNN>/)")
    parser.add_argument("--no-reference", action="store_true", help="Use raw MV2/MV4 only; do not re-reference with MV1/MV3")
    parser.add_argument("--force-reference", action="store_true", help="Always re-reference (ignore quality checks)")
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
