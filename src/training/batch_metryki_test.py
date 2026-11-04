"""
Metrics and plots for the clean test set (videos provided by the instructor).
Only processes the testp* folders in data/reports/.
FPS is read from the MP4 file (not hard-coded).

For each session it generates:
  - macierz_konfuzji.png
  - wykres_czasowy.png

Usage:
    python src/training/batch_metryki_test.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR     = os.path.join(BASE_DIR, "data", "reports")
TEST_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "test_frames")

SMOOTH_N       = 20
SMOOTH_THRESH  = 0.70
SHORT_THRESH_S = 3.0

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
ONLY_SESSION = "testp5"   # e.g. "testp2" – only process that session; None = all testp*
# ─────────────────────────────────────────────────────────────────────────────


def get_fps(video_name: str) -> float:
    path = os.path.join(TEST_VIDEOS_DIR, f"{video_name}.mp4")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def _runs(arr):
    arr = np.asarray(arr, dtype=int)
    if len(arr) == 0:
        return []
    runs, s, cur = [], 0, arr[0]
    for i in range(1, len(arr)):
        if arr[i] != cur:
            runs.append((int(cur), s, i - 1))
            cur, s = arr[i], i
    runs.append((int(cur), s, len(arr) - 1))
    return runs


def compute_metrics(df, fps, smooth_n=SMOOTH_N,
                    smooth_thresh=SMOOTH_THRESH, short_s=SHORT_THRESH_S):
    df = df.reset_index(drop=True).copy()

    rolling = df["prediction"].rolling(window=smooth_n, min_periods=1).mean()
    df["pred_s"] = (rolling >= smooth_thresh).astype(int)

    gt   = df["ground_truth"].values
    pred = df["pred_s"].values

    acc_raw    = float(accuracy_score(gt, df["prediction"].values))
    acc_smooth = float(accuracy_score(gt, pred))

    cm = confusion_matrix(gt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    gt_runs   = _runs(gt)
    pred_runs = _runs(pred)
    gt_s      = pd.Series(gt)
    pred_s    = pd.Series(pred)

    fa = sum(1 for l, s, e in pred_runs
             if l == 1 and gt_s.iloc[s:e+1].mean() < 0.5)
    total_sec = len(df) / fps
    fa_per_h  = fa / total_sec * 3600 if total_sec > 0 else float("nan")

    det_times = []
    for l, s, e in gt_runs:
        if l == 1:
            seg = pred_s.iloc[s:e+1].values
            if seg.any():
                det_times.append(int(np.argmax(seg)) / fps)
    avg_det = float(np.mean(det_times)) if det_times else float("nan")

    bb = []
    for l, s, e in gt_runs:
        if l == 1:
            seg = pred_s.iloc[s:e+1].values
            bb.append(int(np.argmax(seg)) / fps if seg.any() else (e - s + 1) / fps)
    avg_bb = float(np.mean(bb)) if bb else float("nan")

    SHORT_F = int(short_s * fps)
    fp_runs = [(l, s, e) for l, s, e in pred_runs
               if l == 1 and gt_s.iloc[s:e+1].mean() < 0.5]
    short_fp = sum(1 for _, s, e in fp_runs if (e - s + 1) < SHORT_F)
    pct_sh   = (short_fp / len(fp_runs) * 100) if fp_runs else 0.0

    fmt = lambda v, d=2: round(v, d) if not (isinstance(v, float) and np.isnan(v)) else "N/D"

    return df, cm, {
        "Frames":              len(df),
        "FPS":                 round(fps, 1),
        "Bad % (GT)":          round(gt_s.mean() * 100, 1),
        "Acc. raw":            round(acc_raw * 100, 2),
        "Acc. smooth":         round(acc_smooth * 100, 2),
        "Sensitivity":         fmt(round(sensitivity * 100, 2)),
        "Specificity":         fmt(round(specificity * 100, 2)),
        "FA/h":                fmt(round(fa_per_h, 1), 1),
        "Detection [s]":        fmt(avg_det),
        "Bad-before-alarm [s]": fmt(avg_bb),
        "% Short FA":             round(pct_sh, 1),
    }


def plot_confusion_matrix(cm, session_name, out_dir, metrics):
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Good (0)", "Bad (1)"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="#ccc")
    ax.set_xlabel("Prediction (smoothed)", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    acc  = metrics["Acc. smooth"]
    sens = metrics["Sensitivity"]
    spec = metrics["Specificity"]
    ax.set_title(
        f"{session_name}\nAcc={acc}%  Sensitivity={sens}%  Specificity={spec}%",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "macierz_konfuzji.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_time_series(df, session_name, fps, out_dir, metrics):
    # Time axis in seconds – based on the actual frame number in the video
    t = df["frame_num"] / fps

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Top panel: ground truth
    axes[0].fill_between(t, df["ground_truth"], alpha=0.7,
                         color="#e74c3c", label="Bad posture (GT=1)")
    axes[0].fill_between(t, 1 - df["ground_truth"], alpha=0.4,
                         color="#2ecc71", label="Good posture (GT=0)")
    axes[0].set_ylabel("Ground Truth", fontsize=10)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{session_name} – ground truth vs. prediction",
                      fontsize=11, fontweight="bold")

    # Bottom panel: smoothed prediction
    axes[1].fill_between(t, df["pred_s"], alpha=0.7,
                         color="#e67e22", label="Prediction: bad (1)")
    axes[1].fill_between(t, 1 - df["pred_s"], alpha=0.4,
                         color="#3498db", label="Prediction: good (0)")
    axes[1].set_ylabel("Prediction (smoothed)", fontsize=10)
    axes[1].set_xlabel("Time", fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right", fontsize=8)

    # MM:SS formatter on both panels – tick interval depends on the video length
    import matplotlib.ticker as ticker
    duration = float(t.max() - t.min())
    tick_interval = 60 if duration < 600 else 120 if duration < 1500 else 300
    def fmt_mmss(x, _):
        m, s = divmod(int(max(x, 0)), 60)
        return f"{m:02d}:{s:02d}"
    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_mmss))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        ax.tick_params(axis="x", labelbottom=True, labelsize=8)
        ax.set_xlabel("Time [MM:SS]", fontsize=9)

    acc  = metrics["Acc. smooth"]
    fa   = metrics["FA/h"]
    fig.text(0.01, 0.01,
             f"Acc. smooth={acc}%  |  FA/h={fa}  |  "
             f"Sensitivity={metrics['Sensitivity']}%  |  Specificity={metrics['Specificity']}%",
             fontsize=9, color="#555")

    plt.tight_layout()
    path = os.path.join(out_dir, "wykres_czasowy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_summary(result_df, out_dir):
    """Comparison chart for all 4 sessions."""
    sessions = result_df["Session"].tolist()
    x = np.arange(len(sessions))
    width = 0.25

    acc_raw    = pd.to_numeric(result_df["Acc. raw"],    errors="coerce").tolist()
    acc_smooth = pd.to_numeric(result_df["Acc. smooth"], errors="coerce").tolist()
    sens       = pd.to_numeric(result_df["Sensitivity"], errors="coerce").tolist()
    spec       = pd.to_numeric(result_df["Specificity"], errors="coerce").tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: accuracy
    axes[0].bar(x - width/2, acc_raw,    width, label="Acc. raw",    color="#3498db", alpha=0.85)
    axes[0].bar(x + width/2, acc_smooth, width, label="Acc. smooth", color="#2980b9", alpha=0.85)
    axes[0].axhline(y=50, color="gray", linestyle="--", linewidth=1, label="random classifier (50%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sessions)
    axes[0].set_ylabel("Accuracy [%]")
    axes[0].set_title("Classification accuracy", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 105)
    for i, (r, s) in enumerate(zip(acc_raw, acc_smooth)):
        axes[0].text(i - width/2, r + 1, f"{r:.1f}", ha="center", fontsize=8)
        axes[0].text(i + width/2, s + 1, f"{s:.1f}", ha="center", fontsize=8)

    # Right: sensitivity vs specificity
    axes[1].bar(x - width/2, sens, width, label="Sensitivity (detecting bad)", color="#e74c3c", alpha=0.85)
    axes[1].bar(x + width/2, spec, width, label="Specificity (detecting good)", color="#2ecc71", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sessions)
    axes[1].set_ylabel("Value [%]")
    axes[1].set_title("Sensitivity and specificity", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 105)
    for i, (se, sp) in enumerate(zip(sens, spec)):
        if not np.isnan(se):
            axes[1].text(i - width/2, se + 1, f"{se:.1f}", ha="center", fontsize=8)
        if not np.isnan(sp):
            axes[1].text(i + width/2, sp + 1, f"{sp:.1f}", ha="center", fontsize=8)

    plt.suptitle("Clean test set – session comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "podsumowanie_test_zewnetrzny.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved summary chart: {path}")


# ── Main loop ────────────────────────────────────────────────────────────────
rows    = []
skipped = []

test_dirs = sorted([
    d for d in os.listdir(REPORTS_DIR)
    if d.startswith("testp") and os.path.isdir(os.path.join(REPORTS_DIR, d))
])

if ONLY_SESSION:
    test_dirs = [d for d in test_dirs if d == ONLY_SESSION]
    if not test_dirs:
        print(f"Folder '{ONLY_SESSION}' does not exist in {REPORTS_DIR}.")
        sys.exit(1)

if not test_dirs:
    print(f"No testp* folders in {REPORTS_DIR}.")
    print("Run this first: python src/core/evaluate_external_test.py")
    sys.exit(1)

for d in test_dirs:
    csv_path = os.path.join(REPORTS_DIR, d, "wyniki_detekcji.csv")
    if not os.path.isfile(csv_path):
        skipped.append(f"{d} (no CSV)")
        continue

    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw[(df_raw["prediction"] != -1) & (df_raw["ground_truth"] != -1)].reset_index(drop=True)

    if len(df_raw) == 0:
        skipped.append(f"{d} (no labeled frames)")
        continue

    fps = get_fps(d)
    out_dir = os.path.join(REPORTS_DIR, d)

    print(f"\n{d}  ({len(df_raw)} frames, FPS={fps:.1f})")
    df_proc, cm, m = compute_metrics(df_raw, fps)

    plot_confusion_matrix(cm, d, out_dir, m)
    plot_time_series(df_proc, d, fps, out_dir, m)

    rows.append({"Session": d, **m})

if not rows:
    print("\nNo data.")
    sys.exit(1)

result = pd.DataFrame(rows)

# ── Table ────────────────────────────────────────────────────────────────────
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 180)

print()
print("=" * 160)
print("  METRICS – CLEAN TEST SET (videos from the instructor)")
print(f"  Parameters: SMOOTH_N={SMOOTH_N}, threshold={SMOOTH_THRESH:.0%}")
print("=" * 160)
print(result.to_string(index=False))
print("=" * 160)

numeric_cols = ["Acc. raw", "Acc. smooth", "Sensitivity", "Specificity", "% Short FA"]
print()
print("Average across people:")
for col in numeric_cols:
    vals = pd.to_numeric(result[col], errors="coerce").dropna()
    if len(vals):
        print(f"  {col:<25} avg={vals.mean():.2f}  min={vals.min():.2f}  max={vals.max():.2f}")
for col in ["FA/h", "Detection [s]", "Bad-before-alarm [s]"]:
    vals = pd.to_numeric(result[col], errors="coerce").dropna()
    if len(vals):
        print(f"  {col:<25} avg={vals.mean():.2f}  min={vals.min():.2f}  max={vals.max():.2f}")

# ── Summary chart ─────────────────────────────────────────────────────────────
plot_summary(result, REPORTS_DIR)

# ── Save CSV ─────────────────────────────────────────────────────────────────
out_csv = os.path.join(REPORTS_DIR, "metryki_test_zewnetrzny.csv")
result.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

if skipped:
    print(f"\nSkipped: {skipped}")
