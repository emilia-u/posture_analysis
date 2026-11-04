"""
benchmark.py – performance measurement of the posture analysis pipeline.

Measures:
  - the time of each frame processing stage (MediaPipe, RF, normalization)
  - total time per frame
  - process CPU and RAM usage during the test
  - generates a chart and saves a CSV with the results

Usage:
    python src/training/benchmark.py

Requirements: pip install psutil
"""

import sys, os, time, warnings
import cv2, joblib, numpy as np
import mediapipe as mp
import psutil
import json
import threading
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from src.core.normalizer import normalize_features

# ── Configuration ─────────────────────────────────────────────────────────────
N_FRAMES      = 300          # number of frames to measure
PROFILE_NAME  = "laptop_nisko_prawie90"     # calibration profile to load
SMOOTH_N      = 20           # smoothing buffer size
RESULTS_DIR   = os.path.join(BASE_DIR, "data", "reports")
# ─────────────────────────────────────────────────────────────────────────────

# Load the model and scaler
MODEL_PATH  = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
LANDMARKS_TO_USE = {"nose": 0, "left_shoulder": 11, "right_shoulder": 12}

print("[1/4] Loading model...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# Load the calibration profile
profile_path = os.path.join(BASE_DIR, "data", "profiles", f"{PROFILE_NAME}.json")
with open(profile_path, "r", encoding="utf-8") as f:
    calibration = json.load(f)["calibration"]

print(f"[2/4] Profile '{PROFILE_NAME}' loaded.")


def extract_features(landmarks):
    features = []
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])
    return features


# ── CPU/RAM monitoring in a separate thread ───────────────────────────────────
cpu_samples  = []
ram_samples  = []
_stop_monitor = threading.Event()

def _monitor():
    proc = psutil.Process(os.getpid())
    while not _stop_monitor.is_set():
        cpu_samples.append(proc.cpu_percent(interval=None))
        ram_samples.append(proc.memory_info().rss / 1024 / 1024)  # MB
        time.sleep(0.1)

monitor_thread = threading.Thread(target=_monitor, daemon=True)
# ─────────────────────────────────────────────────────────────────────────────

# ── Benchmark ─────────────────────────────────────────────────────────────────
print("[3/4] Starting the camera and beginning measurement...")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

timings = {
    "capture":     [],
    "mediapipe":   [],
    "extraction":  [],
    "normalization": [],
    "scaling":     [],
    "prediction":  [],
    "smoothing":   [],
    "total":       [],
}

predictions_buf = []
frame_count     = 0

# Warm-up – the first 10 frames are not measured
mp_pose = mp.solutions.pose
with mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
) as pose:
    # warm-up
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            pose.process(rgb)

    monitor_thread.start()

    while frame_count < N_FRAMES:
        t0 = time.perf_counter()

        # 1. Frame capture
        ret, frame = cap.read()
        if not ret:
            continue
        t1 = time.perf_counter()

        # 2. MediaPipe
        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        t2 = time.perf_counter()

        if not results.pose_landmarks:
            continue  # frames without a detection are not measured

        # 3. Feature extraction
        raw_features = extract_features(results.pose_landmarks.landmark)
        t3 = time.perf_counter()

        # 4. Normalization
        norm_features = normalize_features(raw_features, calibration)
        t4 = time.perf_counter()

        # 5. Scaling
        X = np.array(norm_features).reshape(1, -1)
        if scaler:
            X = scaler.transform(X)
        t5 = time.perf_counter()

        # 6. RF prediction
        pred = model.predict(X)[0]
        t6 = time.perf_counter()

        # 7. Smoothing
        predictions_buf.append(pred)
        if len(predictions_buf) > SMOOTH_N:
            predictions_buf.pop(0)
        _ = 1 if predictions_buf.count(1) >= int(SMOOTH_N * 0.7) else 0
        t7 = time.perf_counter()

        # Record timings (in ms)
        timings["capture"].append((t1 - t0) * 1000)
        timings["mediapipe"].append((t2 - t1) * 1000)
        timings["extraction"].append((t3 - t2) * 1000)
        timings["normalization"].append((t4 - t3) * 1000)
        timings["scaling"].append((t5 - t4) * 1000)
        timings["prediction"].append((t6 - t5) * 1000)
        timings["smoothing"].append((t7 - t6) * 1000)
        timings["total"].append((t7 - t0) * 1000)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Frame {frame_count}/{N_FRAMES} "
                  f"| Avg total: {np.mean(timings['total']):.1f} ms")

_stop_monitor.set()
cap.release()
monitor_thread.join()

print(f"[4/4] Measurement finished. Frames: {frame_count}")


# ── Statistical results ───────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Stage':<20} {'Avg [ms]':>9} {'Med. [ms]':>9} "
      f"{'P95 [ms]':>9} {'Max [ms]':>9}")
print("-"*60)

stats_rows = []
for stage, vals in timings.items():
    arr  = np.array(vals)
    mean = np.mean(arr)
    med  = np.median(arr)
    p95  = np.percentile(arr, 95)
    mx   = np.max(arr)
    print(f"{stage:<20} {mean:>9.2f} {med:>9.2f} {p95:>9.2f} {mx:>9.2f}")
    stats_rows.append({"etap": stage, "srednia_ms": round(mean, 3),
                        "mediana_ms": round(med, 3),
                        "p95_ms": round(p95, 3), "max_ms": round(mx, 3)})

total_mean = np.mean(timings["total"])
fps_real   = 1000 / total_mean
print("="*60)
print(f"\nAverage time per frame: {total_mean:.2f} ms  →  {fps_real:.1f} FPS (theoretical max)")

if cpu_samples:
    cpu_mean = np.mean(cpu_samples[5:])   # skip the first samples (warm-up)
    ram_mean = np.mean(ram_samples[5:])
    ram_max  = np.max(ram_samples[5:])
    print(f"Average CPU usage:     {cpu_mean:.1f}%")
    print(f"Average RAM usage:     {ram_mean:.1f} MB  (max: {ram_max:.1f} MB)")
else:
    cpu_mean = ram_mean = ram_max = 0

# ── Save CSV ──────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
df_stats = pd.DataFrame(stats_rows)
df_stats.to_csv(os.path.join(RESULTS_DIR, "benchmark_wyniki.csv"), index=False)
print(f"\nCSV saved: data/reports/benchmark_wyniki.csv")

# ── Chart ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: time per stage (without total)
stages_plot = [s for s in timings if s != "total"]
means_plot  = [np.mean(timings[s]) for s in stages_plot]
colors      = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
               "#f39c12", "#1abc9c", "#e67e22"]

axes[0].barh(stages_plot, means_plot, color=colors)
axes[0].set_xlabel("Average time [ms]")
axes[0].set_title("Processing time by stage")
for i, v in enumerate(means_plot):
    axes[0].text(v + 0.05, i, f"{v:.2f} ms", va="center", fontsize=9)

# Right: histogram of total time per frame
axes[1].hist(timings["total"], bins=40, color="#2c3e50", edgecolor="white", alpha=0.85)
axes[1].axvline(total_mean, color="red", linestyle="--",
                label=f"Mean: {total_mean:.1f} ms")
axes[1].axvline(np.percentile(timings["total"], 95), color="orange",
                linestyle="--", label=f"P95: {np.percentile(timings['total'], 95):.1f} ms")
axes[1].set_xlabel("Total time per frame [ms]")
axes[1].set_ylabel("Number of frames")
axes[1].set_title("Distribution of total processing time")
axes[1].legend()

plt.suptitle(
    f"Pipeline performance benchmark  |  n={N_FRAMES} frames\n"
    f"CPU: {cpu_mean:.1f}%  |  RAM: {ram_mean:.1f} MB  |  "
    f"Avg FPS: {fps_real:.1f}",
    fontsize=11
)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "benchmark_wykres.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Chart saved: data/reports/benchmark_wykres.png")
plt.show()
