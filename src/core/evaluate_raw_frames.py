"""
Ewaluacja sesji nagranych przez record_video.py (surowe klatki JPG).
Uzupełnia istniejący CSV z ground truth o predykcje modelu.

Logika inference identyczna jak server.py i evaluate_external_test.py.

Uruchomienie:
  python src/core/evaluate_raw_frames.py
"""

import json
import os
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.core.normalizer import normalize_features

RAW_FRAMES_DIR = os.path.join(BASE_DIR, "data", "raw_frames")
PROFILES_DIR   = os.path.join(BASE_DIR, "data", "profiles")
REPORTS_DIR    = os.path.join(BASE_DIR, "data", "reports")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "models", "scaler.pkl")

SMOOTH_N           = 20
INFERENCE_INTERVAL = 0.2   # sekund między inferencjami (jak server: 5 Hz)
ASSUMED_FPS        = 30.0  # FPS kamery przy nagrywaniu
CALIB_SECONDS      = 5.0
CALIB_VIS_MIN      = 0.6
CALIB_SAMPLES_MIN  = 10

# ── KONFIGURACJA ──────────────────────────────────────────────────────────────
SESSION_NAME       = "testp5"   # ← zmień tutaj (musi być ten sam co w record_video.py)
CALIB_START_FRAME  = 0          # od której klatki (0-based) kalibrować
#                                 (zmień jeśli na początku siedziałaś w złej pozycji)
# ─────────────────────────────────────────────────────────────────────────────

LANDMARKS_TO_USE = {"nose": 0, "left_shoulder": 11, "right_shoulder": 12}


def extract_features(landmarks):
    features = []
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])
    return features


def calibrate_from_frames(frames_dir, sorted_files, pose, start_frame=0):
    max_frames = int(ASSUMED_FPS * CALIB_SECONDS)
    samples = []

    for fname in sorted_files[start_frame: start_frame + max_frames]:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            vis = (lm[11].visibility + lm[12].visibility) / 2.0
            if vis > CALIB_VIS_MIN:
                sample = {}
                for name, idx in LANDMARKS_TO_USE.items():
                    p = lm[idx]
                    sample[f"{name}_x"] = p.x
                    sample[f"{name}_y"] = p.y
                    sample[f"{name}_z"] = p.z
                samples.append(sample)

    if len(samples) < CALIB_SAMPLES_MIN:
        print(f"  UWAGA: tylko {len(samples)} próbek kalibracyjnych "
              f"(minimum {CALIB_SAMPLES_MIN}). Kontynuuję.")
    if not samples:
        return None

    calibration_data = {
        key: float(np.mean([s[key] for s in samples]))
        for key in samples[0]
    }

    os.makedirs(PROFILES_DIR, exist_ok=True)
    profile = {
        "name":         SESSION_NAME,
        "calibration":  calibration_data,
        "created_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(PROFILES_DIR, f"{SESSION_NAME}.json"), "w",
              encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"  Kalibracja: {len(samples)} próbek (kl. {start_frame}–"
          f"{start_frame + max_frames}) → data/profiles/{SESSION_NAME}.json")
    return calibration_data


def evaluate_frames(frames_dir, sorted_files, pose, calibration, model, scaler):
    inference_every_n = max(1, int(ASSUMED_FPS * INFERENCE_INTERVAL))
    total = len(sorted_files)

    results          = []
    predictions_buffer = []
    last_label       = 0
    last_raw_pred    = 0
    last_confidence  = 1.0

    for frame_idx, fname in enumerate(sorted_files, start=1):
        if frame_idx % 500 == 0:
            print(f"  Klatka {frame_idx}/{total}...")

        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None:
            results.append({"frame_num": frame_idx,
                            "prediction": -1, "confidence": -1, "smoothed_pred": -1})
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        person_visible = bool(res.pose_landmarks)

        if person_visible and (frame_idx % inference_every_n == 0):
            raw_features  = extract_features(res.pose_landmarks.landmark)
            norm_features = normalize_features(raw_features, calibration)
            X = np.array(norm_features).reshape(1, -1)
            if scaler:
                X = scaler.transform(X)

            last_raw_pred   = int(model.predict(X)[0])
            proba           = model.predict_proba(X)[0]
            last_confidence = float(proba[last_raw_pred])

            predictions_buffer.append(last_raw_pred)
            if len(predictions_buffer) > SMOOTH_N:
                predictions_buffer.pop(0)
            last_label = 1 if predictions_buffer.count(1) >= int(SMOOTH_N * 0.7) else 0

        prediction_out = last_raw_pred if person_visible else -1
        results.append({
            "frame_num":    frame_idx,
            "prediction":   prediction_out,
            "confidence":   round(last_confidence, 4) if person_visible else -1,
            "smoothed_pred": last_label if person_visible else -1,
        })

    return results


def main():
    frames_dir  = os.path.join(RAW_FRAMES_DIR, SESSION_NAME)
    reports_dir = os.path.join(REPORTS_DIR,    SESSION_NAME)
    csv_path    = os.path.join(reports_dir, "wyniki_detekcji.csv")

    if not os.path.isdir(frames_dir):
        print(f"BŁĄD: brak folderu {frames_dir}")
        return

    sorted_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not sorted_files:
        print(f"BŁĄD: brak plików JPG w {frames_dir}")
        return

    print(f"Sesja:  {SESSION_NAME}")
    print(f"Klatek: {len(sorted_files)}")

    # Wczytaj ground truth zapisany przez record_video.py
    if not os.path.exists(csv_path):
        print(f"UWAGA: brak {csv_path} – ground_truth zostanie ustawiony na -1")
        gt_df = None
    else:
        gt_df = pd.read_csv(csv_path)
        labeled = (gt_df["ground_truth"] != -1).sum()
        print(f"Ground truth: {labeled}/{len(gt_df)} klatek oznaczonych")

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    mp_pose = mp.solutions.pose

    print(f"\nKalibracja od klatki {CALIB_START_FRAME} (0.7/0.7)...")
    with mp_pose.Pose(min_detection_confidence=0.7,
                      min_tracking_confidence=0.7,
                      model_complexity=1) as pose_calib:
        calibration = calibrate_from_frames(frames_dir, sorted_files,
                                             pose_calib, CALIB_START_FRAME)

    if calibration is None:
        print("BŁĄD: kalibracja nie powiodła się.")
        return

    print("\nEwaluacja (0.6/0.6)...")
    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6,
                      model_complexity=1) as pose_eval:
        results = evaluate_frames(frames_dir, sorted_files, pose_eval,
                                   calibration, model, scaler)

    pred_df = pd.DataFrame(results)

    if gt_df is not None and "ground_truth" in gt_df.columns:
        pred_df = pred_df.merge(gt_df[["frame_num", "ground_truth"]],
                                on="frame_num", how="left")
        pred_df["ground_truth"] = pred_df["ground_truth"].fillna(-1).astype(int)
    else:
        pred_df["ground_truth"] = -1

    pred_df = pred_df[["frame_num", "ground_truth", "prediction",
                        "confidence", "smoothed_pred"]]

    os.makedirs(reports_dir, exist_ok=True)
    pred_df.to_csv(csv_path, index=False)

    valid   = pred_df[pred_df["prediction"] != -1]
    bad_pct = valid["prediction"].mean() * 100 if len(valid) else float("nan")
    labeled = (pred_df["ground_truth"] != -1).sum()

    print(f"\nZapisano {len(pred_df)} klatek → {csv_path}")
    print(f"Raw predykcja: {bad_pct:.1f}% klatek = zła postura")
    print(f"Ground truth:  {labeled}/{len(pred_df)} klatek oznaczonych")
    print(f"\nNastępny krok: python src/training/batch_metryki_test.py")


if __name__ == "__main__":
    main()
