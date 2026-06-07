"""
Ewaluacja czystego zbioru testowego – filmiki od prowadzącego.
Logika inference identyczna jak server.py (finalny produkt).

Dla każdego pliku testp{N}.mp4:
  1. Kalibracja profilu z pierwszych kilku sekund (osoba siedzi prosto).
  2. Pełna analiza wideo z inference co INFERENCE_INTERVAL sekund (jak server).
  3. Zapis do data/reports/testp{N}/wyniki_detekcji.csv

Uruchomienie:
  python src/core/evaluate_external_test.py
"""

import os
import sys
import json
import time

import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.core.normalizer import normalize_features

# ── Konfiguracja – identyczna jak server.py ───────────────────────────────────
TEST_VIDEOS_DIR    = os.path.join(BASE_DIR, "data", "test_frames")
PROFILES_DIR       = os.path.join(BASE_DIR, "data", "profiles")
REPORTS_DIR        = os.path.join(BASE_DIR, "data", "reports")
MODEL_PATH         = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH        = os.path.join(BASE_DIR, "models", "scaler.pkl")

SMOOTH_N           = 20
INFERENCE_INTERVAL = 0.2   # sekund między inferencjami (server: 5 Hz)
CALIB_SECONDS      = 5.0   # ile sekund kalibracji
CALIB_VIS_MIN      = 0.6   # minimalna widoczność ramion (jak server)
CALIB_SAMPLES_MIN  = 10    # minimalna liczba próbek (jak server)

# ── Per-video: od której sekundy zacząć kalibrację ───────────────────────────
# Domyślnie 0.0 (od początku). Zmień jeśli na początku filmiku jest zła postura.
CALIB_START_SEC = {
    "testp1": 180.0,
    "testp2": 185.0,
    "testp3": 120.0,
    "testp4": 80.0,
}

# ── Które filmy przetworzyć (None = wszystkie) ────────────────────────────────
# Ustaw np. ["testp2"] żeby przeliczyć tylko jeden filmik bez nadpisywania reszty
RUN_ONLY = ["testp2", "testp4"]

# Dokładnie te same landmarki co server.py
LANDMARKS_TO_USE = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12
}
# ─────────────────────────────────────────────────────────────────────────────


def extract_features(landmarks):
    """Identyczna funkcja jak w server.py."""
    features = []
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])
    return features


def calibrate_from_video(cap, pose, fps, profile_name, start_sec=0.0):
    """
    Kalibracja z CALIB_SECONDS sekund wideo zaczynając od start_sec.
    Logika widoczności i minimum próbek jak w server.py.
    """
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    max_frames = int(fps * CALIB_SECONDS)
    samples = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
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
        print(f"  UWAGA: zebrano tylko {len(samples)} próbek kalibracyjnych "
              f"(minimum {CALIB_SAMPLES_MIN}). Kontynuuję z tym co jest.")
    if not samples:
        return None

    calibration_data = {}
    for key in samples[0]:
        calibration_data[key] = float(np.mean([s[key] for s in samples]))

    os.makedirs(PROFILES_DIR, exist_ok=True)
    profile = {
        "name": profile_name,
        "calibration": calibration_data,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(PROFILES_DIR, f"{profile_name}.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"  Kalibracja: {len(samples)} próbek z {start_sec}s–{start_sec+CALIB_SECONDS}s → "
          f"data/profiles/{profile_name}.json")
    return calibration_data


def evaluate_video(cap, pose, calibration, fps, model, scaler):
    """
    Analiza wideo zgodna z server.py:
    - MediaPipe na każdej klatce (0.6/0.6)
    - Inference modelu co INFERENCE_INTERVAL sekund (5 Hz)
    - Między inferencjami: last_label się utrzymuje (jak server)
    - Smoothing buffer 20 predykcji, próg int(20 * 0.7) = 14
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    inference_every_n = max(1, int(fps * INFERENCE_INTERVAL))

    results = []
    predictions_buffer = []
    last_label = 0
    last_raw_pred = 0   # carry-forward między inferencjami (jak last_label w server)
    last_confidence = 1.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  Klatka {frame_idx}/{total}...")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        person_visible = bool(res.pose_landmarks)

        if person_visible and (frame_idx % inference_every_n == 0):
            raw_features = extract_features(res.pose_landmarks.landmark)
            norm_features = normalize_features(raw_features, calibration)

            X = np.array(norm_features).reshape(1, -1)
            if scaler:
                X = scaler.transform(X)

            last_raw_pred = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            last_confidence = float(proba[last_raw_pred])

            predictions_buffer.append(last_raw_pred)
            if len(predictions_buffer) > SMOOTH_N:
                predictions_buffer.pop(0)

            last_label = 1 if predictions_buffer.count(1) >= int(SMOOTH_N * 0.7) else 0

        # prediction: carry-forward gdy osoba widoczna, -1 gdy brak osoby
        prediction_out = last_raw_pred if person_visible else -1

        results.append({
            "frame_num":     frame_idx,
            "ground_truth":  -1,
            "prediction":    prediction_out,
            "confidence":    round(last_confidence, 4) if person_visible else -1,
            "smoothed_pred": last_label if person_visible else -1,
        })

    return results


def process_test_video(video_path, video_name, model, scaler):
    print(f"\n{'='*60}")
    print(f"Przetwarzanie: {video_name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  BŁĄD: nie można otworzyć {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.1f}, łączna liczba klatek: {total} "
          f"(~{total/fps/60:.1f} min)")
    print(f"  Inference co {INFERENCE_INTERVAL}s = co {int(fps*INFERENCE_INTERVAL)} klatki")

    mp_pose = mp.solutions.pose

    # Kalibracja: 0.7/0.7 (jak server przy tworzeniu profilu)
    start_sec = CALIB_START_SEC.get(video_name, 0.0)
    print(f"  Kalibracja od {start_sec}s (0.7/0.7)...")
    with mp_pose.Pose(min_detection_confidence=0.7,
                      min_tracking_confidence=0.7,
                      model_complexity=1) as pose_calib:
        calibration = calibrate_from_video(cap, pose_calib, fps, video_name, start_sec)

    if calibration is None:
        print("  BŁĄD: kalibracja nie powiodła się.")
        cap.release()
        return

    # Ewaluacja: 0.6/0.6 (jak server w sesji)
    print("  Ewaluacja wideo (0.6/0.6)...")
    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6,
                      model_complexity=1) as pose_eval:
        results = evaluate_video(cap, pose_eval, calibration, fps, model, scaler)

    cap.release()

    out_dir = os.path.join(REPORTS_DIR, video_name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "wyniki_detekcji.csv")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    valid = df[df["prediction"] != -1]
    bad_pct = valid["prediction"].mean() * 100 if len(valid) else float("nan")
    print(f"  Zapisano {len(df)} klatek → {out_csv}")
    print(f"  Raw predykcja: {bad_pct:.1f}% klatek = zła postura")
    print(f"  UWAGA: uzupełnij ground_truth przez add_ground_truth.py!")


def main():
    print("Wczytywanie modelu...")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    video_files = sorted([
        f for f in os.listdir(TEST_VIDEOS_DIR)
        if f.lower().endswith(".mp4")
    ])
    if not video_files:
        print(f"Brak plików .mp4 w {TEST_VIDEOS_DIR}")
        return

    if RUN_ONLY:
        video_files = [f for f in video_files if os.path.splitext(f)[0] in RUN_ONLY]
        print(f"Tryb selektywny – przetwarzam tylko: {RUN_ONLY}")
    else:
        print(f"Znaleziono {len(video_files)} filmik(ów): {video_files}")

    for fname in video_files:
        video_name = os.path.splitext(fname)[0]
        process_test_video(os.path.join(TEST_VIDEOS_DIR, fname), video_name, model, scaler)

    print("\n" + "="*60)
    print("GOTOWE. Następne kroki:")
    print("  1. python src/core/add_ground_truth.py  (uzupełnij segmenty)")
    print("  2. python src/training/batch_metryki.py")
    print("="*60)


if __name__ == "__main__":
    main()
