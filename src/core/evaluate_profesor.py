import sys
import os
import cv2
import joblib
import pandas as pd
import numpy as np
import mediapipe as mp
import json

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = r"C:\Users\emilk\Downloads\profesor.mp4"
PROFILE_NAME = "profesor"
OUTPUT_CSV = "wyniki_profesor.csv"
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.core.normalizer import normalize_features

MODEL_PATH = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
PROFILES_DIR = os.path.join(BASE_DIR, "data", "profiles")

LANDMARKS_TO_USE = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12
}


def extract_features(landmarks):
    features = []
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])
    return features


def main():
    print("--- STARTING ANALYSIS OF THE PROFESSOR'S VIDEO ---")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    profile_path = os.path.join(PROFILES_DIR, f"{PROFILE_NAME}.json")
    with open(profile_path, "r", encoding="utf-8") as f:
        calibration = json.load(f)["calibration"]
    print(f"Loaded profile: {PROFILE_NAME}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    mp_pose = mp.solutions.pose

    # For on-the-fly smoothing (so results are clean right away)
    predictions_buffer = []

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Analyzing frame: {frame_idx}/{total_frames}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            label = -1
            smoothed_label = -1

            if res.pose_landmarks:
                raw_features = extract_features(res.pose_landmarks.landmark)
                norm_features = normalize_features(raw_features, calibration)

                X = np.array(norm_features).reshape(1, -1)
                if scaler:
                    X = scaler.transform(X)

                pred = int(model.predict(X)[0])

                # Anti-noise buffer (20 frames)
                predictions_buffer.append(pred)
                if len(predictions_buffer) > 20:
                    predictions_buffer.pop(0)

                smoothed_label = 1 if predictions_buffer.count(1) >= 14 else 0

            results.append({
                "frame": frame_idx,
                "model_label": smoothed_label
            })

    cap.release()

    df = pd.DataFrame(results)
    output_path = os.path.join(BASE_DIR, "data", OUTPUT_CSV)
    try:
        df.to_csv(output_path, index=False)
    except:
        df.to_csv(OUTPUT_CSV, index=False)
        output_path = OUTPUT_CSV

    print(f"\n✅ DONE! Results saved to: {output_path}")


if __name__ == "__main__":
    main()