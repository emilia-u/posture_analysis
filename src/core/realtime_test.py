import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import time
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
CSV_PATH    = os.path.join(BASE_DIR, "data", "posture_data.csv")

# add the project path so the src imports work
import sys
sys.path.insert(0, BASE_DIR)

from src.core.session_manager import select_or_create_profile, ensure_calibration
from src.core.normalizer import normalize_features

LANDMARKS_TO_USE = {
    "nose":           0,
    "left_eye":       2,
    "right_eye":      5,
    "left_ear":       7,
    "right_ear":      8,
    "left_shoulder":  11,
    "right_shoulder": 12,
}

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

BAD_POSTURE_ALERT_SECONDS = 3   # how many seconds of bad posture before an alert


def extract_features(landmarks) -> list:
    features = []
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])
    return features


def draw_ui(frame, label: int, confidence: float,
            bad_seconds: float, alert_threshold: float,
            user_name: str, corrections: int, flash: str = ""):

    if label == 0:
        text, color, bg = "GOOD POSTURE", (0, 220, 0), (0, 60, 0)
    else:
        text, color, bg = "BAD POSTURE!", (0, 0, 220), (0, 0, 60)

    cv2.rectangle(frame, (0, 0), (640, 70), bg, -1)
    cv2.putText(frame, text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    cv2.putText(frame, f"{user_name}  |  {confidence:.1%}",
                (380, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # "bad posture" bar — fills up over 3 seconds
    if label == 1 and bad_seconds > 0:
        bar_w     = 640 - 30
        progress  = min(bad_seconds / alert_threshold, 1.0)
        filled    = int(bar_w * progress)
        cv2.rectangle(frame, (15, 75), (15 + bar_w, 90), (60, 30, 30), -1)
        bar_color = (0, 100, 255) if progress < 0.7 else (0, 0, 220)
        cv2.rectangle(frame, (15, 75), (15 + filled, 90), bar_color, -1)
        cv2.putText(frame, f"Alert in: {max(0, alert_threshold - bad_seconds):.1f}s",
                    (15, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # ALERT when the threshold is exceeded
    if bad_seconds >= alert_threshold:
        cv2.rectangle(frame, (100, 180), (540, 260), (0, 0, 180), -1)
        cv2.rectangle(frame, (100, 180), (540, 260), (0, 0, 255), 3)
        cv2.putText(frame, "FIX YOUR POSTURE!", (120, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # bottom bar
    cv2.rectangle(frame, (0, frame.shape[0] - 55), (640, frame.shape[0]), (30, 30, 30), -1)
    cv2.putText(frame, "G = mark GOOD  |  B = mark BAD  |  Q = quit",
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"Corrections: {corrections}",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    if flash:
        cv2.putText(frame, flash, (180, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def save_correction(features, correct_label, csv_path):
    columns = []
    for name in LANDMARKS_TO_USE.keys():
        columns += [f"{name}_x", f"{name}_y", f"{name}_z"]
    row           = dict(zip(columns, features))
    row["label"]  = correct_label
    df_new        = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)


def main():
    # ── profile selection ─────────────────────────────────────
    profile = select_or_create_profile()
    if profile is None:
        print("Goodbye!")
        return

    # ── calibration (required) ────────────────────────────────
    profile = ensure_calibration(profile)
    if profile is None:
        print("Cannot start without calibration.")
        return

    calibration = profile["calibration"]
    user_name   = profile["name"]

    # ── model ─────────────────────────────────────────────────
    print("\nLoading model...")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    print("✅ Ready!\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    predictions    = []
    SMOOTH_N       = 20
    label          = 0
    confidence     = 1.0
    corrections    = 0
    flash_msg      = ""
    flash_until    = 0.0
    last_features  = None

    # bad posture timer
    bad_since      = None
    bad_seconds    = 0.0

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    ) as pose:

        print(f"📷 Started for: {user_name} | Q = quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

                raw_features  = extract_features(results.pose_landmarks.landmark)
                last_features = raw_features

                # normalize relative to the calibration
                norm_features = normalize_features(raw_features, calibration)

                X = np.array(norm_features).reshape(1, -1)
                if scaler:
                    X = scaler.transform(X)

                pred       = model.predict(X)[0]
                proba      = model.predict_proba(X)[0]
                confidence = proba[pred]

                predictions.append(pred)
                if len(predictions) > SMOOTH_N:
                    predictions.pop(0)
                label = 1 if predictions.count(1) >= int(SMOOTH_N * 0.7) else 0

            # ── bad posture timer ─────────────────────────────
            now = time.time()
            if label == 1:
                if bad_since is None:
                    bad_since = now
                bad_seconds = now - bad_since
            else:
                bad_since   = None
                bad_seconds = 0.0

            # flash message expires
            flash_msg = flash_msg if now < flash_until else ""

            draw_ui(frame, label, confidence,
                    bad_seconds, BAD_POSTURE_ALERT_SECONDS,
                    user_name, corrections, flash_msg)

            cv2.imshow("PostureGuard", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("g"), ord("G")) and last_features:
                save_correction(last_features, 0, CSV_PATH)
                corrections += 1
                flash_msg    = "SAVED: GOOD!"
                flash_until  = now + 1.5

            elif key in (ord("b"), ord("B")) and last_features:
                save_correction(last_features, 1, CSV_PATH)
                corrections += 1
                flash_msg    = "SAVED: BAD!"
                flash_until  = now + 1.5

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nClosed. Corrections saved: {corrections}")
    if corrections > 0:
        print("Remember to retrain the model in Jupyter!")


if __name__ == "__main__":
    main()