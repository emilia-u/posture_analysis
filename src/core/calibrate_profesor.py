import os
import cv2
import numpy as np
import json
import mediapipe as mp
import time

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = r"C:\Users\emilk\Downloads\profesor.mp4"
PROFILE_NAME = "profesor"
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Make sure the profiles path is correct for your folder structure
PROFILES_DIR = os.path.join(BASE_DIR, "data", "profiles")

LANDMARKS_TO_USE = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12
}

def main():
    print(f"--- STARTING CALIBRATION FROM THE PROFESSOR'S VIDEO ---")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open the video.")
        return

    mp_pose = mp.solutions.pose
    samples = []

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1) as pose:
        # Take 20 frames from the first seconds of the video (where the professor sits upright)
        while len(samples) < 20:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                visibility = (lm[11].visibility + lm[12].visibility) / 2.0

                if visibility > 0.6:
                    sample = {}
                    for name, idx in LANDMARKS_TO_USE.items():
                        p = lm[idx]
                        sample[f"{name}_x"] = p.x
                        sample[f"{name}_y"] = p.y
                        sample[f"{name}_z"] = p.z
                    samples.append(sample)
                    print(f"Captured reference frame: {len(samples)}/20")

    cap.release()

    if len(samples) == 20:
        calibration_data = {}
        keys = samples[0].keys()
        for key in keys:
            values = [s[key] for s in samples]
            calibration_data[key] = float(np.mean(values))

        os.makedirs(PROFILES_DIR, exist_ok=True)
        profile_path = os.path.join(PROFILES_DIR, f"{PROFILE_NAME}.json")

        profile_content = {
            "name": PROFILE_NAME,
            "calibration": calibration_data,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_content, f, indent=2, ensure_ascii=False)

        print(f"\n✅ SUCCESS! Created calibration profile for: {PROFILE_NAME}")
    else:
        print("\n❌ Error: Could not collect samples. Check that the person is visible at the start of the video.")

if __name__ == "__main__":
    main()