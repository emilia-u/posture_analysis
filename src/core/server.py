import sys
import os
import time
import cv2
import joblib
import numpy as np
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.core.normalizer import normalize_features
from datetime import datetime
from src.core.profile_manager import save_session, get_sessions, delete_profile

MODEL_PATH = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
PROFILES_DIR = os.path.join(BASE_DIR, "data", "profiles")

LANDMARKS_TO_USE = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12
}


def extract_features(landmarks):
    features = []
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])
    return features


# Load the model trained in Jupyter right at startup
print("\n[AI] Loading the model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


async def posture_server(websocket):
    print("\n[SERVER] Interface connected!")


    def get_profiles_list():
        profiles = []
        if os.path.exists(PROFILES_DIR):
            for f in glob.glob(os.path.join(PROFILES_DIR, "*.json")):
                profiles.append(os.path.basename(f).replace(".json", ""))
        return profiles

    # MAIN APPLICATION SESSION LOOP
    while True:
        print("[SERVER] Sending the profile list and waiting for a selection...")
        try:
            await websocket.send(json.dumps({"type": "profiles_list", "data": get_profiles_list()}))
        except websockets.exceptions.ConnectionClosed:
            return

        profile_name = None

        # Wait for a user action (select a profile or create a new one)
        while True:
            try:
                msg = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                print("\n[SERVER] User interface disconnected.")
                return

            client_data = json.loads(msg)
            # Handle profile DELETION
            if client_data.get("type") == "delete_profile":
                del_name = client_data.get("name")
                if del_name:
                    if delete_profile(del_name):
                        print(f"\n[SERVER] Deleted profile from disk: {del_name}")

                # After deleting, refresh the list in the interface
                await websocket.send(json.dumps({"type": "profiles_list", "data": get_profiles_list()}))
                continue
            # Handle creating a new profile
            if client_data.get("type") == "create_profile":
                new_profile_name = client_data.get("name", "").strip()
                avatar_type = client_data.get("avatar_type", "cat")
                if new_profile_name:
                    print(f"\n[SERVER] Starting calibration for profile: {new_profile_name}")
                    cap = cv2.VideoCapture(0)
                    mp_pose = mp.solutions.pose
                    mp_drawing = mp.solutions.drawing_utils

                    samples = []
                    collecting = False
                    start_time = None
                    calibration_data = None

                    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1) as pose:
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                await asyncio.sleep(0.01)
                                continue

                            frame = cv2.flip(frame, 1)
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(rgb)

                            visibility = 0.0
                            progress = 0.0

                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                                )

                                lm = results.pose_landmarks.landmark
                                visibility = (lm[11].visibility + lm[12].visibility) / 2.0

                                if collecting and visibility > 0.6:
                                    sample = {}
                                    for name, idx in LANDMARKS_TO_USE.items():
                                        p = lm[idx]
                                        sample[f"{name}_x"] = p.x
                                        sample[f"{name}_y"] = p.y
                                        sample[f"{name}_z"] = p.z
                                    samples.append(sample)

                            if collecting and start_time:
                                elapsed = time.time() - start_time
                                progress = min(elapsed / 5.0, 1.0)

                                if elapsed >= 5.0:
                                    if len(samples) >= 10:
                                        keys = samples[0].keys()
                                        calibration_data = {}
                                        for key in keys:
                                            values = [s[key] for s in samples]
                                            calibration_data[key] = float(np.mean(values))
                                        print(f"✅ Collected {len(samples)} calibration samples!")
                                    else:
                                        print("❌ Not enough samples!")
                                        samples = []
                                        collecting = False
                                        start_time = None

                                    if calibration_data:
                                        break

                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')

                            send_data = {
                                "type": "calibration_frame",
                                "image": frame_base64,
                                "visibility": float(visibility),
                                "collecting": collecting,
                                "progress": float(progress)
                            }

                            try:
                                await websocket.send(json.dumps(send_data))
                                await asyncio.sleep(0.03)

                                try:
                                    cmd_msg = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                                    cmd = json.loads(cmd_msg)
                                    if cmd.get("type") == "start_calibration" and not collecting:
                                        collecting = True
                                        start_time = time.time()
                                        samples = []
                                        print("▶ Started collecting samples...")
                                    elif cmd.get("type") == "cancel_calibration":
                                        print("❌ Calibration cancelled")
                                        cap.release()
                                        await websocket.send(json.dumps({"type": "profiles_list", "data": get_profiles_list()}))
                                        break
                                except asyncio.TimeoutError:
                                    pass

                            except websockets.exceptions.ConnectionClosed:
                                cap.release()
                                return

                    cap.release()

                    if calibration_data:
                        if not os.path.exists(PROFILES_DIR):
                            os.makedirs(PROFILES_DIR)

                        profile_path = os.path.join(PROFILES_DIR, f"{new_profile_name}.json")
                        profile_content = {
                            "name": new_profile_name,
                            "calibration": calibration_data,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "avatar_type": avatar_type
                        }

                        with open(profile_path, "w", encoding="utf-8") as f:
                            json.dump(profile_content, f, indent=2, ensure_ascii=False)

                        print(f"✅ Profile '{new_profile_name}' saved!")
                        await websocket.send(json.dumps({"type": "profile_created", "name": new_profile_name}))
                continue

            if client_data.get("type") == "get_profiles":
                await websocket.send(json.dumps({"type": "profiles_list", "data": get_profiles_list()}))
                continue

            if client_data.get("type") == "select_profile":
                profile_name = client_data["name"]
                break

        # CONTINUE WITH THE SELECTED PROFILE
        profile_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        calibration = profile_data.get("calibration")
        if not calibration:
            print(f"[ERROR] Profile {profile_name} has no calibration!")
            continue

        STRETCH_LIMIT = 600    # 10 minutes of bad posture → alert
        RECOVERY_LIMIT = 120   # 2 minutes of good posture → reset the counter
        print(f"\n[SERVER] Loaded profile '{profile_name}'. Starting the camera!")

        cap = cv2.VideoCapture(0)
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        predictions = []
        SMOOTH_N = 20
        INFERENCE_INTERVAL = 0.2  # 5 Hz
        bad_since = None
        bad_seconds = 0.0

        total_bad_seconds = 0.0   # reset after every alert (trigger)
        session_total_bad = 0.0   # total bad-posture time in the session (saved)
        warning_sent = False

        last_time = time.time()
        last_inference_time = 0.0
        last_label = 0
        last_confidence = 1.0

        good_since = None
        good_seconds = 0.0

        session_active = True
        session_start = datetime.now()
        session_events = []
        session_alerts = 0
        total_good_seconds = 0.0
        last_label_for_events = -1

        await websocket.send(json.dumps({
            "type": "session_config",
            "stretch_limit": STRETCH_LIMIT,
            "avatar_type": profile_data.get("avatar_type", "cat")
        }))

        with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
            while session_active:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                label = last_label
                confidence = last_confidence
                person_visible = bool(results.pose_landmarks)

                now = time.time()
                dt = now - last_time
                last_time = now

                if not person_visible:
                    bad_since = None
                    bad_seconds = 0.0
                    good_since = None
                    good_seconds = 0.0

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )
                    if now - last_inference_time >= INFERENCE_INTERVAL:
                        last_inference_time = now
                        raw_features = extract_features(results.pose_landmarks.landmark)
                        norm_features = normalize_features(raw_features, calibration)

                        X = np.array(norm_features).reshape(1, -1)
                        if scaler: X = scaler.transform(X)

                        pred = model.predict(X)[0]
                        proba = model.predict_proba(X)[0]
                        last_confidence = float(proba[pred])

                        predictions.append(pred)
                        if len(predictions) > SMOOTH_N: predictions.pop(0)
                        last_label = 1 if predictions.count(1) >= int(SMOOTH_N * 0.7) else 0

                    label = last_label
                    confidence = last_confidence

                    if label == 1:  # --- BAD POSTURE ---
                        good_since = None
                        good_seconds = 0.0

                        if bad_since is None:
                            bad_since = now
                        bad_seconds = now - bad_since

                        if bad_seconds >= 3.0:
                            total_bad_seconds += dt
                            session_total_bad += dt

                    else:  # --- GOOD POSTURE ---
                        bad_since = None
                        bad_seconds = 0.0

                        total_good_seconds += dt

                        if good_since is None:
                            good_since = now
                        good_seconds = now - good_since

                        if good_seconds >= RECOVERY_LIMIT:
                            if total_bad_seconds > 0:
                                print(
                                    f"\n[SERVER] 🌿 Posture recovered! Resetting the fatigue counter from {total_bad_seconds:.1f}s to 0.")
                                total_bad_seconds = 0.0
                                warning_sent = False
                            good_since = now

                    # Track posture state changes (events for statistics)
                    if label != last_label_for_events:
                        session_events.append({
                            "state": "bad" if label == 1 else "good",
                            "time": datetime.now().strftime("%H:%M:%S")
                        })
                        last_label_for_events = label

                    if bad_seconds >= 3.0:
                        print(f"[ALERT] You are ignoring the cat! Accumulated: {total_bad_seconds:.1f}s / {STRETCH_LIMIT}s")

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                send_data = {
                    "type": "video_frame",
                    "label": int(label),
                    "visible": person_visible,
                    "confidence": confidence,
                    "bad_seconds": float(bad_seconds),
                    "total_bad_seconds": float(total_bad_seconds),
                    "total_good_seconds": float(total_good_seconds),
                    "stretch_progress": round(total_bad_seconds / STRETCH_LIMIT, 3),
                    "image": frame_base64
                }

                try:
                    await websocket.send(json.dumps(send_data))


                    # --- ALERT (100% of the limit) ---
                    if total_bad_seconds >= STRETCH_LIMIT:
                        print("\n[SERVER] Bad posture limit reached! Sending a stretch alert.")
                        await websocket.send(json.dumps({"type": "stretch_alert"}))
                        session_alerts += 1
                        total_bad_seconds = 0.0
                        warning_sent = False

                    await asyncio.sleep(0.03)

                    # LISTEN FOR THE "CHANGE PROFILE" OR "STRETCH DONE" COMMAND
                    try:
                        cmd_msg = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                        cmd = json.loads(cmd_msg)
                        if cmd.get("type") == "stop_session":
                            print("\n[SERVER] Camera session stopped. Returning to the menu.")
                            session_duration = (datetime.now() - session_start).total_seconds()
                            if session_duration >= 600:
                                save_session(profile_name, {
                                    "date": session_start.strftime("%Y-%m-%d"),
                                    "start_time": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "good_seconds": round(total_good_seconds, 1),
                                    "bad_seconds": round(session_total_bad, 1),
                                    "alerts": session_alerts,
                                    "events": session_events
                                })
                                print(f"[SERVER] Session saved for profile: {profile_name}")
                            session_active = False
                        elif cmd.get("type") == "stretch_done":
                            print("\n[SERVER] User finished the stretch. Resuming analysis.")
                            last_time = time.time()  # Reset the timer after the pause for the stretch screen
                            bad_since = None
                        elif cmd.get("type") == "get_stats":
                            sessions = get_sessions(profile_name)
                            await websocket.send(json.dumps({"type": "stats_data", "sessions": sessions}))
                    except asyncio.TimeoutError:
                        pass


                except websockets.exceptions.ConnectionClosed:
                    print("\n[SERVER] Application window closed. Python is ready for the next connection.")
                    session_duration = (datetime.now() - session_start).total_seconds()
                    if session_duration >= 600:
                        save_session(profile_name, {
                            "date": session_start.strftime("%Y-%m-%d"),
                            "start_time": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "good_seconds": round(total_good_seconds, 1),
                            "bad_seconds": round(session_total_bad, 1),
                            "alerts": session_alerts,
                            "events": session_events
                        })
                        print(f"[SERVER] Session saved (window closed): {profile_name}")
                    cap.release()
                    return

        cap.release()
        # After leaving the camera loop the script returns to the start of the "MAIN SESSION LOOP"


async def main():
    print("\n🚀 Opening the WebSocket gates (waiting for Electron)...")
    async with websockets.serve(posture_server, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())