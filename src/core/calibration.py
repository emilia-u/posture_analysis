import cv2
import mediapipe as mp
import numpy as np
import time

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

CALIBRATION_SECONDS = 5   # how many seconds we collect samples for


def run_calibration(user_name: str) -> dict | None:
    """
    Opens the calibration window.
    Returns a dict with the averaged baseline coordinates,
    or None if the user cancelled.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    samples       = []   # collected samples
    collecting    = False
    start_time    = None
    result        = None

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1
    ) as pose:

        print(f"\n🎯 Calibration for: {user_name}")
        print("Sit up straight and press SPACE to start...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # shoulder visibility
            visibility = 0.0
            if results.pose_landmarks:
                lm        = results.pose_landmarks.landmark
                left_vis  = lm[11].visibility
                right_vis = lm[12].visibility
                visibility = (left_vis + right_vis) / 2.0

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

                # collecting samples
                if collecting and visibility > 0.6:
                    sample = {}
                    for name, idx in LANDMARKS_TO_USE.items():
                        p = lm[idx]
                        sample[f"{name}_x"] = p.x
                        sample[f"{name}_y"] = p.y
                        sample[f"{name}_z"] = p.z
                    samples.append(sample)

            # check whether the calibration time has elapsed
            if collecting and start_time:
                elapsed  = time.time() - start_time
                progress = min(elapsed / CALIBRATION_SECONDS, 1.0)

                if elapsed >= CALIBRATION_SECONDS:
                    # finish calibration
                    if len(samples) >= 10:
                        result = _average_samples(samples)
                        print(f"✅ Collected {len(samples)} calibration samples!")
                    else:
                        print("❌ Not enough samples — try again!")
                        samples    = []
                        collecting = False
                        start_time = None
                        progress   = 0.0
                    if result:
                        break
            else:
                progress = 0.0

            # draw the calibration UI
            _draw_calibration_ui(frame, user_name, collecting, progress, visibility)
            cv2.imshow("PostureGuard - Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and not collecting and visibility > 0.6:
                collecting = True
                start_time = time.time()
                samples    = []
                print("▶ Collecting samples...")
            elif key == ord("q"):
                print("Calibration cancelled.")
                break

    cap.release()
    cv2.destroyAllWindows()
    return result


def _average_samples(samples: list[dict]) -> dict:
    """Averages the collected calibration samples."""
    keys   = samples[0].keys()
    result = {}
    for key in keys:
        values      = [s[key] for s in samples]
        result[key] = float(np.mean(values))
    return result


def _draw_calibration_ui(frame, user_name: str, collecting: bool,
                          progress: float, visibility: float):
    h, w = frame.shape[:2]

    # Semi-transparent background for the top bar (gradient effect)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (25, 20, 35), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Neon border at the top
    cv2.rectangle(frame, (0, 0), (w, 3), (120, 255, 0), -1)

    # Title with emoji and shadow
    title_text = f"🎯 Profile Calibration: {user_name}"
    cv2.putText(frame, title_text, (22, 52), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 3)  # shadow
    cv2.putText(frame, title_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)  # text

    # Shoulder visibility with an icon
    vis_color = (0, 255, 120) if visibility > 0.7 else (0, 165, 255) if visibility > 0.4 else (0, 80, 220)
    vis_icon = "✓" if visibility > 0.7 else "⚠"
    vis_text = f"{vis_icon} Detection: {'Excellent!' if visibility > 0.7 else 'Adjust your position'}"

    # Background for the visibility status
    status_bg = overlay.copy()
    cv2.rectangle(status_bg, (10, 100), (w - 10, 135), (20, 20, 30), -1)
    cv2.addWeighted(status_bg, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (10, 100), (w - 10, 135), vis_color, 2, cv2.LINE_AA)

    cv2.putText(frame, vis_text, (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, vis_color, 2, cv2.LINE_AA)

    if not collecting:
        # Nicely formatted instructions
        instr_bg = overlay.copy()
        cv2.rectangle(instr_bg, (10, 155), (w - 10, 230), (20, 20, 30), -1)
        cv2.addWeighted(instr_bg, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "📸 Sit up STRAIGHT and look at the camera",
                    (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "⌨  Press SPACE to start",
                    (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)
    else:
        # Progress bar with a gradient
        bar_x, bar_y, bar_w, bar_h = 20, 160, w - 40, 40

        # Bar background
        prog_bg = overlay.copy()
        cv2.rectangle(prog_bg, (bar_x - 5, bar_y - 5), (bar_x + bar_w + 5, bar_y + bar_h + 5), (20, 20, 30), -1)
        cv2.addWeighted(prog_bg, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 120), 2, cv2.LINE_AA)

        # Bar fill with a green gradient
        filled = int(bar_w * progress)
        if filled > 0:
            # Gradient from yellow-green to green
            color = (0, int(180 + 75 * progress), int(220 - 100 * progress))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 255, 150), 2, cv2.LINE_AA)

        # Text on the bar
        secs_left = CALIBRATION_SECONDS - int(progress * CALIBRATION_SECONDS)
        percent_text = f"{int(progress * 100)}%"
        cv2.putText(frame, percent_text, (bar_x + bar_w // 2 - 30, bar_y + bar_h // 2 + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"⏱ Hold still... {secs_left}s",
                    (bar_x, bar_y + bar_h + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2, cv2.LINE_AA)

    # Bottom bar with instructions
    bottom_overlay = overlay.copy()
    cv2.rectangle(bottom_overlay, (0, h - 45), (w, h), (25, 20, 35), -1)
    cv2.addWeighted(bottom_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (0, h - 45), (w, h - 42), (120, 255, 0), -1)

    cv2.putText(frame, "❌ Press Q to cancel",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 200), 1, cv2.LINE_AA)