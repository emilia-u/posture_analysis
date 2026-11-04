import csv
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SESSION_NAME = "testp5"   # ← change here (testp5, testp6, ...)
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "raw_frames", SESSION_NAME)
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports",    SESSION_NAME)
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

_LABEL_COLOR = {-1: (160, 160, 160), 0: (0, 210, 0), 1: (0, 0, 220)}
_LABEL_TEXT  = {-1: "? unknown  [G]=good  [B]=bad",
                 0: "GOOD posture  [B]=switch",
                 1: "BAD posture   [G]=switch"}


def _wait_for_calibration(cap, window_name):
    """Shows a live preview (nothing is saved) until the user presses SPACE.
    Returns True when ready to record, False if the user cancelled."""
    print("Calibration: get into frame, press [SPACE] to start recording ([Q] = cancel)")
    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        cv2.rectangle(frame, (0, 0), (640, 42), (0, 0, 0), -1)
        cv2.putText(frame, "Calibration - [SPACE]=start  [Q]=cancel", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 210, 255), 2)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


def record_raw_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    window_name = f"RECORDING: {SESSION_NAME}"

    if not _wait_for_calibration(cap, window_name):
        cap.release()
        cv2.destroyAllWindows()
        return

    frame_count   = 0
    current_label = -1
    labels        = []

    print(f"Recording → {OUTPUT_DIR}")
    print("Keys: [G] good posture  [B] bad posture  [U] unknown  [Q] quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            color = _LABEL_COLOR[current_label]
            text  = _LABEL_TEXT[current_label]
            cv2.rectangle(frame, (0, 0), (560, 42), (0, 0, 0), -1)
            cv2.putText(frame, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.putText(frame, f"fr. {frame_count}", (8, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(window_name, frame)

            frame_name = f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, frame_name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            labels.append(current_label)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                current_label = 0
                print(f"  fr. {frame_count:6d} → GOOD (0)")
            elif key == ord('b'):
                current_label = 1
                print(f"  fr. {frame_count:6d} → BAD  (1)")
            elif key == ord('u'):
                current_label = -1
                print(f"  fr. {frame_count:6d} → unknown (-1)")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSaved {frame_count} frames.")

    csv_path = os.path.join(REPORTS_DIR, "wyniki_detekcji.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_num", "ground_truth"])
        for i, label in enumerate(labels):
            writer.writerow([i + 1, label])

    labeled = sum(1 for l in labels if l != -1)
    print(f"Ground truth saved: {csv_path}")
    print(f"Labeled {labeled}/{frame_count} frames "
          f"({labeled / frame_count * 100:.1f}%)" if frame_count else "")
    print(f"\nNext step: python src/core/evaluate_raw_frames.py")


if __name__ == "__main__":
    record_raw_frames()