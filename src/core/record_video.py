import csv
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── KONFIGURACJA ──────────────────────────────────────────────────────────────
SESSION_NAME = "testp5"   # ← zmień tutaj (testp5, testp6, ...)
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "raw_frames", SESSION_NAME)
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports",    SESSION_NAME)
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

_LABEL_COLOR = {-1: (160, 160, 160), 0: (0, 210, 0), 1: (0, 0, 220)}
_LABEL_TEXT  = {-1: "? nieznana  [G]=dobra  [B]=zla",
                 0: "DOBRA postura  [B]=zmien",
                 1: "ZLA postura    [G]=zmien"}


def _wait_for_calibration(cap, window_name):
    """Pokazuje podgląd na żywo (nic nie jest zapisywane) dopóki user nie naciśnie SPACJI.
    Zwraca True gdy gotowe do nagrywania, False gdy użytkownik anulował."""
    print("Kalibracja: ustaw się w kadrze, naciśnij [SPACJA] aby rozpocząć nagrywanie ([Q] = anuluj)")
    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        cv2.rectangle(frame, (0, 0), (640, 42), (0, 0, 0), -1)
        cv2.putText(frame, "Kalibracja - [SPACJA]=start  [Q]=anuluj", (8, 30),
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

    window_name = f"NAGRYWANIE: {SESSION_NAME}"

    if not _wait_for_calibration(cap, window_name):
        cap.release()
        cv2.destroyAllWindows()
        return

    frame_count   = 0
    current_label = -1
    labels        = []

    print(f"Nagrywanie → {OUTPUT_DIR}")
    print("Klawisze: [G] dobra postura  [B] zła postura  [U] nieznana  [Q] koniec\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            color = _LABEL_COLOR[current_label]
            text  = _LABEL_TEXT[current_label]
            cv2.rectangle(frame, (0, 0), (560, 42), (0, 0, 0), -1)
            cv2.putText(frame, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.putText(frame, f"kl. {frame_count}", (8, 70),
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
                print(f"  kl. {frame_count:6d} → DOBRA (0)")
            elif key == ord('b'):
                current_label = 1
                print(f"  kl. {frame_count:6d} → ZŁA  (1)")
            elif key == ord('u'):
                current_label = -1
                print(f"  kl. {frame_count:6d} → nieznana (-1)")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nZapisano {frame_count} klatek.")

    csv_path = os.path.join(REPORTS_DIR, "wyniki_detekcji.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_num", "ground_truth"])
        for i, label in enumerate(labels):
            writer.writerow([i + 1, label])

    labeled = sum(1 for l in labels if l != -1)
    print(f"Ground truth zapisany: {csv_path}")
    print(f"Oznaczono {labeled}/{frame_count} klatek "
          f"({labeled / frame_count * 100:.1f}%)" if frame_count else "")
    print(f"\nNastępny krok: python src/core/evaluate_raw_frames.py")


if __name__ == "__main__":
    record_raw_frames()