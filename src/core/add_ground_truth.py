"""
Tool for adding ground truth based on timestamps.

Fill in the GROUND_TRUTH dict in the CONFIGURATION section.
Segment format: ("HH:MM:SS", "HH:MM:SS", label)
  label: 0 = good posture, 1 = bad posture

Usage:
  python src/core/add_ground_truth.py
"""

import os
import cv2
import pandas as pd

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR     = os.path.join(BASE_DIR, "data", "reports")
TEST_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "test_frames")


def parse_time(t: str) -> float:
    """'HH:MM:SS' or 'MM:SS' → seconds (float)."""
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def get_fps(video_name: str) -> float:
    path = os.path.join(TEST_VIDEOS_DIR, f"{video_name}.mp4")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0
        print(f"  WARNING: could not read FPS from {path}, assuming {fps}")
    return fps


def apply_ground_truth(video_name: str, segments: list[tuple[str, str, int]]):
    csv_path = os.path.join(REPORTS_DIR, video_name, "wyniki_detekcji.csv")
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found – run evaluate_external_test.py first")
        return

    fps = get_fps(video_name)
    print(f"  Video FPS: {fps:.2f}")

    df = pd.read_csv(csv_path)
    if "ground_truth" not in df.columns:
        df["ground_truth"] = -1

    for (t_from, t_to, label) in segments:
        s_from = parse_time(t_from)
        s_to   = parse_time(t_to)
        # frame_num is 1-indexed, time of frame N = N/fps
        frm_from = max(1, int(s_from * fps) + 1)
        frm_to   = int(s_to * fps)
        mask = (df["frame_num"] >= frm_from) & (df["frame_num"] <= frm_to)
        df.loc[mask, "ground_truth"] = label
        count = mask.sum()
        print(f"  {t_from}–{t_to} → label={label}  "
              f"(fr. {frm_from}–{frm_to}, {count} frames)")

    df.to_csv(csv_path, index=False)
    labeled = (df["ground_truth"] != -1).sum()
    total   = len(df)
    print(f"  Saved. Labeled {labeled}/{total} frames ({labeled/total*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# Format: ("HH:MM:SS", "HH:MM:SS", label)
#   0 = good posture,  1 = bad posture
# ═══════════════════════════════════════════════════════════════════
GROUND_TRUTH = {
    "testp1": [
        ("00:00:00", "00:03:05", -1),
        ("00:03:05", "00:07:50",  0),
        ("00:07:50", "00:10:16",  1),
        ("00:10:16", "00:11:34",  0),
        ("00:11:34", "00:21:27",  1),
        ("00:21:27", "00:24:10",  0),
        ("00:24:10", "00:25:04",  1),
    ],
    "testp2": [
        ("00:00:00", "00:03:10", -1),
        ("00:03:10", "00:06:16",  0),
        ("00:06:16", "00:11:35",  1),
        ("00:11:35", "00:15:00",  0),
    ],
    "testp3": [
        ("00:00:00", "00:02:05", -1),
        ("00:02:05", "00:02:40",  0),
        ("00:02:40", "00:03:18",  1),
        ("00:03:18", "00:06:21",  0),
        ("00:06:21", "00:07:54",  1),
        ("00:07:54", "00:09:26",  0),
        ("00:09:26", "00:10:10",  1),
        ("00:10:10", "00:10:30",  0),
        ("00:10:30", "00:11:00",  1),
        ("00:11:00", "00:11:20",  0),
        ("00:11:20", "00:12:21",  1),
    ],
    "testp4": [
        ("00:00:00", "00:01:25", -1),   
        ("00:01:25", "00:09:30",  0),
        ("00:09:30", "00:10:55",  1),
        ("00:10:55", "00:13:10",  0),
        ("00:13:10", "00:17:30",  1),
        ("00:17:30", "00:18:40",  0),
        ("00:18:40", "00:19:27",  1),
        ("00:19:27", "00:20:00",  0),
    ],
    "testp6": [
        ("00:00:00", "00:00:32", 0),
        ("00:00:32", "00:00:47", 1),
        ("00:00:47", "00:01:16", 0),
        ("00:01:16", "00:01:25", 1),
    ],
}
# ═══════════════════════════════════════════════════════════════════


def main():
    configured = {k: v for k, v in GROUND_TRUTH.items() if v}
    if not configured:
        print("Fill in GROUND_TRUTH before running.")
        return

    for video_name, segments in configured.items():
        print(f"\n{video_name}:")
        apply_ground_truth(video_name, segments)

    print("\nDone! Now run: python src/training/batch_metryki.py")


if __name__ == "__main__":
    main()
