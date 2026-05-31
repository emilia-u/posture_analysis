"""
Zbiorcze obliczanie rozszerzonych metryk dla wszystkich sesji testowych.
Wczytuje istniejace wyniki_detekcji.csv (nie uruchamia ponownie MediaPipe).

Uruchomienie:
    python src/training/batch_metryki.py
"""

import os, sys
import numpy as np
import pandas as pd

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")

# ── Parametry wygładzania (identyczne jak server.py) ────────────────────────
# Nagrania raw_frames sa ~30 fps (np. laptop_nizej_2: 58031 kl. / 33 min ≈ 29 fps).
# Server.py przetwarza w 10 Hz, ale tutaj oceniamy surowe klatki wideo.
FPS            = 30
SMOOTH_N       = 20
SMOOTH_THRESH  = 0.70
SHORT_THRESH_S = 3.0

# ────────────────────────────────────────────────────────────────────────────

def _runs(arr):
    arr = np.asarray(arr, dtype=int)
    if len(arr) == 0:
        return []
    runs, s, cur = [], 0, arr[0]
    for i in range(1, len(arr)):
        if arr[i] != cur:
            runs.append((int(cur), s, i - 1))
            cur, s = arr[i], i
    runs.append((int(cur), s, len(arr) - 1))
    return runs


def compute_extended(df, fps=FPS, smooth_n=SMOOTH_N,
                     smooth_thresh=SMOOTH_THRESH, short_s=SHORT_THRESH_S):
    df = df.reset_index(drop=True).copy()

    # Wygładzanie
    rolling = df["prediction"].rolling(window=smooth_n, min_periods=1).mean()
    df["pred_s"] = (rolling >= smooth_thresh).astype(int)

    gt   = df["ground_truth"].values
    pred = df["pred_s"].values

    from sklearn.metrics import accuracy_score
    acc_raw    = float(accuracy_score(gt, df["prediction"].values))
    acc_smooth = float(accuracy_score(gt, pred))

    gt_runs   = _runs(gt)
    pred_runs = _runs(pred)
    gt_s      = pd.Series(gt)
    pred_s    = pd.Series(pred)

    # Falszywe alerty / h
    fa = sum(1 for l, s, e in pred_runs
             if l == 1 and gt_s.iloc[s:e+1].mean() < 0.5)
    total_sec  = len(df) / fps
    fa_per_h   = fa / total_sec * 3600 if total_sec > 0 else float("nan")

    # Sredni czas wykrycia
    det_times = []
    for l, s, e in gt_runs:
        if l == 1:
            seg = pred_s.iloc[s:e+1].values
            if seg.any():
                det_times.append(int(np.argmax(seg)) / fps)
    avg_det = float(np.mean(det_times)) if det_times else float("nan")

    # Sredni czas zlej postawy przed alarmem
    bb = []
    for l, s, e in gt_runs:
        if l == 1:
            seg = pred_s.iloc[s:e+1].values
            bb.append(int(np.argmax(seg)) / fps if seg.any() else (e-s+1)/fps)
    avg_bb = float(np.mean(bb)) if bb else float("nan")

    # Procent krotkich falsz. aktywacji
    SHORT_F = int(short_s * fps)
    fp_runs = [(l, s, e) for l, s, e in pred_runs
               if l == 1 and gt_s.iloc[s:e+1].mean() < 0.5]
    short_fp = sum(1 for _, s, e in fp_runs if (e-s+1) < SHORT_F)
    pct_sh   = (short_fp / len(fp_runs) * 100) if fp_runs else 0.0

    return {
        "Klatek":              len(df),
        "Bad %":               round(gt_s.mean() * 100, 1),
        "Acc. raw":            round(acc_raw * 100, 2),
        "Acc. smooth":         round(acc_smooth * 100, 2),
        "FA/h":                round(fa_per_h, 1) if not np.isnan(fa_per_h) else "N/D",
        "Wykrycie [s]":        round(avg_det, 2) if not np.isnan(avg_det) else "N/D",
        "Bad-przed-alarm [s]": round(avg_bb, 2)  if not np.isnan(avg_bb)  else "N/D",
        "% Kr.FA":             round(pct_sh, 1),
    }


# ── Glowna petla ─────────────────────────────────────────────────────────────
rows = []
skipped = []

for d in sorted(os.listdir(REPORTS_DIR)):
    csv_path = os.path.join(REPORTS_DIR, d, "wyniki_detekcji.csv")
    if not os.path.isfile(csv_path):
        skipped.append(d)
        continue
    df = pd.read_csv(csv_path)
    # Wyklucz klatki bez osoby w kadrze (pred=-1) i bez zdefiniowanej etykiety (gt=-1)
    df = df[(df["prediction"] != -1) & (df["ground_truth"] != -1)].reset_index(drop=True)
    if len(df) == 0:
        skipped.append(d)
        continue
    m = compute_extended(df)
    rows.append({"Sesja": d, **m})

if not rows:
    print("Brak danych. Uruchom evaluate.ipynb dla co najmniej jednej sesji.")
    sys.exit(1)

result = pd.DataFrame(rows)

# ── Wyswietlenie tabeli ───────────────────────────────────────────────────────
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 160)

print()
print("=" * 140)
print(f"  ZBIORCZE METRYKI ROZSZERZONE")
print(f"  Parametry: SMOOTH_N={SMOOTH_N}, prog={SMOOTH_THRESH:.0%}, FPS={FPS}")
print("=" * 140)
print(result.to_string(index=False))
print("=" * 140)

# Statystyki podsumowujace (pomijamy kolumny z N/D)
numeric_cols = ["Acc. raw", "Acc. smooth", "% Kr.FA"]
print()
print("Srednia po sesjach (kolumny numeryczne):")
numeric_df = result[numeric_cols].apply(pd.to_numeric, errors="coerce")
for col in numeric_cols:
    print(f"  {col:<22} srednia={numeric_df[col].mean():.2f}  "
          f"min={numeric_df[col].min():.2f}  max={numeric_df[col].max():.2f}")

fa_vals = pd.to_numeric(result["FA/h"], errors="coerce").dropna()
if len(fa_vals):
    print(f"  {'FA/h':<22} srednia={fa_vals.mean():.1f}  "
          f"min={fa_vals.min():.1f}  max={fa_vals.max():.1f}")

det_vals = pd.to_numeric(result["Wykrycie [s]"], errors="coerce").dropna()
if len(det_vals):
    print(f"  {'Wykrycie [s]':<22} srednia={det_vals.mean():.2f}  "
          f"min={det_vals.min():.2f}  max={det_vals.max():.2f}")

# ── Zapis ────────────────────────────────────────────────────────────────────
out_csv = os.path.join(REPORTS_DIR, "zbiorcze_metryki_rozszerzone.csv")
result.to_csv(out_csv, index=False)
print(f"\nZapisano: {out_csv}")

if skipped:
    print(f"\nPominięto (brak CSV): {skipped}")
