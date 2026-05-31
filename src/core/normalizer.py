import numpy as np

LANDMARKS_TO_USE = [
    "nose", "left_shoulder", "right_shoulder",
]


def normalize_features(features: list[float], calibration: dict) -> list[float]:
    """
    Normalizuje wektor cech względem danych kalibracyjnych.
    Zamiast surowych współrzędnych model dostaje odchylenia od normy.
    """
    normalized = []
    for name in LANDMARKS_TO_USE:
        for axis in ["x", "y", "z"]:
            key       = f"{name}_{axis}"
            raw_val   = features[len(normalized)]
            base_val  = calibration.get(key, 0.0)
            normalized.append(raw_val - base_val)
    return normalized

