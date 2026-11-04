import json
import os
import shutil
from datetime import datetime

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILES_DIR  = os.path.join(BASE_DIR, "data", "profiles")


def _ensure_dir():
    os.makedirs(PROFILES_DIR, exist_ok=True)


def get_all_profiles() -> list[dict]:
    """Returns a list of all saved profiles."""
    _ensure_dir()
    profiles = []
    for fname in os.listdir(PROFILES_DIR):
        if fname.endswith(".json"):
            path = os.path.join(PROFILES_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                profiles.append(json.load(f))
    return sorted(profiles, key=lambda p: p.get("last_used", ""), reverse=True)


def get_profile(name: str) -> dict | None:
    """Loads a profile by name. Returns None if it does not exist."""
    _ensure_dir()
    path = os.path.join(PROFILES_DIR, f"{name.lower()}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_profile(profile: dict):
    """Saves the profile to a JSON file."""
    _ensure_dir()
    name = profile["name"].lower()
    path = os.path.join(PROFILES_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def create_profile(name: str, avatar_type: str = "cat") -> dict:
    """Creates a new empty profile (without calibration)."""
    profile = {
        "name": name,
        "created":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_used":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "calibration": None,   # filled in after calibration
        "sensitivity": "medium",
        "avatar_type": avatar_type
    }
    save_profile(profile)
    return profile


def update_last_used(name: str):
    """Updates the profile's last-used date."""
    profile = get_profile(name)
    if profile:
        profile["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_profile(profile)


def save_calibration(name: str, calibration_data: dict):
    """Saves calibration data to the profile."""
    profile = get_profile(name)
    if profile is None:
        raise ValueError(f"Profile '{name}' does not exist!")
    profile["calibration"] = calibration_data
    profile["calibrated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_profile(profile)
    print(f"✅ Calibration saved for profile: {name}")


def has_calibration(name: str) -> bool:
    """Checks whether the profile has a saved calibration."""
    profile = get_profile(name)
    return profile is not None and profile.get("calibration") is not None


def delete_profile(name: str) -> bool:
    """Deletes a profile and its sessions. Returns True on success."""
    profile_name_lower = name.lower()

    # Delete the profile file
    profile_path = os.path.join(PROFILES_DIR, f"{profile_name_lower}.json")
    if os.path.exists(profile_path):
        os.remove(profile_path)

    # Delete the sessions folder
    sessions_folder = os.path.join(SESSIONS_DIR, profile_name_lower)
    if os.path.exists(sessions_folder):
        shutil.rmtree(sessions_folder)
        print(f"✅ Deleted sessions for profile: {name}")

    # Return True if the profile was deleted
    return not os.path.exists(profile_path)


SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sessions")


def save_session(profile_name: str, session_data: dict):
    """Saves session data to a JSON file."""
    profile_sessions_dir = os.path.join(SESSIONS_DIR, profile_name.lower())
    os.makedirs(profile_sessions_dir, exist_ok=True)
    filename = session_data["start_time"].replace(":", "-").replace(" ", "_") + ".json"
    path = os.path.join(profile_sessions_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)


def get_sessions(profile_name: str, limit: int = 200) -> list[dict]:
    """Returns the last N sessions for a profile, newest first."""
    profile_sessions_dir = os.path.join(SESSIONS_DIR, profile_name.lower())
    if not os.path.exists(profile_sessions_dir):
        return []
    sessions = []
    for fname in os.listdir(profile_sessions_dir):
        if fname.endswith(".json"):
            path = os.path.join(profile_sessions_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    sessions.append(json.load(f))
                except Exception:
                    pass
    sessions.sort(key=lambda s: s.get("start_time", ""), reverse=True)
    return sessions[:limit]