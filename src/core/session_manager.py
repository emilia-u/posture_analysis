import cv2
import os
from src.core.profile_manager import (
    get_all_profiles, get_profile, create_profile,
    save_calibration, has_calibration, update_last_used
)
from src.core.calibration import run_calibration


def select_or_create_profile() -> dict | None:
    """
    Shows the profile selection screen in an OpenCV window.
    Returns the selected/created profile, or None if cancelled.
    """
    while True:
        profiles = get_all_profiles()

        if not profiles:
            print("\n👋 No profiles — creating the first profile...")
            return _create_new_profile_flow()

        # show the selection screen
        choice = _show_profile_selection(profiles)

        if choice == "new":
            result = _create_new_profile_flow()
            if result:
                return result

        elif choice == "quit":
            return None

        elif choice is not None:
            # an existing profile was selected
            profile = get_profile(choice)
            update_last_used(choice)
            print(f"\n✅ Logged in as: {profile['name']}")
            return profile


def ensure_calibration(profile: dict) -> dict | None:
    """
    Checks whether the profile has a calibration.
    If not, forces a calibration.
    Returns the updated profile, or None if cancelled.
    """
    if has_calibration(profile["name"]):
        print(f"✅ Calibration OK for: {profile['name']}")
        return profile

    print(f"\n⚠️  Profile '{profile['name']}' has no calibration!")
    print("Calibration is required before using the application.")

    calibration_data = run_calibration(profile["name"])

    if calibration_data is None:
        print("❌ Calibration cancelled — cannot start the application.")
        return None

    save_calibration(profile["name"], calibration_data)
    profile["calibration"] = calibration_data
    return profile


def _show_profile_selection(profiles: list[dict]) -> str | None:
    """
    Draws the profile selection screen in an OpenCV window.
    Returns the selected profile name, 'new', or 'quit'.
    """
    WINDOW = "PostureGuard - Select profile"
    W, H   = 640, 480

    selected = None

    while selected is None:
        frame = _draw_profile_screen(profiles, W, H)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(30) & 0xFF

        # keys 1-9 select a profile
        for i, profile in enumerate(profiles[:9]):
            if key == ord(str(i + 1)):
                selected = profile["name"]

        if key == ord("n") or key == ord("N"):
            selected = "new"
        elif key == ord("q") or key == ord("Q"):
            selected = "quit"

    cv2.destroyWindow(WINDOW)
    return selected


def _draw_profile_screen(profiles: list[dict], W: int, H: int):
    """Draws the profile selection screen."""
    frame = __import__("numpy").zeros((H, W, 3), dtype=__import__("numpy").uint8)
    frame[:] = (25, 25, 35)  # dark background

    # title
    cv2.putText(frame, "PostureGuard", (180, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
    cv2.putText(frame, "Select a profile:", (220, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)

    # profile list
    for i, profile in enumerate(profiles[:9]):
        y         = 160 + i * 50
        name      = profile["name"]
        last_used = profile.get("last_used", "")[:10]
        has_cal   = "✓" if profile.get("calibration") else "!"
        cal_color = (0, 200, 0) if profile.get("calibration") else (0, 100, 220)

        # row background
        cv2.rectangle(frame, (40, y - 28), (W - 40, y + 12), (40, 40, 55), -1)
        cv2.rectangle(frame, (40, y - 28), (W - 40, y + 12), (60, 60, 80), 1)

        cv2.putText(frame, f"{i + 1}.", (55, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 150), 1)
        cv2.putText(frame, name, (90, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"last used: {last_used}", (300, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        cv2.putText(frame, f"[{has_cal}]", (580, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, cal_color, 2)

    # bottom options
    y_bottom = H - 80
    cv2.line(frame, (40, y_bottom), (W - 40, y_bottom), (60, 60, 80), 1)
    cv2.putText(frame, "N = New profile",
                (55, y_bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)
    cv2.putText(frame, "Q = Quit",
                (380, y_bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)
    cv2.putText(frame, "[✓] = calibrated   [!] = needs calibration",
                (55, y_bottom + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    return frame


def _create_new_profile_flow() -> dict | None:
    """Flow for creating a new profile via the terminal."""
    print("\n" + "="*40)
    print("  NEW PROFILE")
    print("="*40)
    name = input("Enter your name (or Q to cancel): ").strip()

    if name.lower() == "q" or not name:
        return None

    # check whether the profile already exists
    existing = get_profile(name)
    if existing:
        print(f"⚠️  Profile '{name}' already exists!")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            return None

    profile = create_profile(name)
    print(f"✅ Created profile: {name}")
    return profile