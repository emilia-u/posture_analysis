import subprocess
import sys
import time

def main():
    print("🚀 Starting the Cat Posture Guardian...")

    # 1. Start the AI server in the background
    # We use sys.executable to make sure we use the active environment (.venv)
    print("🧠 Starting the AI server...")
    server_process = subprocess.Popen([sys.executable, "src/core/server.py"])

    # Give the server a few seconds to load the ML model
    time.sleep(5)

    # 2. Start the Electron interface
    print("🎨 Starting the graphical interface...")
    # shell=True is required on Windows so that the 'npm' command is recognised
    electron_process = subprocess.Popen("npm start", shell=True)

    try:
        # This blocks the script and waits until the user closes the Electron window
        electron_process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        # 3. Cleanup: when you close the app, we shut down the background server
        print("\n🛑 Closing window. Shutting down the AI server...")
        server_process.terminate()
        server_process.wait()
        print("✨ Application closed cleanly. Goodbye!")

if __name__ == "__main__":
    main()