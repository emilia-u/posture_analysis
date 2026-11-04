"""
benchmark_startup.py – measures the startup time of the whole application
and total resource usage (Python + Electron) in steady state.

Measures:
  - ML model loading time
  - time until the WebSocket server is ready (port 8765)
  - Electron window startup time
  - time to establish the UI <-> WebSocket connection
  - total CPU and RAM (Python + Electron + all child processes)

Usage:
    python src/training/benchmark_startup.py
"""

import subprocess, sys, os, time, socket, threading
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "reports")
WS_PORT = 8765
MONITOR_DURATION = 45   # seconds of resource monitoring after the connection is established

# ── Thread synchronization ────────────────────────────────────────────────────
_ui_connected_event = threading.Event()
_timestamps = {}          # key → time.perf_counter()

def _monitor_server_stdout(proc):
    """Reads the server stdout and records timestamps of key events."""
    for raw in iter(proc.stdout.readline, b""):
        txt = raw.decode("utf-8", errors="replace").strip()
        if not txt:
            continue
        print(f"    [server] {txt}")
        now = time.perf_counter()
        if "Loading the model" in txt and "model_load_start" not in _timestamps:
            _timestamps["model_load_start"] = now
        if "Opening the WebSocket gates" in txt and "ws_server_ready" not in _timestamps:
            _timestamps["ws_server_ready"] = now
        if "Interface connected" in txt and "ui_connected" not in _timestamps:
            _timestamps["ui_connected"] = now
            _ui_connected_event.set()

def _wait_for_port(port, timeout=30.0):
    """Polls the port every 50 ms; returns once it accepts connections."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=0.1):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    raise TimeoutError(f"Port {port} did not open within {timeout:.0f}s")

def _all_pids(*procs):
    """Returns the set of PIDs of the root processes and all their children."""
    pids = set()
    for p in procs:
        try:
            pp = psutil.Process(p.pid)
            pids.add(pp.pid)
            for child in pp.children(recursive=True):
                pids.add(child.pid)
        except psutil.NoSuchProcess:
            pass
    return pids

# ── START ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("APPLICATION STARTUP BENCHMARK")
print("=" * 60)

t0 = time.perf_counter()
_timestamps["t0"] = t0

# 1. Python server
print("\n[1/4] Starting the Python server (server.py)...")
server_proc = subprocess.Popen(
    [sys.executable, os.path.join(BASE_DIR, "src", "core", "server.py")],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=BASE_DIR
)
threading.Thread(target=_monitor_server_stdout, args=(server_proc,), daemon=True).start()

# 2. Wait for the WebSocket port
print(f"[2/4] Waiting for port {WS_PORT} to be ready...")
try:
    _wait_for_port(WS_PORT, timeout=30)
except TimeoutError as e:
    print(f"ERROR: {e}")
    server_proc.terminate()
    sys.exit(1)
if "ws_port_ready" not in _timestamps:
    _timestamps["ws_port_ready"] = time.perf_counter()
print(f"    → port ready after {_timestamps['ws_port_ready'] - t0:.2f}s")

# 3. Electron
print("[3/4] Starting Electron (npm start)...")
_timestamps["electron_launch"] = time.perf_counter()
electron_proc = subprocess.Popen(
    "npm start", shell=True, cwd=BASE_DIR,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

# 4. Wait for the UI <-> WebSocket connection
print("[4/4] Waiting for the Electron → WebSocket connection (max 30s)...")
_ui_connected_event.wait(timeout=30)
if "ui_connected" not in _timestamps:
    _timestamps["ui_connected"] = time.perf_counter()
    print("    → (timeout – recording the current time)")
else:
    print(f"    → UI connected after {_timestamps['ui_connected'] - t0:.2f}s")

# ── Resource monitoring ───────────────────────────────────────────────────────
print(f"\nMonitoring resources for {MONITOR_DURATION}s...")

# Initialize cpu_percent (the first call always returns 0.0)
time.sleep(0.5)
for pid in _all_pids(server_proc, electron_proc):
    try:
        psutil.Process(pid).cpu_percent(interval=None)
    except psutil.NoSuchProcess:
        pass

cpu_samples, ram_samples = [], []
t_monitor_start = time.perf_counter()

while time.perf_counter() - t_monitor_start < MONITOR_DURATION:
    total_cpu = total_ram = 0.0
    for pid in _all_pids(server_proc, electron_proc):
        try:
            p = psutil.Process(pid)
            total_cpu += p.cpu_percent(interval=None)
            total_ram += p.memory_info().rss / 1024 / 1024
        except psutil.NoSuchProcess:
            pass
    cpu_samples.append(total_cpu)
    ram_samples.append(total_ram)
    time.sleep(0.5)

# ── Shutdown ──────────────────────────────────────────────────────────────────
print("\nClosing the application...")
for p in (electron_proc, server_proc):
    try:
        p.terminate(); p.wait(timeout=5)
    except Exception:
        pass

# ── Text report ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS – STARTUP TIME")
print("=" * 60)

phases = [
    ("Python library imports",            "t0",              "model_load_start"),
    ("ML model loading + WS start",       "model_load_start","ws_port_ready"),
    ("Electron init + connection",        "electron_launch", "ui_connected"),
]

print(f"\n{'Phase':<42} {'Phase time [s]':>13} {'Since start [s]':>13}")
print("-" * 70)
for label, key_start, key_end in phases:
    ts = _timestamps.get(key_start, t0)
    te = _timestamps.get(key_end)
    if te:
        delta = te - ts
        cumul = te - t0
        print(f"{label:<42} {delta:>13.2f} {cumul:>13.2f}")

t_ready = _timestamps.get("ui_connected", t0 + 99) - t0
print(f"\n→ Total time until the UI is ready: {t_ready:.2f}s")

steps = [
    ("ML model loading (start)",       "model_load_start"),
    ("WebSocket server ready (port)",  "ws_port_ready"),
    ("Electron started",               "electron_launch"),
    ("UI connected to WebSocket",      "ui_connected"),
]

WARMUP = min(10, len(cpu_samples))
cpu_arr = np.array(cpu_samples[WARMUP:]) if len(cpu_samples) > WARMUP else np.array(cpu_samples)
ram_arr = np.array(ram_samples[WARMUP:]) if len(ram_samples) > WARMUP else np.array(ram_samples)

cpu_mean = float(np.mean(cpu_arr)) if len(cpu_arr) else 0
ram_mean = float(np.mean(ram_arr)) if len(ram_arr) else 0
ram_max  = float(np.max(ram_samples)) if ram_samples else 0

print(f"\nSteady-state resources (Python + Electron combined):")
print(f"  CPU avg:  {cpu_mean:.1f}%")
print(f"  RAM avg:  {ram_mean:.1f} MB  (max: {ram_max:.1f} MB)")

# ── Chart ─────────────────────────────────────────────────────────────────────
t_axis = np.arange(len(ram_samples)) * 0.5
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(t_axis, ram_samples, color="#3498db", linewidth=1.5)
axes[0].axhline(ram_mean, color="red", linestyle="--",
                label=f"Avg: {ram_mean:.0f} MB")
axes[0].set_xlabel("Time since connection [s]")
axes[0].set_ylabel("RAM [MB]")
axes[0].set_title("RAM usage – combined (Python + Electron)")
axes[0].legend()

axes[1].plot(t_axis, cpu_samples, color="#e74c3c", linewidth=1.5)
axes[1].axhline(cpu_mean, color="navy", linestyle="--",
                label=f"Avg: {cpu_mean:.1f}%")
axes[1].set_xlabel("Time since connection [s]")
axes[1].set_ylabel("CPU [%]")
axes[1].set_title("CPU usage – combined (Python + Electron)")
axes[1].legend()

plt.suptitle(
    f"Whole-application resources  |  Startup time: {t_ready:.1f}s  |  "
    f"RAM avg: {ram_mean:.0f} MB  |  CPU avg: {cpu_mean:.1f}%",
    fontsize=11
)
plt.tight_layout()
os.makedirs(RESULTS_DIR, exist_ok=True)
out_img = os.path.join(RESULTS_DIR, "benchmark_startup_wykres.png")
plt.savefig(out_img, dpi=150, bbox_inches="tight")
print(f"\nChart saved: data/reports/benchmark_startup_wykres.png")

df = pd.DataFrame({"czas_s": t_axis, "cpu_pct": cpu_samples, "ram_mb": ram_samples})
out_csv = os.path.join(RESULTS_DIR, "benchmark_startup_wyniki.csv")
df.to_csv(out_csv, index=False)
print(f"CSV saved:   data/reports/benchmark_startup_wyniki.csv")

import json
startup_record = {s_label: round(_timestamps[s_key] - t0, 3)
                  for s_label, s_key in steps if s_key in _timestamps}
startup_record["ram_srednia_mb"]  = round(ram_mean, 1)
startup_record["ram_max_mb"]      = round(ram_max, 1)
startup_record["cpu_srednie_pct"] = round(cpu_mean, 1)
out_json = os.path.join(RESULTS_DIR, "benchmark_startup_timestamps.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(startup_record, f, ensure_ascii=False, indent=2)
print(f"Timestamps saved: data/reports/benchmark_startup_timestamps.json")
