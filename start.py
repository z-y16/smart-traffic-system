"""
start.py — run the whole system with one command
================================================

The system is two programs: the **CV node** (`broadcast_server.py`), which
watches the road and drives the hardware, and the **dashboard**
(`SmartTrafficSystem/`), which displays and controls it. They are separate
because they do not have to run on the same machine — the node needs the camera
and the GPU, the dashboard only needs a browser.

Most of the time they do run on the same machine, and then starting them by
hand means two terminals, a directory change, and remembering to tell the
dashboard where the node is. This does all of that:

    python start.py

and stops both cleanly with a single Ctrl-C, so the workbooks are written and
the serial port is released.

Anything it does not recognise is passed straight through to the CV node, so
every option in ``broadcast_server.py --help`` still works:

    python start.py --source road.mp4          # start on a recorded video
    python start.py --source rtsp://...        # start on a roadside camera
    python start.py --no-arduino --no-display  # no hardware, no preview window
    python start.py --imgsz 1280               # slower, better at distance

    python start.py --no-dashboard             # just the CV node
    python start.py --no-server                # just the dashboard (node is elsewhere)
    python start.py --node 192.168.1.25        # dashboard for a node on another PC
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CV_NODE = ROOT_DIR / "broadcast_server.py"
DASHBOARD = ROOT_DIR / "SmartTrafficSystem" / "app.py"

#: Windows needs children in their own process group to be sent a Ctrl-C
#: deliberately rather than receiving ours by accident.
NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

#: How long to let each program shut itself down before insisting.
SHUTDOWN_GRACE_SECONDS = 20.0


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Split our own options from the ones meant for the CV node."""
    parser = argparse.ArgumentParser(
        description="Start the Smart Traffic CV node and dashboard together.",
        epilog="Any other option is passed through to broadcast_server.py.",
    )
    parser.add_argument("--stream-port", type=int, default=8502,
                        help="port the CV node serves video and telemetry on")
    parser.add_argument("--dashboard-port", type=int, default=8501,
                        help="port the dashboard is served on")
    parser.add_argument("--node", default="127.0.0.1",
                        help="address of the CV node the dashboard should talk to; "
                             "change this only when the node runs on another machine")
    parser.add_argument("--max-upload-mb", type=int, default=2000,
                        help="largest video that may be uploaded; raises the limit on "
                             "both the dashboard (Streamlit's own default is a rather "
                             "small 200) and the CV node")
    parser.add_argument("--no-server", action="store_true",
                        help="do not start the CV node (it is already running, or elsewhere)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not start the dashboard")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    parser.add_argument("--no-logs", action="store_true",
                        help="do not open the folder holding the vehicle logs")
    return parser.parse_known_args(argv)


def node_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    """Build the command line for the CV node.

    ``--max-upload-mb`` is the one option both programs want, and they mean the
    same thing by it — the largest video a person may hand the system. The
    number is therefore given to both, rather than the launcher quietly
    keeping it and leaving the node on its own default.
    """
    return [sys.executable, str(CV_NODE),
            "--stream-port", str(args.stream_port),
            "--max-upload-mb", str(args.max_upload_mb),
            *passthrough]


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when something is already listening on ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def wait_for_port(port: int, label: str, seconds: float = 90.0,
                  host: str = "127.0.0.1") -> bool:
    """Wait for a program to start listening, reporting progress as it goes."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if port_in_use(port, host):
            return True
        time.sleep(0.5)
    print(f"[START] {label} did not come up within {seconds:.0f}s.", flush=True)
    return False


def local_address() -> str:
    """This machine's address on the network, for reaching it from a phone."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))   # no packet is sent; this just picks a route
            return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def launch(command: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a child program in its own process group."""
    return subprocess.Popen(command, cwd=str(ROOT_DIR), creationflags=NEW_GROUP,
                            env={**os.environ, **(env or {})})


def stop(process: subprocess.Popen | None, label: str) -> None:
    """Ask a child to stop, and insist if it will not.

    The CV node has real work to do on the way out — finishing the workbooks,
    retiring the vehicles still in frame, closing the serial port — so it is
    asked politely first and given time to finish.
    """
    if process is None or process.poll() is not None:
        return

    print(f"[START] Stopping {label}...", flush=True)
    try:
        if NEW_GROUP:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
    except (OSError, ValueError):
        process.terminate()

    try:
        process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        print(f"[START] {label} is taking too long; terminating.", flush=True)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def install_interrupt_handlers() -> None:
    """Make Ctrl-Break shut everything down as tidily as Ctrl-C does.

    Windows kills a Python process outright on Ctrl-Break, which here would
    mean the launcher vanishing and leaving both children orphaned — still
    holding the camera, the serial port and their ports — with no way to stop
    them but Task Manager.
    """
    def interrupt(_signum, _frame) -> None:
        """Turn the signal into the exception the wait loop already handles."""
        raise KeyboardInterrupt

    for name in ("SIGBREAK", "SIGTERM"):
        handled = getattr(signal, name, None)
        if handled is not None:
            try:
                signal.signal(handled, interrupt)
            except (OSError, ValueError):   # not supported on this platform
                pass


def open_logs_folder() -> None:
    """Open the folder holding the vehicle and telemetry logs in the file manager.

    The logs are the point of a run, so they are opened alongside the dashboard
    rather than left for someone to go hunting for afterwards. Failing to open a
    window is never worth stopping a run over, so every error here is reported
    and swallowed.
    """
    try:
        if sys.platform == "win32":
            os.startfile(ROOT_DIR)  # noqa: S606 - opening a known directory
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(ROOT_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(ROOT_DIR)])
    except Exception as exc:  # noqa: BLE001 - a missing file manager is not fatal
        print(f"[START] Could not open the logs folder ({exc}).", flush=True)
        print(f"[START] They are in {ROOT_DIR}", flush=True)


def main() -> int:
    """Start both programs, then wait until one of them stops or Ctrl-C is pressed."""
    args, passthrough = parse_args()
    install_interrupt_handlers()

    if not args.no_server and not CV_NODE.exists():
        print(f"[START] Cannot find {CV_NODE}.", flush=True)
        return 1
    if not args.no_dashboard and not DASHBOARD.exists():
        print(f"[START] Cannot find {DASHBOARD}.", flush=True)
        return 1

    for port, what, skip in ((args.stream_port, "CV node", args.no_server),
                             (args.dashboard_port, "dashboard", args.no_dashboard)):
        if not skip and port_in_use(port):
            print(f"[START] Port {port} is already in use — the {what} may already "
                  f"be running. Close it, or choose another port.", flush=True)
            return 1

    print("=" * 68, flush=True)
    print(" SMART TRAFFIC SYSTEM".center(68), flush=True)
    print("=" * 68, flush=True)

    server: subprocess.Popen | None = None
    dashboard: subprocess.Popen | None = None

    try:
        if not args.no_server:
            print("[START] CV node starting (loading the model takes a few seconds)...", flush=True)
            server = launch(node_command(args, passthrough))
            if not wait_for_port(args.stream_port, "CV node"):
                stop(server, "CV node")
                return 1
            print(f"[START] CV node ready on port {args.stream_port}.", flush=True)
        elif passthrough:
            print(f"[START] Ignoring {' '.join(passthrough)} — those are CV node "
                  "options and the CV node was not started.", flush=True)

        if not args.no_dashboard:
            print("[START] Dashboard starting...", flush=True)
            dashboard = launch(
                [sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
                 "--server.port", str(args.dashboard_port),
                 "--server.headless", "true",
                 "--server.maxUploadSize", str(args.max_upload_mb),
                 "--browser.gatherUsageStats", "false"],
                # This is the setting that is forgotten most often when the two
                # are started by hand, and its symptom — "stream offline" on a
                # dashboard sitting next to a perfectly healthy node — sends
                # people looking in entirely the wrong place.
                env={"TRAFFIC_STREAM_HOST": args.node,
                     "TRAFFIC_STREAM_PORT": str(args.stream_port)},
            )
            if not wait_for_port(args.dashboard_port, "dashboard"):
                stop(dashboard, "dashboard")
                stop(server, "CV node")
                return 1

        url = f"http://localhost:{args.dashboard_port}"
        print("-" * 68, flush=True)
        if not args.no_dashboard:
            print(f"  Dashboard      {url}", flush=True)
            print(f"  From a phone   http://{local_address()}:{args.dashboard_port}", flush=True)
        if not args.no_server:
            print(f"  Live video     http://localhost:{args.stream_port}/video_feed", flush=True)
            print(f"  Telemetry      http://localhost:{args.stream_port}/telemetry", flush=True)
        print(f"  Vehicle log    {ROOT_DIR / 'traffic_vehicles.csv'}", flush=True)
        print(f"  All-time log   {ROOT_DIR / 'traffic_all_vehicles.csv'}", flush=True)
        print("-" * 68, flush=True)
        print("  Press Ctrl-C here to stop everything cleanly.", flush=True)
        print("-" * 68, flush=True)

        if not args.no_dashboard and not args.no_browser:
            webbrowser.open(url)
        if not args.no_logs:
            open_logs_folder()

        # Wait until either child exits on its own, or Ctrl-C arrives.
        while True:
            for process, label in ((server, "CV node"), (dashboard, "dashboard")):
                if process is not None and process.poll() is not None:
                    print(f"[START] The {label} stopped (exit code {process.returncode}).", flush=True)
                    return process.returncode or 0
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[START] Shutting down...", flush=True)
        return 0
    finally:
        stop(dashboard, "dashboard")
        stop(server, "CV node")
        print("[START] Stopped.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
