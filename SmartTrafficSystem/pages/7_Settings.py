"""Settings page — camera, serial, detection, and theme configuration."""

import streamlit as st

from backend.services.settings_service import SettingsService
from backend.services.system_status_service import LIVE_STREAM_HOST, LIVE_STREAM_PORT
from components.navigation import render_sidebar_nav
from components.page_header import render_page_header
from config.settings import get_settings
from utils.session import init_session_state
from utils.theme import apply_dark_theme

settings = get_settings()

st.set_page_config(
    page_title=f"{settings.app_title} | Settings",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()
init_session_state()

# Edit the same Settings instance the running services hold, so a save takes
# effect immediately instead of only after a restart.
settings = st.session_state.settings

if "settings_service" not in st.session_state:
    st.session_state.settings_service = SettingsService(settings)

settings_service: SettingsService = st.session_state.settings_service

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings Controls")
    st.divider()

    if st.button("Reset to Defaults", width="stretch", type="secondary"):
        settings_service.reset_to_defaults()
        st.toast("Settings reset to defaults", icon="🔄")
        st.rerun()

    st.divider()
    render_sidebar_nav()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
render_page_header(
    "⚙️ Settings",
    "Camera, serial, detection, and theme configuration",
)

# ---------------------------------------------------------------------------
# Get current settings
# ---------------------------------------------------------------------------
current_settings = settings_service.get_settings()

# ---------------------------------------------------------------------------
# General Settings
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">General Settings</div>', unsafe_allow_html=True)

general_col1, general_col2 = st.columns(2)

with general_col1:
    app_title = st.text_input("Application Title", value=current_settings.app_title)

with general_col2:
    theme = st.selectbox("Theme", ["dark", "light"], index=0 if current_settings.theme == "dark" else 1)

simulation_mode = st.toggle(
    "Simulation Mode",
    value=current_settings.simulation_mode,
    help="Enable simulation mode when hardware is unavailable.",
)

# ---------------------------------------------------------------------------
# Camera Settings
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Camera Settings</div>', unsafe_allow_html=True)

camera_col1, camera_col2 = st.columns(2)

with camera_col1:
    # Probing devices takes seconds and prints driver errors, so the scan is
    # explicit and its result is remembered for the rest of the session.
    if st.button("🔍 Rescan cameras", width="stretch"):
        st.session_state.detected_cameras = settings_service.get_available_cameras(scan=True)
        st.toast(f"Found camera index: {st.session_state.detected_cameras}", icon="📷")

    available_cameras = st.session_state.get("detected_cameras") or (
        settings_service.get_available_cameras()
    )
    if current_settings.camera_index not in available_cameras:
        available_cameras = sorted({*available_cameras, current_settings.camera_index})

    camera_index = st.selectbox(
        "Camera Index",
        available_cameras,
        index=available_cameras.index(current_settings.camera_index),
    )

with camera_col2:
    refresh_interval = st.slider(
        "Refresh Interval (ms)",
        min_value=500,
        max_value=5000,
        value=current_settings.refresh_interval_ms,
        step=100,
    )

# ---------------------------------------------------------------------------
# Serial Settings
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Serial Settings</div>', unsafe_allow_html=True)

serial_col1, serial_col2 = st.columns(2)

with serial_col1:
    available_ports = settings_service.get_available_serial_ports()
    serial_port = st.selectbox(
        "Serial Port",
        available_ports,
        index=available_ports.index(current_settings.serial_port)
        if current_settings.serial_port in available_ports
        else 0,
        help="Shown for reference. The CV node owns the serial link — Windows "
        "lets only one process hold a COM port — so the port it uses is set "
        "with `--arduino-port` (see START HERE.bat), not here.",
    )

with serial_col2:
    available_baud_rates = settings_service.get_available_baud_rates()
    baud_rate = st.selectbox(
        "Baud Rate",
        available_baud_rates,
        index=available_baud_rates.index(current_settings.baud_rate) if current_settings.baud_rate in available_baud_rates else 4,
    )

# ---------------------------------------------------------------------------
# Live Broadcast (CV node) Settings
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Live Broadcast Node</div>', unsafe_allow_html=True)

stream_col1, stream_col2 = st.columns(2)

with stream_col1:
    stream_host = st.text_input(
        "Broadcast Host",
        value=current_settings.stream_host,
        help="IP address of the PC running broadcast_server.py. "
        "Use 127.0.0.1 if it is this same machine. Find it with `ipconfig`.",
    )

with stream_col2:
    stream_port = st.number_input(
        "Broadcast Port",
        min_value=1,
        max_value=65535,
        value=int(current_settings.stream_port),
        step=1,
        help="The --stream-port used by broadcast_server.py (default 8502).",
    )

st.caption(
    f"Current live endpoints: `http://{LIVE_STREAM_HOST}:{LIVE_STREAM_PORT}/video_feed` — "
    f"{'connected ✅' if st.session_state.status_service.is_live_stream_connected else 'offline ⚠️'}. "
    "Changing the host or port takes effect after restarting the dashboard."
)

# ---------------------------------------------------------------------------
# Detection Settings
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Detection Settings</div>', unsafe_allow_html=True)

detection_threshold = st.slider(
    "Detection Threshold",
    min_value=0.50,
    max_value=0.99,
    value=current_settings.detection_threshold,
    step=0.01,
    help="Minimum confidence threshold for YOLO detections shown in the simulated feed. "
    "The live pipeline's own threshold is set with broadcast_server.py --conf.",
)

# ---------------------------------------------------------------------------
# Lane Settings
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Lane Settings</div>', unsafe_allow_html=True)

lanes = st.text_input(
    "Lane Names",
    value=", ".join(current_settings.lanes),
    help="Comma-separated list of lane names.",
)

# ---------------------------------------------------------------------------
# Save Button
# ---------------------------------------------------------------------------
st.divider()

save_col1, save_col2, save_col3 = st.columns([1, 1, 2])

with save_col1:
    if st.button("💾 Save Settings", width="stretch", type="primary"):
        updated_lanes = [lane.strip() for lane in lanes.split(",") if lane.strip()]
        settings_service.update_settings(
            app_title=app_title,
            theme=theme,
            simulation_mode=simulation_mode,
            camera_index=camera_index,
            refresh_interval_ms=refresh_interval,
            serial_port=serial_port,
            baud_rate=baud_rate,
            detection_threshold=detection_threshold,
            stream_host=stream_host.strip(),
            stream_port=int(stream_port),
            lanes=updated_lanes,
        )
        settings_service.save_settings()
        st.toast("Settings saved successfully!", icon="✅")
        st.rerun()

with save_col2:
    if st.button("🔄 Reload", width="stretch", type="secondary"):
        st.rerun()

# ---------------------------------------------------------------------------
# Current Settings Display
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Current Settings</div>', unsafe_allow_html=True)

settings_dict = current_settings.to_dict()
st.json(settings_dict)

# ---------------------------------------------------------------------------
# Integration info
# ---------------------------------------------------------------------------
with st.expander("Settings Integration", expanded=False):
    st.markdown(
        """
        This module manages application settings with persistence to JSON.
        Settings are saved to `data/settings.json` and loaded on startup.

        **Available settings:**
        - Application title and theme
        - Simulation mode toggle
        - Camera index and refresh interval
        - Serial port and baud rate
        - Detection threshold
        - Lane configuration
        """
    )
