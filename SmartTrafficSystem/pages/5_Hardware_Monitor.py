"""Hardware Monitor page — ESP32, Arduino, and peripheral device status."""

import streamlit as st

from backend.services.hardware_monitor_service import HardwareMonitorService
from components.kpi_card import render_kpi_card
from components.navigation import render_sidebar_nav
from components.page_header import render_page_header
from config.settings import get_settings
from utils.session import init_session_state
from utils.theme import apply_dark_theme

settings = get_settings()

# Voltage and heartbeat drift slowly; twice a second kept the server permanently
# busy redrawing identical cards.
HARDWARE_REFRESH_SECONDS: float = 2.0

st.set_page_config(
    page_title=f"{settings.app_title} | Hardware Monitor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()
init_session_state()

if "hardware_service" not in st.session_state:
    st.session_state.hardware_service = HardwareMonitorService(settings)

if "hardware_auto_refresh" not in st.session_state:
    st.session_state.hardware_auto_refresh = True

#: Passed to `st.fragment(run_every=...)`; None stops the timer, which is
#: how the Live Refresh toggle turns updating off.
REFRESH_SECONDS: float | None = (HARDWARE_REFRESH_SECONDS
                                 if st.session_state.hardware_auto_refresh else None)

hardware_service: HardwareMonitorService = st.session_state.hardware_service

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Hardware Controls")
    st.divider()

    st.toggle(
        "Live Refresh",
        key="hardware_auto_refresh",
        help="Continuously update hardware status.",
    )

    st.divider()
    render_sidebar_nav()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
badge = "Simulation Active" if settings.simulation_mode else "Hardware Connected"
render_page_header(
    "🔧 Hardware Monitor",
    "ESP32, Arduino, and peripheral device status",
    badge=badge,
)

# ---------------------------------------------------------------------------
# Live readings
# ---------------------------------------------------------------------------


@st.fragment(run_every=REFRESH_SECONDS)
def render_hardware_status() -> None:
    """Draw every live hardware reading.

    A fragment, so Streamlit reruns this block on the timer and lets the
    script itself finish. The `sleep(); st.rerun()` loop this replaces kept
    the script permanently mid-run, and a websocket blip — a backgrounded
    tab, a brief network drop — orphaned the queued rerun, freezing the
    page until a widget was touched.
    """
    state = hardware_service.get_hardware_state()

    # ---------------------------------------------------------------------------
    # Connection Status Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Device Connections</div>', unsafe_allow_html=True)

    conn_col1, conn_col2 = st.columns(2)

    with conn_col1:
        esp32_color = "🟢" if state.esp32_connected else "🔴"
        esp32_text = "Connected" if state.esp32_connected else "Disconnected"
        render_kpi_card(label="ESP32", value=esp32_text, icon=esp32_color)

    with conn_col2:
        arduino_color = "🟢" if state.arduino_connected else "🔴"
        arduino_text = "Connected" if state.arduino_connected else "Disconnected"
        render_kpi_card(label="Arduino", value=arduino_text, icon=arduino_color)

    # ---------------------------------------------------------------------------
    # Traffic Signal Section — mirrors the physical lamp on the Arduino
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Traffic Signal</div>', unsafe_allow_html=True)

    # Fetching metrics is what populates last_telemetry; this page has no other
    # reason to call it, so it must ask explicitly.
    st.session_state.status_service.get_dashboard_metrics()
    telemetry = st.session_state.status_service.last_telemetry
    signal_live = bool(telemetry)
    phase = str(telemetry.get("light_phase", "") or "—")
    phase_icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(phase, "⚪")
    hardware_block = telemetry.get("hardware", {}) if signal_live else {}

    signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)

    with signal_col1:
        render_kpi_card(label="Signal Phase", value=phase, icon=phase_icon)

    with signal_col2:
        if telemetry.get("light_transitioning"):
            countdown = f"{telemetry.get('light_remaining', 0):.0f}s → {telemetry.get('light_target', '')}"
        else:
            countdown = "Settled"
        render_kpi_card(label="Transition", value=countdown, icon="⏳")

    with signal_col3:
        strip_on = bool(hardware_block.get("emergency_strip"))
        render_kpi_card(
            label="Emergency Strip",
            value="ACTIVE" if strip_on else "Off",
            icon="🚨" if strip_on else "⚫",
        )

    with signal_col4:
        manual = telemetry.get("light_manual")
        render_kpi_card(label="Control Mode", value=manual or "Automatic",
                        icon="🕹️" if manual else "🤖")

    if signal_live:
        lcd1 = str(hardware_block.get("lcd_line1", "")).rstrip()
        lcd2 = str(hardware_block.get("lcd_line2", "")).rstrip()
        st.code(f"|{lcd1:<16}|\n|{lcd2:<16}|", language=None)
        st.caption(
            "Exactly what the 16×2 LCD on the Arduino is showing. The signal is "
            "demand-responsive: GREEN while there is traffic to serve, RED on an "
            "empty road, and five seconds of YELLOW on every change between them. "
            "An emergency vehicle does not take the signal over — emergency traffic "
            "passes on any indication — it lights the dedicated strip instead."
        )
    else:
        st.info("Broadcast server offline — signal state unavailable.")

    # ---------------------------------------------------------------------------
    # Peripheral Status Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Peripheral Status</div>', unsafe_allow_html=True)

    periph_col1, periph_col2, periph_col3 = st.columns(3)

    with periph_col1:
        servo_color = "🟢" if state.servo_status == "Idle" else "🟡" if state.servo_status == "Active" else "🔴"
        render_kpi_card(label="Servo", value=state.servo_status, icon=servo_color)

    with periph_col2:
        led_icon = {"Police": "🔵", "Ambulance": "🔴", "Speed": "🟡"}.get(
            state.led_strip_status, "⚪"
        )
        render_kpi_card(label="LED Guide", value=state.led_strip_status, icon=led_icon)

    with periph_col3:
        lcd_color = "🟢" if state.lcd_status == "Displaying" else "⚪" if state.lcd_status == "Idle" else "🔴"
        render_kpi_card(label="LCD", value=state.lcd_status, icon=lcd_color)

    # ---------------------------------------------------------------------------
    # Network Status Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Network Status</div>', unsafe_allow_html=True)

    wifi_col1, wifi_col2 = st.columns(2)

    with wifi_col1:
        wifi_color = "🟢" if state.wifi_status == "Connected" else "🟡" if state.wifi_status == "Weak Signal" else "🔴"
        render_kpi_card(label="WiFi", value=state.wifi_status, icon=wifi_color)

    with wifi_col2:
        render_kpi_card(label="Firmware Version", value=state.firmware_version, icon="📡")

    # ---------------------------------------------------------------------------
    # Power Metrics Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Power Metrics</div>', unsafe_allow_html=True)

    power_col1, power_col2 = st.columns(2)

    with power_col1:
        voltage_color = "🟢" if state.voltage > 4.5 else "🟡" if state.voltage > 4.0 else "🔴"
        render_kpi_card(label="Voltage", value=f"{state.voltage}V", icon=voltage_color)

    with power_col2:
        current_color = "🟢" if state.current < 2.0 else "🟡" if state.current < 3.0 else "🔴"
        render_kpi_card(label="Current", value=f"{state.current}A", icon=current_color)

    # ---------------------------------------------------------------------------
    # Heartbeat Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">System Heartbeat</div>', unsafe_allow_html=True)

    heartbeat_col1, heartbeat_col2 = st.columns(2)

    with heartbeat_col1:
        render_kpi_card(label="Heartbeat Counter", value=str(state.heartbeat), icon="💓")

    with heartbeat_col2:
        power_watts = round(state.voltage * state.current, 2)
        render_kpi_card(label="Power Consumption", value=f"{power_watts}W", icon="⚡")

    # ---------------------------------------------------------------------------
    # Status Summary
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Status Summary</div>', unsafe_allow_html=True)

    if state.esp32_connected and state.arduino_connected:
        st.success("✅ All devices connected and operational")
    elif state.esp32_connected or state.arduino_connected:
        st.warning("⚠️ Some devices disconnected - check connections")
    else:
        st.error("❌ No devices connected - check hardware")

    if state.voltage < 4.0:
        st.error("⚠️ Low voltage detected - check power supply")
    elif state.current > 3.0:
        st.warning("⚠️ High current draw - check for short circuits")

    # ---------------------------------------------------------------------------
    # Integration info
    # ---------------------------------------------------------------------------


render_hardware_status()

with st.expander("Hardware Integration Interface", expanded=False):
    st.markdown(
        """
        This module monitors the status of all connected hardware devices.
        Replace `HardwareSimulator` with real hardware telemetry when ready —
        the UI will continue to work without changes.

        **Monitored devices:**
        - ESP32 (main controller)
        - Arduino (peripheral controller)
        - Servo motors
        - LED strips
        - LCD display
        - WiFi connection
        - Power metrics (voltage, current)
        """
    )
