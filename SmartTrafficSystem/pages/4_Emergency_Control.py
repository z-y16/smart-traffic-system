"""Emergency Control page — priority lane management and hardware commands."""

import streamlit as st

from backend.services.emergency_control_service import EmergencyControlService
from components.kpi_card import render_kpi_card
from components.navigation import render_sidebar_nav
from components.page_header import render_page_header
from config.settings import get_settings
from utils.session import init_session_state
from utils.theme import apply_dark_theme

settings = get_settings()

# Each rerun re-polls telemetry and redraws the whole page; twice a second kept
# the server permanently busy for state that changes far more slowly.
EMERGENCY_REFRESH_SECONDS: float = 2.0

st.set_page_config(
    page_title=f"{settings.app_title} | Emergency Control",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()
init_session_state()

if "emergency_service" not in st.session_state:
    st.session_state.emergency_service = EmergencyControlService(settings)

if "emergency_auto_refresh" not in st.session_state:
    st.session_state.emergency_auto_refresh = True

#: Passed to `st.fragment(run_every=...)`; None stops the timer, which is
#: how the Live Refresh toggle turns updating off.
REFRESH_SECONDS: float | None = (EMERGENCY_REFRESH_SECONDS
                                 if st.session_state.emergency_auto_refresh else None)

emergency_service: EmergencyControlService = st.session_state.emergency_service

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚨 Emergency Controls")
    st.divider()

    st.toggle(
        "Live Refresh",
        key="emergency_auto_refresh",
        help="Continuously update emergency status.",
    )

    st.divider()
    render_sidebar_nav()

# ---------------------------------------------------------------------------
# Header and live state
# ---------------------------------------------------------------------------
# The header is inside the fragment because its badge reports the live link,
# which is only known once the state has been fetched.


@st.fragment(run_every=REFRESH_SECONDS)
def render_emergency_view() -> None:
    """Draw the live emergency state, hardware readouts and controls.

    A fragment, so Streamlit reruns this block on the timer and lets the
    script itself finish. The `sleep(); st.rerun()` loop this replaces kept
    the script permanently mid-run, and a websocket blip — a backgrounded
    tab, a brief network drop — orphaned the queued rerun, freezing the
    page until a widget was touched.
    """
    state = emergency_service.get_emergency_state()
    is_live = emergency_service.is_live

    badge = "Live Detection" if is_live else (
        "Simulation Active" if settings.simulation_mode else "Hardware Connected"
    )
    render_page_header(
        "🚨 Emergency Control",
        "Priority lane management and hardware commands",
        badge=badge,
    )

    # ---------------------------------------------------------------------------
    # Emergency Detection Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Emergency Detection Status</div>', unsafe_allow_html=True)

    detection = state.detection
    detected_col1, detected_col2, detected_col3, detected_col4 = st.columns(4)

    with detected_col1:
        status_color = "🟢" if not detection.detected else "🔴"
        status_text = "No Detection" if not detection.detected else "DETECTED"
        render_kpi_card(label="Emergency Vehicle", value=status_text, icon=status_color)

    with detected_col2:
        conf_text = f"{detection.confidence:.0%}" if detection.detected else "—"
        render_kpi_card(label="Confidence", value=conf_text, icon="📊")

    with detected_col3:
        render_kpi_card(label="Detected Lane", value=detection.detected_lane, icon="🛣️")

    with detected_col4:
        arrival_text = f"{detection.estimated_arrival_seconds}s" if detection.detected else "—"
        render_kpi_card(label="Est. Arrival", value=arrival_text, icon="⏱️")

    if detection.detected:
        if is_live:
            telemetry = emergency_service.last_telemetry
            count = telemetry.get("emergency_vehicle_count", 1)
            st.error(
                f"🚨 **{detection.vehicle_type}** detected by the live camera "
                f"({count} vehicle(s) flagged) — priority lane should be cleared."
            )
            flagged = [d for d in telemetry.get("detections", []) if d.get("emergency")]
            if flagged:
                st.markdown("**Flagged vehicles**")
                for vehicle in flagged:
                    speed = vehicle.get("speed_kmh")
                    speed_text = "speed pending" if speed is None else f"{speed} km/h"
                    colour = vehicle.get("siren_colour", "none")
                    rate = vehicle.get("siren_rate_hz", 0) or 0
                    siren_text = (f", **{colour}** beacon at {rate:.1f} Hz"
                                  if colour not in (None, "none") else "")
                    st.markdown(
                        f"- ID **{vehicle.get('id')}** — "
                        f"**{str(vehicle.get('emergency_type', 'unknown')).title()}**, "
                        f"{speed_text}{siren_text}, "
                        f"confidence {vehicle.get('emergency_confidence', 0):.0%}, "
                        f"recognised by *{vehicle.get('emergency_evidence', 'unknown')}*"
                    )

            # Marked vehicles running no beacon are deliberately not given
            # priority; listing them separately makes that a decision rather than
            # a silence.
            marked = [d for d in telemetry.get("detections", [])
                      if d.get("emergency_suspected")]
            if marked:
                st.info(
                    f"{len(marked)} vehicle(s) look like emergency vehicles but are "
                    "running no siren — no priority given: "
                    + ", ".join(f"ID {d.get('id')} "
                                f"({str(d.get('emergency_type', '')).title()})"
                                for d in marked)
                )
        else:
            st.success(f"🚨 **{detection.vehicle_type}** detected in {detection.detected_lane}")
    else:
        st.info("✅ No emergency vehicle currently detected")

    if is_live:
        telemetry_now = emergency_service.last_telemetry
        st.caption(
            "Detection and hardware status are read live from the broadcast server. "
            "Emergency mode is raised by a **working siren light**, whose colour "
            "names the vehicle; livery recognition only refines the type. "
            f"Siren detector: {telemetry_now.get('siren_detection', 'unknown')}. "
            f"Recogniser: {telemetry_now.get('emergency_recognition', 'unknown')}. "
            "Lane assignment and arrival estimate remain simulated — a single camera "
            "has no per-lane sensor."
        )
        if telemetry_now.get("siren_undersampled"):
            st.warning(
                f"⚠️ The camera node is sampling at "
                f"{telemetry_now.get('siren_sample_hz', 0):.0f} frames/s, too slow to "
                "see a beacon flash reliably — a siren could pass undetected."
            )

    # ---------------------------------------------------------------------------
    # Hardware Status Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Hardware Status</div>', unsafe_allow_html=True)

    hardware = state.hardware
    hw_col1, hw_col2, hw_col3, hw_col4, hw_col5 = st.columns(5)

    with hw_col1:
        signal_color = "🟢" if hardware.current_signal == "Green" else "🔴" if hardware.current_signal == "Red" else "🟡"
        render_kpi_card(label="Signal", value=hardware.current_signal, icon=signal_color)

    with hw_col2:
        bump_color = "🟢" if hardware.speed_bump_status == "Lowered" else "🔴"
        render_kpi_card(label="Speed Bump", value=hardware.speed_bump_status, icon=bump_color)

    with hw_col3:
        render_kpi_card(
            label="Lane Divider",
            value=hardware.lane_allocation,
            icon="↔️",
        )

    with hw_col4:
        led_icon = {"Police": "🔵", "Ambulance": "🔴", "Speed": "🟡"}.get(
            hardware.led_status, "⚪"
        )
        render_kpi_card(label="LED Guide", value=hardware.led_status, icon=led_icon)

    with hw_col5:
        lcd_color = "🟢" if hardware.lcd_status == "Displaying Message" else "⚪"
        render_kpi_card(label="LCD", value=hardware.lcd_status, icon=lcd_color)

    # ---------------------------------------------------------------------------
    # Control Buttons Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">Manual Controls</div>', unsafe_allow_html=True)

    st.markdown("**Lane Divider**")
    st.caption(
        "Move the barrier one lane at a time across the six highway lanes. "
        "Left hands the extra lane to opposite-direction traffic, right hands "
        "it to the forward direction, and the middle position is an even "
        "3 / 3 split. The state shown above updates once the controller "
        "acknowledges the servo has finished moving."
    )

    def move_divider(direction: str, icon: str) -> None:
        """Send one divider step and report what the controller made of it."""
        result = emergency_service.move_lane_divider(direction)
        position = result.get("lane_allocation", "its current position")
        if not result.get("success"):
            st.error(result.get("error", "Lane changer did not acknowledge"))
        elif result.get("at_limit"):
            # A toast rather than st.warning: this fragment redraws on a
            # two-second timer, which would wipe an inline message before it
            # had been read.
            st.toast(
                f"Divider is already as far {direction.lower()} as it goes "
                f"— {position}",
                icon="🚧",
            )
        else:
            st.toast(f"Divider moved {direction.lower()} — {position}", icon=icon)

    divider_col1, divider_col2 = st.columns(2)
    with divider_col1:
        if st.button("⬅️ Move Left", width="stretch"):
            move_divider("LEFT", "⬅️")
    with divider_col2:
        if st.button("➡️ Move Right", width="stretch"):
            move_divider("RIGHT", "➡️")

    st.divider()

    control_col1, control_col2 = st.columns(2)

    with control_col1:
        st.markdown("**Emergency Mode**")
        emergency_btn_col1, emergency_btn_col2 = st.columns(2)
        with emergency_btn_col1:
            if st.button("🚨 Activate Emergency", width="stretch", type="primary"):
                lane = state.detection.detected_lane if state.detection.detected else "Lane A"
                result = emergency_service.activate_emergency_mode(lane)
                if result.get("success"):
                    st.toast(f"Emergency mode activated for {lane}", icon="🚨")
                else:
                    st.error("Failed to activate emergency mode")
        with emergency_btn_col2:
            if st.button("✅ Deactivate", width="stretch"):
                result = emergency_service.deactivate_emergency_mode()
                if result.get("success"):
                    st.toast("Emergency mode deactivated", icon="✅")
                else:
                    st.error("Failed to deactivate emergency mode")

    with control_col2:
        st.markdown("**Speed Bump Control**")
        bump_btn_col1, bump_btn_col2 = st.columns(2)
        with bump_btn_col1:
            if st.button("⬆️ Raise", width="stretch"):
                result = emergency_service.raise_speed_bump()
                if result.get("success"):
                    st.toast("Speed bump raised", icon="⬆️")
                else:
                    st.error(result.get("error", "Speed-bump controller did not acknowledge"))
        with bump_btn_col2:
            if st.button("⬇️ Lower", width="stretch"):
                result = emergency_service.lower_speed_bump()
                if result.get("success"):
                    st.toast("Speed bump lowered", icon="⬇️")
                else:
                    st.error(result.get("error", "Speed-bump controller did not acknowledge"))

        st.markdown("**LED Guiding Lights**")
        led_btn_col1, led_btn_col2, led_btn_col3, led_btn_col4 = st.columns(4)
        with led_btn_col1:
            if st.button("🔵 Police", width="stretch"):
                result = emergency_service.set_led_mode("Police")
                if result.get("success"):
                    st.toast("Police blue flashing activated", icon="🔵")
                else:
                    st.error(result.get("error", "LED controller did not acknowledge"))
        with led_btn_col2:
            if st.button("🔴 Ambulance", width="stretch"):
                result = emergency_service.set_led_mode("Ambulance")
                if result.get("success"):
                    st.toast("Ambulance red flashing activated", icon="🔴")
                else:
                    st.error(result.get("error", "LED controller did not acknowledge"))
        with led_btn_col3:
            if st.button("🟡 Speed", width="stretch"):
                result = emergency_service.set_led_mode("Speed")
                if result.get("success"):
                    st.toast("Speed yellow flashing activated", icon="🟡")
                else:
                    st.error(result.get("error", "LED controller did not acknowledge"))
        with led_btn_col4:
            if st.button("⚫ LED Off", width="stretch"):
                result = emergency_service.set_led_mode("Off")
                if result.get("success"):
                    st.toast("LED guide switched off", icon="⚫")
                else:
                    st.error(result.get("error", "LED controller did not acknowledge"))

    # ---------------------------------------------------------------------------
    # LCD Message Section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-title">LCD Display</div>', unsafe_allow_html=True)

    lcd_col1, lcd_col2 = st.columns([3, 1])
    with lcd_col1:
        lcd_message = st.text_input("Message to Display", value="EMERGENCY VEHICLE APPROACHING")
    with lcd_col2:
        if st.button("Send to LCD", width="stretch", type="secondary"):
            result = emergency_service.send_lcd_message(lcd_message)
            if result.get("success"):
                st.toast(f"Message sent: {lcd_message}", icon="📺")
            else:
                st.error(result.get("error", "Failed to send message"))

    # ---------------------------------------------------------------------------
    # Integration info
    # ---------------------------------------------------------------------------


render_emergency_view()

with st.expander("Member 3 Integration Interface", expanded=False):
    st.markdown(
        """
        Hardware commands are not sent from this dashboard directly. The CV node
        (`broadcast_server.py`) owns the Arduino serial port — only one process
        can — so commands are POSTed to its `/command` endpoint and forwarded
        down the same link. That is also what lets the dashboard run on a
        different machine from the camera.

        In simulation mode nothing leaves the process: `SimulatedCommunication`
        answers instead, so the UI is identical with no hardware attached.

        **Available commands:**
        - ACTIVATE_EMERGENCY / DEACTIVATE_EMERGENCY
        - MOVE_LANE_DIVIDER (LEFT / RIGHT) — one lane per press
        - SET_LANE_ALLOCATION (BALANCED / FORWARD_4 / OPPOSITE_4)
        - RAISE_SPEED_BUMP / LOWER_SPEED_BUMP
        - LED_MODE (Police / Ambulance / Speed / Off)
        - LED_COLOR (Red/Green/Yellow)
        - LCD_MESSAGE
        """
    )
