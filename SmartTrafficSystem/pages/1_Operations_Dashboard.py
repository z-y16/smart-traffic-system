"""
Operations Dashboard — live system health and traffic overview.

This was the entry point until the front page took that slot. It is the same
page, moved: the entry point is now a title page for the project, and the
figures that change every second belong behind it rather than underneath it.
"""

import streamlit as st

from backend.models.system_state import EmergencyStatus
from components.kpi_card import render_kpi_grid
from components.navigation import render_sidebar_nav
from components.page_header import render_page_header
from config.settings import get_settings
from utils.datetime_utils import format_current_date, format_current_time, format_timestamp
from utils.session import init_session_state
from utils.theme import apply_dark_theme, density_class, status_class

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
settings = get_settings()

st.set_page_config(
    page_title=f"{settings.app_title} | Operations",
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()
init_session_state()

#: Passed to `st.fragment(run_every=...)`; None stops the timer, which is
#: how the Auto Refresh toggle turns updating off.
REFRESH_SECONDS: float | None = (settings.refresh_interval_ms / 1000
                                 if st.session_state.auto_refresh else None)

# ---------------------------------------------------------------------------
# Fetch metrics (before sidebar so live-connection state is fresh this run)
# ---------------------------------------------------------------------------
# The figures themselves are read again inside the fragment below, which is
# what keeps them advancing; this call is what the sidebar's mode line needs.
st.session_state.status_service.get_dashboard_metrics()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {settings.app_icon} Control Center")
    st.caption("Smart Traffic Management System")
    st.divider()

    st.toggle(
        "Auto Refresh",
        key="auto_refresh",
        help="Automatically refresh dashboard metrics.",
    )

    if st.button("Refresh Now", width="stretch", type="primary"):
        st.rerun()

    st.divider()
    # ── Session recording ───────────────────────────────────────────────────
    # The CV node always logs to the all-sessions archive; this captures a
    # named slice of it so a demo run can be found again later.
    recorder = st.session_state.recording_service
    record_status = recorder.status()
    node_online = recorder.last_error is None
    st.markdown("**⏺️ Session Recording**")
    if not node_online:
        st.caption("Broadcast server offline")
    elif record_status.get("recording"):
        st.caption(f"Recording **{record_status.get('name', '')}** — "
                   f"{record_status.get('rows', 0)} samples")
    else:
        st.caption(f"Idle — {record_status.get('rows', 0)} samples in the "
                   "current session")

    if record_status.get("recording"):
        if st.button("⏹️ Stop Recording", width="stretch", key="dash_stop_recording",
                     disabled=not node_online):
            result = recorder.stop()
            st.toast(f"Saved {result.get('rows', 0)} samples", icon="💾")
            st.rerun()
    else:
        if st.button("⏺️ Record Session", width="stretch",
                     key="dash_start_recording", disabled=not node_online):
            recorder.start("")
            st.toast("Recording started", icon="⏺️")
            st.rerun()

    st.divider()
    render_sidebar_nav()

    st.divider()
    live_connected = st.session_state.status_service.is_live_stream_connected
    mode_label = "Live Broadcast Connected" if live_connected else (
        "Simulation Mode" if settings.simulation_mode else "Live Mode (stream offline)"
    )
    st.info(f"**Mode:** {mode_label}")

    if live_connected:
        # Fetched on demand (not every auto-refresh) to avoid re-downloading
        # the whole workbook every couple of seconds.
        if st.button("Prepare Excel Log for Download", width="stretch"):
            st.session_state.excel_export_bytes = (
                st.session_state.analytics_service.fetch_export_bytes()
            )
        if st.session_state.get("excel_export_bytes"):
            st.download_button(
                "⬇️ This Session (Excel)",
                data=st.session_state.excel_export_bytes,
                file_name="traffic_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        # The master workbook holds every session ever recorded, so it is
        # fetched only on request rather than on every auto-refresh.
        if st.button("Prepare All-Sessions Workbook", width="stretch"):
            st.session_state.master_export_bytes = recorder.fetch_master_excel()
        if st.session_state.get("master_export_bytes"):
            st.download_button(
                "⬇️ All Sessions (Excel)",
                data=st.session_state.master_export_bytes,
                file_name="traffic_all_sessions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
badge = "Simulation Active" if settings.simulation_mode else "Live"
render_page_header(
    title=f"{settings.app_icon} Operations Dashboard",
    subtitle="Real-time traffic monitoring and system health overview",
    badge=badge,
)

# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------


@st.fragment(run_every=REFRESH_SECONDS)
def render_dashboard() -> None:
    """Draw every live figure on the dashboard.

    A fragment, so Streamlit reruns this block on the timer and lets the
    script itself finish. The `sleep(); st.rerun()` loop this replaces kept
    the script permanently mid-run, and a websocket blip — a backgrounded
    tab, a brief network drop — orphaned the queued rerun, freezing the
    figures until a widget was touched.
    """
    metrics = st.session_state.status_service.get_dashboard_metrics()
    health = metrics.system_health
    traffic = metrics.traffic

    time_col, date_col, updated_col = st.columns(3)
    with time_col:
        st.metric(label="Current Time", value=format_current_time())
    with date_col:
        st.metric(label="Current Date", value=format_current_date())
    with updated_col:
        st.metric(label="Last Updated", value=format_timestamp(metrics.timestamp))

    st.markdown('<div class="section-title">System Health</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # System status KPIs
    # ---------------------------------------------------------------------------
    system_kpis = [
        {
            "label": "System Status",
            "value": health.system_status.value,
            "icon": "🖥️",
            "status": health.system_status.value,
        },
        {
            "label": "Camera Status",
            "value": health.camera_status.value,
            "icon": "📷",
            "status": health.camera_status.value,
        },
        {
            "label": "YOLO Status",
            "value": health.yolo_status.value,
            "icon": "🤖",
            "status": health.yolo_status.value,
        },
        {
            "label": "ESP32 Status",
            "value": health.esp32_status.value,
            "icon": "📡",
            "status": health.esp32_status.value,
        },
        {
            "label": "Serial Status",
            "value": health.serial_status.value,
            "icon": "🔌",
            "status": health.serial_status.value,
        },
        {
            "label": "Database Status",
            "value": health.database_status.value,
            "icon": "🗄️",
            "status": health.database_status.value,
        },
    ]
    render_kpi_grid(system_kpis, columns=3)

    st.markdown('<div class="section-title">Traffic Overview</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Traffic KPIs
    # ---------------------------------------------------------------------------
    emergency_display = metrics.emergency_status.value
    emergency_icon = "🚨" if metrics.emergency_status != EmergencyStatus.NONE else "✅"

    traffic_kpis = [
        {
            "label": "Traffic Density",
            "value": traffic.traffic_density,
            "icon": "🚗",
            "density": traffic.traffic_density,
        },
        {
            "label": "Vehicle Count",
            "value": str(traffic.vehicle_count),
            "icon": "🔢",
            "delta": "Live count",
        },
        {
            "label": "Emergency Status",
            "value": emergency_display,
            "icon": emergency_icon,
            "delta": "Priority lane control",
        },
        {
            "label": "Emergency Vehicles",
            "value": str(traffic.emergency_vehicle_count),
            "icon": "🚑",
            "delta": "Light-bar detection",
        },
        {
            "label": "Current Lane",
            "value": traffic.current_lane,
            "icon": "🛣️",
        },
        {
            "label": "Average Speed",
            "value": f"{traffic.average_speed_kmh:.1f} km/h",
            "icon": "⚡",
        },
        {
            "label": "Detection FPS",
            "value": f"{traffic.fps:.1f}",
            "icon": "🎬",
            "delta": "YOLO pipeline",
        },
        {
            # The signal is demand-responsive: green while there is traffic to
            # serve, red on an empty road, five seconds of yellow in between.
            "label": "Traffic Signal",
            "value": traffic.light_phase or "—",
            "icon": {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(
                traffic.light_phase, "⚪"),
            "delta": (f"{traffic.light_remaining:.0f}s → {traffic.light_target}"
                      if traffic.light_transitioning else "Settled"),
        },
    ]
    render_kpi_grid(traffic_kpis, columns=3)

    st.markdown('<div class="section-title">Resource Utilization</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Resource usage with progress bars
    # ---------------------------------------------------------------------------
    resource_col1, resource_col2 = st.columns(2)

    with resource_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">💻</div>
                <div class="kpi-label">CPU Usage</div>
                <div class="kpi-value">{metrics.cpu_usage_percent:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(metrics.cpu_usage_percent / 100.0, 1.0))

    with resource_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">🧠</div>
                <div class="kpi-label">RAM Usage</div>
                <div class="kpi-value">{metrics.ram_usage_percent:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(metrics.ram_usage_percent / 100.0, 1.0))

    # ---------------------------------------------------------------------------
    # Quick status summary
    # ---------------------------------------------------------------------------
    with st.expander("System Summary", expanded=False):
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown("**Subsystem Connectivity**")
            for name, status in [
                ("Camera", health.camera_status),
                ("YOLO Model", health.yolo_status),
                ("ESP32", health.esp32_status),
                ("Serial", health.serial_status),
                ("Database", health.database_status),
            ]:
                css = status_class(status.value)
                st.markdown(
                    f'- {name}: <span class="{css}">{status.value}</span>',
                    unsafe_allow_html=True,
                )
        with summary_col2:
            st.markdown("**Traffic Snapshot**")
            density_css = density_class(traffic.traffic_density)
            st.markdown(
                f"""
                - Density: <span class="{density_css}">{traffic.traffic_density}</span>
                - Vehicles: **{traffic.vehicle_count}**
                - Active Lane: **{traffic.current_lane}**
                - Avg Speed: **{traffic.average_speed_kmh:.1f} km/h**
                - Emergency: **{emergency_display}**
                """,
                unsafe_allow_html=True,
            )


render_dashboard()
