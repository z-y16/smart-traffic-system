"""System Logs page — searchable event log with filtering and export."""

import pandas as pd
import streamlit as st

from backend.models.system_state import LogLevel
from backend.services.system_log_service import SystemLogService
from components.kpi_card import render_kpi_card
from components.navigation import render_sidebar_nav
from components.page_header import render_page_header
from config.settings import get_settings
from utils.session import init_session_state
from utils.theme import apply_dark_theme

settings = get_settings()

#: How often the log fragments redraw while Auto Refresh is on.
LOG_REFRESH_SECONDS: float = 1.0

st.set_page_config(
    page_title=f"{settings.app_title} | System Logs",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()
init_session_state()

if "log_service" not in st.session_state:
    st.session_state.log_service = SystemLogService(settings)

if "log_auto_refresh" not in st.session_state:
    st.session_state.log_auto_refresh = False

#: Passed to `st.fragment(run_every=...)`; None stops the timer, which is
#: how the Auto Refresh toggle turns updating off.
REFRESH_SECONDS: float | None = (
    LOG_REFRESH_SECONDS if st.session_state.log_auto_refresh else None)

log_service: SystemLogService = st.session_state.log_service

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📋 Log Controls")
    st.divider()

    st.toggle(
        "Auto Refresh",
        key="log_auto_refresh",
        help="Automatically add new log entries.",
    )

    if st.button("Clear All Logs", width="stretch", type="secondary"):
        log_service.clear_logs()
        st.toast("Logs cleared", icon="🗑️")
        st.rerun()

    st.divider()
    render_sidebar_nav()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
badge = "Simulation Active" if settings.simulation_mode else "Live Logs"
render_page_header(
    "📋 System Logs",
    "Searchable event log with filtering and export",
    badge=badge,
)

# ---------------------------------------------------------------------------
# Log Statistics
# ---------------------------------------------------------------------------


@st.fragment(run_every=REFRESH_SECONDS)
def render_log_statistics() -> None:
    """Draw the per-level log counts."""
    st.markdown('<div class="section-title">Log Statistics</div>', unsafe_allow_html=True)

    counts = log_service.get_log_count_by_level()
    total_logs = sum(counts.values())

    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

    with stat_col1:
        render_kpi_card(label="Total Logs", value=str(total_logs), icon="📊")

    with stat_col2:
        render_kpi_card(label="INFO", value=str(counts[LogLevel.INFO]), icon="🔵")

    with stat_col3:
        render_kpi_card(label="WARNING", value=str(counts[LogLevel.WARNING]), icon="🟡")

    with stat_col4:
        render_kpi_card(label="ERROR", value=str(counts[LogLevel.ERROR]), icon="🔴")

    with stat_col5:
        render_kpi_card(label="CRITICAL", value=str(counts[LogLevel.CRITICAL]), icon="⚫")


render_log_statistics()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    st.selectbox(
        "Log Level",
        ["All"] + [level.value for level in LogLevel],
        key="log_level_filter",
    )

with filter_col2:
    st.selectbox(
        "Source",
        ["All"] + log_service.get_available_sources(),
        key="log_source_filter",
    )

with filter_col3:
    st.text_input("Search", placeholder="Search logs...", key="log_search")

# ---------------------------------------------------------------------------
# Log table
# ---------------------------------------------------------------------------
# The filters above stay outside the fragment: they are keyed, so this reads
# their values from session state, and a redraw on the timer never lands in the
# middle of someone typing a search.


@st.fragment(run_every=REFRESH_SECONDS)
def render_log_table() -> None:
    """Draw the filtered log entries."""
    level_filter = st.session_state.get("log_level_filter", "All")
    source_filter = st.session_state.get("log_source_filter", "All")
    search_query = st.session_state.get("log_search", "")

    logs = log_service.get_logs(
        level_filter=None if level_filter == "All" else LogLevel(level_filter),
        source_filter=None if source_filter == "All" else source_filter,
        search_query=search_query or None,
        limit=100,
    )

    st.markdown('<div class="section-title">Log Entries</div>', unsafe_allow_html=True)

    if logs:
        rows = [
            {
                "Timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Level": log.level.value,
                "Source": log.source,
                "Module": log.module,
                "Message": log.message,
            }
            for log in logs
        ]
        df = pd.DataFrame(rows)

        def _color_level(level: str) -> str:
            """Apply color styling based on log level."""
            if level == "CRITICAL":
                return "background-color: rgba(0, 0, 0, 0.3); color: #ff4d4d; font-weight: bold;"
            if level == "ERROR":
                return "background-color: rgba(239, 68, 68, 0.15); color: #ff6b6b;"
            if level == "WARNING":
                return "background-color: rgba(234, 179, 8, 0.15); color: #fbbf24;"
            if level == "INFO":
                return "background-color: rgba(59, 130, 246, 0.15); color: #60a5fa;"
            return ""

        styled = df.style.apply(
            lambda row: [_color_level(row["Level"])] * len(row),
            axis=1,
        )
        st.dataframe(styled, width="stretch", hide_index=True, height=400)

        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Export to CSV",
            data=csv,
            file_name=f"system_logs_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("No logs match the current filters.")


render_log_table()

# ---------------------------------------------------------------------------
# Integration info
# ---------------------------------------------------------------------------
with st.expander("Log Integration Interface", expanded=False):
    st.markdown(
        """
        This module manages system logs with filtering and export capabilities.
        Replace `LogSimulator` with real logging infrastructure when ready —
        the UI will continue to work without changes.

        **Features:**
        - Log level filtering (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - Source filtering (Camera, YOLO, ESP32, Arduino, Serial, Database, System, Emergency)
        - Full-text search across messages, sources, and modules
        - Export to CSV for analysis
        - Real-time log statistics
        """
    )
