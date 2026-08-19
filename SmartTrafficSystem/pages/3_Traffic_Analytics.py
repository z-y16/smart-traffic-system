"""Traffic Analytics page — session recording, archive insights, and export.

Two distinct bodies of data are presented here, and the difference matters:

* **Live session analytics** come from the broadcast server's rolling in-memory
  buffer (roughly the last hour) and re-bin as you change the interval.
* **Archive analytics** come from the all-sessions record on disk — every run
  ever logged — which is what makes trends across days visible at all.
"""

from datetime import datetime

import streamlit as st

from backend.services import archive_analytics as aa
from backend.services.analytics_service import AnalyticsService
from backend.services.recording_service import RecordingService
from components.charts import (
    congestion_histogram_chart,
    congestion_trend_chart,
    hourly_demand_chart,
    level_share_chart,
    phase_share_chart,
    rolling_trend_chart,
    session_comparison_chart,
    speed_timeline_chart,
    speed_vs_congestion_chart,
    traffic_density_chart,
    traffic_heatmap,
    vehicle_count_timeline,
    vehicle_mix_chart,
    vehicle_types_pie,
    waiting_time_chart,
)
from components.kpi_card import render_kpi_grid
from components.navigation import render_sidebar_nav
from components.page_header import render_page_header
from config.settings import get_settings
from utils.session import init_session_state
from utils.theme import apply_dark_theme

settings = get_settings()

st.set_page_config(
    page_title=f"{settings.app_title} | Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()
init_session_state()

if "analytics_service" not in st.session_state:
    st.session_state.analytics_service = AnalyticsService(settings)
if "recording_service" not in st.session_state:
    st.session_state.recording_service = RecordingService()

analytics_service: AnalyticsService = st.session_state.analytics_service
recorder: RecordingService = st.session_state.recording_service

# ---------------------------------------------------------------------------
# Sidebar — recording control and analysis scope
# ---------------------------------------------------------------------------
record_status = recorder.status()
node_online = recorder.last_error is None
sessions_frame = recorder.sessions() if node_online else None

with st.sidebar:
    st.markdown("### ⏺️ Session Recording")
    # The controls are always rendered and disabled when the node is
    # unreachable, rather than disappearing — a control that vanishes looks
    # like a missing feature, a disabled one explains itself.
    offline_help = None if node_online else "Broadcast server offline"

    if not node_online:
        st.warning("Broadcast server offline — recording unavailable.")
    elif record_status.get("recording"):
        st.success(
            f"**Recording:** {record_status.get('name', '')}  \n"
            f"{record_status.get('rows', 0)} samples · "
            f"{record_status.get('elapsed_s', 0):.0f}s"
        )
    else:
        st.caption(
            f"Idle — logging to the archive as session "
            f"`{record_status.get('id', '')}` "
            f"({record_status.get('rows', 0)} samples)."
        )

    st.text_input("Recording name", key="recording_name",
                  placeholder="e.g. morning rush test", disabled=not node_online)

    if record_status.get("recording"):
        if st.button("⏹️ Stop Recording", width="stretch", type="primary",
                     key="stop_recording", disabled=not node_online,
                     help=offline_help):
            result = recorder.stop()
            st.toast(f"Saved {result.get('rows', 0)} samples", icon="💾")
            st.rerun()
    else:
        if st.button("⏺️ Start Recording", width="stretch", type="primary",
                     key="start_recording", disabled=not node_online,
                     help=offline_help):
            recorder.start(st.session_state.get("recording_name", ""))
            st.toast("Recording started", icon="⏺️")
            st.rerun()

    if st.button("💾 Save Session Now", width="stretch", key="save_now",
                 disabled=not node_online, help=offline_help):
        written = recorder.save_now()
        st.toast(
            "Workbooks written"
            if written.get("session") or written.get("master")
            else "Nothing to write yet",
            icon="💾",
        )

    st.divider()
    st.markdown("### 📈 Analytics Controls")

    scope_options = ["This session", "All sessions"]
    session_labels: dict[str, str] = {}
    if sessions_frame is not None and not sessions_frame.empty:
        for _, entry in sessions_frame.iterrows():
            name = str(entry.get("Session Name", "") or "")
            stamp = str(entry.get("Session", ""))
            label = f"{stamp}" + (f" — {name}" if name and name != "auto" else "")
            session_labels[label] = stamp
        scope_options += list(session_labels)

    scope = st.selectbox("Data scope", options=scope_options, index=0, key="scope",
                         help="'All sessions' reads the archive on disk — every "
                              "run ever recorded, not just the live hour.")

    time_range = st.selectbox(
        "Time Range",
        options=[6, 12, 24, 48],
        index=2,
        format_func=lambda h: f"Last {h} hours",
        key="time_range",
    )
    interval = st.selectbox(
        "Data Interval",
        options=[5, 15, 30],
        index=1,
        format_func=lambda m: f"Every {m} minutes",
        key="interval",
    )

    if st.button("Regenerate Data", width="stretch", type="primary", key="regenerate"):
        analytics_service.refresh(hours=time_range, interval_minutes=interval)
        recorder.invalidate()
        st.toast("Analytics data refreshed!", icon="📈")

    st.divider()
    render_sidebar_nav()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = analytics_service.get_analytics(hours=time_range, interval_minutes=interval)
is_live = analytics_service.is_live_data

if scope == "All sessions":
    archive_raw = recorder.records()
    scope_note = "every session ever recorded"
elif scope == "This session":
    archive_raw = recorder.records(session=record_status.get("id") or None)
    scope_note = "the session currently being logged"
else:
    archive_raw = recorder.records(session=session_labels.get(scope))
    scope_note = f"session {session_labels.get(scope, scope)}"

archive = aa.prepare(archive_raw)
stats = aa.headline_stats(archive)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
render_page_header(
    "📈 Traffic Analytics",
    "Historical trends, congestion analysis, and data export",
    badge="Live Session Data" if is_live else "Simulation Data",
)

if is_live:
    st.success(
        f"Showing real analytics recorded from the live broadcast stream "
        f"({len(data.vehicle_timeline)} data points this session). "
        "Waiting time is estimated from congestion — the camera has no direct wait-time sensor."
    )
else:
    st.info(
        "Live broadcast stream unreachable — showing simulated demo data instead. "
        "Start `broadcast_server.py` and reload to see real analytics."
    )

# ---------------------------------------------------------------------------
# Archive headline + findings
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Recorded Archive</div>', unsafe_allow_html=True)

if archive.empty:
    st.info(
        "No archived telemetry for this scope yet. The broadcast server writes "
        "every sample to `traffic_all_sessions.csv`; press **Start Recording** "
        "to capture a named run, or just leave the server going and reload."
    )
else:
    st.caption(f"Analysing {scope_note} — {stats['samples']:,} samples "
               f"({stats['duration_min']:.0f} minutes) across "
               f"{stats['sessions']} session(s).")

    render_kpi_grid([
        {"label": "Samples", "value": f"{stats['samples']:,}", "icon": "📊"},
        {"label": "Duration", "value": f"{stats['duration_min']:.0f} min", "icon": "⏳"},
        {"label": "Sessions", "value": str(stats["sessions"]), "icon": "🗂️"},
        {"label": "Avg Vehicles", "value": f"{stats['avg_vehicles']:.1f}", "icon": "🚗"},
        {"label": "Peak Vehicles", "value": str(stats["peak_vehicles"]), "icon": "📈"},
        {"label": "Avg Speed", "value": f"{stats['avg_speed_kmh']:.1f} km/h", "icon": "⚡"},
        {"label": "Peak Speed", "value": f"{stats['peak_speed_kmh']:.0f} km/h", "icon": "🏁"},
        {"label": "Avg Congestion", "value": f"{stats['avg_congestion_pct']:.0f}%",
         "icon": "🔥"},
        {"label": "Time Congested", "value": f"{stats['congested_pct']:.0f}%", "icon": "🚧"},
        {"label": "Busiest Hour", "value": str(stats["busiest_hour"]), "icon": "🕒"},
        {"label": "Emergency Samples", "value": str(stats["emergency_samples"]),
         "icon": "🚑"},
        {"label": "Peak Congestion", "value": f"{stats['peak_congestion_pct']:.0f}%",
         "icon": "⚠️"},
    ], columns=4)

    st.markdown("#### What the data says")
    st.caption(
        "Generated from the archive above — the charts show what happened, "
        "these sentences say what it means."
    )
    for finding in aa.interpret(archive):
        st.markdown(f"- {finding}")

# ---------------------------------------------------------------------------
# Archive charts
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Archive Insights</div>', unsafe_allow_html=True)

tab_demand, tab_speedcong, tab_signal, tab_spread, tab_compare, tab_shape = st.tabs([
    "🕒 Demand by Hour",
    "📉 Speed vs Congestion",
    "🚦 Signal Behaviour",
    "📊 Distribution",
    "🗂️ Session Comparison",
    "🌊 Session Shape",
])

with tab_demand:
    st.plotly_chart(hourly_demand_chart(aa.hourly_profile(archive)), width="stretch")
    st.markdown(
        "**How to read this.** Bars are the average number of vehicles in frame "
        "during each clock hour, across every session in scope; the orange line "
        "is average congestion on the right-hand axis. "
        "**Why it is useful:** the tallest bars are when the road is under most "
        "demand, so they tell you when *not* to schedule roadworks or deliveries, "
        "and which hours would benefit most from extra green time. "
        "Hours with few samples are less trustworthy — check the Raw Data table "
        "below before acting on a spike."
    )

with tab_speedcong:
    st.plotly_chart(speed_vs_congestion_chart(aa.speed_vs_congestion(archive)),
                    width="stretch")
    st.markdown(
        "**How to read this.** Samples are grouped into congestion bands, and "
        "each point is the mean measured speed in that band. "
        "**Why it is useful:** this is the practical form of the speed/flow "
        "relationship. A flat line means the road is absorbing its traffic "
        "without slowing down. Where the line bends downward is the point at "
        "which congestion starts costing real journey time — that inflection is "
        "the threshold worth designing an intervention around, and it is far "
        "more actionable than an average speed on its own."
    )

with tab_signal:
    signal_left, signal_right = st.columns(2)
    with signal_left:
        st.plotly_chart(phase_share_chart(aa.phase_distribution(archive)),
                        width="stretch")
    with signal_right:
        st.plotly_chart(level_share_chart(aa.level_distribution(archive)),
                        width="stretch")
    st.markdown(
        "**How to read this.** Sampling is once per second, so a bar height is "
        "literally seconds spent in that state. Left: what the traffic light was "
        "showing. Right: how congested the road actually was. "
        "**Why it is useful:** the signal is demand-responsive — GREEN whenever "
        "there is traffic to serve, RED on an empty road, YELLOW for five "
        "seconds in between. A record dominated by RED means the road is mostly "
        "empty and the light is correctly resting; one dominated by GREEN means "
        "demand is high enough that it rarely gets to. A large YELLOW share is a "
        "warning sign: it means the light is changing often, and every change "
        "costs five seconds of transition."
    )

with tab_spread:
    st.plotly_chart(congestion_histogram_chart(aa.congestion_histogram(archive)),
                    width="stretch")
    st.markdown(
        "**How to read this.** Each bar counts the seconds spent at that "
        "congestion level. "
        "**Why it is useful:** an average hides the shape of the problem. One "
        "tall bar near zero with a thin tail to the right is a road that is "
        "usually fine with occasional spikes — manage the spikes. A broad hump "
        "in the middle or a second bump on the right is a road under sustained "
        "strain, which needs capacity rather than timing changes. These two "
        "cases can share an identical mean congestion figure."
    )

with tab_compare:
    comparison = aa.session_comparison(
        sessions_frame if sessions_frame is not None else None)
    metric = st.selectbox(
        "Compare sessions by",
        options=["Mean Congestion %", "Peak Vehicles", "Mean Speed (km/h)", "Samples"],
        key="compare_metric",
    )
    st.plotly_chart(session_comparison_chart(comparison, metric), width="stretch")
    if comparison is not None and not comparison.empty:
        st.dataframe(comparison.drop(columns=["label"], errors="ignore"),
                     width="stretch", hide_index=True)
    st.markdown(
        "**How to read this.** One bar per recorded session, most recent on the "
        "right. "
        "**Why it is useful:** a single session tells you about one afternoon; "
        "the trend across sessions tells you whether the road is getting worse. "
        "Record under comparable conditions — same camera position, same time of "
        "day — or you are comparing the weather rather than the traffic."
    )

with tab_shape:
    st.plotly_chart(rolling_trend_chart(aa.rolling_trend(archive)), width="stretch")
    mix_left, mix_right = st.columns([1.4, 1])
    with mix_left:
        st.plotly_chart(vehicle_mix_chart(aa.vehicle_mix(archive)), width="stretch")
    with mix_right:
        st.markdown("**Traffic composition**")
        mix = aa.vehicle_mix(archive)
        if not mix.empty:
            total = mix["count"].sum()
            for _, row in mix.iterrows():
                share = row["count"] / total * 100 if total else 0
                st.markdown(f"- **{row['vehicle_type']}**: "
                            f"{row['count']:.2f} avg in frame ({share:.0f}%)")
        st.caption(
            "Averages, not totals: a vehicle appears in every sample it stays in "
            "frame, so a sum would measure loitering rather than volume."
        )
    st.markdown(
        "**How to read this.** A one-minute rolling mean over the one-second "
        "samples, so short-lived detection noise is smoothed away while genuine "
        "surges survive. "
        "**Why it is useful:** this is the narrative of a session — when traffic "
        "built up, how long it stayed, and whether speed fell as it did. Heavy "
        "vehicles matter out of proportion to their number: a bus occupies the "
        "road like three cars, which is why the congestion index is weighted by "
        "class rather than a plain count."
    )

# ---------------------------------------------------------------------------
# Live session KPIs and export
# ---------------------------------------------------------------------------
avg_vehicles = round(data.vehicle_timeline["vehicle_count"].mean(), 1)
peak_vehicles = int(data.vehicle_timeline["vehicle_count"].max())
avg_congestion = round(data.congestion_trend["congestion_index"].mean(), 1)
avg_wait = round(data.waiting_time["avg_waiting_seconds"].mean(), 1)
dominant_type = data.vehicle_types.loc[data.vehicle_types["count"].idxmax(), "vehicle_type"]

has_speed = data.speed_timeline is not None and not data.speed_timeline.empty
if has_speed:
    measured_speeds = data.speed_timeline.loc[
        data.speed_timeline["avg_speed_kmh"] > 0, "avg_speed_kmh"
    ]
    avg_speed = round(measured_speeds.mean(), 1) if not measured_speeds.empty else 0.0
    peak_speed = round(data.speed_timeline["avg_speed_kmh"].max(), 1)
else:
    avg_speed = peak_speed = 0.0

st.markdown('<div class="section-title">Live Session</div>', unsafe_allow_html=True)

summary_kpis = [
    {"label": "Avg Vehicle Count", "value": str(avg_vehicles), "icon": "🚗"},
    {"label": "Peak Vehicle Count", "value": str(peak_vehicles), "icon": "📈"},
    {"label": "Avg Congestion", "value": f"{avg_congestion}%", "icon": "🔥"},
    {"label": "Avg Speed", "value": f"{avg_speed} km/h", "icon": "⚡"},
    {"label": "Peak Speed", "value": f"{peak_speed} km/h", "icon": "🏁"},
    {"label": "Avg Waiting Time", "value": f"{avg_wait}s", "icon": "⏱️"},
    {"label": "Top Vehicle Type", "value": dominant_type, "icon": "🏷️"},
    {"label": "Data Points", "value": str(len(data.vehicle_timeline)), "icon": "📊"},
]
render_kpi_grid(summary_kpis, columns=4)

export_col1, export_col2, export_col3, export_col4 = st.columns([1.8, 1, 1, 1])
with export_col1:
    st.markdown('<div class="section-title">Analytics Charts</div>',
                unsafe_allow_html=True)
with export_col2:
    csv_data = analytics_service.export_csv(hours=time_range, interval_minutes=interval)
    st.download_button(
        label="⬇️ Export CSV",
        data=csv_data,
        file_name=f"traffic_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width="stretch",
    )
with export_col3:
    excel_bytes = analytics_service.fetch_export_bytes() if is_live else None
    if excel_bytes:
        st.download_button(
            label="⬇️ This Session",
            data=excel_bytes,
            file_name=f"traffic_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            help="Excel workbook covering only the run in progress",
        )
    else:
        st.button("⬇️ This Session", disabled=True, width="stretch",
                  help="Live stream offline")
with export_col4:
    master_bytes = recorder.fetch_master_excel() if node_online else None
    if master_bytes:
        st.download_button(
            label="⬇️ All Sessions",
            data=master_bytes,
            file_name=f"traffic_all_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            help="Excel workbook containing every session ever recorded",
        )
    else:
        st.button("⬇️ All Sessions", disabled=True, width="stretch",
                  help="Live stream offline")

# ---------------------------------------------------------------------------
# Live charts — tab layout
# ---------------------------------------------------------------------------
tab_timeline, tab_speed, tab_density, tab_types, tab_congestion, tab_wait, tab_heatmap = st.tabs(
    [
        "📉 Timeline",
        "⚡ Speed",
        "🚦 Density",
        "🥧 Vehicle Types",
        "🔥 Congestion",
        "⏱️ Waiting Time",
        "🗺️ Heatmap",
    ]
)

with tab_timeline:
    st.plotly_chart(vehicle_count_timeline(data.vehicle_timeline), width="stretch")

with tab_speed:
    if has_speed:
        st.plotly_chart(speed_timeline_chart(data.speed_timeline), width="stretch")
        if is_live:
            speed_source = st.session_state.status_service.last_telemetry.get(
                "speed_accuracy", "size-based auto calibration"
            )
            st.caption(
                f"Measured in km/h from the live camera using {speed_source}. "
                "Samples with no vehicles in frame are excluded from the average."
            )
        else:
            st.caption("Simulated speeds — start the broadcast server for real measurements.")
    else:
        st.info("No speed data in this dataset.")

with tab_density:
    st.plotly_chart(traffic_density_chart(data.density_timeline), width="stretch")

with tab_types:
    type_col1, type_col2 = st.columns([1.5, 1])
    with type_col1:
        st.plotly_chart(vehicle_types_pie(data.vehicle_types), width="stretch")
    with type_col2:
        st.markdown("**Type Breakdown**")
        total = data.vehicle_types["count"].sum()
        for _, row in data.vehicle_types.iterrows():
            pct = row["count"] / total * 100
            st.markdown(f"- **{row['vehicle_type']}**: {row['count']} ({pct:.1f}%)")

with tab_congestion:
    st.plotly_chart(congestion_trend_chart(data.congestion_trend), width="stretch")
    st.caption("Shaded region indicates critical congestion (≥ 70%).")

with tab_wait:
    st.plotly_chart(waiting_time_chart(data.waiting_time), width="stretch")
    if is_live:
        st.caption("Estimated from congestion index — no direct wait-time sensor on the live camera.")

with tab_heatmap:
    st.plotly_chart(traffic_heatmap(data.heatmap), width="stretch")
    st.caption("Placeholder heatmap — lane traffic intensity by hour of day.")

# ---------------------------------------------------------------------------
# Raw data expander
# ---------------------------------------------------------------------------
with st.expander("View Raw Data", expanded=False):
    dataset_map = {
        "Vehicle Timeline": data.vehicle_timeline,
        "Density Timeline": data.density_timeline,
        "Vehicle Types": data.vehicle_types,
        "Congestion Trend": data.congestion_trend,
        "Waiting Time": data.waiting_time,
        "Heatmap": data.heatmap,
    }
    if has_speed:
        dataset_map["Speed Timeline (km/h)"] = data.speed_timeline
    if not archive.empty:
        dataset_map["Archive — Hourly Profile"] = aa.hourly_profile(archive)
        dataset_map["Archive — Speed vs Congestion"] = aa.speed_vs_congestion(archive)
        dataset_map["Archive — Phase Share"] = aa.phase_distribution(archive)
        dataset_map["Archive — Raw Telemetry"] = archive
    dataset = st.selectbox("Dataset", list(dataset_map), key="dataset")
    st.dataframe(dataset_map[dataset], width="stretch", hide_index=True)
