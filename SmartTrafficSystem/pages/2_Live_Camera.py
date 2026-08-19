"""Live Camera page — live MJPEG broadcast feed with simulated fallback."""

import pandas as pd
import streamlit as st

from backend.services.camera_service import CameraService  # noqa: F401  (type hint)
from backend.services.system_status_service import VIDEO_FEED_URL
from backend.services.video_source_service import VideoSourceService
from components.navigation import render_sidebar_nav
from components.page_header import render_page_header
from config.settings import get_settings
from utils.session import init_session_state
from utils.theme import apply_dark_theme

settings = get_settings()

# How often the live fragments redraw themselves while Live Refresh is on.
CAMERA_REFRESH_SECONDS: float = 0.5

st.set_page_config(
    page_title=f"{settings.app_title} | Live Camera",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()
init_session_state()

# Rebind to the one Settings instance the services hold. Writing the sidebar's
# threshold onto a fresh get_settings() object left the slider with no effect.
settings = st.session_state.settings

if "camera_auto_refresh" not in st.session_state:
    st.session_state.camera_auto_refresh = True

#: Passed to `st.fragment(run_every=...)`. None stops the timer, which is how
#: the Live Refresh toggle turns updating off.
REFRESH_SECONDS: float | None = (CAMERA_REFRESH_SECONDS
                                 if st.session_state.camera_auto_refresh else None)

if "show_boxes" not in st.session_state:
    st.session_state.show_boxes = True

if "show_ids" not in st.session_state:
    st.session_state.show_ids = True

if "show_lanes" not in st.session_state:
    st.session_state.show_lanes = True

if "show_labels" not in st.session_state:
    st.session_state.show_labels = True

if "show_source_panel" not in st.session_state:
    st.session_state.show_source_panel = False

camera_service: CameraService = st.session_state.camera_service
video_service: VideoSourceService = st.session_state.video_source_service

#: How each kind of source is badged in the picker.
SOURCE_ICONS: dict[str, str] = {
    "camera": "📷", "file": "🎞️", "stream": "🌐", "none": "⚫",
}


def describe_source(source: dict) -> str:
    """One line naming the source and, for a video, where it has got to."""
    if not source or source.get("kind") in (None, "none"):
        return "⚫ **No video source** — nothing is being watched yet."

    icon = SOURCE_ICONS.get(source.get("kind", ""), "❓")
    label = source.get("label") or "unknown"
    size = f"{source.get('width', 0)}×{source.get('height', 0)}"

    if not source.get("is_file"):
        return f"{icon} **{label}** — live, {size}"

    state = ("paused" if source.get("paused")
             else "finished" if source.get("finished") else "playing")
    return (f"{icon} **{label}** — {source.get('position_text', '0:00')} / "
            f"{source.get('duration_text', '?')} · {state} · "
            f"{source.get('native_fps', 0):.0f} fps, {size}")


def remember_source_result(result: dict, success: str) -> None:
    """Store the outcome of a source command for the next run to display.

    Widget callbacks run before the page does, so anything they want to say
    has to be left in session state rather than written to the screen.
    """
    if result.get("error"):
        st.session_state.source_error = result["error"]
    else:
        st.session_state.source_notice = success


def apply_source_change(result: dict, success: str) -> None:
    """Report the outcome of a source command and re-render with the new state."""
    remember_source_result(result, success)
    st.rerun(scope="app")


def apply_playback_rate() -> None:
    """Send the newly chosen playback speed to the CV node.

    This is a change callback rather than a comparison in the page body on
    purpose. Comparing the widget against the node's reported rate and posting
    on a mismatch loops forever whenever the two disagree — which they do for
    at least one rerun after every change, and permanently if the node ever
    declines.
    """
    rate = float(st.session_state.playback_rate_choice)
    remember_source_result(
        st.session_state.video_source_service.set_rate(rate),
        "Playing as fast as possible." if rate == 0 else f"Playback speed set to {rate:g}×.",
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📹 Camera Controls")
    st.divider()

    # Keyed rather than assigned from the return value: the fragments' refresh
    # interval is read at the top of the script, so the toggle has to have
    # written session state before the body runs, not halfway down it.
    st.toggle(
        "Live Refresh",
        key="camera_auto_refresh",
        help="Continuously update the live numbers.",
    )

    st.markdown("**Overlay Options**")
    st.session_state.show_boxes = st.checkbox("Bounding Boxes", value=st.session_state.show_boxes)
    st.session_state.show_ids = st.checkbox("Tracking IDs", value=st.session_state.show_ids)
    st.session_state.show_lanes = st.checkbox("Lane Markings", value=st.session_state.show_lanes)
    st.session_state.show_labels = st.checkbox("Class Labels", value=st.session_state.show_labels)

    st.divider()
    st.markdown("**Detection Threshold**")
    # Keyed, with no `value=` argument: passing a default that this page then
    # mutates changes the widget's identity every run, which reset the slider
    # and made it impossible to drag back down.
    if "detection_threshold" not in st.session_state:
        st.session_state.detection_threshold = float(settings.detection_threshold)

    threshold = st.slider(
        "Confidence",
        min_value=0.50,
        max_value=0.99,
        step=0.01,
        key="detection_threshold",
        help="Minimum YOLO confidence to display detections.",
    )
    settings.detection_threshold = threshold

    if st.button("Refresh Frame", width="stretch", type="primary"):
        st.rerun()

    st.divider()
    render_sidebar_nav()

# ---------------------------------------------------------------------------
# Live stream status
# ---------------------------------------------------------------------------
status_service = st.session_state.status_service
metrics = status_service.get_dashboard_metrics()
live_connected = status_service.is_live_stream_connected

badge = "Live Feed" if live_connected else ("Simulation Active" if settings.simulation_mode else "Offline")
render_page_header(
    "📹 Live Camera Feed",
    "Real-time traffic monitoring with YOLO detection overlays",
    badge=badge,
)

# ---------------------------------------------------------------------------
# Video source — the live camera, an uploaded video, or a camera URL
# ---------------------------------------------------------------------------
# The whole pipeline runs identically on all three, so this is a picker rather
# than a mode switch: whatever is chosen, the same detector, the same km/h, the
# same signal and the same logs follow it.
#
# The status comes from the telemetry poll that has already happened, so
# watching the feed costs no extra requests; the catalogue of cameras and
# uploads is only fetched while the panel is open.
def current_source() -> tuple[dict, bool]:
    """The node's source block, and whether the node itself is unreachable."""
    if status_service.is_live_stream_connected:
        live = status_service.last_telemetry.get("source") or {}
        if live:
            return live, False
    return video_service.status(), video_service.last_error is not None


@st.fragment(run_every=REFRESH_SECONDS)
def render_source_status() -> None:
    """Name the source and how far through it the node is.

    Reads the snapshot the live-view fragment already fetched rather than
    polling again, so keeping this line live costs no extra request.
    """
    source, node_offline = current_source()

    source_col, toggle_col = st.columns([4, 1])
    with source_col:
        if node_offline:
            st.markdown("⚠️ **CV node unreachable** — start `broadcast_server.py` on "
                        "the camera machine, or correct its address on the Settings page.")
        else:
            st.markdown(describe_source(source))
    with toggle_col:
        wants_panel = st.toggle(
            "Change source",
            value=st.session_state.show_source_panel,
            help="Upload a video, paste a camera URL, or pick a connected camera.",
        )
    # The panel itself is rendered outside this fragment, so showing or
    # hiding it takes a whole-page rerun rather than a fragment one.
    if wants_panel != st.session_state.show_source_panel:
        st.session_state.show_source_panel = wants_panel
        st.rerun(scope="app")


# Drawn here rather than inside the fragment above: the fragment redraws on a
# timer, and a message popped from session state cannot survive that — it would
# be wiped half a second after the upload it is reporting on. In the page body
# it stays put until something actually happens.
if notice := st.session_state.pop("source_notice", None):
    st.success(notice)
if problem := st.session_state.pop("source_error", None):
    st.error(problem)

render_source_status()

if st.session_state.show_source_panel:
    # The status line lives in a fragment, so its copy of the source block is
    # out of scope here; the panel reads it again for the position controls.
    source, _node_offline = current_source()
    # Streamlit runs every tab body on every rerun, so the catalogue is fetched
    # once here rather than by each tab that needs it.
    catalogue = video_service.catalogue()
    upload_tab, url_tab, camera_tab, library_tab = st.tabs(
        ["📁 Upload a video", "🌐 Camera / stream URL", "📷 Connected camera",
         "🗂️ Uploaded videos"]
    )

    with upload_tab:
        uploaded = st.file_uploader(
            "Road footage to analyse",
            type=["mp4", "mov", "avi", "mkv", "webm", "m4v", "wmv", "mpg", "mpeg"],
            help="The video is sent to the CV node, decoded once to check it "
                 "plays, and then run through the full pipeline.",
        )
        loop_upload = st.checkbox("Loop when it reaches the end", value=False,
                                  key="loop_upload")
        if st.button("▶ Analyse this video", type="primary", disabled=uploaded is None,
                     width="stretch"):
            with st.spinner(f"Uploading {uploaded.name} and opening it…"):
                result = video_service.upload(uploaded.name, uploaded, loop=loop_upload)
            stored = result.get("stored", uploaded.name)
            # The node stores each distinct video once, so re-sending one you
            # already uploaded reuses it rather than filling the disk.
            apply_source_change(result, (
                f"Now analysing {stored} — already on the node, so it was reused "
                "rather than stored a second time."
                if result.get("reused") else f"Now analysing {stored}."
            ))
        st.caption(
            "Streamlit accepts uploads up to 200 MB by default — raise it with "
            "`streamlit run app.py --server.maxUploadSize 2000`, or for a very "
            "large file copy it next to `broadcast_server.py` and type its name "
            "on the **Camera / stream URL** tab instead."
        )

    with url_tab:
        url = st.text_input(
            "Address",
            placeholder="rtsp://user:pass@192.168.1.40:554/stream1",
            help="An RTSP or MJPEG camera, an HLS playlist, a YouTube live "
                 "traffic camera, or the name of a video file on the CV node.",
        )
        loop_url = st.checkbox("Loop when it reaches the end", value=False, key="loop_url")
        if st.button("🔌 Connect", type="primary", disabled=not url.strip(),
                     width="stretch"):
            with st.spinner(f"Opening {url.strip()}…"):
                result = video_service.use(url.strip(), loop=loop_url)
            apply_source_change(result, f"Now watching {url.strip()}.")
        st.caption(
            "**RTSP** `rtsp://user:pass@host:554/stream1` · **IP camera MJPEG** "
            "`http://host:8080/video` · **HLS** `https://host/live.m3u8` · "
            "**YouTube live** needs `pip install yt-dlp` on the CV node · "
            "**a file already on that machine** `road.mp4`"
        )

    with camera_tab:
        cameras = catalogue.get("cameras", [])
        if cameras:
            chosen = st.selectbox(
                "Camera",
                options=[camera["spec"] for camera in cameras],
                format_func=lambda spec: next(
                    (f"[{c['index']}] {c['name']}" for c in cameras
                     if c["spec"] == spec), spec),
            )
            if st.button("📷 Use this camera", type="primary", width="stretch"):
                with st.spinner("Opening the camera…"):
                    result = video_service.use(chosen)
                apply_source_change(result, "Switched to the live camera.")
        else:
            st.info(
                "No cameras were named. Install `pygrabber` on the CV node "
                "(`pip install pygrabber`) to list them, or enter an index "
                "such as `0` on the **Camera / stream URL** tab."
            )
        st.caption("Windows renumbers cameras when devices are replugged, so a "
                   "name survives a restart better than an index does.")

    with library_tab:
        uploads = catalogue.get("uploads", [])
        if not uploads:
            st.info("Nothing uploaded yet, and no footage in the node's "
                    "`vedios/` folder.")
        for entry in uploads:
            name_col, use_col, drop_col = st.columns([4, 1, 1])
            folder = entry.get("folder", "")
            name_col.markdown(f"**{entry['name']}** · {entry['size_mb']} MB · "
                              f"{entry['uploaded_at']}"
                              + (f" · `{folder}/`" if folder else ""))
            if use_col.button("▶", key=f"use_{entry['name']}", help="Analyse this video"):
                apply_source_change(video_service.use(entry["spec"]),
                                    f"Now analysing {entry['name']}.")
            # Only footage the dashboard uploaded is deletable from here. The
            # rest is the source material somebody put on the machine by hand,
            # and a stray click should not be able to destroy it.
            if entry.get("removable", True):
                if drop_col.button("🗑", key=f"del_{entry['name']}", help="Delete it"):
                    apply_source_change(video_service.delete(entry["name"]),
                                        f"Deleted {entry['name']}.")
            else:
                drop_col.button("🗑", key=f"del_{entry['name']}", disabled=True,
                                help="Source footage — delete it on the CV node")
        st.caption("Videos in the node's `uploads/` and `vedios/` folders. Drop a "
                   "file into `vedios/` and it appears here without uploading.")

    if source.get("is_file"):
        st.markdown("**Position**")
        target = st.slider("Jump to", 0.0, 100.0,
                           value=float(source.get("progress", 0.0)) * 100.0,
                           step=0.5, format="%.1f%%", label_visibility="collapsed")
        if st.button("Jump", width="stretch"):
            apply_source_change(video_service.seek(target / 100.0),
                                f"Jumped to {target:.0f}%.")

        # Keyed, with no `value=`: passing a default that the node then changes
        # would give the widget a new identity on every rerun and make it
        # impossible to hold a setting.
        if "playback_rate_choice" not in st.session_state:
            st.session_state.playback_rate_choice = float(source.get("playback_rate", 1.0))
        rates = sorted({0.25, 0.5, 1.0, 2.0, 4.0,
                        st.session_state.playback_rate_choice} - {0.0}) + [0.0]
        st.select_slider(
            "Playback speed",
            options=rates,
            key="playback_rate_choice",
            on_change=apply_playback_rate,
            format_func=lambda rate: "as fast as possible" if rate == 0 else f"{rate:g}×",
            help="Only how quickly frames are fed to the detector. Speeds are "
                 "measured against the video's own clock, so the km/h reported "
                 "do not change with it.",
        )

    st.caption("↻ The live numbers keep refreshing on their own while this panel "
               "is open — they redraw separately, so typing here is not interrupted.")


@st.fragment(run_every=REFRESH_SECONDS)
def render_live_view() -> None:
    """Draw the feed, the live numbers and the detection table.

    This is a fragment so that Streamlit reruns *it* on the timer rather
    than the whole page. A page-level `sleep(); st.rerun()` loop leaves the
    script permanently mid-run, and when the browser's websocket blips —
    a backgrounded tab, a network hiccup — the queued rerun is orphaned and
    the numbers stop advancing until a widget is touched. A fragment lets
    the script finish, so the timer survives a reconnect.
    """
    metrics = status_service.get_dashboard_metrics()
    live_connected = status_service.is_live_stream_connected
    source, _node_offline = current_source()   # reachability is reported above

    feed_col, stats_col = st.columns([2.5, 1])

    if live_connected:
        telemetry = status_service.last_telemetry

        with feed_col:
            st.markdown(
                f'<img src="{VIDEO_FEED_URL}" style="width:100%; border-radius:10px; '
                f'border: 2px solid #334155;">',
                unsafe_allow_html=True,
            )

            if source.get("is_file"):
                # A recorded video is the one source with somewhere to go, so it
                # gets transport controls. They are here rather than in the panel
                # above because they are used while watching, not while choosing.
                st.progress(min(1.0, float(source.get("progress", 0.0))))
                back, play, forward, again, loop_col = st.columns(5)
                paused = bool(source.get("paused"))
                if back.button("⏪ 5s", width="stretch"):
                    apply_source_change(video_service.nudge(-5), "Jumped back 5 seconds.")
                if play.button("▶ Play" if paused else "⏸ Pause", width="stretch",
                               type="primary"):
                    apply_source_change(video_service.resume() if paused else video_service.pause(),
                                        "Resumed." if paused else "Paused.")
                if forward.button("5s ⏩", width="stretch"):
                    apply_source_change(video_service.nudge(5), "Jumped forward 5 seconds.")
                if again.button("⏮ Restart", width="stretch"):
                    apply_source_change(video_service.restart(), "Restarted from the beginning.")
                if loop_col.button("🔁 Loop: " + ("on" if source.get("loop") else "off"),
                                   width="stretch"):
                    apply_source_change(video_service.set_loop(not source.get("loop")),
                                        "Looping " + ("off." if source.get("loop") else "on."))
                if source.get("finished") and not source.get("loop"):
                    st.info("The video has played out. Restart it, or load another "
                            "source above — the node keeps running either way.")
                st.caption(
                    f"Source: {source.get('label', 'video file')} | "
                    f"{source.get('frames_read', 0)} frames analysed"
                    + (f", {source['frames_skipped']} skipped to stay in real time"
                       if source.get("frames_skipped") else "")
                )
            else:
                st.caption(f"Source: {source.get('label', 'LIVE BROADCAST')} | {VIDEO_FEED_URL}")

        with stats_col:
            traffic = metrics.traffic
            emergency_active = metrics.emergency_status.value != "None"

            st.metric("Vehicle Count", traffic.vehicle_count)
            st.metric("Traffic Density", traffic.traffic_density)
            st.metric(
                "Average Speed",
                f"{traffic.average_speed_kmh:.1f} km/h",
                delta=f"peak {telemetry.get('max_speed_kmh', 0):.0f} km/h",
                delta_color="off",
            )
            st.metric("Emergency Vehicles", traffic.emergency_vehicle_count)

            if emergency_active:
                kinds = telemetry.get("emergency_types") or []
                kind_text = ", ".join(k.title() for k in kinds) or "Emergency vehicle"
                colours = telemetry.get("siren_colours") or []
                siren_text = (f" — {', '.join(colours)} beacon flashing" if colours
                              else "")
                st.error(
                    f"🚨 **{kind_text}** detected "
                    f"({traffic.emergency_vehicle_count} vehicle(s)){siren_text}"
                )
            else:
                st.success("✅ No emergency vehicles")

            # A marked vehicle with its lights off is deliberately given no
            # priority, but hiding it entirely would look like a miss.
            marked = int(telemetry.get("marked_no_siren", 0) or 0)
            if marked:
                st.info(f"{marked} marked vehicle(s) in view with no siren running "
                        "— no priority given")

            # Beacons flash 1–4 times a second. If the pipeline is sampling slower
            # than that, a siren can pass unseen, and saying so is more useful than
            # a confident "no emergency vehicles".
            if telemetry.get("siren_undersampled"):
                st.warning(
                    f"⚠️ Sampling at {telemetry.get('siren_sample_hz', 0):.0f} frames/s — "
                    "too slow to see a beacon flash reliably. Lower `--imgsz` or "
                    "use a smaller model."
                )

            st.markdown("**Live Telemetry**")
            st.markdown(f"- Density: **{traffic.traffic_density}**")
            st.markdown(f"- Vehicles: **{traffic.vehicle_count}** "
                        f"({telemetry.get('moving_vehicles', 0)} moving, "
                        f"{telemetry.get('stopped_vehicles', 0)} stopped)")
            st.markdown(f"- Avg Speed: **{traffic.average_speed_kmh:.1f} km/h**")
            st.markdown(f"- Congestion Index: **{telemetry.get('congestion_index', 0):.2f}**")
            st.markdown(f"- Seen this session: **{telemetry.get('unique_vehicles', 0)}** vehicles")
            st.markdown(f"- Pipeline: **{telemetry.get('fps', 0):.1f} FPS** "
                        f"({telemetry.get('inference_ms', 0):.0f} ms/frame)")
            st.markdown(f"- Emergency: **{metrics.emergency_status.value}** "
                        f"({traffic.emergency_vehicle_count} vehicle(s))")

            # Vehicles the detector found and the monitored region then threw
            # away. Shown because a region drawn around one carriageway
            # silently discards whatever rides beside it, and a class that
            # keeps to that lane reads as a class the detector cannot see.
            outside = telemetry.get("outside_roi_counts") or {}
            if outside:
                worst = ", ".join(
                    f"{n} {name}" for name, n in
                    sorted(outside.items(), key=lambda kv: -kv[1])[:3])
                st.caption(f"↔ {telemetry.get('outside_roi', 0)} detected outside "
                           f"the monitored region ({worst}) — not counted. Redraw "
                           "the region with `calibrate_speed.py` if a lane is missing.")

        # -----------------------------------------------------------------------
        # Live detection table straight from the broadcast telemetry
        # -----------------------------------------------------------------------
        st.markdown('<div class="section-title">Active Detections</div>', unsafe_allow_html=True)

        detections = telemetry.get("detections", [])
        if detections:
            rows = [
                {
                    "ID": d.get("id"),
                    "Class": str(d.get("class", "")).upper(),
                    "Speed (km/h)": "—" if d.get("speed_kmh") is None else d["speed_kmh"],
                    "Confidence": f"{d.get('confidence', 0):.0%}",
                    # A trailing "?" marks a vehicle recognised by its livery whose
                    # beacon is not running: shown, but given no priority.
                    "Emergency": (str(d.get("emergency_type", "yes")).upper()
                                  if d.get("emergency")
                                  else (f"{str(d.get('emergency_type', '')).upper()}?"
                                        if d.get("emergency_suspected") else "—")),
                    "Siren": (f"{d.get('siren_colour')} @ {d.get('siren_rate_hz', 0):.1f} Hz"
                              if d.get("emergency") and d.get("siren_colour") not in
                              (None, "none") else "—"),
                    "Recognised By": d.get("emergency_evidence", "—")
                    if (d.get("emergency") or d.get("emergency_suspected")) else "—",
                    "BBox": "({}, {}) → ({}, {})".format(*d.get("bbox", [0, 0, 0, 0])),
                }
                for d in detections
            ]
            frame_df = pd.DataFrame(rows)

            def _highlight_live_emergency(row: pd.Series) -> list[str]:
                """Red for a running siren, amber for livery without one."""
                if row["Emergency"] == "—":
                    return [""] * len(row)
                colour = ("rgba(245, 158, 11, 0.15)" if row["Emergency"].endswith("?")
                          else "rgba(239, 68, 68, 0.15)")
                return [f"background-color: {colour}"] * len(row)

            st.dataframe(
                frame_df.style.apply(_highlight_live_emergency, axis=1),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No confirmed vehicles in the current frame.")

        with st.expander("How speed is measured in km/h", expanded=False):
            st.markdown(
                f"""
                Pixels per second mean nothing on their own, so the broadcast
                server converts every vehicle's motion into real metres before
                reporting a speed.

                **Current mode: `{telemetry.get('speed_source', 'unknown')}` —
                {telemetry.get('speed_accuracy', 'unknown accuracy')}**

                - **homography** — the road plane was calibrated with
                  `calibrate_speed.py`, so each vehicle's ground-contact point is
                  projected onto the real road surface. Perspective is fully
                  accounted for.
                - **auto** — no calibration file yet. Each vehicle's own bounding
                  box is compared against the typical real height of its class
                  (a car is ~1.5 m tall), giving a metres-per-pixel scale at that
                  vehicle's exact position.

                Speed is measured across a short time window rather than between
                consecutive frames, so box jitter does not inflate the reading,
                and physically impossible values (from tracker ID switches) are
                discarded.
                """
            )

        with st.expander("How ambulances, police cars and fire trucks are detected",
                         expanded=False):
            st.markdown(
                f"""
                The YOLO model recognises only generic classes (car, truck, bus,
                motorcycle) — COCO has no ambulance or police-car class. Two cues
                are combined, with a deliberate hierarchy between them.

                **1. The siren light — what the vehicle is *doing*. This is the
                trigger.**
                A vehicle earns a cleared lane when it is responding to a call, and
                the visible sign of that is its beacon switching on and off. Each
                vehicle's roofline is measured for pure red and pure blue light,
                and the resulting signal must *switch* — nearly vanishing between
                flashes, at least three times, at 0.7–6 Hz, sustained for over a
                second — not merely brighten and dim. That last distinction is what
                separates a beacon from a car passing under street lights, and the
                beacon's **colour names the vehicle**: blue → police, red →
                ambulance (fire engine on a large vehicle), alternating red+blue →
                police.
                Current detector: `{telemetry.get('siren_detection', 'unknown')}`.

                **2. The livery — what the vehicle *looks like*. This names, it
                does not trigger.**
                CLIP compares each vehicle against descriptions of ambulances,
                police cars, fire trucks and ordinary traffic. It is used to tell
                an ambulance from a fire engine when both show a red beacon, and to
                raise confidence when both cues agree.
                Current recogniser: `{telemetry.get('emergency_recognition', 'unknown')}`.

                **Why livery alone is not enough.** An ambulance parked outside a
                hospital is still an ambulance, and does not need the junction held
                for it. Worse, "does this look like a police car?" is a question a
                dark saloon at night answers yes to: on the night footage in
                `vedios/`, appearance alone flagged 29% of all ordinary traffic.
                Requiring a working beacon brought that to zero while still
                catching every planted beacon. A marked vehicle with its lights off
                is shown with a **?** and highlighted amber — visible, but given no
                priority. Start the node with `--livery-triggers-emergency` to let
                livery trigger as well, which suits a camera watching a hospital
                approach rather than a public road.

                The **Recognised By** column shows exactly which cue fired, and
                **Siren** shows the colour and measured flash rate behind it.
                """
            )
    else:
        st.warning(
            f"⚠️ Live broadcast stream offline ({VIDEO_FEED_URL}). Showing simulated feed instead."
        )

        frame = camera_service.render_frame(
            show_boxes=st.session_state.show_boxes,
            show_ids=st.session_state.show_ids,
            show_lanes=st.session_state.show_lanes,
            show_labels=st.session_state.show_labels,
        )

        with feed_col:
            st.image(
                frame.image_bytes,
                caption=f"Source: {frame.source.upper()} | {frame.frame_width}×{frame.frame_height}",
                width="stretch",
            )

            legend_col1, legend_col2, legend_col3 = st.columns(3)
            with legend_col1:
                st.markdown("🟩 **Normal vehicle** — green bounding box")
            with legend_col2:
                st.markdown("🟥 **Emergency vehicle** — red bounding box")
            with legend_col3:
                st.markdown("🟨 **Lane markings** — dashed center lines")

        with stats_col:
            emergency_count = sum(1 for d in frame.detections if d.is_emergency)
            normal_count = len(frame.detections) - emergency_count

            st.metric("FPS", f"{frame.fps:.1f}")
            st.metric("Vehicles Detected", len(frame.detections))
            st.metric("Emergency Vehicles", emergency_count)

            if emergency_count > 0:
                st.error("🚨 Emergency vehicle detected in feed!")
            else:
                st.success("✅ No emergency vehicles")

            st.markdown("**Detection Summary**")
            st.markdown(f"- Normal: **{normal_count}**")
            st.markdown(f"- Emergency: **{emergency_count}**")
            avg_conf = (
                sum(d.confidence for d in frame.detections) / len(frame.detections)
                if frame.detections
                else 0.0
            )
            st.markdown(f"- Avg Confidence: **{avg_conf:.0%}**")

        # -----------------------------------------------------------------------
        # Detection table
        # -----------------------------------------------------------------------
        st.markdown('<div class="section-title">Active Detections</div>', unsafe_allow_html=True)

        if frame.detections:
            rows = [
                {
                    "ID": d.tracking_id,
                    "Class": d.vehicle_class.value.upper(),
                    "Lane": d.lane,
                    "Confidence": f"{d.confidence:.0%}",
                    "Speed (km/h)": d.speed_kmh,
                    "Emergency": "YES" if d.is_emergency else "—",
                    "BBox": f"({d.bbox.x1},{d.bbox.y1})→({d.bbox.x2},{d.bbox.y2})",
                }
                for d in frame.detections
            ]
            df = pd.DataFrame(rows)

            def _highlight_emergency(row: pd.Series) -> list[str]:
                """Apply red background styling to emergency vehicle rows."""
                if row["Emergency"] == "YES":
                    return ["background-color: rgba(239, 68, 68, 0.15)"] * len(row)
                return [""] * len(row)

            styled = df.style.apply(_highlight_emergency, axis=1)
            st.dataframe(styled, width="stretch", hide_index=True)
        else:
            st.info("No detections above the current confidence threshold.")

    # ---------------------------------------------------------------------------
    # Integration info
    # ---------------------------------------------------------------------------
    with st.expander("Member 1 Integration Interface", expanded=False):
        st.markdown(
            """
            This module exposes a `DetectionProvider` interface for the YOLO pipeline.
            Replace `SimulatedDetectionProvider` with a real implementation when ready —
            the UI will continue to work without changes.

            **Available data points:**
            - Vehicle count, classes, tracking IDs
            - Traffic density, emergency detection
            - YOLO confidence scores
            """
        )


render_live_view()
