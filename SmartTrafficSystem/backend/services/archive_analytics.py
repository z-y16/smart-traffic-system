"""Derived analytics over the all-sessions telemetry archive.

Every function here is a pure ``DataFrame -> DataFrame`` transform of the raw
telemetry rows the CV node logs once per second. Keeping them free of Streamlit
and of HTTP means each one can be tested directly against a synthetic archive,
which is what ``test_analytics.py`` does.

The questions these datasets are meant to answer are stated on each function,
because a chart nobody can interpret is worse than no chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Raw telemetry column names, mirroring ``LOG_HEADERS`` in traffic_vision.py.
COL_SESSION = "Session"
COL_SESSION_NAME = "Session Name"
COL_DATE = "Date"
COL_TIME = "Timestamp"
COL_CI = "Congestion Index"
COL_LEVEL = "Level"
COL_PHASE = "Light Phase"
COL_VEHICLES = "Total Vehicles"
COL_AVG_SPEED = "Avg Speed (km/h)"
COL_MAX_SPEED = "Max Speed (km/h)"
COL_MOVING = "Moving"
COL_STOPPED = "Stopped"
COL_DENSITY = "Weighted Density (PCE)"
COL_EMERGENCY = "Emergency"

VEHICLE_CLASS_COLUMNS: list[str] = ["Cars", "Motorcycles", "Buses", "Trucks"]

#: Congestion index above which the road is treated as genuinely congested.
CONGESTED_THRESHOLD = 0.60


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw archive frame: real datetimes and numeric columns.

    The archive arrives as JSON, so numbers may be strings and the timestamp is
    split across a date and a time column. Everything downstream assumes this
    function has already run.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    for column in (COL_CI, COL_VEHICLES, COL_AVG_SPEED, COL_MAX_SPEED,
                   COL_MOVING, COL_STOPPED, COL_DENSITY, *VEHICLE_CLASS_COLUMNS):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    if COL_DATE in out.columns and COL_TIME in out.columns:
        out["datetime"] = pd.to_datetime(
            out[COL_DATE].astype(str) + " " + out[COL_TIME].astype(str),
            errors="coerce",
        )
    elif COL_TIME in out.columns:
        out["datetime"] = pd.to_datetime(out[COL_TIME], errors="coerce")
    else:
        out["datetime"] = pd.NaT

    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    out["hour"] = out["datetime"].dt.hour
    out["weekday"] = out["datetime"].dt.day_name()
    return out.reset_index(drop=True)


def hourly_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """*When is this road busiest?* — mean traffic per hour of day.

    Averaging every sample that fell in a given clock hour across every
    recorded session turns a pile of one-second readings into a daily demand
    curve, which is the single most useful thing for deciding when to widen a
    junction or retime a signal.
    """
    if frame.empty or "hour" not in frame:
        return pd.DataFrame(columns=["hour", "avg_vehicles", "avg_congestion",
                                     "avg_speed_kmh", "samples"])

    speed = frame[COL_AVG_SPEED].where(frame[COL_AVG_SPEED] > 0)
    grouped = frame.assign(_speed=speed).groupby("hour")
    profile = pd.DataFrame({
        "hour": sorted(frame["hour"].unique()),
    }).set_index("hour")
    profile["avg_vehicles"] = grouped[COL_VEHICLES].mean().round(2)
    profile["avg_congestion"] = (grouped[COL_CI].mean() * 100).round(1)
    profile["avg_speed_kmh"] = grouped["_speed"].mean().round(1).fillna(0.0)
    profile["samples"] = grouped.size()
    return profile.reset_index()


def speed_vs_congestion(frame: pd.DataFrame, bins: int = 12) -> pd.DataFrame:
    """*How much does congestion actually cost in speed?*

    Bins samples by congestion index and reports the mean speed in each bin.
    The resulting curve is the practical form of the speed/flow relationship:
    if it is flat, congestion is not yet hurting throughput; where it bends
    downward is the point at which the road starts failing.
    """
    if frame.empty or COL_CI not in frame:
        return pd.DataFrame(columns=["congestion_pct", "avg_speed_kmh", "samples"])

    working = frame[frame[COL_AVG_SPEED] > 0].copy()
    if working.empty:
        return pd.DataFrame(columns=["congestion_pct", "avg_speed_kmh", "samples"])

    working["bin"] = pd.cut(working[COL_CI].clip(0, 1), bins=bins,
                            labels=False, include_lowest=True)
    grouped = working.groupby("bin")
    result = pd.DataFrame({
        "congestion_pct": (grouped["bin"].first() * (100 / bins)).round(1),
        "avg_speed_kmh": grouped[COL_AVG_SPEED].mean().round(1),
        "samples": grouped.size(),
    }).reset_index(drop=True)
    return result


def phase_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """*How is the signal spending its time?* — seconds in each phase.

    Sampling is 1 Hz, so a row count is a duration in seconds. A signal sitting
    mostly on red means the road is usually empty; mostly green means demand is
    high enough that the light rarely gets to rest.
    """
    if frame.empty or COL_PHASE not in frame:
        return pd.DataFrame(columns=["phase", "seconds", "share_pct"])

    counts = frame[COL_PHASE].replace("", np.nan).dropna().value_counts()
    if counts.empty:
        return pd.DataFrame(columns=["phase", "seconds", "share_pct"])
    total = int(counts.sum())
    return pd.DataFrame({
        "phase": counts.index.astype(str),
        "seconds": counts.values.astype(int),
        "share_pct": (counts.values / total * 100).round(1),
    })


def level_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """*How often is the road actually busy?* — seconds at each congestion level."""
    if frame.empty or COL_LEVEL not in frame:
        return pd.DataFrame(columns=["level", "seconds", "share_pct"])

    counts = frame[COL_LEVEL].replace("", np.nan).dropna().value_counts()
    if counts.empty:
        return pd.DataFrame(columns=["level", "seconds", "share_pct"])
    total = int(counts.sum())
    order = {"FREE": 0, "MODERATE": 1, "HEAVY": 2, "EMERGENCY": 3}
    result = pd.DataFrame({
        "level": counts.index.astype(str),
        "seconds": counts.values.astype(int),
        "share_pct": (counts.values / total * 100).round(1),
    })
    return result.sort_values("level", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)


def vehicle_mix(frame: pd.DataFrame) -> pd.DataFrame:
    """*What kind of traffic is this?* — mean count of each vehicle class.

    Mean rather than sum, because the same vehicle appears in every sample it
    stays in frame; a sum would just measure how long vehicles loiter.
    """
    present = [c for c in VEHICLE_CLASS_COLUMNS if c in frame.columns]
    if frame.empty or not present:
        return pd.DataFrame(columns=["vehicle_type", "count"])

    means = frame[present].mean().round(3)
    means = means[means > 0]
    if means.empty:
        return pd.DataFrame({"vehicle_type": ["No vehicles recorded"], "count": [1]})
    return pd.DataFrame({
        "vehicle_type": means.index.astype(str),
        "count": means.values,
    }).sort_values("count", ascending=False).reset_index(drop=True)


def session_comparison(sessions: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """*Is it getting worse?* — the most recent sessions side by side.

    Takes the per-session summary rows straight from the CV node so the
    dashboard and the master workbook always agree on what a session contained.
    """
    if sessions is None or sessions.empty:
        return pd.DataFrame(columns=["label", "Samples", "Peak Vehicles",
                                     "Mean Speed (km/h)", "Mean Congestion"])

    frame = sessions.copy()
    if "Session" in frame.columns:
        frame = frame.sort_values("Session").tail(limit)
    name = frame.get("Session Name", pd.Series([""] * len(frame), index=frame.index))
    stamp = frame.get("Session", pd.Series([""] * len(frame), index=frame.index))
    frame["label"] = [
        f"{n}" if n and n != "auto" else str(s)[-6:]
        for n, s in zip(name.astype(str), stamp.astype(str))
    ]
    for column in ("Samples", "Peak Vehicles", "Mean Speed (km/h)", "Mean Congestion"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["Mean Congestion %"] = (frame.get("Mean Congestion", 0) * 100).round(1)
    return frame.reset_index(drop=True)


def congestion_histogram(frame: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """*Is congestion steady or spiky?* — distribution of the congestion index.

    A single tall bar near zero with a long thin tail is an occasional problem;
    a broad hump in the middle is a road that is permanently under strain. The
    two need completely different responses, and an average hides the
    difference between them.
    """
    if frame.empty or COL_CI not in frame:
        return pd.DataFrame(columns=["congestion_pct", "samples"])

    values = frame[COL_CI].clip(0, 1) * 100
    counts, edges = np.histogram(values, bins=bins, range=(0, 100))
    centres = ((edges[:-1] + edges[1:]) / 2).round(1)
    return pd.DataFrame({"congestion_pct": centres, "samples": counts.astype(int)})


def rolling_trend(frame: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """*What is the shape of this session?* — smoothed vehicles and congestion.

    A one-minute rolling mean over 1 Hz samples removes the frame-to-frame
    noise that makes a raw plot unreadable without hiding real surges.
    """
    if frame.empty or "datetime" not in frame:
        return pd.DataFrame(columns=["datetime", "vehicles", "congestion_pct",
                                     "speed_kmh"])

    window = max(1, int(window_seconds))
    out = pd.DataFrame({"datetime": frame["datetime"]})
    out["vehicles"] = frame[COL_VEHICLES].rolling(window, min_periods=1).mean().round(2)
    out["congestion_pct"] = (
        frame[COL_CI].rolling(window, min_periods=1).mean() * 100).round(1)
    speed = frame[COL_AVG_SPEED].where(frame[COL_AVG_SPEED] > 0)
    out["speed_kmh"] = speed.rolling(window, min_periods=1).mean().round(1).fillna(0.0)
    return out.reset_index(drop=True)


def headline_stats(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarise an archive slice into the numbers worth putting on a card."""
    if frame.empty:
        return {
            "samples": 0, "duration_min": 0.0, "sessions": 0,
            "avg_vehicles": 0.0, "peak_vehicles": 0,
            "avg_speed_kmh": 0.0, "peak_speed_kmh": 0.0,
            "avg_congestion_pct": 0.0, "peak_congestion_pct": 0.0,
            "congested_pct": 0.0, "emergency_samples": 0, "busiest_hour": "—",
        }

    speeds = frame[COL_AVG_SPEED].where(frame[COL_AVG_SPEED] > 0).dropna()
    profile = hourly_profile(frame)
    if profile.empty:
        busiest = "—"
    else:
        peak_row = profile.loc[profile["avg_vehicles"].idxmax()]
        busiest = f"{int(peak_row['hour']):02d}:00"

    emergency = frame.get(COL_EMERGENCY)
    emergency_count = (int((emergency.astype(str).str.upper() == "YES").sum())
                       if emergency is not None else 0)

    return {
        "samples": int(len(frame)),
        # Sampling is 1 Hz, so one row is one second.
        "duration_min": round(len(frame) / 60.0, 1),
        "sessions": int(frame[COL_SESSION].nunique()) if COL_SESSION in frame else 0,
        "avg_vehicles": round(float(frame[COL_VEHICLES].mean()), 2),
        "peak_vehicles": int(frame[COL_VEHICLES].max()),
        "avg_speed_kmh": round(float(speeds.mean()), 1) if not speeds.empty else 0.0,
        "peak_speed_kmh": round(float(frame[COL_MAX_SPEED].max()), 1),
        "avg_congestion_pct": round(float(frame[COL_CI].mean()) * 100, 1),
        "peak_congestion_pct": round(float(frame[COL_CI].max()) * 100, 1),
        "congested_pct": round(
            float((frame[COL_CI] >= CONGESTED_THRESHOLD).mean()) * 100, 1),
        "emergency_samples": emergency_count,
        "busiest_hour": busiest,
    }


def interpret(frame: pd.DataFrame) -> list[str]:
    """Turn the archive into plain-language findings.

    The charts show *what* happened; these sentences say *what it means*, which
    is the part a reader without the context of the project would otherwise
    have to work out for themselves.
    """
    stats = headline_stats(frame)
    if not stats["samples"]:
        return ["No data recorded yet — start the broadcast server and record a session."]

    findings: list[str] = []

    findings.append(
        f"**{stats['samples']:,} samples** ({stats['duration_min']:.0f} minutes) "
        f"across **{stats['sessions']} session(s)**, averaging "
        f"**{stats['avg_vehicles']:.1f} vehicles** in frame and peaking at "
        f"**{stats['peak_vehicles']}**."
    )

    congested = stats["congested_pct"]
    if congested >= 40:
        findings.append(
            f"The road was congested (index ≥ {CONGESTED_THRESHOLD:.0%}) for "
            f"**{congested:.0f}% of the time** — this is a persistent problem, "
            "not an occasional peak. Capacity, not signal timing, is the limit."
        )
    elif congested >= 10:
        findings.append(
            f"Congestion appeared **{congested:.0f}% of the time** — intermittent "
            "peaks rather than a permanently overloaded road, which is usually "
            "worth addressing with signal timing before anything structural."
        )
    else:
        findings.append(
            f"Congestion was rare (**{congested:.0f}% of samples**); the road ran "
            "essentially free-flowing for the whole record."
        )

    if stats["avg_speed_kmh"] > 0:
        findings.append(
            f"Mean measured speed was **{stats['avg_speed_kmh']:.1f} km/h**, "
            f"peaking at **{stats['peak_speed_kmh']:.0f} km/h**."
        )

    curve = speed_vs_congestion(frame)
    if len(curve) >= 3:
        low = curve.head(max(1, len(curve) // 3))["avg_speed_kmh"].mean()
        high = curve.tail(max(1, len(curve) // 3))["avg_speed_kmh"].mean()
        if low > 0 and high < low * 0.75:
            findings.append(
                f"Speed falls from **{low:.0f} km/h** when clear to **{high:.0f} km/h** "
                "under the heaviest congestion recorded — a "
                f"**{(1 - high / low) * 100:.0f}% loss**, so congestion here is "
                "genuinely costing journey time."
            )
        elif low > 0:
            findings.append(
                f"Speed holds up under load (**{low:.0f} → {high:.0f} km/h**), so the "
                "road is still coping with the demand it is seeing."
            )

    if stats["busiest_hour"] != "—":
        findings.append(
            f"The busiest hour on record is **{stats['busiest_hour']}**; scheduling "
            "roadworks or deliveries away from it is the cheapest possible "
            "intervention."
        )

    phases = phase_distribution(frame)
    if not phases.empty:
        top = phases.iloc[phases["seconds"].idxmax()]
        findings.append(
            f"The signal spent most of its time on **{top['phase']}** "
            f"(**{top['share_pct']:.0f}%** of the record)."
        )

    if stats["emergency_samples"]:
        findings.append(
            f"Emergency vehicles were present in **{stats['emergency_samples']} samples**, "
            "each of which lit the dedicated clear-the-lane strip."
        )

    return findings
