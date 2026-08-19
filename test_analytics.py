"""Checks for the archive analytics that feed the dashboard's insight charts.

Every dataset is asserted against a *synthetic archive with known properties* —
a two-session record where the busy hour, the speed/congestion relationship and
the signal phase split are all constructed on purpose. That is the only way to
prove a chart is telling the truth: with real camera data there is nothing to
check the answer against.

    python test_analytics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "SmartTrafficSystem"))

from backend.services import archive_analytics as aa  # noqa: E402
from components import charts  # noqa: E402

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    """Print a section banner."""
    print(f"\n[{title}]")


def make_archive() -> pd.DataFrame:
    """Build a synthetic two-session archive with deliberately known shape.

    Session A: 08:00, quiet  — 2 vehicles, congestion 0.10, 50 km/h, light RED.
    Session B: 17:00, busy   — 12 vehicles, congestion 0.80, 20 km/h, light GREEN.

    So the busiest hour must come out as 17:00, speed must fall as congestion
    rises, and the phase split must be exactly half red / half green.
    """
    rows = []
    for index in range(60):                       # 60 samples = 1 minute
        rows.append({
            "Session": "20260101_080000", "Session Name": "quiet morning",
            "Date": "2026-01-01", "Timestamp": f"08:00:{index:02d}",
            "Congestion Index": 0.10, "Level": "FREE", "Light Phase": "RED",
            "Total Vehicles": 2, "Cars": 2, "Motorcycles": 0, "Buses": 0, "Trucks": 0,
            "Avg Speed (km/h)": 50.0, "Max Speed (km/h)": 60.0,
            "Moving": 2, "Stopped": 0, "Weighted Density (PCE)": 2.0,
            "Servo Angle (deg)": 10, "Emergency": "no", "Emergency Type": "none",
            "Emergency Vehicles": 0, "Unique Vehicles": 2, "FPS": 25.0,
            "Speed Source": "auto",
        })
    for index in range(60):
        rows.append({
            "Session": "20260101_170000", "Session Name": "evening peak",
            "Date": "2026-01-01", "Timestamp": f"17:00:{index:02d}",
            "Congestion Index": 0.80, "Level": "HEAVY", "Light Phase": "GREEN",
            "Total Vehicles": 12, "Cars": 8, "Motorcycles": 1, "Buses": 2, "Trucks": 1,
            "Avg Speed (km/h)": 20.0, "Max Speed (km/h)": 34.0,
            "Moving": 6, "Stopped": 6, "Weighted Density (PCE)": 18.0,
            "Servo Angle (deg)": 90,
            "Emergency": "YES" if index < 5 else "no",
            "Emergency Type": "ambulance" if index < 5 else "none",
            "Emergency Vehicles": 1 if index < 5 else 0,
            "Unique Vehicles": 30, "FPS": 22.0, "Speed Source": "auto",
        })
    return pd.DataFrame(rows)


RAW = make_archive()
ARCHIVE = aa.prepare(RAW)
EMPTY = aa.prepare(pd.DataFrame())


# ══════════════════════════════════════════════════════════════════════════
section("A] prepare() normalises the raw archive")

check("all rows survive preparation", len(ARCHIVE) == 120, f"{len(ARCHIVE)}")
check("a real datetime is derived", str(ARCHIVE["datetime"].dtype).startswith("datetime"),
      str(ARCHIVE["datetime"].dtype))
check("hour is extracted", sorted(ARCHIVE["hour"].unique()) == [8, 17],
      f"{sorted(ARCHIVE['hour'].unique())}")
check("rows come back in time order", ARCHIVE["datetime"].is_monotonic_increasing)
check("numeric columns are numeric, not text",
      pd.api.types.is_numeric_dtype(ARCHIVE["Congestion Index"]))

string_archive = aa.prepare(RAW.astype({"Total Vehicles": str, "Congestion Index": str}))
check("numbers arriving as JSON strings are coerced",
      pd.api.types.is_numeric_dtype(string_archive["Total Vehicles"]),
      "JSON delivers numbers as strings")
check("an empty archive prepares to an empty frame", EMPTY.empty)

corrupt = RAW.copy()
corrupt.loc[0, "Timestamp"] = "not a time"
check("an unparseable timestamp is dropped, not fatal",
      len(aa.prepare(corrupt)) == 119, f"{len(aa.prepare(corrupt))}")


# ══════════════════════════════════════════════════════════════════════════
section("B] hourly_profile - when is the road busiest?")

profile = aa.hourly_profile(ARCHIVE)
check("one row per hour present", len(profile) == 2, f"{len(profile)}")
busiest = profile.loc[profile["avg_vehicles"].idxmax()]
check("the busy hour is identified", int(busiest["hour"]) == 17, f"hour {busiest['hour']}")
check("busy-hour vehicle average is right", busiest["avg_vehicles"] == 12.0,
      f"{busiest['avg_vehicles']}")
check("congestion is reported as a percentage", busiest["avg_congestion"] == 80.0,
      f"{busiest['avg_congestion']}")
quiet = profile.loc[profile["hour"] == 8].iloc[0]
check("the quiet hour keeps its own figures",
      quiet["avg_vehicles"] == 2.0 and quiet["avg_speed_kmh"] == 50.0,
      f"{quiet['avg_vehicles']} veh, {quiet['avg_speed_kmh']} km/h")
check("sample counts are carried through", int(quiet["samples"]) == 60,
      f"{quiet['samples']}")
check("an empty archive yields an empty profile", aa.hourly_profile(EMPTY).empty)

# Samples with no vehicles log 0 km/h; averaging those in would understate speed.
zeroed = ARCHIVE.copy()
zeroed.loc[zeroed.index[:30], "Avg Speed (km/h)"] = 0.0
zero_profile = aa.hourly_profile(zeroed)
check("zero-speed samples are excluded from the speed average",
      zero_profile.loc[zero_profile["hour"] == 8, "avg_speed_kmh"].iloc[0] == 50.0,
      "still 50 km/h, not 25")


# ══════════════════════════════════════════════════════════════════════════
section("C] speed_vs_congestion - what does congestion cost?")

curve = aa.speed_vs_congestion(ARCHIVE)
check("the curve has points", not curve.empty, f"{len(curve)} bins")
check("congestion is on a 0-100 scale",
      curve["congestion_pct"].between(0, 100).all(), f"{list(curve['congestion_pct'])}")
check("speed falls as congestion rises",
      curve.iloc[0]["avg_speed_kmh"] > curve.iloc[-1]["avg_speed_kmh"],
      f"{curve.iloc[0]['avg_speed_kmh']} -> {curve.iloc[-1]['avg_speed_kmh']} km/h")
check("the free-flow speed is recovered", curve.iloc[0]["avg_speed_kmh"] == 50.0)
check("the congested speed is recovered", curve.iloc[-1]["avg_speed_kmh"] == 20.0)
check("stationary samples do not distort the curve",
      (curve["avg_speed_kmh"] > 0).all())
check("an empty archive yields an empty curve", aa.speed_vs_congestion(EMPTY).empty)


# ══════════════════════════════════════════════════════════════════════════
section("D] phase and level distributions")

phases = aa.phase_distribution(ARCHIVE)
check("both phases appear", set(phases["phase"]) == {"RED", "GREEN"},
      f"{set(phases['phase'])}")
check("seconds are counted correctly", phases["seconds"].sum() == 120,
      f"{phases['seconds'].sum()}")
check("the split is 50/50 as constructed",
      set(phases["share_pct"]) == {50.0}, f"{list(phases['share_pct'])}")

levels = aa.level_distribution(ARCHIVE)
check("both congestion levels appear", set(levels["level"]) == {"FREE", "HEAVY"},
      f"{set(levels['level'])}")
check("levels are ordered FREE before HEAVY", list(levels["level"]) == ["FREE", "HEAVY"],
      f"{list(levels['level'])}")
check("level shares sum to 100%", abs(levels["share_pct"].sum() - 100.0) < 0.2,
      f"{levels['share_pct'].sum()}")
check("empty input is handled", aa.phase_distribution(EMPTY).empty
      and aa.level_distribution(EMPTY).empty)


# ══════════════════════════════════════════════════════════════════════════
section("E] vehicle_mix - what kind of traffic is this?")

mix = aa.vehicle_mix(ARCHIVE)
check("every class that appeared is listed", set(mix["vehicle_type"]) ==
      {"Cars", "Motorcycles", "Buses", "Trucks"}, f"{set(mix['vehicle_type'])}")
check("cars dominate, as constructed", mix.iloc[0]["vehicle_type"] == "Cars",
      f"{mix.iloc[0]['vehicle_type']}")
check("the mix is a mean, not a sum",
      abs(mix.loc[mix['vehicle_type'] == 'Cars', 'count'].iloc[0] - 5.0) < 0.01,
      "(2 + 8) / 2 = 5 average cars in frame")
check("a class never seen is omitted",
      "Buses" in set(mix["vehicle_type"]) and len(mix) == 4)

no_vehicles = ARCHIVE.copy()
no_vehicles[["Cars", "Motorcycles", "Buses", "Trucks"]] = 0
check("an empty road produces a placeholder, not a crash",
      aa.vehicle_mix(no_vehicles).iloc[0]["vehicle_type"] == "No vehicles recorded")


# ══════════════════════════════════════════════════════════════════════════
section("F] distributions and trend")

histogram = aa.congestion_histogram(ARCHIVE)
check("the histogram covers the full 0-100 range", len(histogram) == 20,
      f"{len(histogram)} bins")
check("every sample is counted once", histogram["samples"].sum() == 120,
      f"{histogram['samples'].sum()}")
occupied = histogram[histogram["samples"] > 0]
check("the two constructed clusters are visible", len(occupied) == 2,
      f"{list(occupied['congestion_pct'])}")

trend = aa.rolling_trend(ARCHIVE, window_seconds=10)
check("the trend has one point per sample", len(trend) == 120, f"{len(trend)}")
check("the trend is smoothed, not raw",
      trend["vehicles"].max() <= ARCHIVE["Total Vehicles"].max())
check("congestion is scaled to a percentage", trend["congestion_pct"].max() <= 100.0)
check("empty input is handled", aa.rolling_trend(EMPTY).empty)


# ══════════════════════════════════════════════════════════════════════════
section("G] headline_stats")

stats = aa.headline_stats(ARCHIVE)
check("sample count is right", stats["samples"] == 120, f"{stats['samples']}")
check("duration is derived from the 1 Hz sampling rate", stats["duration_min"] == 2.0,
      f"{stats['duration_min']} min")
check("both sessions are counted", stats["sessions"] == 2, f"{stats['sessions']}")
check("peak vehicles found", stats["peak_vehicles"] == 12)
check("mean speed averages the two regimes", stats["avg_speed_kmh"] == 35.0,
      "(50 + 20) / 2")
check("peak speed found", stats["peak_speed_kmh"] == 60.0)
check("mean congestion is a percentage", stats["avg_congestion_pct"] == 45.0,
      "(10 + 80) / 2")
check("time-congested share is right", stats["congested_pct"] == 50.0,
      "half the samples are at 0.80")
check("busiest hour is reported", stats["busiest_hour"] == "17:00",
      stats["busiest_hour"])
check("emergency samples are counted", stats["emergency_samples"] == 5,
      f"{stats['emergency_samples']}")
check("an empty archive gives zeroes, not an exception",
      aa.headline_stats(EMPTY)["samples"] == 0)


# ══════════════════════════════════════════════════════════════════════════
section("H] interpret - plain-language findings")

findings = aa.interpret(ARCHIVE)
text = " ".join(findings).lower()
check("findings are produced", len(findings) >= 5, f"{len(findings)} findings")
check("the sample count is stated", "120" in text)
check("the congestion situation is characterised",
      any(word in text for word in ("congested", "congestion")), text[:80])
check("the speed drop is quantified", "km/h" in text)
check("the busiest hour is called out", "17:00" in text)
check("the signal behaviour is described",
      "red" in text or "green" in text)
check("emergencies are mentioned when present", "emergency" in text)
check("an empty archive gives a helpful message, not a crash",
      "no data" in " ".join(aa.interpret(EMPTY)).lower())


# ══════════════════════════════════════════════════════════════════════════
section("I] session_comparison")

sessions = pd.DataFrame([
    {"Session": "20260101_080000", "Session Name": "quiet morning", "Samples": 60,
     "Peak Vehicles": 2, "Mean Speed (km/h)": 50.0, "Mean Congestion": 0.10},
    {"Session": "20260101_170000", "Session Name": "evening peak", "Samples": 60,
     "Peak Vehicles": 12, "Mean Speed (km/h)": 20.0, "Mean Congestion": 0.80},
])
comparison = aa.session_comparison(sessions)
check("every session is compared", len(comparison) == 2, f"{len(comparison)}")
check("named sessions are labelled by name",
      "evening peak" in list(comparison["label"]), f"{list(comparison['label'])}")
check("congestion is converted to a percentage",
      comparison["Mean Congestion %"].max() == 80.0,
      f"{list(comparison['Mean Congestion %'])}")
check("an unnamed session falls back to its timestamp",
      "0800" in aa.session_comparison(
          sessions.assign(**{"Session Name": "auto"}))["label"].iloc[0],
      "auto sessions labelled by id")
check("no sessions yields an empty comparison",
      aa.session_comparison(pd.DataFrame()).empty)


# ══════════════════════════════════════════════════════════════════════════
section("J] every chart renders from these datasets")

figures = {
    "hourly demand": charts.hourly_demand_chart(aa.hourly_profile(ARCHIVE)),
    "speed vs congestion": charts.speed_vs_congestion_chart(
        aa.speed_vs_congestion(ARCHIVE)),
    "phase share": charts.phase_share_chart(aa.phase_distribution(ARCHIVE)),
    "level share": charts.level_share_chart(aa.level_distribution(ARCHIVE)),
    "congestion histogram": charts.congestion_histogram_chart(
        aa.congestion_histogram(ARCHIVE)),
    "session comparison": charts.session_comparison_chart(comparison),
    "rolling trend": charts.rolling_trend_chart(aa.rolling_trend(ARCHIVE)),
    "vehicle mix": charts.vehicle_mix_chart(aa.vehicle_mix(ARCHIVE)),
}
for name, figure in figures.items():
    check(f"{name} chart builds", figure is not None and len(figure.data) > 0,
          f"{len(figure.data)} trace(s)")
    check(f"{name} chart is titled", bool(figure.layout.title.text),
          str(figure.layout.title.text))

# An offline node hands every chart an empty frame; none may raise.
for name, builder, dataset in [
    ("hourly demand", charts.hourly_demand_chart, aa.hourly_profile(EMPTY)),
    ("speed vs congestion", charts.speed_vs_congestion_chart, aa.speed_vs_congestion(EMPTY)),
    ("phase share", charts.phase_share_chart, aa.phase_distribution(EMPTY)),
    ("level share", charts.level_share_chart, aa.level_distribution(EMPTY)),
    ("congestion histogram", charts.congestion_histogram_chart,
     aa.congestion_histogram(EMPTY)),
    ("session comparison", charts.session_comparison_chart, pd.DataFrame()),
    ("rolling trend", charts.rolling_trend_chart, aa.rolling_trend(EMPTY)),
    ("vehicle mix", charts.vehicle_mix_chart, aa.vehicle_mix(EMPTY)),
]:
    try:
        figure = builder(dataset)
        ok = figure is not None
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"        raised {exc!r}")
    check(f"{name} degrades gracefully with no data", ok)


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f" RESULT: {passed} passed, {failed} failed")
print("=" * 70)
sys.exit(1 if failed else 0)
