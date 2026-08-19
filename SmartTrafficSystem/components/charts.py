"""Plotly chart components with dark theme styling."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DARK_LAYOUT: dict = {
    "paper_bgcolor": "rgba(15, 23, 42, 0)",
    "plot_bgcolor": "rgba(30, 41, 59, 0.6)",
    "font": {"color": "#e2e8f0", "family": "Inter, sans-serif"},
    "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
    "xaxis": {"gridcolor": "rgba(148, 163, 184, 0.15)", "zerolinecolor": "rgba(148, 163, 184, 0.2)"},
    "yaxis": {"gridcolor": "rgba(148, 163, 184, 0.15)", "zerolinecolor": "rgba(148, 163, 184, 0.2)"},
}

DENSITY_COLORS: dict[str, str] = {
    "Low": "#34d399",
    "Moderate": "#38bdf8",
    "High": "#fbbf24",
    "Critical": "#f87171",
}

PIE_COLORS: list[str] = ["#38bdf8", "#34d399", "#fbbf24", "#a78bfa", "#f87171"]


def _apply_dark(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent dark theme layout to a Plotly figure."""
    fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left"}, **DARK_LAYOUT)
    return fig


def vehicle_count_timeline(df: pd.DataFrame) -> go.Figure:
    """Render vehicle count over time as an area line chart."""
    fig = px.area(
        df,
        x="timestamp",
        y="vehicle_count",
        labels={"timestamp": "Time", "vehicle_count": "Vehicles"},
        color_discrete_sequence=["#2dd4bf"],
    )
    fig.update_traces(line=dict(width=2))
    return _apply_dark(fig, "Vehicle Count Timeline")


def traffic_density_chart(df: pd.DataFrame) -> go.Figure:
    """Render traffic density score over time."""
    fig = px.bar(
        df,
        x="timestamp",
        y="density_score",
        color="density",
        color_discrete_map=DENSITY_COLORS,
        labels={"timestamp": "Time", "density_score": "Density Level", "density": "Level"},
    )
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": 1.12, "x": 0})
    return _apply_dark(fig, "Traffic Density")


def vehicle_types_pie(df: pd.DataFrame) -> go.Figure:
    """Render vehicle type distribution as a donut chart."""
    fig = px.pie(
        df,
        names="vehicle_type",
        values="count",
        hole=0.45,
        color_discrete_sequence=PIE_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_dark(fig, "Vehicle Types")


def congestion_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Render congestion index trend line."""
    fig = px.line(
        df,
        x="timestamp",
        y="congestion_index",
        labels={"timestamp": "Time", "congestion_index": "Congestion Index (%)"},
        color_discrete_sequence=["#f97316"],
    )
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248, 113, 113, 0.12)", line_width=0)
    fig.update_traces(line=dict(width=2.5))
    return _apply_dark(fig, "Congestion Trend")


def speed_timeline_chart(df: pd.DataFrame, free_flow_kmh: float = 50.0) -> go.Figure:
    """Render measured average vehicle speed in km/h over time."""
    fig = px.line(
        df,
        x="timestamp",
        y="avg_speed_kmh",
        labels={"timestamp": "Time", "avg_speed_kmh": "Average Speed (km/h)"},
        color_discrete_sequence=["#38bdf8"],
    )
    fig.update_traces(line=dict(width=2.5), fill="tozeroy",
                      fillcolor="rgba(56, 189, 248, 0.12)")
    # Reference line for the speed the system treats as completely free flowing.
    fig.add_hline(
        y=free_flow_kmh,
        line_dash="dot",
        line_color="rgba(52, 211, 153, 0.7)",
        annotation_text=f"free flow ({free_flow_kmh:.0f} km/h)",
        annotation_position="top left",
    )
    return _apply_dark(fig, "Average Vehicle Speed")


def waiting_time_chart(df: pd.DataFrame) -> go.Figure:
    """Render average waiting time trend."""
    fig = px.line(
        df,
        x="timestamp",
        y="avg_waiting_seconds",
        labels={"timestamp": "Time", "avg_waiting_seconds": "Avg Wait (seconds)"},
        color_discrete_sequence=["#818cf8"],
    )
    fig.update_traces(line=dict(width=2.5))
    return _apply_dark(fig, "Average Waiting Time")


def traffic_heatmap(df: pd.DataFrame) -> go.Figure:
    """Render lane-by-hour traffic intensity heatmap."""
    heatmap_df = df.set_index("hour")
    fig = px.imshow(
        heatmap_df.T,
        labels={"x": "Hour", "y": "Lane", "color": "Vehicle Count"},
        color_continuous_scale=["#0f172a", "#0e7490", "#2dd4bf", "#fbbf24", "#ef4444"],
        aspect="auto",
    )
    fig.update_xaxes(side="bottom")
    return _apply_dark(fig, "Traffic Heatmap (Lane × Hour)")


# ══════════════════════════════════════════════════════════════════════════
# Archive analytics — charts over the all-sessions record
# ══════════════════════════════════════════════════════════════════════════

#: Signal phase colours, matching the physical lamps.
PHASE_COLORS: dict[str, str] = {
    "GREEN": "#34d399",
    "YELLOW": "#fbbf24",
    "RED": "#f87171",
}

LEVEL_COLORS: dict[str, str] = {
    "FREE": "#34d399",
    "MODERATE": "#fbbf24",
    "HEAVY": "#f87171",
    "EMERGENCY": "#60a5fa",
}


def _empty(title: str, message: str = "No data yet") -> go.Figure:
    """A styled placeholder so a missing dataset never renders a broken chart."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font={"size": 15, "color": "#94a3b8"},
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply_dark(fig, title)


def hourly_demand_chart(df: pd.DataFrame) -> go.Figure:
    """Vehicles and congestion by hour of day, on twin axes.

    Answers "when is this road busiest?" — the demand curve that tells you when
    to avoid scheduling work, and when extra green time would actually help.
    """
    if df is None or df.empty:
        return _empty("Demand by Hour of Day")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["hour"], y=df["avg_vehicles"], name="Avg vehicles",
        marker_color="#2dd4bf", opacity=0.85,
        hovertemplate="%{x}:00 — %{y:.1f} vehicles<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["hour"], y=df["avg_congestion"], name="Avg congestion %",
        yaxis="y2", mode="lines+markers", line=dict(color="#f97316", width=2.5),
        hovertemplate="%{x}:00 — %{y:.0f}% congestion<extra></extra>",
    ))
    fig.update_layout(
        yaxis2={"overlaying": "y", "side": "right", "title": "Congestion (%)",
                "range": [0, 100], "showgrid": False},
        yaxis={"title": "Vehicles"},
        xaxis={"title": "Hour of day", "dtick": 1},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        bargap=0.25,
    )
    return _apply_dark(fig, "Demand by Hour of Day")


def speed_vs_congestion_chart(df: pd.DataFrame) -> go.Figure:
    """Mean speed against congestion — the practical speed/flow curve.

    Where this line bends downward is the point at which congestion starts
    costing real journey time; a flat line means the road is still coping.
    """
    if df is None or df.empty:
        return _empty("Speed vs Congestion")

    fig = px.line(
        df, x="congestion_pct", y="avg_speed_kmh", markers=True,
        labels={"congestion_pct": "Congestion index (%)",
                "avg_speed_kmh": "Mean speed (km/h)"},
        color_discrete_sequence=["#38bdf8"],
    )
    fig.update_traces(line=dict(width=2.5),
                      hovertemplate="%{x:.0f}% congestion — %{y:.1f} km/h<extra></extra>")
    fig.add_vrect(x0=60, x1=100, fillcolor="rgba(248, 113, 113, 0.10)", line_width=0,
                  annotation_text="congested", annotation_position="top right")
    return _apply_dark(fig, "Speed vs Congestion")


def phase_share_chart(df: pd.DataFrame) -> go.Figure:
    """How long the signal held each phase, as a horizontal bar."""
    if df is None or df.empty:
        return _empty("Signal Phase Share")

    fig = px.bar(
        df, x="seconds", y="phase", orientation="h", color="phase",
        color_discrete_map=PHASE_COLORS, text="share_pct",
        labels={"seconds": "Seconds", "phase": "Phase"},
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside",
                      hovertemplate="%{y}: %{x:,} s<extra></extra>")
    fig.update_layout(showlegend=False)
    return _apply_dark(fig, "Signal Phase Share")


def level_share_chart(df: pd.DataFrame) -> go.Figure:
    """How long the road spent at each congestion level."""
    if df is None or df.empty:
        return _empty("Time at Each Congestion Level")

    fig = px.bar(
        df, x="level", y="seconds", color="level",
        color_discrete_map=LEVEL_COLORS, text="share_pct",
        labels={"seconds": "Seconds", "level": "Congestion level"},
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside",
                      hovertemplate="%{x}: %{y:,} s<extra></extra>")
    fig.update_layout(showlegend=False)
    return _apply_dark(fig, "Time at Each Congestion Level")


def congestion_histogram_chart(df: pd.DataFrame) -> go.Figure:
    """Distribution of the congestion index across every sample.

    Distinguishes a road with occasional spikes from one that is permanently
    strained — two situations with the same average and different answers.
    """
    if df is None or df.empty:
        return _empty("Congestion Distribution")

    fig = px.bar(
        df, x="congestion_pct", y="samples",
        labels={"congestion_pct": "Congestion index (%)", "samples": "Samples (s)"},
        color_discrete_sequence=["#a78bfa"],
    )
    fig.update_traces(hovertemplate="%{x:.0f}% — %{y:,} samples<extra></extra>")
    fig.add_vline(x=60, line_dash="dot", line_color="rgba(248, 113, 113, 0.8)",
                  annotation_text="congested", annotation_position="top right")
    return _apply_dark(fig, "Congestion Distribution")


def session_comparison_chart(df: pd.DataFrame, metric: str = "Mean Congestion %",
                             ) -> go.Figure:
    """Compare one metric across recent sessions, to spot a trend over time."""
    if df is None or df.empty or metric not in df.columns:
        return _empty("Session Comparison")

    fig = px.bar(
        df, x="label", y=metric,
        labels={"label": "Session", metric: metric},
        color_discrete_sequence=["#38bdf8"],
    )
    fig.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
    fig.update_layout(xaxis={"tickangle": -35})
    return _apply_dark(fig, f"{metric} by Session")


def rolling_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Smoothed vehicles, congestion and speed over the selected record."""
    if df is None or df.empty:
        return _empty("Session Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["vehicles"], name="Vehicles",
        mode="lines", line=dict(color="#2dd4bf", width=2),
        fill="tozeroy", fillcolor="rgba(45, 212, 191, 0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["congestion_pct"], name="Congestion %",
        mode="lines", line=dict(color="#f97316", width=2), yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["speed_kmh"], name="Speed km/h",
        mode="lines", line=dict(color="#818cf8", width=1.6, dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        yaxis={"title": "Vehicles"},
        yaxis2={"overlaying": "y", "side": "right", "title": "Congestion % / km/h",
                "showgrid": False},
        xaxis={"title": "Time"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return _apply_dark(fig, "Session Trend (1-minute rolling mean)")


def vehicle_mix_chart(df: pd.DataFrame) -> go.Figure:
    """Average composition of traffic by vehicle class."""
    if df is None or df.empty:
        return _empty("Vehicle Mix")

    fig = px.pie(df, names="vehicle_type", values="count", hole=0.45,
                 color_discrete_sequence=PIE_COLORS)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      hovertemplate="%{label}: %{value:.2f} avg in frame<extra></extra>")
    return _apply_dark(fig, "Vehicle Mix (mean in frame)")
