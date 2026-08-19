"""KPI card rendering components."""

import streamlit as st

from utils.theme import density_class, status_class


def render_kpi_card(
    *,
    label: str,
    value: str,
    icon: str = "📊",
    delta: str | None = None,
    value_class: str = "",
) -> None:
    """Render a styled KPI card with optional delta text.

    Arguments are keyword-only: ``label``, ``value`` and ``icon`` are all
    strings, so a positional call silently renders them in the wrong slots.
    """
    delta_html = f'<div class="kpi-delta {value_class}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {value_class}">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_grid(items: list[dict[str, str]], columns: int = 4) -> None:
    """Render a responsive grid of KPI cards."""
    rows = [items[i : i + columns] for i in range(0, len(items), columns)]
    for row in rows:
        cols = st.columns(len(row))
        for col, item in zip(cols, row):
            with col:
                value_class = item.get("value_class", "")
                if item.get("status"):
                    value_class = status_class(item["status"])
                if item.get("density"):
                    value_class = density_class(item["density"])
                render_kpi_card(
                    label=item["label"],
                    value=item["value"],
                    icon=item.get("icon", "📊"),
                    delta=item.get("delta"),
                    value_class=value_class,
                )
