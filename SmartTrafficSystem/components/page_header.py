"""Page header component for consistent page layout."""

import streamlit as st


def render_page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    """Render a styled page header with optional status badge."""
    badge_html = f'<span class="sim-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="dashboard-header">
            <h1>{title}</h1>
            <p>{subtitle} {badge_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
