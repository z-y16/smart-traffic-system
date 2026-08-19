"""
Smart Traffic Management System — front page

The first thing anyone sees, in the demo room and on a laptop: what the
project is, who built it, what it was aiming at and what it measured. The
live figures are one click away on the Operations Dashboard rather than here,
because a title page that redraws itself every two seconds is a dashboard
wearing a title.

Names and roles come from ``config/project.py`` — edit them there.
"""

import html

import streamlit as st

from backend.database.db_manager import initialize_database
from components.navigation import render_sidebar_nav
from config import project
from config.settings import get_settings
from utils.session import init_session_state
from utils.theme import apply_dark_theme

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
settings = get_settings()

st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()
init_session_state()
# This is the entry point, so it is where the schema is created — every other
# page assumes the tables already exist.
initialize_database()


def esc(text: str) -> str:
    """Escape text going into the raw HTML blocks below.

    The roster is edited by hand in a Python file, so an ampersand in a name
    is a realistic thing to meet and would otherwise break the markup around
    it rather than simply displaying.
    """
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
# Deliberately thin. The controls that belong to a running system — recording,
# exports, auto-refresh — live on the pages that use them; this page has
# nothing to control.
with st.sidebar:
    st.markdown(f"### {settings.app_icon} {esc(settings.app_title)}")
    st.caption(f"{project.PROGRAMME} · {project.STAGE}")
    st.divider()
    render_sidebar_nav()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
# Whether the CV node is answering. Shown as a chip rather than a warning: on
# a title page it is a statement of what is running, not a fault to fix.
#
# The fetch is what makes the chip mean anything -- the flag it sets records
# the last telemetry call, so reading it without making one reports whatever
# some other page happened to see. Once per page load, and this page has no
# refresh timer, so it is one request rather than a stream of them.
st.session_state.status_service.get_dashboard_metrics()
live_connected = st.session_state.status_service.is_live_stream_connected
if live_connected:
    status_chip = '<span class="hero-chip is-live">● Live system connected</span>'
elif settings.simulation_mode:
    status_chip = '<span class="hero-chip is-offline">● Simulation mode</span>'
else:
    status_chip = '<span class="hero-chip is-offline">● CV node not running</span>'

st.markdown(
    f"""
    <div class="title-hero">
        <div class="hero-eyebrow">{esc(project.PROGRAMME)} · {esc(project.STAGE)}</div>
        <h1 class="hero-title">{esc(project.PROJECT_TITLE)}</h1>
        <p class="hero-tagline">{esc(project.PROJECT_TAGLINE)}</p>
        <div class="hero-meta">
            {status_chip}
            <span class="hero-chip">{esc(project.THEME)}</span>
            <span class="hero-chip">{esc(project.DURATION)}</span>
            <span class="hero-chip">{len(project.TEAM)} members</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# The group
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">The Group</div>', unsafe_allow_html=True)

cards = "".join(
    f"""
    <div class="team-card{'' if member.filled else ' is-unfilled'}">
        <div class="team-avatar">{esc(member.initials)}</div>
        <div class="team-name">{esc(member.name)}</div>
        <div class="team-role">{esc(member.role)}</div>
        <div class="team-contribution">{esc(member.contribution)}</div>
    </div>
    """
    for member in project.TEAM
)
st.markdown(f'<div class="team-grid">{cards}</div>', unsafe_allow_html=True)

# A front page with "Add name" on it in front of an examiner is the failure
# worth being loud about, so the gap is named here rather than left to be
# noticed on the day.
if not project.roster_is_complete():
    missing = project.missing_names()
    st.warning(
        f"**{missing} name{'s' if missing != 1 else ''} still to fill in.** "
        "Edit `TEAM` in `SmartTrafficSystem/config/project.py` — the front "
        "page and the About page both read that one list, and this message "
        "disappears once every slot has a name."
    )

# ---------------------------------------------------------------------------
# Aim and objective
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Aim &amp; Objective</div>', unsafe_allow_html=True)

aim_col, objective_col = st.columns(2)
with aim_col:
    st.markdown(
        f"""
        <div class="aim-card">
            <div class="aim-label">Project Aim</div>
            <div class="aim-body">{esc(project.PROJECT_AIM)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with objective_col:
    st.markdown(
        f"""
        <div class="aim-card">
            <div class="aim-label">Group Objective</div>
            <div class="aim-body">{esc(project.GROUP_OBJECTIVE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Measured results
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Measured Results</div>', unsafe_allow_html=True)

results = "".join(
    f"""
    <div class="result-card">
        <div class="result-figure">{esc(result.figure)}</div>
        <div class="result-label">{esc(result.label)}</div>
        <div class="result-note">{esc(result.note)}</div>
    </div>
    """
    for result in project.HEADLINE_RESULTS
)
st.markdown(f'<div class="result-grid">{results}</div>', unsafe_allow_html=True)
st.caption(f"Measured {project.MEASURED_ON}. Every figure, and the method "
           "behind it, is in `METRICS.md`.")

# ---------------------------------------------------------------------------
# Where to go next
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Open the System</div>', unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    st.page_link("pages/1_Operations_Dashboard.py",
                 label="Operations Dashboard", icon="📊")
    st.caption("Live system health, traffic figures and the signal state.")
with nav_col2:
    st.page_link("pages/2_Live_Camera.py", label="Live Camera", icon="📹")
    st.caption("The annotated video feed, with speeds and tracked IDs.")
with nav_col3:
    st.page_link("pages/4_Emergency_Control.py", label="Emergency Control", icon="🚨")
    st.caption("Priority-lane response and the hardware controls.")
