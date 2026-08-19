"""Who made this and what it is called.

Everything the front page says about the *project* rather than about the
traffic lives here: the title, the aim, the group, the measured results. It is
separated from :mod:`config.settings` because none of it is a setting — nobody
changes it between runs, and it must not end up in ``settings.json`` where a
saved preference could quietly overwrite a member's name.

**To put the group's names on the front page, edit `TEAM` below and nothing
else.** The page reads the roster from here, the About page reads the same
list, and until every name is filled in the front page says so in a banner
rather than showing a placeholder to an examiner.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Identity ────────────────────────────────────────────────────────────────

PROGRAMME = "Group Design Project"
STAGE = "Final Submission"

PROJECT_TITLE = (
    "Computer Vision-Based Smart Traffic Congestion Analytics "
    "with Adaptive Signal Optimization"
)

#: One sentence, for the line under the title. Longer than a tagline and
#: shorter than the aim, because the front page has to be readable from the
#: back of a room.
PROJECT_TAGLINE = (
    "A camera, a GPU and three microcontrollers that watch real traffic, "
    "measure it in km/h, and change the signals to suit it."
)

PROJECT_AIM = (
    "To develop a Computer Vision-Based Smart Traffic Congestion Analytics "
    "System with Adaptive Signal Optimization that monitors real-time traffic "
    "conditions and improves urban traffic flow efficiency through intelligent "
    "signal control."
)

GROUP_OBJECTIVE = (
    "To develop a fully functional smart traffic congestion analytics "
    "prototype capable of reducing simulated traffic waiting time by at least "
    "20% before final project evaluation."
)

DURATION = "14 weeks"
THEME = "Smart City & AI Traffic Optimization"


# ── The group ───────────────────────────────────────────────────────────────

#: What an unfilled roster slot holds. The front page checks for this exact
#: string, so changing it here changes nothing else.
PLACEHOLDER = "Add name"


@dataclass(frozen=True)
class Member:
    """One member of the group, as shown on the front page."""

    name: str
    role: str
    contribution: str

    @property
    def filled(self) -> bool:
        """Whether a real name has been put in this slot yet."""
        return bool(self.name.strip()) and self.name.strip() != PLACEHOLDER

    @property
    def initials(self) -> str:
        """Up to two initials for the avatar, or a question mark if unfilled."""
        if not self.filled:
            return "?"
        parts = [word for word in self.name.split() if word]
        return "".join(word[0].upper() for word in parts[:2]) or "?"


#: The group, in the order they appear on the page.
#:
#: Names are what needs filling in; the roles below are the five areas the
#: work actually divided into, so reassign them freely — they are labels for
#: the page, not a claim about who did what.
TEAM: tuple[Member, ...] = (
    Member(
        name="Zeyad Khairy",
        role="Computer Vision & Detection",
        contribution="Vehicle detection and tracking, km/h speed measurement "
                     "from road-plane calibration, and the congestion index.",
    ),
    Member(
        name=PLACEHOLDER,
        role="Dashboard & System Integration",
        contribution="The Streamlit control centre, the database, telemetry "
                     "logging and the link to the CV node.",
    ),
    Member(
        name=PLACEHOLDER,
        role="Hardware & Control",
        contribution="The ESP32 hub, the Arduino signal heads, the servo lane "
                     "changer, the LED strip and the LCDs.",
    ),
    Member(
        name=PLACEHOLDER,
        role="Emergency Vehicle Priority",
        contribution="Flashing-beacon detection, livery recognition, and the "
                     "priority response that clears a lane.",
    ),
    Member(
        name=PLACEHOLDER,
        role="Testing, Calibration & Documentation",
        contribution="Speed calibration, the automated test suites, the "
                     "measured metrics and the report.",
    ),
)


def roster_is_complete() -> bool:
    """Whether every slot in :data:`TEAM` has a real name in it."""
    return all(member.filled for member in TEAM)


def missing_names() -> int:
    """How many roster slots are still placeholders."""
    return sum(1 for member in TEAM if not member.filled)


# ── Measured results ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Result:
    """One headline figure, with the qualifier that keeps it honest."""

    figure: str
    label: str
    note: str


#: The four figures worth putting on a front page. Every one of them is in
#: METRICS.md with the method that produced it; the notes are what stops a
#: number being read as a stronger claim than it is.
HEADLINE_RESULTS: tuple[Result, ...] = (
    Result("36.6 fps", "Detector throughput", "yolo11m @ 1280 px, FP16, RTX 4050"),
    Result("±5%", "Speed accuracy", "calibrated — exact against ground truth"),
    Result("0", "False alarms, night footage", "down from 29% of all traffic"),
    Result("833", "Automated checks", "across 7 suites"),
)

#: When the figures above were measured, and on what.
MEASURED_ON = "9 August 2026 · RTX 4050 laptop GPU, FP16"
