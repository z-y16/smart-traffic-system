"""Congestion-driven traffic light state machine.

The signal is **demand-responsive**: it shows GREEN while there is traffic
waiting and falls back to RED once the road is empty, so an empty approach
never holds a green that nobody is using. Every change between GREEN and RED
passes through a fixed YELLOW interval.

    HEAVY / MODERATE traffic  ->  GREEN   (let the queue discharge)
    FREE (empty road)         ->  RED     (nothing to serve)
    any change between them   ->  YELLOW for `yellow_seconds`

This module deliberately knows nothing about cameras, serial ports or Flask.
It is a pure function of (congestion level, clock), which is what makes it
testable against a synthetic clock — see ``test_traffic_light.py``.

Extending to a full intersection
--------------------------------
One :class:`TrafficLightController` drives one approach. A four-way junction
is four controllers plus a policy object that decides which approach may hold
GREEN at any moment; the per-approach timing rules here do not change. The
controller exposes :meth:`request_phase` for exactly that purpose, so a future
intersection coordinator can drive the phase directly instead of deriving it
from congestion. Nothing in this file assumes it is the only signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    """The lamp that is currently lit.

    Inherits from ``str`` so the value drops straight into JSON telemetry and
    the serial protocol without a conversion step.
    """

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


# Congestion levels that mean "there is traffic to serve".  Everything else
# (i.e. FREE) is treated as an empty road.  Flip this to just ("HEAVY",) to
# make MODERATE traffic wait at red.
DEFAULT_GREEN_LEVELS: tuple[str, ...] = ("MODERATE", "HEAVY")

# A merging side road is metered the other way round. The signal is on the
# ramp, not on the carriageway it feeds, so it holds merging traffic back
# exactly when the main road is busiest and lets it in once there is room:
#
#     HEAVY main road  ->  RED    (close the merge, do not add to the queue)
#     MODERATE / FREE  ->  GREEN  (there is room, let the side road in)
#
# Nothing else about the controller changes -- the same committed yellow and
# the same minimum hold apply -- because which levels call for green is the
# only thing that differs between a junction signal and a ramp meter.
MERGE_GREEN_LEVELS: tuple[str, ...] = ("FREE", "MODERATE")


@dataclass
class TrafficLightConfig:
    """Timing and mapping rules for one signal head."""

    #: How long YELLOW is displayed on every GREEN<->RED change.
    yellow_seconds: float = 5.0
    #: Minimum time to hold GREEN or RED before another transition may start.
    #: Without this a level that flips during the yellow interval could produce
    #: a green that lasts a single frame.
    min_phase_seconds: float = 3.0
    #: Congestion levels that call for GREEN.
    green_levels: tuple[str, ...] = DEFAULT_GREEN_LEVELS
    #: Phase shown at start-up, before any traffic has been measured.
    initial_phase: Phase = Phase.RED

    def to_dict(self) -> dict:
        """Return the config as plain JSON-friendly values."""
        return {
            "yellow_seconds": self.yellow_seconds,
            "min_phase_seconds": self.min_phase_seconds,
            "green_levels": list(self.green_levels),
            "initial_phase": self.initial_phase.value,
        }


@dataclass
class LightState:
    """Everything a consumer needs to render or transmit the signal."""

    phase: Phase
    #: Phase the controller is moving towards; equals ``phase`` when settled.
    target: Phase
    #: Seconds left of the yellow interval, 0.0 when not transitioning.
    remaining: float
    #: True while YELLOW is being displayed.
    transitioning: bool
    #: Seconds the current phase has been displayed.
    elapsed: float

    def to_dict(self) -> dict:
        """Return the state as plain JSON-friendly values."""
        return {
            "phase": self.phase.value,
            "target": self.target.value,
            "remaining": round(self.remaining, 1),
            "transitioning": self.transitioning,
            "elapsed": round(self.elapsed, 1),
        }


@dataclass
class TrafficLightController:
    """Drives one signal head from the measured congestion level.

    Transitions are **committed**: once YELLOW starts it always runs for the
    full ``yellow_seconds`` and always lands on the phase it was aiming at,
    exactly like a real signal, which never aborts halfway through an amber.
    If the traffic changed its mind during the interval, the controller simply
    starts a fresh transition afterwards -- subject to ``min_phase_seconds``.
    """

    config: TrafficLightConfig = field(default_factory=TrafficLightConfig)

    def __post_init__(self) -> None:
        """Start settled on the configured initial phase."""
        self._phase: Phase = self.config.initial_phase
        self._pending: Phase = self.config.initial_phase
        self._changed_at: float | None = None   # when the current phase began
        self._forced: Phase | None = None       # manual override, if any

    # -- queries ---------------------------------------------------------------

    @property
    def phase(self) -> Phase:
        """The lamp currently lit."""
        return self._phase

    @property
    def is_transitioning(self) -> bool:
        """True while the yellow interval is running."""
        return self._phase is Phase.YELLOW

    def target_for(self, level: str) -> Phase:
        """Return the phase a given congestion level calls for."""
        return Phase.GREEN if level in self.config.green_levels else Phase.RED

    # -- driving ---------------------------------------------------------------

    def update(self, level: str, now: float) -> LightState:
        """Advance the state machine and return the resulting state.

        Safe to call at the full frame rate; it only acts when a deadline has
        actually passed.
        """
        if self._changed_at is None:          # first call establishes the clock
            self._changed_at = now

        elapsed = max(0.0, now - self._changed_at)

        if self._phase is Phase.YELLOW:
            if elapsed >= self.config.yellow_seconds:
                self._enter(self._pending, now)
                elapsed = 0.0
        else:
            wanted = self._forced if self._forced is not None else self.target_for(level)
            if wanted is not self._phase and elapsed >= self.config.min_phase_seconds:
                self._pending = wanted
                self._enter(Phase.YELLOW, now)
                elapsed = 0.0

        remaining = (max(0.0, self.config.yellow_seconds - elapsed)
                     if self._phase is Phase.YELLOW else 0.0)
        target = self._pending if self._phase is Phase.YELLOW else self._phase
        return LightState(
            phase=self._phase,
            target=target,
            remaining=remaining,
            transitioning=self._phase is Phase.YELLOW,
            elapsed=elapsed,
        )

    def request_phase(self, phase: Phase | None) -> None:
        """Force a phase (manual override), or pass ``None`` to release it.

        The forced phase is still reached through the normal yellow interval,
        so an override cannot make the signal jump straight from green to red.
        This is also the entry point a future intersection coordinator would
        use to drive each approach directly.
        """
        self._forced = phase

    @property
    def forced(self) -> Phase | None:
        """The manually requested phase, or ``None`` when running automatically."""
        return self._forced

    def rebase(self, now: float) -> None:
        """Re-anchor the dwell clock to ``now`` without changing the phase.

        Every deadline here is a comparison against the clock that set it, so
        a caller whose clock jumps must say so. That happens when the video
        source changes: a recorded video runs on its own timeline, and without
        this the signal would either freeze waiting out a dwell that can never
        elapse, or skip one that had not.
        """
        self._changed_at = now

    def _enter(self, phase: Phase, now: float) -> None:
        """Switch to ``phase`` and restart the dwell clock."""
        self._phase = phase
        self._changed_at = now


# ── Presentation helpers ──────────────────────────────────────────────────────

#: 16-column LCD text for each phase, indexed by phase value.
_PHASE_WORD = {Phase.GREEN: "GREEN", Phase.YELLOW: "YELLOW", Phase.RED: "RED"}


def lcd_lines(state: LightState, level: str, congestion_index: float,
              emergency_text: str | None = None) -> tuple[str, str]:
    """Render the two 16-character LCD rows for a light state.

    Line 1 is always the signal itself, counting down during yellow so an
    observer can see the transition progressing. Line 2 carries the emergency
    banner when one is active, and the congestion reading otherwise.
    """
    if state.transitioning:
        line1 = f"YELLOW {int(state.remaining + 0.999):d}s>{_PHASE_WORD[state.target]}"
    else:
        line1 = f"LIGHT: {_PHASE_WORD[state.phase]}"

    line2 = emergency_text if emergency_text else f"{level} CI:{congestion_index:.2f}"
    return line1[:16].ljust(16), line2[:16].ljust(16)
