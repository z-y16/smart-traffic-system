#!/usr/bin/env python3
"""Build the A1 GROUP poster for the Smart Traffic GDP.

    python make_group_poster.py      ->  GDP_Group_Poster_A1.pdf

This is the whole-group poster: the vision and analytics subsystem, the smart
retractable speed bump with emergency-vehicle priority, and the physical road
prototype with its movable lane divider - plus every item on the panel's
checklist (aim and measurable objectives, justification, methodology, block
diagram, integration, working principle, innovation, testing and data, results
against objectives, professional practice, health and safety, sustainability,
UN SDGs, costs, market, troubleshooting, sources of error, limitations, future
work, conclusion, references and prototype evidence).

`make_poster.py` stays as it is and still builds the original three-column
poster; this script imports its text engine, its block model and two of its
charts rather than repeating them.

Editing:
  * TEAM   - names and TP numbers
  * COSTS  - the bill of materials, one row per line
  * SHOTS  - photograph slots; drop a file at the path and it appears,
             otherwise a labelled placeholder prints what to photograph.
"""

from __future__ import annotations

import os
import math

from reportlab.lib.units import mm
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas

import make_poster as mp
from make_poster import (
    Para, Bullet, Gap, Rule, KeyLine, Table, Figure, Chart, Section,
    parse_runs, wrap_runs, draw_lines, text_height, yt, _sz,
    network_pattern, chevron, footer_motif, draw_apu_logo,
    chart_speed_vs_congestion, chart_signal_machine,
    INK, INK_SOFT, RULE, BAR, BAR_DEEP, NAVY, PERI, CORAL, CORAL_PALE,
    APU_RED, TEAL, GREEN, AMBER, RED, BLUE, PAPER, TINT, TINT_WARM,
    F, FB, FI, FBI, FL, FM, FMB,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "GDP_Group_Poster_A1.pdf")


# ───────────────────────────── page geometry ────────────────────────────────
#
# A1 portrait, four columns. The page size is deliberately identical to
# make_poster.py so its yt() helper and every imported block behave the same.

PAGE_W_MM, PAGE_H_MM = mp.PAGE_W_MM, mp.PAGE_H_MM
PAGE_W, PAGE_H = mp.PAGE_W, mp.PAGE_H

MARGIN = 13.0
GUTTER = 5.5
NCOL = 4
COL_W = (PAGE_W_MM - 2 * MARGIN - (NCOL - 1) * GUTTER) / NCOL      # 137.9 mm
COL_X = [MARGIN + i * (COL_W + GUTTER) for i in range(NCOL)]
FULL_W = PAGE_W_MM - 2 * MARGIN

HEADER_H   = 60.0                 # white band: wordmark, title, logo
TEAM_TOP   = 62.0
TEAM_H     = 20.0
KPI_TOP    = TEAM_TOP + TEAM_H + 3.0
KPI_H      = 18.4
BLOCK_TOP  = KPI_TOP + KPI_H + 3.5                # full-width block diagram
BLOCK_H    = 90.0
BODY_TOP   = BLOCK_TOP + BLOCK_H + 4.5

FOOTER_H   = 22.0
FOOTER_TOP = PAGE_H_MM - FOOTER_H

CLOSE_H    = 54.0                 # conclusion + references
CLOSE_TOP  = FOOTER_TOP - 3.0 - CLOSE_H
GAPS_H     = 56.0                 # limitations / sources of error / future
GAPS_TOP   = CLOSE_TOP - 3.5 - GAPS_H
TROUBLE_H  = 46.0                 # problems encountered & troubleshooting
TROUBLE_TOP = GAPS_TOP - 3.5 - TROUBLE_H
BODY_BOTTOM = TROUBLE_TOP - 4.0


# ───────────────────────── type scale for 138 mm columns ────────────────────

BODY = 7.6
LEAD = 9.9

GREY_BG = HexColor("#EDF1F4")
PANEL   = HexColor("#F6F9FB")
GOOD    = HexColor("#1F7A4D")
WARN    = HexColor("#B8791A")
TERM_BG = HexColor("#10161B")
TERM_FG = HexColor("#D6E2EA")
TERM_HI = HexColor("#7FD1A3")


class P(Para):
    def __init__(self, text, size=BODY, lead=LEAD, space=2.8, **kw):
        super().__init__(text, size=size, lead=lead, space=space, **kw)


class B(Bullet):
    def __init__(self, text, size=BODY, lead=LEAD, space=2.0, **kw):
        super().__init__(text, size=size, lead=lead, space=space, **kw)


class K(KeyLine):
    def __init__(self, text, size=BODY, lead=LEAD, space=3.2, **kw):
        super().__init__(text, size=size, lead=lead, space=space, **kw)


class T(Table):
    def __init__(self, cols, rows, size=7.0, lead=8.8, space=2.8, **kw):
        super().__init__(cols, rows, size=size, lead=lead, space=space, **kw)


class S(Section):
    """A section box with the title bar scaled for a four-column grid."""

    BAR_H = 8.0
    PAD_X = 3.2
    PAD_T = 2.8
    PAD_B = 2.4
    TITLE_SIZE = 11.0

    def draw(self, c, x_mm, top_mm, w_mm):
        h = self.height(w_mm)
        c.setFillColor(PAPER)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.7)
        c.rect(x_mm * mm, yt(top_mm + h), w_mm * mm, (h - self.BAR_H) * mm,
               stroke=1, fill=1)
        c.setFillColor(self.bar)
        c.rect(x_mm * mm, yt(top_mm + self.BAR_H), w_mm * mm, self.BAR_H * mm,
               stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, self.TITLE_SIZE)
        c.drawString((x_mm + self.PAD_X) * mm, yt(top_mm + self.BAR_H) + 2.2 * mm,
                     self.title)
        if self.tag:
            c.setFont(F, 7.2)
            tw = pdfmetrics.stringWidth(self.tag, F, 7.2)
            c.setFillColor(Color(1, 1, 1, 0.74))
            c.drawString((x_mm + w_mm - self.PAD_X) * mm - tw,
                         yt(top_mm + self.BAR_H) + 2.5 * mm, self.tag)

        y = top_mm + self.BAR_H + self.PAD_T
        iw = self.inner_w(w_mm)
        for b in self.blocks:
            b.draw(c, x_mm + self.PAD_X, y, iw)
            y += b.height(iw)


# ─────────────────────────── extra block types ──────────────────────────────

class Term:
    """A serial-monitor transcript, set in the console font."""

    def __init__(self, lines, title="Serial Monitor", size=6.5, lead=8.0,
                 space=3.4, hi=()):
        self.lines, self.title = lines, title
        self.size, self.lead, self.space = size, lead, space
        self.hi = set(hi)
        self.pad = 2.2
        self.bar = 4.6

    def height(self, w_mm):
        return self.bar + self.pad * 2 + len(self.lines) * self.lead / mm + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        h = self.height(w_mm) - self.space
        c.setFillColor(TERM_BG)
        c.roundRect(x_mm * mm, yt(top_mm + h), w_mm * mm, h * mm, 1.2 * mm,
                    stroke=0, fill=1)
        c.setFillColor(HexColor("#1E2A33"))
        c.rect(x_mm * mm, yt(top_mm + self.bar), w_mm * mm, self.bar * mm,
               stroke=0, fill=1)
        for i, col in enumerate((HexColor("#E06C5A"), HexColor("#D9A63C"),
                                 HexColor("#5FA96B"))):
            c.setFillColor(col)
            c.circle((x_mm + 2.4 + i * 2.2) * mm, yt(top_mm + self.bar / 2),
                     0.65 * mm, stroke=0, fill=1)
        c.setFillColor(HexColor("#93A6B2"))
        c.setFont(FB, 5.8)
        c.drawString((x_mm + 10.0) * mm, yt(top_mm + self.bar) + 1.4 * mm,
                     self.title)

        y = top_mm + self.bar + self.pad
        for i, ln in enumerate(self.lines):
            c.setFillColor(TERM_HI if i in self.hi else TERM_FG)
            c.setFont(FMB if i in self.hi else FM, self.size)
            c.drawString((x_mm + self.pad) * mm, yt(y) - self.lead + 2.0, ln)
            y += self.lead / mm


class Shots:
    """Photographs in a row. A missing file prints a labelled slot instead."""

    def __init__(self, items, caption, h_mm, space=3.6, gap=2.0):
        self.items, self.caption, self.h_mm = items, caption, h_mm
        self.space, self.gap = space, gap

    def height(self, w_mm):
        cap = text_height(parse_runs(self.caption, base=FI), 7.0, 8.8,
                          w_mm * mm) / mm
        return self.h_mm + 4.0 + cap + 1.6 + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        n = len(self.items)
        cw = (w_mm - self.gap * (n - 1)) / n
        for i, (path, label, hint) in enumerate(self.items):
            x = x_mm + i * (cw + self.gap)
            if path and os.path.exists(path):
                from PIL import Image
                im = Image.open(path).convert("RGB")
                iw, ih = im.size
                target = cw / self.h_mm
                if iw / ih > target:
                    nw = int(ih * target)
                    im = im.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
                else:
                    nh = int(iw / target)
                    off = max(0, (ih - nh) // 2)
                    im = im.crop((0, off, iw, off + nh))
                c.drawImage(ImageReader(im), x * mm, yt(top_mm + self.h_mm),
                            cw * mm, self.h_mm * mm, mask="auto")
            else:
                c.setFillColor(PANEL)
                c.rect(x * mm, yt(top_mm + self.h_mm), cw * mm,
                       self.h_mm * mm, stroke=0, fill=1)
                c.setStrokeColor(HexColor("#9FB3C0"))
                c.setLineWidth(0.7)
                c.setDash(2, 2)
                c.rect(x * mm, yt(top_mm + self.h_mm), cw * mm,
                       self.h_mm * mm, stroke=1, fill=0)
                c.setDash()
                c.setFillColor(HexColor("#8CA2B0"))
                c.setFont(FB, 8.4)
                c.drawCentredString((x + cw / 2) * mm,
                                    yt(top_mm + self.h_mm / 2) + 1.2 * mm,
                                    "PHOTO SLOT")
                lines = wrap_runs(parse_runs(hint, base=F), 6.2, (cw - 5) * mm)
                draw_lines(c, lines, (x + 2.5) * mm,
                           yt(top_mm + self.h_mm / 2) - 0.4 * mm, 6.2, 7.6,
                           HexColor("#8CA2B0"), "center", (cw - 5) * mm)
            c.setStrokeColor(RULE)
            c.setLineWidth(0.6)
            c.rect(x * mm, yt(top_mm + self.h_mm), cw * mm, self.h_mm * mm,
                   stroke=1, fill=0)
            c.setFillColor(BAR)
            c.rect(x * mm, yt(top_mm + self.h_mm + 4.0), cw * mm, 4.0 * mm,
                   stroke=0, fill=1)
            c.setFillColor(PAPER)
            c.setFont(FB, 6.6)
            tw = pdfmetrics.stringWidth(label, FB, 6.6)
            c.drawString(x * mm + (cw * mm - tw) / 2,
                         yt(top_mm + self.h_mm + 4.0) + 1.2 * mm, label)
        lines = wrap_runs(parse_runs(self.caption, base=FI), 7.0, w_mm * mm)
        draw_lines(c, lines, x_mm * mm, yt(top_mm + self.h_mm + 4.0 + 1.6),
                   7.0, 8.8, INK_SOFT)


class Verdict:
    """Objective -> target -> measured -> met/partly, as a stack of rows."""

    def __init__(self, rows, size=7.2, lead=9.0, space=3.2):
        self.rows, self.size, self.lead, self.space = rows, size, lead, space

    TEXT_INSET = 24.0          # room for the MET / PARTLY MET stamp

    def _body(self, row):
        oid, target, measured, ok = row
        return f"**{oid}**  {target}   →   {measured}"

    def _h(self, row, w_mm):
        return text_height(parse_runs(self._body(row)), self.size, self.lead,
                           (w_mm - self.TEXT_INSET) * mm) / mm + 2.8

    def height(self, w_mm):
        return sum(self._h(r, w_mm) for r in self.rows) + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        y = top_mm
        for row in self.rows:
            ok = row[3]
            h = self._h(row, w_mm)
            col = GOOD if ok else WARN
            c.setFillColor(PANEL if ok else HexColor("#FDF6EA"))
            c.rect(x_mm * mm, yt(y + h - 0.9), w_mm * mm, (h - 0.9) * mm,
                   stroke=0, fill=1)
            c.setFillColor(col)
            c.rect(x_mm * mm, yt(y + h - 0.9), 1.4 * mm, (h - 0.9) * mm,
                   stroke=0, fill=1)
            c.setFont(FB, 6.8)
            c.drawRightString((x_mm + w_mm - 1.8) * mm, yt(y + 4.6),
                              "MET" if ok else "PARTLY MET")
            lines = wrap_runs(parse_runs(self._body(row)), self.size,
                              (w_mm - self.TEXT_INSET) * mm)
            draw_lines(c, lines, (x_mm + 3.0) * mm, yt(y + 0.4), self.size,
                       self.lead, INK)
            y += h


# ────────────────────────── diagram primitives ──────────────────────────────
#
# Every diagram is drawn straight onto the canvas in millimetres measured from
# the top of the page, so it stays vector and shares the poster's palette.
# Each chart takes (canvas, x_mm, top_mm, w_mm, h_mm).

def _ctext(c, txt, x, y, w, font, size, colour):
    """Centred text, baseline y mm from the page top."""
    c.setFont(font, size)
    c.setFillColor(colour)
    tw = pdfmetrics.stringWidth(txt, font, size)
    c.drawString(x * mm + (w * mm - tw) / 2, yt(y), txt)


def _txt(c, txt, x, y, font, size, colour, align="l"):
    c.setFont(font, size)
    c.setFillColor(colour)
    tw = pdfmetrics.stringWidth(txt, font, size)
    px = x * mm
    if align == "r":
        px -= tw
    elif align == "c":
        px -= tw / 2
    c.drawString(px, yt(y), txt)


def _box(c, x, y, w, h, label, fill, fg=PAPER, size=7.0, sub=None,
         subcol=None, r=1.2, border=None, dash=None):
    """A rounded label box. x/y/w/h in mm, y measured from the page top."""
    if dash:
        c.setDash(*dash)
    if fill is not None:
        c.setFillColor(fill)
    if border is not None:
        c.setStrokeColor(border)
        c.setLineWidth(0.7)
    c.roundRect(x * mm, yt(y + h), w * mm, h * mm, r * mm,
                stroke=1 if border is not None else 0,
                fill=1 if fill is not None else 0)
    c.setDash()
    if sub:
        _ctext(c, label, x, y + h / 2 - 0.4, w, FB, size, fg)
        _ctext(c, sub, x, y + h / 2 + 2.9, w, F, size - 1.3,
               subcol or (Color(1, 1, 1, 0.95) if fg == PAPER else INK_SOFT))
    else:
        _ctext(c, label, x, y + h / 2 + size / 6.4, w, FB, size, fg)


def _head(c, x, y, w, txt, colour=INK_SOFT, size=6.2):
    _txt(c, txt, x, y, FB, size, colour)


def _arrow(c, x0, y0, x1, y1, colour=INK_SOFT, lw=0.8, dash=None, head=1.6):
    c.setStrokeColor(colour)
    c.setLineWidth(lw)
    if dash:
        c.setDash(*dash)
    c.line(x0 * mm, yt(y0), x1 * mm, yt(y1))
    c.setDash()
    ang = math.atan2(-(yt(y1) - yt(y0)), (x1 - x0) * mm)
    L = head * mm
    p = c.beginPath()
    p.moveTo(x1 * mm, yt(y1))
    p.lineTo(x1 * mm - L * math.cos(ang - 0.42), yt(y1) + L * math.sin(ang - 0.42))
    p.lineTo(x1 * mm - L * math.cos(ang + 0.42), yt(y1) + L * math.sin(ang + 0.42))
    p.close()
    c.setFillColor(colour)
    c.drawPath(p, stroke=0, fill=1)


def _elbow(c, x0, y0, x1, y1, colour=INK_SOFT, lw=0.8, dash=None, bend=None):
    """Horizontal, then vertical, then a short horizontal arrow into (x1, y1)."""
    bx = bend if bend is not None else (x0 + x1) / 2.0
    c.setStrokeColor(colour)
    c.setLineWidth(lw)
    if dash:
        c.setDash(*dash)
    c.line(x0 * mm, yt(y0), bx * mm, yt(y0))
    c.line(bx * mm, yt(y0), bx * mm, yt(y1))
    c.setDash()
    _arrow(c, bx, y1, x1, y1, colour, lw)


# ─────────────────────────── the block diagram ──────────────────────────────

def chart_block(c, x, top, w, h):
    """The complete system: camera -> measurement -> decision -> actuators,
    with the emergency radio path drawn as the separate route it is."""

    ZW1 = 70.0
    ZX2, ZW2 = 81.0, 152.0
    ZX3, ZW3 = 244.0, 74.0
    ZX4, ZW4 = 329.0, 86.0
    ZX5, ZW5 = 426.0, 142.0

    sx, sy = w / 568.0, h / 92.0

    def X(v):
        return x + v * sx

    def W(v):
        return v * sx

    def Y(v):
        return top + v * sy

    def H(v):
        return v * sy

    _head(c, X(0), Y(3.0), W(ZW1), "1 · VIDEO SOURCE")
    _head(c, X(ZX2), Y(3.0), W(ZW2), "2 · MEASUREMENT  (CV node, laptop GPU)")
    _head(c, X(ZX3), Y(3.0), W(ZW3), "3 · DECISION")
    _head(c, X(ZX4), Y(3.0), W(ZW4), "4 · RELAY")
    _head(c, X(ZX5), Y(3.0), W(ZW5), "5 · ACTUATORS  (physical prototype)")

    # 1 — sources
    for i, s in enumerate(("USB camera", "recorded video file", "roadside stream URL")):
        _box(c, X(0), Y(14 + i * 14), W(ZW1), H(11.0), s, PANEL, INK, 7.0,
             border=RULE)
        _arrow(c, X(ZW1), Y(19.5 + i * 14), X(ZX2) - 0.6, Y(36.0))

    # 2 — the vision pipeline
    _box(c, X(ZX2), Y(6), W(ZW2), H(60.0), "", BAR_DEEP, r=1.8)
    _ctext(c, "CV NODE  ·  broadcast_server.py", X(ZX2), Y(12.4), W(ZW2),
           FB, 8.6, PAPER)
    _ctext(c, "Python 3.13  ·  PyTorch + CUDA  ·  RTX 4050, FP16",
           X(ZX2), Y(16.8), W(ZW2), F, 6.4, Color(1, 1, 1, 0.72))
    chips = [
        ("YOLO11-m", "detect @ 1280 px", BAR),
        ("BoT-SORT", "one ID per vehicle", BAR),
        ("homography", "pixels → metres", TEAL),
        ("speed + index", "km/h  ·  0 → 1", TEAL),
        ("CLIP livery", "what a vehicle is", CORAL),
        ("beacon detector", "what it is doing", CORAL),
    ]
    cw = (ZW2 - 12.0) / 2.0
    for i, (a, b, col) in enumerate(chips):
        _box(c, X(ZX2 + 4.0 + (i % 2) * (cw + 4.0)), Y(21.0 + (i // 2) * 13.5),
             W(cw), H(11.0), a, col, PAPER, 7.0, sub=b)
    _ctext(c, "27.1 fps end to end   ·   23.0 tracked vehicles per frame   ·   "
              "every second written to CSV / XLSX",
           X(ZX2), Y(63.6), W(ZW2), FB, 6.4, CORAL_PALE)

    # 3 — decisions
    dec = [("congestion index", "0 → 1, once per second", BAR),
           ("signal decision", "phase, dwell, commitment", NAVY),
           ("emergency flag", "liveried AND flashing", CORAL)]
    for i, (a, b, col) in enumerate(dec):
        _box(c, X(ZX3), Y(12 + i * 18), W(ZW3), H(14.0), a, PANEL, INK, 7.2,
             sub=b, border=col)
        _arrow(c, X(ZX2 + ZW2), Y(36.0), X(ZX3) - 0.6, Y(19.0 + i * 18))

    # 4 — the relay and the control centre
    _box(c, X(ZX4), Y(12), W(ZW4), H(17.0), "ESP32 HUB", NAVY, PAPER, 8.0,
         sub="USB · COM6 · 7-field line")
    _box(c, X(ZX4), Y(41), W(ZW4), H(17.0), "STREAMLIT DASHBOARD", BAR, PAPER, 7.4,
         sub="HTTP :8502 · 9 pages · Excel")
    for i in range(3):
        _arrow(c, X(ZX3 + ZW3), Y(19.0 + i * 18), X(ZX4) - 0.6, Y(20.5))
    # telemetry reaches the dashboard from the node itself, not from the relay
    c.setStrokeColor(BAR)
    c.setLineWidth(0.8)
    c.setDash(2, 2)
    c.line(X(ZX2 + ZW2 / 2) * mm, yt(Y(66.4)), X(ZX2 + ZW2 / 2) * mm, yt(Y(69.6)))
    c.line(X(ZX2 + ZW2 / 2) * mm, yt(Y(69.6)), X(ZX4 + ZW4 / 2) * mm, yt(Y(69.6)))
    c.setDash()
    _arrow(c, X(ZX4 + ZW4 / 2), Y(69.6), X(ZX4 + ZW4 / 2), Y(58.6), BAR, 0.8)
    _txt(c, "HTTP telemetry  ·  the node owns the only serial port, so the "
            "dashboard asks it to act", X(ZX3 - 6), Y(68.6), FI, 6.0, BAR)

    # 5 — the three controllers
    unos = [("SIGNAL UNO", "two signal heads  ·  status LCD", GREEN),
            ("LANE UNO", "two divider servos  ·  LED strip  ·  I²C LCD", TEAL),
            ("BUMP UNO", "retractable bump servo  ·  45° / 90°", CORAL)]
    for i, (a, b, col) in enumerate(unos):
        _box(c, X(ZX5), Y(8 + i * 20), W(ZW5), H(16.0), a, col, PAPER, 8.0, sub=b)
        _elbow(c, X(ZX4 + ZW4), Y(20.5), X(ZX5) - 0.6, Y(16.0 + i * 20),
               bend=X(ZX4 + ZW4 + 5.0))
    _txt(c, "A0 / A1", X(ZX4 + ZW4 + 7.0), Y(50.0), FB, 6.0, INK_SOFT)

    # the independent emergency path
    _box(c, X(0), Y(72), W(568), H(20.0), "", TINT_WARM, r=1.4)
    c.setFillColor(CORAL)
    c.rect(X(0) * mm, yt(Y(92)), 1.6 * mm, H(20) * mm, stroke=0, fill=1)
    _txt(c, "LOCAL INPUTS  ·  INDEPENDENT EMERGENCY PATH", X(3.5), Y(77.0),
         FB, 6.6, CORAL)
    _txt(c, "— the bump reads its own sensors and its own radio, so it still "
            "flattens for an ambulance with the laptop switched off",
         X(75), Y(77.0), FI, 6.2, INK_SOFT)
    lane = [(6.0, 62.0, "EMERGENCY VEHICLE", "ambulance · police · fire", PANEL, INK),
            (74.0, 52.0, "ARDUINO NANO", "transmitter", PANEL, INK),
            (132.0, 44.0, "SX1278 TX", "433 MHz", CORAL, PAPER),
            (218.0, 44.0, "SX1278 RX", "433 MHz", CORAL, PAPER),
            (270.0, 58.0, "HC-SR04", "vehicle within 20 cm", PANEL, INK),
            (334.0, 62.0, "PUSH BUTTON", "congestion mode", PANEL, INK)]
    for bx, bw, a, b, fill, fg in lane:
        _box(c, X(bx), Y(80.0), W(bw), H(10.0), a, fill, fg, 6.6, sub=b,
             border=RULE if fill is PANEL else None)
    _arrow(c, X(68), Y(85.0), X(74) - 0.6, Y(85.0))
    _arrow(c, X(126), Y(85.0), X(132) - 0.6, Y(85.0))
    _arrow(c, X(176), Y(85.0), X(218) - 0.6, Y(85.0), CORAL, 1.1)
    _txt(c, "command  'E'", X(197), Y(83.0), FB, 6.2, CORAL, align="c")
    c.setStrokeColor(INK_SOFT)
    c.setLineWidth(0.8)
    for cx in (240.0, 299.0, 365.0):                 # RX, ultrasonic, button
        c.line(X(cx) * mm, yt(Y(90.2)), X(cx) * mm, yt(Y(91.4)))
        c.line(X(cx) * mm, yt(Y(91.4)), X(497) * mm, yt(Y(91.4)))
    c.line(X(497) * mm, yt(Y(91.4)), X(497) * mm, yt(Y(70.0)))
    _arrow(c, X(497), Y(70.0), X(497), Y(64.6))
    _txt(c, "three local inputs", X(501), Y(69.0), FI, 6.0, INK_SOFT)


# ───────────────────────────── the other figures ────────────────────────────

METHOD = [
    ("Requirement analysis",
     "Problem framed, smart-bump and camera-traffic literature reviewed, "
     "measurable objectives fixed."),
    ("System design",
     "Architecture split into vision, relay and actuators; road and divider drawn "
     "in CAD; interfaces, pins and power allocated."),
    ("Build",
     "Road fabricated, linkage 3D-printed, bump hinged to a servo; CV pipeline "
     "and three firmware images written."),
    ("Subsystem testing",
     "Servo travel, the 20 cm threshold, the radio link, detection accuracy and "
     "signal timing each proved alone."),
    ("Integration",
     "One measurement drives four actuators; priority ladder, common ground and "
     "separate actuator supply settled here."),
    ("Validation & documentation",
     "833 automated checks, 69 recorded sessions, evaluation against O1–O7, then "
     "report, poster and demonstration."),
]


def chart_method(c, x, top, w, h):
    step = h / len(METHOD)
    r = 3.1
    cx = x + r + 1.0
    c.setStrokeColor(HexColor("#C3D2DC"))
    c.setLineWidth(1.4)
    c.line(cx * mm, yt(top + step / 2), cx * mm, yt(top + h - step / 2))
    for i, (title, body) in enumerate(METHOD):
        y = top + i * step
        c.setFillColor(BAR if i < 5 else CORAL)
        c.circle(cx * mm, yt(y + step / 2 - 2.2), r * mm, stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 7.4)
        c.drawCentredString(cx * mm, yt(y + step / 2 - 2.2) - 2.4, str(i + 1))
        _txt(c, title, x + 2 * r + 3.4, y + 4.2, FB, 8.0, BAR_DEEP)
        lines = wrap_runs(parse_runs(body), 6.9, (w - 2 * r - 4.0) * mm)
        draw_lines(c, lines, (x + 2 * r + 3.4) * mm, yt(y + 5.0), 6.9, 8.4, INK)


def chart_loop(c, x, top, w, h):
    """SEE -> MEASURE -> DECIDE -> ACT, closed once a second."""
    chips = [("SEE", "detect + track", BAR),
             ("MEASURE", "km/h · index", TEAL),
             ("DECIDE", "phase · lane · bump", NAVY),
             ("ACT", "servos · lights · log", CORAL)]
    gap = 3.0
    cw = (w - 3 * gap) / 4.0
    for i, (a, b, col) in enumerate(chips):
        bx = x + i * (cw + gap)
        _box(c, bx, top, cw, 14.0, a, col, PAPER, 8.0, sub=b)
        if i:
            _arrow(c, bx - gap + 0.4, top + 7.0, bx - 0.5, top + 7.0)
    c.setStrokeColor(INK_SOFT)
    c.setLineWidth(0.8)
    c.setDash(2, 2)
    c.line((x + w - cw / 2) * mm, yt(top + 14.0), (x + w - cw / 2) * mm, yt(top + 21.0))
    c.line((x + w - cw / 2) * mm, yt(top + 21.0), (x + cw / 2) * mm, yt(top + 21.0))
    c.setDash()
    _arrow(c, x + cw / 2, top + 21.0, x + cw / 2, top + 14.6)
    _txt(c, "the loop closes once a second — the next decision is made on the "
            "next measurement, not on a timer",
         x + w / 2, top + 24.6, FI, 6.4, INK_SOFT, align="c")


def chart_priority(c, x, top, w, h):
    """The bump's control ladder: the first rule that matches wins."""
    rows = [("1", "EMERGENCY", "valid LoRa command 'E' received", "90°  FLAT",
             HexColor("#B23A2E")),
            ("2", "CONGESTION", "push button, or hint on A0 / A1", "90°  FLAT",
             AMBER),
            ("3", "AUTOMATIC", "vehicle detected within 20 cm", "45°  RAISED",
             BAR)]
    rh = 11.0
    gap = 1.8
    chip_w = 26.0
    for i, (n, name, cond, out, col) in enumerate(rows):
        y = top + i * (rh + gap)
        c.setFillColor(col)
        c.rect(x * mm, yt(y + rh), 7.0 * mm, rh * mm, stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 9.0)
        c.drawCentredString((x + 3.5) * mm, yt(y + rh / 2) - 1.3 * mm, n)
        c.setFillColor(PANEL)
        c.rect((x + 7.0) * mm, yt(y + rh), (w - 7.0 - chip_w) * mm, rh * mm,
               stroke=0, fill=1)
        _txt(c, name, x + 9.6, y + 4.6, FB, 7.8, col)
        _txt(c, cond, x + 9.6, y + 9.2, F, 6.8, INK)
        _box(c, x + w - chip_w, y, chip_w, rh, out, col, PAPER, 7.6)
    yb = top + 3 * (rh + gap)
    _txt(c, "otherwise  ·  no vehicle, no button, no radio  →  90° flat, held",
         x, yb + 3.4, FI, 6.8, INK_SOFT)


def chart_bump(c, x, top, w, h):
    """Side elevation of the bump in both calibrated positions."""
    pw = (w - 5.0) / 2.0
    for k, (title, raised) in enumerate((("FLAT  ·  servo 90°", False),
                                         ("RAISED  ·  servo 45°", True))):
        px = x + k * (pw + 5.0)
        c.setFillColor(PANEL)
        c.rect(px * mm, yt(top + h), pw * mm, h * mm, stroke=0, fill=1)
        c.setFillColor(CORAL if raised else BAR)
        c.rect(px * mm, yt(top + 5.0), pw * mm, 5.0 * mm, stroke=0, fill=1)
        _ctext(c, title, px, top + 3.6, pw, FB, 7.0, PAPER)

        road_y = top + h - 9.0
        c.setFillColor(HexColor("#464C52"))
        c.rect((px + 2.0) * mm, yt(road_y + 3.0), (pw - 4.0) * mm, 3.0 * mm,
               stroke=0, fill=1)
        hinge_x = px + 9.0
        plate_l = pw * 0.42

        c.setStrokeColor(HexColor("#C2472F"))
        c.setLineWidth(2.2)
        if raised:
            ang = math.radians(26.0)
            ex = hinge_x + plate_l * math.cos(ang)
            ey = road_y - plate_l * math.sin(ang)
            c.line(hinge_x * mm, yt(road_y), ex * mm, yt(ey))
            c.setDash(1.5, 1.5)
            c.setStrokeColor(HexColor("#9AA6AE"))
            c.setLineWidth(0.6)
            c.line(hinge_x * mm, yt(road_y), (hinge_x + plate_l) * mm, yt(road_y))
            c.setDash()
            _txt(c, "45°", hinge_x + 7.0, road_y - 2.4, FB, 6.6, HexColor("#C2472F"))
        else:
            c.line(hinge_x * mm, yt(road_y), (hinge_x + plate_l) * mm, yt(road_y))
        c.setFillColor(HexColor("#C2472F"))
        c.circle(hinge_x * mm, yt(road_y), 0.9 * mm, stroke=0, fill=1)

        _box(c, hinge_x - 4.0, road_y + 3.4, 12.0, 5.4, "servo", NAVY, PAPER, 6.0)
        _box(c, px + pw - 20.0, top + 9.0, 17.0, 5.4, "HC-SR04", TEAL, PAPER, 6.0)
        c.setStrokeColor(TEAL)
        c.setLineWidth(0.5)
        c.setDash(1.2, 1.2)
        c.line((px + pw - 11.5) * mm, yt(top + 14.4), (px + pw - 17.0) * mm, yt(road_y))
        c.line((px + pw - 11.5) * mm, yt(top + 14.4), (px + pw - 6.0) * mm, yt(road_y))
        c.setDash()
        _txt(c, "20 cm", px + pw - 11.5, road_y - 1.6, F, 5.8, TEAL, align="c")
        if raised:
            _txt(c, "vehicle detected", px + pw - 11.5, top + 19.4, FI, 5.8,
                 INK_SOFT, align="c")
        else:
            _txt(c, "clear road, or override", px + pw - 11.5, top + 19.4, FI,
                 5.8, INK_SOFT, align="c")


def chart_road(c, x, top, w, h):
    """Plan view of the physical prototype and where every device sits."""
    c.setFillColor(HexColor("#8FBF72"))
    c.rect(x * mm, yt(top + h), w * mm, h * mm, stroke=0, fill=1)
    road_t, road_b = top + 12.0, top + h - 14.0
    c.setFillColor(HexColor("#464C52"))
    c.rect(x * mm, yt(road_b), w * mm, (road_b - road_t) * mm, stroke=0, fill=1)

    mid = (road_t + road_b) / 2.0
    c.setStrokeColor(PAPER)
    c.setLineWidth(0.9)
    c.line(x * mm, yt(mid - 0.7), (x + w) * mm, yt(mid - 0.7))
    c.line(x * mm, yt(mid + 0.7), (x + w) * mm, yt(mid + 0.7))
    c.setDash(2.4, 2.4)
    c.setLineWidth(0.7)
    for ly in (road_t + (mid - road_t) / 2, road_b - (road_b - mid) / 2):
        c.line(x * mm, yt(ly), (x + w) * mm, yt(ly))
    c.setDash()

    # the movable divider
    c.setFillColor(CORAL)
    c.rect((x + w * 0.30) * mm, yt(mid + 1.6), (w * 0.16) * mm, 3.2 * mm,
           stroke=0, fill=1)
    _arrow(c, x + w * 0.38, mid - 2.6, x + w * 0.38, mid - 6.4, CORAL, 0.9, head=1.3)
    _arrow(c, x + w * 0.38, mid + 4.8, x + w * 0.38, mid + 8.6, CORAL, 0.9, head=1.3)
    _txt(c, "movable lane divider  ·  two servos", x + w * 0.38, mid - 7.6, FB,
         6.0, PAPER, align="c")

    # the retractable bump
    c.setFillColor(HexColor("#E4A11B"))
    c.rect((x + w * 0.66) * mm, yt(mid - 0.9), (w * 0.05) * mm,
           (mid - road_t - 0.9) * mm, stroke=0, fill=1)
    _txt(c, "retractable bump", x + w * 0.685, road_t - 1.4, FB, 6.0,
         HexColor("#6B4A08"), align="c")

    # signal heads
    for sy in (road_t - 4.2, road_b + 1.2):
        _box(c, x + w * 0.13, sy, 8.0, 3.0, "", HexColor("#22282D"), r=0.6)
        for i, col in enumerate((RED, AMBER, GREEN)):
            c.setFillColor(col)
            c.circle((x + w * 0.13 + 2.0 + i * 2.0) * mm, yt(sy + 1.5),
                     0.55 * mm, stroke=0, fill=1)
    _txt(c, "signal heads", x + w * 0.13, road_t - 5.4, FB, 6.0,
         HexColor("#2C3E22"))

    # LED strip along the verge
    for i in range(18):
        c.setFillColor([CORAL, PAPER, TEAL][i % 3])
        c.rect((x + w * 0.52 + i * 2.1) * mm, yt(road_b + 4.2), 1.4 * mm,
               1.4 * mm, stroke=0, fill=1)
    _txt(c, "addressable LED strip  ·  'make space' pattern", x + w * 0.52,
         road_b + 7.6, FB, 6.0, HexColor("#2C3E22"))

    # camera and its field of view
    c.setFillColor(HexColor("#1B222A"))
    c.rect((x + w - 12.0) * mm, yt(top + 7.0), 8.0 * mm, 4.0 * mm, stroke=0, fill=1)
    c.circle((x + w - 8.0) * mm, yt(top + 5.0), 1.5 * mm, stroke=0, fill=1)
    c.setFillColor(Color(1, 1, 1, 0.22))
    p = c.beginPath()
    p.moveTo((x + w - 8.0) * mm, yt(top + 7.0))
    p.lineTo((x + w * 0.30) * mm, yt(road_b))
    p.lineTo((x + w * 0.86) * mm, yt(road_b))
    p.close()
    c.drawPath(p, stroke=0, fill=1)
    _txt(c, "camera", x + w - 13.0, top + 6.4, FB, 6.0, HexColor("#2C3E22"),
         align="r")
    _txt(c, "360 × 480 mm printed carriageway  ·  merge lanes at 30°  ·  "
            "everything below is driven from one measurement",
         x + 1.0, top + h - 1.6, FI, 6.0, PAPER)


SDGS = [
    ("3", "Good health\n& well-being", "#4C9F38"),
    ("4", "Quality\neducation", "#C5192D"),
    ("7", "Affordable &\nclean energy", "#FCC30B"),
    ("8", "Decent work &\neconomic growth", "#A21942"),
    ("9", "Industry, innovation\n& infrastructure", "#FD6925"),
    ("10", "Reduced\ninequalities", "#DD1367"),
    ("11", "Sustainable cities\n& communities", "#FD9D24"),
    ("12", "Responsible\nconsumption", "#BF8B2E"),
    ("13", "Climate\naction", "#3F7E44"),
    ("15", "Life\non land", "#56C02B"),
    ("16", "Peace, justice &\nstrong institutions", "#00689D"),
    ("17", "Partnerships\nfor the goals", "#19486A"),
]


def chart_sdg(c, x, top, w, h):
    cols, gap = 4, 2.2
    tw = (w - (cols - 1) * gap) / cols
    th = (h - 2 * gap) / 3.0
    for i, (num, name, col) in enumerate(SDGS):
        bx = x + (i % cols) * (tw + gap)
        by = top + (i // cols) * (th + gap)
        c.setFillColor(HexColor(col))
        c.rect(bx * mm, yt(by + th), tw * mm, th * mm, stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 11.6)
        c.drawString((bx + 1.8) * mm, yt(by + th - 1.8) + 0.6 * mm, num)
        yy = by + 4.4
        for ln in name.split("\n"):
            _txt(c, ln, bx + 1.8, yy, FB, 5.5, Color(1, 1, 1, 0.95))
            yy += 2.6


# ══════════════════════════════ the content ═════════════════════════════════

TITLE = ("COMPUTER VISION-BASED SMART TRAFFIC MANAGEMENT WITH A SMART "
         "RETRACTABLE SPEED BUMP AND A MOVABLE LANE DIVIDER")
SUBTITLE = ("One camera measures the road in km/h — and the signals, the lane "
            "divider and a retractable speed bump all act on that one "
            "measurement, while an ambulance can flatten the bump over its own "
            "radio link")
FACULTY = "School of Computing  ·  Asia Pacific University of Technology & Innovation"
PROGRAMME = "Group Design Project  ·  Final Submission"

#: name, TP number
TEAM = [
    ("Zeyad Khairy", "TP074127"),
    ("Anfaz Mohamed", "TP073356"),
    ("Naif Mohamed", "TP073471"),
    ("Abdelrahman Osama", "TP072967"),
    ("Adem Becheikh", "TP074416"),
]

HEADLINES = [
    ("27.1", "fps end to end", "YOLO11-m @ 1280 px, FP16"),
    ("±5%", "speed accuracy", "calibrated, vs. ground truth"),
    ("0", "night false alarms", "down from 29% of all traffic"),
    ("45° / 90°", "bump raised / flat", "calibrated against the plate"),
    ("831/833", "automated checks pass", "7 suites, reproducible"),
    ("21,526", "vehicles logged", "across 69 recorded sessions"),
]

#: The bill of materials. item, quantity, unit price in RM.
COSTS = [
    ("Arduino Uno R3 — signal, lane and bump controllers", 3, 35.00),
    ("Arduino Nano — emergency vehicle transmitter", 1, 20.00),
    ("ESP32 DevKit V1 — relay hub", 1, 30.00),
    ("SX1278 433 MHz LoRa modules (TX + RX)", 2, 25.00),
    ("Servos — MG90S bump, 2 × SG90 divider", 3, 11.00),
    ("HC-SR04 ultrasonic sensor", 1, 8.00),
    ("WS2812B addressable LED strip, 1 m", 1, 35.00),
    ("16×2 I²C LCD modules", 2, 14.00),
    ("Signal-head LEDs, resistors, push button", 1, 36.00),
    ("5 V 1 A adapter, USB cables, breadboard, wiring", 1, 60.00),
    ("3D-printed linkage, printed road mat, cardboard bump", 1, 80.00),
]
COST_TOTAL = sum(q * u for _, q, u in COSTS)

EVID = os.path.join(HERE, "evidence")
NIGHT_SHOT = os.path.join(HERE, "Screenshot 2026-08-09 014446.png")
SHOT_BUMP_FLAT = os.path.join(EVID, "bump_flat.jpg")
SHOT_BUMP_UP = os.path.join(EVID, "bump_raised.jpg")
SHOT_ROAD = os.path.join(EVID, "road_prototype.jpg")
SHOT_LORA = os.path.join(EVID, "lora_transmitter.jpg")
SHOT_SERIAL = os.path.join(EVID, "bump_serial.png")
SHOT_SYSTEM = os.path.join(EVID, "system_running.jpg")

REFERENCES = [
    "World Health Organization. *Global Status Report on Road Safety 2023.* "
    "WHO, Geneva, 2023.",
    "Mokhtari, M. et al. “Intelligent Traffic Control with Smart Speed Bumps.” "
    "arXiv preprint, 2023.",
    "Edeva AB. *Actibump — How It Works.* edeva.se/en/actibump (accessed 2026).",
    "Sustainable Avenue. “Smart Speed Bumps Flatten for Vehicles Driving Within "
    "Speed Limits.” 2023.",
    "Jocher, G. and Qiu, J. *Ultralytics YOLO11.* Ultralytics, 2024.",
    "Aharon, N., Orfaig, R. and Bobrovsky, B.-Z. “BoT-SORT: Robust Associations "
    "Multi-Pedestrian Tracking.” arXiv:2206.14651, 2022.",
    "Radford, A. et al. “Learning Transferable Visual Models from Natural "
    "Language Supervision.” *ICML*, 2021. arXiv:2103.00020.",
    "Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*, "
    "2nd ed. Cambridge University Press, 2004.",
    "Semtech Corp. *SX1276/77/78/79 Low Power Long Range Transceiver — "
    "Datasheet.* Rev. 7, 2020.",
    "ElecFreaks. *HC-SR04 Ultrasonic Ranging Module — Datasheet.*",
    "Arduino S.r.l. *Arduino UNO R3 and Arduino Nano — product reference.*",
    "United Nations. *Transforming Our World: the 2030 Agenda for Sustainable "
    "Development.* UN, 2015.",
    "IEEE. *IEEE Code of Ethics.* IEEE, 2020.",
]

#: symptom -> what was done about it
TROUBLE = [
    ("The LoRa receiver stayed silent",
     "A read of register 0x42 came back **0x00** — the module was not communicating at all. "
     "The SPI wiring, the NSS line, the supply and the radio settings were checked in turn and "
     "the firmware corrected. The receiver now detects and validates the E command."),
    ("The bump moved the wrong way",
     "Servo travel was opposite to the assumed 0°/90°. It was re-measured against the physical "
     "plate and recalibrated to **45° raised, 90° flat**."),
    ("Servos reset their own board",
     "Stall current was sagging the logic rail. Actuators moved to a separate 5 V 1 A supply "
     "with a common ground."),
    ("Empty road reported as HEAVY",
     "The index remembered the last speed it saw. Zero vehicles now means zero congestion, "
     "full stop."),
    ("Slow computer, slow cars",
     "A video was being timed by the wall clock. It now carries its own clock, one tick per "
     "frame."),
    ("29% of night traffic called “police”",
     "An unfair 14-against-15 vote. Made one-on-one, with the flashing beacon — not the livery "
     "— as the trigger."),
    ("The signal flickered between phases",
     "Yellow became committed to its full five seconds, plus a three-second minimum dwell."),
    ("Two programs, one COM port",
     "Windows refuses the second. The node owns the port and the dashboard asks it to act."),
]

ERRORS = [
    "**Calibration.** The homography rests on four clicked points and one typed dimension; a "
    "5% error in that dimension scales every speed by 5% (±5% calibrated, ±25% automatic).",
    "**Perspective and occlusion.** Distant vehicles occupy few pixels and their boxes jitter, "
    "so displacement is taken over 0.8 s and anything over 200 km/h is discarded as an "
    "identity swap.",
    "**Ultrasonic physics.** A ~15° beam, ~3 mm quantisation and a ~0.17%/°C temperature "
    "dependence; soft or angled surfaces reflect poorly and a neighbouring object can trip "
    "the 20 cm gate.",
    "**Open-loop actuators.** Gear backlash, horn slop and plate flex mean the commanded 45° "
    "is not exactly the physical angle — nothing measures the true position.",
    "**Radio.** One character, no application-level acknowledgement; packet loss rises with "
    "distance and with interference on a shared 433 MHz band.",
    "**Timing jitter.** USB serial and Windows scheduling add tens of milliseconds between a "
    "decision and a movement.",
    "**Scale.** An 8 cm bump on a 360 mm road is geometrically but not dynamically similar to "
    "a real vehicle; timings here do not scale directly.",
    "**Machine state.** The same benchmark re-run returned 16% slower with an identical "
    "vehicle yield — throughput describes a machine as much as a model.",
]

LIMITS = [
    "Congestion control from the ESP32 / dashboard over **A0 and A1 is not fully reliable**; "
    "the push-button path is.",
    "**Vehicle-speed detection at the bump was not completed** — speed comes from the camera, "
    "not from the bump's own sensor.",
    "The LoRa link was formally demonstrated at a **bench distance of about 10 cm** only.",
    "The cardboard plate and micro servo suit a **scale prototype**; neither carries a vehicle "
    "load.",
    "**No position feedback** on the bump or the divider — both are open-loop.",
    "**No acknowledgement** is returned to the emergency vehicle, and the command carries no "
    "vehicle identity.",
    "**No LCD** on the bump subsystem, so its state is visible only on the Serial Monitor.",
    "Untested under **real vehicle loads, rain or outdoor lighting**; a single approach, one "
    "camera, and no hand-labelled frames means no mAP is quoted.",
]

FUTURE = [
    "Make the dashboard link deterministic — replace the A0 / A1 hint lines with a **framed, "
    "acknowledged UART or I²C message**.",
    "Add **speed measurement at the bump** with a two-gate ultrasonic pair, so calming can "
    "respond to speed as well as presence.",
    "Replace cardboard with **aluminium or steel plate** on a higher-torque actuator sized for "
    "a real wheel load.",
    "Fit **limit switches or an angle encoder** so the controller knows the plate angle "
    "instead of assuming it.",
    "**Range-test the radio** properly — RSSI and packet loss logged from 10 m to the "
    "hundreds of metres LoRa is designed for.",
    "Add **authenticated, acknowledged emergency packets** with a rolling code and a vehicle "
    "identity, and echo confirmation to the driver.",
    "Build a **weatherproof IP65 enclosure** and repeat response-time and reliability runs "
    "(n ≥ 100) before any outdoor trial.",
    "**Label a few hundred frames** for a real mAP, and port detection to an edge board so a "
    "laptop GPU is not required.",
]


def build_columns():
    """The four body columns, as lists of Sections."""

    # ══ column 1 — why the project exists and how it was run ════════════════

    background = S("Background & Problem Statement", [
        P("Urban junctions still run on **fixed-time plans**. A timer holds a green over an "
          "empty approach and a red over a queue, because the junction cannot see the road it "
          "governs. Where sensing does exist it is usually an **inductive loop** cut into the "
          "carriageway — costly to install, blind to what kind of vehicle passed, and unable "
          "to report a speed."),
        P("Traffic calming is blunter still: a **fixed bump slows everything that crosses "
          "it**, including the ambulance behind, and lane allocation fixed in paint leaves a "
          "road full one way and empty the other right through the peak."),
        B("**Nothing on the road measures the road.** Signal timing, lane allocation and "
          "calming are all set open-loop, from a plan rather than from the traffic present."),
        B("**Calming is indiscriminate**, and **the parts do not talk**: a permanent bump "
          "penalises the one class of traffic that must not be delayed, and the signal, the "
          "divider and the bump are separate installations sharing no measurement."),
        K("**Problem statement.** Urban traffic control is driven by plans instead of "
          "measurements: signals run on timers, lanes are fixed in paint, and speed bumps "
          "penalise emergency vehicles as heavily as speeding ones — because nothing on the "
          "road measures the traffic and shares it with the devices that act on it."),
    ], tag="why this exists")

    aim = S("Aim & Measurable Objectives", [
        P("**Aim.**  To design, build and validate a camera-driven smart traffic prototype "
          "that measures real traffic in engineering units and uses that one measurement to "
          "drive adaptive signals, a movable lane divider and a retractable speed bump with "
          "emergency-vehicle priority."),
        B("**O1 · Measure speed.** Report per-vehicle speed in km/h from a single camera to "
          "within **±5%** of ground truth once calibrated."),
        B("**O2 · Quantify congestion.** Produce one index on **0–1**, updated **every "
          "second**, that falls monotonically as mean speed falls, over **≥10,000** samples."),
        B("**O3 · Adapt the signal.** Drive real signal hardware from that index with a "
          "**committed 5 s yellow** and a **≥3 s minimum dwell**, and no phase flicker when "
          "congestion flips every 0.5 s."),
        B("**O4 · Recognise emergency vehicles.** Grant priority only to a vehicle that is "
          "both liveried **and** flashing, with **zero false alarms** on night footage."),
        B("**O5 · Retract the bump.** Raise it for a vehicle detected within **20 cm** and "
          "flatten it on a **valid LoRa command**, under a fixed **emergency > congestion > "
          "automatic** priority."),
        B("**O6 · Reallocate lanes.** Move the divider left / right / normal on command and "
          "show system state on an LED strip and an LCD, verified over **7 commands**."),
        B("**O7 · Prove it.** Log every second to CSV and XLSX, and validate the software with "
          "**more than 800 automated checks**."),
    ], tag="what counts as done")

    justify = S("Justification for the Project", [
        B("**The problem is lethal and expensive.** The WHO puts road-traffic deaths at "
          "**1.19 million a year**, and congestion costs cities working hours and fuel daily."),
        B("**Minutes decide outcomes.** Survival after out-of-hospital cardiac arrest falls by "
          "roughly **7–10% per minute** before treatment; a bump in front of an ambulance is "
          "not a small inconvenience."),
        B("**Existing sensing is too expensive to spread.** A loop means excavation, a lane "
          "closure and a fixed installation per approach. This prototype costs "
          f"**RM {COST_TOTAL:,.0f}** in parts and reuses cameras many junctions already have."),
        B("**Smart calming exists, but not at this price.** Commercial road plates prove the "
          "principle hydraulically; nothing in that class is emergency-aware at a scale a "
          "council could trial on one street — and this build still produced a reusable "
          "archive of **21,526 tracked vehicles**."),
    ], tag="why it is worth building")

    method = S("Overall System Methodology", [
        Chart(chart_method, 58.0, space=1.6),
        K("Each phase closed only when its output could be **measured** — a servo angle "
          "against the physical plate, a speed against constructed ground truth, a command "
          "on the Serial Monitor."),
    ], tag="six phases")

    practice = S("Professional Engineering Practices", [
        B("**Measure, then claim.** Every figure on this poster has a stated method, including "
          "what it does *not* prove — there is no mAP here because no frames were "
          "hand-labelled, and that is said rather than hidden."),
        B("**Automated regression testing.** 833 checks across 7 suites; the honest best "
          "single run is 831, and why the last two cannot pass in the same run is documented."),
        B("**Version control and a code knowledge graph** keep a 40-file Python system and "
          "three firmware images navigable, with a rollback copy of every superseded build."),
        B("**Interfaces agreed before build.** Pin maps, the seven-field serial line and the "
          "A0 / A1 hint lines were fixed on paper first, so three people could build three "
          "boards in parallel."),
        B("**Fail-loud hardware and ethical data.** Every command returns an acknowledgement, "
          "so an unplugged board is visible rather than silently ignored; only counts, speeds "
          "and classes are stored — no faces, no plates, no retained footage — in line with "
          "the IEEE Code of Ethics and PDPA principles."),
    ], tag="how the work was run")

    # ══ column 2 — how the system works ═════════════════════════════════════

    principle = S("Complete System Working Principle", [
        P("Every frame is turned into an engineering measurement, and every actuator on the "
          "prototype reads that same measurement. Nothing on this road is on a timer."),
        Chart(chart_loop, 26.0, space=2.4),
        B("**SEE.** YOLO11-m detects vehicles at 1280 px; BoT-SORT keeps one identity per "
          "vehicle across frames."),
        B("**MEASURE.** The bottom-centre of each box — the only part touching the road — is "
          "projected onto the road plane through a homography, so displacement is in metres; "
          "speed is taken over **0.8 s**."),
        B("**DECIDE.** Weighted density (how full) and speed deficit (how slow) combine into "
          "one index: an empty road is never congested, and one slow driver is not a jam."),
        B("**ACT.** Signal heads, lane divider, “make space” LED strip and speed bump are all "
          "driven from that index — and the emergency flag outranks all of them."),
        K("**The bump also runs its own loop:** ultrasonic → threshold → servo, ten times a "
          "second, with the radio able to override it at any moment. The calming device keeps "
          "working with the laptop switched off.", fill=TINT_WARM, accent=CORAL),
    ], tag="one loop, four outputs")

    integrate = S("How the Components Integrate", [
        P("The system is deliberately **two programs and four boards**. The CV node must be "
          "where the camera and the GPU are; the dashboard only needs a browser; the boards "
          "own the physical outputs."),
        T([("Link", "l", 0.31), ("Carries", "l", 0.42), ("Endpoint", "l", 0.27)],
          [["Camera → node", "frames: USB, file or stream URL", "broadcast_server.py"],
           ["Node → dashboard", "telemetry + commands, HTTP :8502", "Streamlit, 9 pages"],
           ["Node → ESP32", "seven-field signal line, COM6", "relay hub"],
           ["ESP32 → Signal Uno", "UART, GPIO33 → D7", "2 heads + LCD"],
           ["ESP32 → Lane Uno", "UART, GPIO25 / 26", "2 servos, strip, LCD"],
           ["ESP32 → Bump Uno", "congestion + emergency hints", "analog pins A0 / A1"],
           ["Nano → Bump Uno", "SX1278 LoRa, 433 MHz, command E", "bump servo"]],
          foot="Grounds are tied together. The Unos need power, not a data cable."),
        B("**Only the node holds the COM port.** Windows will not let two programs open the "
          "same port, so a dashboard button does not talk to the Arduino — it asks the node "
          "to. That one rule is what lets the dashboard run on another machine."),
        B("**The emergency path bypasses everything.** The ambulance's Nano talks to the "
          "bump's receiver directly, so priority does not depend on the laptop, the network or "
          "the dashboard being alive. Power is split from data as well: logic on 5 V USB, "
          "servos and the strip on a separate 5 V 1 A supply, one common ground."),
    ], tag="one port, one ground")

    vision = S("Subsystem A — Vision, Speed & Signal", [
        P("Speed is the measurement everything else rests on, so it is taken on the **road "
          "plane in metres**, never in pixels: four corners of a real rectangle are clicked, "
          "their true size typed in, and a homography maps the road plane."),
        Chart(chart_speed_vs_congestion, 42.0,
              "12,919 one-second samples across 66 recorded sessions, binned by congestion "
              "index. Bars are mean speed in km/h; the line is mean vehicles present.",
              space=2.4),
        B("**Two cues decide an emergency.** CLIP asks *what is this vehicle*; the beacon "
          "detector asks *is it on a call right now*. A lamp is on or off and is almost never "
          "caught half-lit, so a red car drifting under a street light is rejected — night "
          "false alarms went from **88 of 299 vehicles to zero**."),
        B("**The signal is demand-responsive, and committed.** Side-road heads hold merging "
          "traffic back only while the main road is busiest; yellow always runs its full "
          "**5 s** and every phase holds a **3 s** minimum, so the lights cannot chatter when "
          "the index crosses a boundary."),
    ], tag="the measurement")

    innovation = S("Innovation, Creativity & Enhancements", [
        B("**One measurement, four actuators.** The signal, the divider, the LED strip and the "
          "bump are not four gadgets — they are four consumers of a single congestion index "
          "and a single emergency flag."),
        B("**Calming that gets out of the way.** A bump that flattens for a validated "
          "emergency command inverts the usual trade-off between calming and response time."),
        B("**Two-cue emergency recognition.** Livery *and* beacon: a parked ambulance outside "
          "a hospital is still an ambulance, and it does not get priority."),
        B("**A radio path that survives the computer.** The emergency link is peer-to-peer, so "
          "the safety-critical behaviour has the fewest dependencies of anything here."),
        B("**Source-agnostic pipeline, with its own clock.** Webcam, uploaded file or roadside "
          "stream run through the same detector and the same log, chosen while the system is "
          "running; and replaying footage on a faster machine no longer makes the traffic look "
          "faster."),
    ], tag="what is new here")

    # ══ column 3 — the two hardware subsystems and the evidence ═════════════

    bump = S("Subsystem B — Smart Retractable Speed Bump", [
        P("An **8 × 4 cm** hinged bump is driven by a servo beneath the carriageway and "
          "controlled by its own **Arduino Uno**. The angles were calibrated against the "
          "physical plate rather than assumed: **90° is completely flat, 45° is fully "
          "raised.**"),
        Chart(chart_bump, 36.0,
              "The two calibrated positions. An HC-SR04 watches the approach and anything "
              "closer than 20 cm counts as a vehicle.", space=2.4),
        P("**Three inputs, one ladder.** The controller tests the rules in a fixed order and "
          "the first match wins, so an emergency can never be outvoted by congestion or by "
          "the sensor.", space=1.8),
        Chart(chart_priority, 40.0, space=1.4),
        Term(["Distance: 34 cm   mode AUTO        servo  90  (flat)",
              "Distance: 18 cm   mode AUTO        servo  45  (raised)",
              "Button pressed  ->  mode CONGESTION,  servo  90  (flat)",
              "LoRa RX packet: 'E'  -> validated  -> mode EMERGENCY",
              "EMERGENCY active  ->  servo  90  (flat), held",
              "EMERGENCY cleared ->  mode AUTO"],
             title="Serial Monitor  ·  bump controller  ·  COM4",
             hi=(3, 4)),
        P("*The trace the controller prints on every state change; the photographed run is in "
          "Prototype Evidence.*", size=6.8, lead=8.4, colour=INK_SOFT, space=2.2),
        B("**The command is validated, not merely received.** The receiver acts on the exact "
          "character `E` and ignores anything else, so noise on a shared 433 MHz band cannot "
          "flatten the bump. On reset, on loss of the link and on any unknown state the servo "
          "is commanded to **90° flat** — the position that cannot trap a vehicle."),
    ], tag="calming that gets out of the way")

    road = S("Subsystem C — Road Prototype & Lane Divider", [
        P("The prototype is a **360 × 480 mm printed carriageway** with 30° merge lanes, "
          "carrying every device the system drives. The divider was drawn in **CAD**, its "
          "linkage **3D-printed**, and two servos move the boundary between the carriageways "
          "toward the busier direction."),
        Chart(chart_road, 54.0,
              "Plan view: signal heads on the approach, the movable divider on the "
              "centreline, the retractable bump across the near carriageway, the LED strip on "
              "the verge, and the camera that measures all of it.", space=2.4),
        T([("Command", "l", 0.26), ("Verified response", "l", 0.63), ("", "c", 0.11)],
          [["LEFT", "divider swings toward the left carriageway", "✓"],
           ["RIGHT", "divider swings toward the right carriageway", "✓"],
           ["NORMAL", "divider returns to the centreline", "✓"],
           ["AMBULANCE", "emergency pattern on the strip, lane cleared", "✓"],
           ["POLICE", "emergency pattern, priority phase requested", "✓"],
           ["ACCIDENT", "hazard pattern, closure shown on the LCD", "✓"],
           ["SPEED", "speed advisory written to the 16×2 I²C LCD", "✓"]],
          foot="Each command was issued from the dashboard and confirmed on the Serial "
               "Monitor.", space=1.6),
    ], tag="the physical road")

    evidence = S("Evidence of the Working Prototype", [
        Shots([(NIGHT_SHOT, "LIVE DETECTION · NIGHT", "")],
              "Night footage, live: the overlay reports the congestion index, the signal phase "
              "and the servo angle, and every box carries an ID, a class and a km/h.", 32.0),
        Shots([(SHOT_BUMP_FLAT, "BUMP · FLAT 90°",
                "bump flat,\nfrom the side\nevidence/\nbump_flat.jpg"),
               (SHOT_BUMP_UP, "BUMP · RAISED 45°",
                "same angle,\nbump raised\nevidence/\nbump_raised.jpg"),
               (SHOT_SERIAL, "SERIAL · 'E' RECEIVED",
                "screenshot of\nthe live run\nevidence/\nbump_serial.png")],
              "The two demonstrated bump positions, and the capture of the E command arriving "
              "and flattening it.", 25.0),
        Shots([(SHOT_ROAD, "ROAD PROTOTYPE",
                "the whole road: divider,\nlights and bump in place\n"
                "evidence/road_prototype.jpg"),
               (SHOT_LORA, "EMERGENCY TRANSMITTER",
                "Nano + SX1278 vehicle unit\nevidence/lora_transmitter.jpg")],
              "The fabricated road, and the emergency vehicle unit.", 27.0),
    ], tag="photographed, not simulated")

    # ══ column 4 — evidence, results and impact ═════════════════════════════

    testing = S("Testing Procedure & Collected Data", [
        P("Testing ran at four levels, and nothing moved up a level until the level below it "
          "passed."),
        T([("Level", "l", 0.19), ("Procedure", "l", 0.55), ("Evidence", "l", 0.26)],
          [["Unit", "servo sweep, ultrasonic against a tape measure, LoRa register read",
            "bench log"],
           ["Subsystem", "priority truth table, signal timing on a synthetic clock, detector "
                         "benchmark", "7 test suites"],
           ["Integration", "one measurement driving all four actuators; dashboard → node → "
                           "board", "acknowledged commands"],
           ["System", "69 recorded sessions, day and night footage, seven divider commands",
            "CSV / XLSX archive"]]),
        T([("Speed validation", "l", 0.52), ("Expected", "r", 0.24), ("Measured", "r", 0.24)],
          [["Homography mode", "18 / 36 / 72", "**18.00 / 36.00 / 72.00**"],
           ["Automatic (height-based)", "18 km/h", "**18.00**"],
           ["Parked vehicle", "0 km/h", "**0.12**"],
           ["Clip at 1×, 2×, ½×, unthrottled", "16.2 km/h", "**16.2**"]],
          foot="Constructed ground truth: the vehicle is synthetic, so its speed is known "
               "rather than estimated."),
        T([("Radio link test", "l", 0.44), ("Observation", "l", 0.36), ("", "c", 0.20)],
          [["Register 0x42 read-back", "0x00 — module silent", "**FAIL**"],
           ["After wiring and firmware fix", "packet received at ~10 cm", "**PASS**"],
           ["Command validation", "E accepted, other bytes ignored", "**PASS**"]],
          foot="The 0x00 read is the diagnostic that found the fault; reception of E is the "
               "final evidence.", space=1.8),
        P("**Data collected.** 69 sessions, **21,526 tracked vehicles** and **12,919 "
          "one-second samples** archived to CSV and XLSX — one row per second and one per "
          "vehicle, with class, dwell time, distance, average and top speed."),
    ], tag="four levels")

    results = S("Results & Evaluation Against the Objectives", [
        Verdict([
            ("O1", "±5% speed",
             "18 / 36 / 72 km/h returned 18.00 / 36.00 / 72.00; ±5% calibrated in the field",
             True),
            ("O2", "index 0–1, ≥10,000 samples",
             "12,919 samples; mean speed falls 42.2 → 10.0 km/h across the five bins", True),
            ("O3", "5 s yellow, 3 s dwell, no flicker",
             "exact against a synthetic clock, congestion flipping every 0.5 s", True),
            ("O4", "zero night false alarms",
             "88 of 299 vehicles (29%) → 0; 166 of 2,511 flagged on unseen footage", True),
            ("O5", "20 cm raise, LoRa flatten",
             "45° ↔ 90° on the sensor; E validated and ranked above both other modes", True),
            ("O6", "divider + LED + LCD, 7 commands",
             "all seven responded, confirmed on the Serial Monitor", True),
            ("O7", ">800 checks, per-second log",
             "831 of 833 in one run; 21,526 vehicles over 69 sessions archived", True),
            ("—", "A0 / A1 congestion hint",
             "intermittent — the button path is reliable, the analog path is not", False),
            ("—", "radio range",
             "shown at ~10 cm on the bench; no field-range test was run", False),
        ]),
        P("*Seven objectives met in full; two stretch items met in part and reported as such.*",
          size=6.9, lead=8.6, colour=INK_SOFT, space=1.0),
    ], tag="measured, not estimated")

    safety = S("Health & Safety Considerations", [
        B("**Low voltage only.** Everything runs at **5 V SELV** from a USB port and a 1 A "
          "adapter; there is no mains wiring anywhere in the prototype."),
        B("**Actuators on their own supply.** Servo stall current is kept off the logic rail "
          "so a board cannot brown out mid-movement, with grounds commonly tied."),
        B("**The bump fails flat.** Power loss, link loss and any unknown state all command "
          "90° — the position that cannot trap or launch a vehicle."),
        B("**Pinch points are designed out.** At this scale the plate is cardboard and "
          "harmless; a full-size plate would need a guarded hinge line, a torque limiter and "
          "an obstruction cut-out before any road trial."),
        B("**Build and data safety.** Soldering with ventilation and eye protection, ESD care "
          "with bare boards, 3D printing in a ventilated room, cabling clear of walkways — "
          "and no faces, plates or retained footage, only counts, speeds and classes."),
    ], tag="5 V, fail-flat")

    green = S("Environmental & Sustainability Considerations", [
        B("**No excavation.** The system retrofits onto cameras and poles that already exist, "
          "avoiding the road-cutting, lane closure and materials an inductive loop needs."),
        B("**Less idling.** Signals that answer the queue, and a bump that flattens for an "
          "emergency vehicle, both cut stop–start running — where urban fuel burn is worst."),
        B("**A small power budget.** Three microcontrollers and an LED strip on 5 V, 1 A; the "
          "GPU stays indoors and can serve several approaches."),
        B("**Materials chosen to be reworked.** Cardboard and PLA, printed rather than moulded "
          "parts, boards reused between builds — and adaptive calming wears road, brakes and "
          "tyres less than a permanent bump does."),
    ], tag="retrofit, not rebuild")

    sdg = S("United Nations Sustainable Development Goals", [
        Chart(chart_sdg, 42.0, space=2.0),
        B("**11 · Sustainable cities** is the core goal — measured, adaptive urban mobility — "
          "with **3 · Good health** and **9 · Infrastructure** following from faster emergency "
          "response and intelligence retrofitted onto existing roads."),
        B("**13 · Climate action**, **12 · Responsible consumption**, **10 · Reduced "
          "inequalities** and **17 · Partnerships**: less idling, no excavation, reused "
          "hardware, at a price that reaches districts which could never fund loops."),
    ], tag="12 goals addressed")

    cost = S("Actual Component Cost", [
        T([("Component", "l", 0.56), ("Qty", "c", 0.10), ("Unit (RM)", "r", 0.16),
           ("Cost (RM)", "r", 0.18)],
          [[item, str(q), f"{u:,.2f}", f"{q * u:,.2f}"] for item, q, u in COSTS]
          + [["**TOTAL**", "", "", f"**{COST_TOTAL:,.2f}**"]],
          bold_rows=(len(COSTS),),
          foot="Malaysian retail prices. The laptop, camera and workshop tools were existing "
               "assets and carry no project cost.", space=1.8),
        K(f"Three controllers, a radio pair, four servos, the road and every sensor come to "
          f"**RM {COST_TOTAL:,.2f}** — less than the site preparation for one inductive loop."),
    ], tag="what it actually cost")

    market = S("Target Market & Marketing Strategy", [
        B("**Primary market — city councils and road authorities** retrofitting junctions "
          "where CCTV is already installed and a loop would mean digging."),
        B("**Secondary — emergency services, campuses and estates**: hospital and fire-station "
          "corridors, gated developments and universities that want calming without "
          "penalising ambulances."),
        B("**Route to market.** One instrumented junction as a paid pilot, a measured "
          "before-and-after report, then tender; kit sold near cost, analytics licensed per "
          "camera."),
        B("**Positioning and partners.** Cheaper than loops, no excavation, works on existing "
          "cameras, and the three subsystems sell separately — a council can start with the "
          "bump alone. CCTV integrators own the installed base; road-safety bodies and "
          "university pilots supply the independent evidence."),
    ], tag="who buys this")

    return [
        [background, aim, justify, method, market, green],
        [principle, integrate, vision, innovation, practice],
        [bump, road, evidence],
        [testing, results, safety, sdg, cost],
    ]


# ══════════════════════════ page furniture ══════════════════════════════════

def draw_header(c):
    c.setFillColor(PAPER)
    c.rect(0, yt(HEADER_H), PAGE_W, HEADER_H * mm, stroke=0, fill=1)
    network_pattern(c, 0, yt(HEADER_H), PAGE_W, PAGE_H, seed=11, n=48, alpha=0.5)
    chevron(c, PAGE_W * 0.5, PAGE_H, 54 * mm, 26 * mm)

    lx = MARGIN * mm
    cy = yt(19.0)
    c.setFillColor(BAR)
    c.setFont(FB, 25)
    c.drawString(lx, cy, "SMART")
    w1 = pdfmetrics.stringWidth("SMART", FB, 25)
    c.setFillColor(CORAL)
    c.drawString(lx + w1 + 4, cy, "TRAFFIC")
    c.setFillColor(BAR)
    c.setFont(FL if mp.HAVE_SEGOE else F, 9.4)
    c.drawString(lx + 1.0, cy - 11, "M E A S U R E D   ·   A D A P T I V E   ·   S A F E")
    c.setStrokeColor(CORAL)
    c.setLineWidth(1.4)
    c.line(lx, cy - 16, lx + 58 * mm, cy - 16)

    draw_apu_logo(c, PAGE_W - MARGIN * mm, yt(16.5), 17 * mm)

    tw = PAGE_W_MM - 2 * MARGIN - 20
    tx = MARGIN + 10
    lines = wrap_runs(parse_runs(TITLE, base=FB), 23.0, tw * mm)
    y_pt = draw_lines(c, lines, tx * mm, yt(31.0), 23.0, 26.5,
                      HexColor("#12181D"), "center", tw * mm)
    y_pt -= 2
    lines = wrap_runs(parse_runs(SUBTITLE, base=FI), 10.6, (tw - 30) * mm)
    y_pt = draw_lines(c, lines, (tx + 15) * mm, y_pt, 10.6, 13.2, BAR,
                      "center", (tw - 30) * mm)
    lines = wrap_runs(parse_runs(FACULTY + "   ·   " + PROGRAMME, base=F),
                      8.8, tw * mm)
    draw_lines(c, lines, tx * mm, y_pt - 1.2, 8.8, 11.0, INK_SOFT, "center", tw * mm)


def draw_team(c):
    x, w, h = MARGIN, FULL_W, TEAM_H
    c.setFillColor(BAR_DEEP)
    c.rect(x * mm, yt(TEAM_TOP + h), w * mm, h * mm, stroke=0, fill=1)
    rail = 24.0
    c.setFillColor(CORAL)
    c.rect(x * mm, yt(TEAM_TOP + h), rail * mm, h * mm, stroke=0, fill=1)
    c.setFillColor(PAPER)
    c.setFont(FB, 10.6)
    c.drawCentredString((x + rail / 2) * mm, yt(TEAM_TOP + h / 2) + 0.6 * mm, "PROJECT")
    c.drawCentredString((x + rail / 2) * mm, yt(TEAM_TOP + h / 2) - 3.8 * mm, "TEAM")

    cx = x + rail
    cw = (w - rail) / len(TEAM)
    for i, (name, tp) in enumerate(TEAM):
        bx = cx + i * cw
        if i:
            c.setStrokeColor(Color(1, 1, 1, 0.22))
            c.setLineWidth(0.6)
            c.line(bx * mm, yt(TEAM_TOP + h - 3.5), bx * mm, yt(TEAM_TOP + 3.5))
        c.setFillColor(CORAL)
        c.circle((bx + 7.0) * mm, yt(TEAM_TOP + h / 2), 3.4 * mm, stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 8.4)
        c.drawCentredString((bx + 7.0) * mm, yt(TEAM_TOP + h / 2) - 2.8, str(i + 1))
        c.setFillColor(PAPER)
        c.setFont(FB, 11.4)
        c.drawString((bx + 12.5) * mm, yt(TEAM_TOP + h / 2) + 0.4 * mm, name)
        c.setFillColor(CORAL_PALE)
        c.setFont(FB, 9.2)
        c.drawString((bx + 12.5) * mm, yt(TEAM_TOP + h / 2) - 4.4 * mm, tp)


def draw_kpi(c):
    x, w, h, top = MARGIN, FULL_W, KPI_H, KPI_TOP
    n = len(HEADLINES)
    gap = 3.0
    cw = (w - gap * (n - 1)) / n
    for i, (fig, label, note) in enumerate(HEADLINES):
        bx = x + i * (cw + gap)
        c.setFillColor(PAPER)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.7)
        c.rect(bx * mm, yt(top + h), cw * mm, h * mm, stroke=1, fill=1)
        c.setFillColor(CORAL if i in (2, 3) else BAR)
        c.rect(bx * mm, yt(top + h), cw * mm, 1.5 * mm, stroke=0, fill=1)
        size = 16.0
        while pdfmetrics.stringWidth(fig, FB, size) > (cw - 8) * mm:
            size -= 0.6
        c.setFillColor(BAR_DEEP)
        c.setFont(FB, size)
        c.drawCentredString((bx + cw / 2) * mm, yt(top + 9.0), fig)
        c.setFillColor(INK)
        c.setFont(FB, 7.8)
        c.drawCentredString((bx + cw / 2) * mm, yt(top + 12.8), label)
        c.setFillColor(INK_SOFT)
        c.setFont(F, 6.6)
        c.drawCentredString((bx + cw / 2) * mm, yt(top + 16.4), note)


def _band(c, top, h, title, tag=None, note=None, bar_h=8.4, bar=BAR_DEEP):
    """A full-width titled box; returns the y where its content may start."""
    x, w = MARGIN, FULL_W
    c.setFillColor(PAPER)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.7)
    c.rect(x * mm, yt(top + h), w * mm, h * mm, stroke=1, fill=1)
    c.setFillColor(bar)
    c.rect(x * mm, yt(top + bar_h), w * mm, bar_h * mm, stroke=0, fill=1)
    c.setFillColor(PAPER)
    c.setFont(FB, 12.0)
    c.drawString((x + 3.6) * mm, yt(top + bar_h) + 2.4 * mm, title)
    if note:
        c.setFillColor(Color(1, 1, 1, 0.80))
        c.setFont(F, 9.0)
        c.drawString((x + 3.6) * mm + pdfmetrics.stringWidth(title, FB, 12.0) + 9,
                     yt(top + bar_h) + 2.5 * mm, note)
    if tag:
        c.setFillColor(Color(1, 1, 1, 0.72))
        c.setFont(F, 8.2)
        c.drawRightString((x + w - 3.6) * mm, yt(top + bar_h) + 2.6 * mm, tag)
    return top + bar_h


def draw_block_band(c):
    top = _band(c, BLOCK_TOP, BLOCK_H, "Complete System Block Diagram",
                tag="five stages, four boards, two radios",
                note="—  one video feed becomes one measurement, and every actuator on the "
                     "prototype is driven from it")
    chart_block(c, MARGIN + 3.0, top + 3.0, FULL_W - 6.0, BLOCK_H - 8.4 - 5.0)


def draw_trouble_band(c):
    top = _band(c, TROUBLE_TOP, TROUBLE_H, "Problems Encountered & Troubleshooting",
                tag="8 of them",
                note="—  almost every rule in this system exists because something went wrong "
                     "first, got measured, and got fixed")
    n = len(TROUBLE)
    gap = 2.2
    cw = (FULL_W - 2 * 3.6 - gap * (n - 1)) / n
    cy = top + 3.0
    ch = TROUBLE_TOP + TROUBLE_H - cy - 3.0
    for i, (bad, good) in enumerate(TROUBLE):
        bx = MARGIN + 3.6 + i * (cw + gap)
        c.setFillColor(HexColor("#FCEEEA"))
        c.rect(bx * mm, yt(cy + ch), cw * mm, ch * mm, stroke=0, fill=1)
        c.setFillColor(HexColor("#F7DED6"))
        c.rect(bx * mm, yt(cy + 1.8), cw * mm, 1.8 * mm, stroke=0, fill=1)
        pad = 2.2
        iw = cw - 2 * pad
        lines = wrap_runs(parse_runs(bad, base=FBI), 7.6, iw * mm)
        y_pt = draw_lines(c, lines, (bx + pad) * mm, yt(cy + 2.2), 7.6, 9.2,
                          HexColor("#A5352A"))
        y_mm = (PAGE_H - y_pt) / mm + 1.4
        c.setFillColor(CORAL)
        c.setFont(FB, 8.6)
        c.drawString((bx + pad) * mm, yt(y_mm + 2.6), "↓")
        lines = wrap_runs(parse_runs(good, base=F), 6.9, (iw - 3.6) * mm)
        draw_lines(c, lines, (bx + pad + 3.6) * mm, yt(y_mm - 0.6), 6.9, 8.4, INK)


def _bullets(c, items, x, top, w, colour=BAR, size=6.8, lead=8.4, gap=1.1):
    y = top
    for txt in items:
        c.setFillColor(colour)
        c.setFont(FB, size)
        c.drawString(x * mm, yt(y) - lead, "•")
        lines = wrap_runs(parse_runs(txt), size, (w - 3.2) * mm)
        y_pt = draw_lines(c, lines, (x + 3.2) * mm, yt(y), size, lead, INK)
        y = (PAGE_H - y_pt) / mm + gap
    return y


def draw_gaps_band(c):
    top = _band(c, GAPS_TOP, GAPS_H, "Sources of Error  ·  Limitations  ·  Future Work",
                tag="what the panel should ask about",
                note="—  the honest edges of the prototype, and what each of them needs next")
    panels = [("Sources of Error", ERRORS, BAR),
              ("Current System Limitations", LIMITS, CORAL),
              ("Future Improvements", FUTURE, TEAL)]
    gap = 4.0
    pw = (FULL_W - 2 * 3.6 - gap * 2) / 3.0
    for i, (title, items, col) in enumerate(panels):
        px = MARGIN + 3.6 + i * (pw + gap)
        c.setFillColor(col)
        c.rect(px * mm, yt(top + 3.0 + 5.4), pw * mm, 5.4 * mm, stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 8.2)
        c.drawString((px + 2.0) * mm, yt(top + 3.0 + 5.4) + 1.5 * mm, title)
        _bullets(c, items, px + 1.0, top + 9.6, pw - 2.0, col)


CONCLUSION = (
    "Seven objectives were set and **seven were met**, with two stretch items met only in "
    "part and reported as such. A single camera and a laptop GPU measure real traffic at "
    "**27.1 fps** and return **±5% speed accuracy** once calibrated — 18, 36 and 72 km/h of "
    "constructed ground truth came back as 18.00, 36.00 and 72.00 **(O1)**. Those speeds "
    "become one congestion index over **12,919 logged one-second samples**, across which mean "
    "speed falls monotonically from 42.2 to 10.0 km/h **(O2)**, and that index drives a signal "
    "that cannot chatter: an exactly committed five-second yellow and a three-second dwell, "
    "proved against a synthetic clock **(O3)**. Requiring a vehicle to be both liveried and "
    "flashing took night false alarms from **88 of 299 vehicles to zero (O4)**.\n"
    "On the physical prototype the bump raises to its calibrated **45°** for a vehicle "
    "detected within 20 cm and returns to **90° flat** on a validated LoRa **E** command, with "
    "emergency priority demonstrated over both congestion and automatic modes **(O5)**; the "
    "movable divider answered all seven commands with LED and LCD state **(O6)**; and the "
    "whole system is logged to CSV and XLSX and checked by **831 of 833 automated tests "
    "(O7)**. What is not yet finished is stated just as plainly: the dashboard's analog hint "
    "path is unreliable, and the radio has only been demonstrated across a bench. The project "
    "therefore closes with a working, measured prototype — and a list of exactly what would "
    "have to be true before it belonged on a real street."
)


def draw_close_band(c):
    top = _band(c, CLOSE_TOP, CLOSE_H, "Conclusion", tag="against the objectives",
                note="—  what was aimed at, what was measured, and what is still open")
    left_w = FULL_W * 0.615
    right_x = MARGIN + left_w + 5.0
    right_w = FULL_W - left_w - 5.0

    y = top + 3.4
    col_gap = 5.0
    cw = (left_w - 3.6 - col_gap) / 2.0
    paras = CONCLUSION.split("\n")
    for i, para in enumerate(paras):
        lines = wrap_runs(parse_runs(para), 7.4, cw * mm)
        draw_lines(c, lines, (MARGIN + 3.6 + i * (cw + col_gap)) * mm, yt(y),
                   7.4, 9.4, INK)

    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.line((right_x - 2.5) * mm, yt(top + 2.5),
           (right_x - 2.5) * mm, yt(CLOSE_TOP + CLOSE_H - 2.5))
    c.setFillColor(BAR_DEEP)
    c.setFont(FB, 9.0)
    c.drawString(right_x * mm, yt(top + 5.6), "References")

    half = (len(REFERENCES) + 1) // 2
    rw = (right_w - 5.0) / 2.0
    for ci, chunk in enumerate((REFERENCES[:half], REFERENCES[half:])):
        y = top + 7.2
        for j, txt in enumerate(chunk):
            n = ci * half + j + 1
            c.setFillColor(BAR)
            c.setFont(FB, 5.9)
            c.drawString((right_x + ci * (rw + 5.0)) * mm, yt(y) - 7.0, f"{n}.")
            lines = wrap_runs(parse_runs(txt), 5.9, (rw - 4.0) * mm)
            y_pt = draw_lines(c, lines, (right_x + ci * (rw + 5.0) + 4.0) * mm,
                              yt(y), 5.9, 7.0, INK)
            y = (PAGE_H - y_pt) / mm + 0.9


def draw_footer(c):
    c.setFillColor(PAPER)
    c.rect(0, 0, PAGE_W, FOOTER_H * mm, stroke=0, fill=1)
    network_pattern(c, 0, 0, PAGE_W, FOOTER_H * mm, seed=23, n=22, alpha=0.5)
    footer_motif(c, PAGE_W * 0.46, 0, 40 * mm, FOOTER_H * mm * 0.82)
    c.setStrokeColor(BAR)
    c.setLineWidth(1.3)
    c.line(MARGIN * mm, yt(FOOTER_TOP), (PAGE_W_MM - MARGIN) * mm, yt(FOOTER_TOP))

    c.setFillColor(BAR_DEEP)
    c.setFont(FB, 13.0)
    c.drawString(MARGIN * mm, yt(PAGE_H_MM - 14.0),
                 "SMART TRAFFIC  ·  MEASURED, ADAPTIVE, SAFE")
    c.setFillColor(INK_SOFT)
    c.setFont(F, 8.6)
    c.drawString(MARGIN * mm, yt(PAGE_H_MM - 7.6),
                 "Group Design Project  ·  School of Computing  ·  "
                 + "   ".join(t for _, t in TEAM))

    c.setFillColor(BAR)
    c.setFont(FB, 12.0)
    c.drawRightString((PAGE_W_MM - MARGIN) * mm, yt(PAGE_H_MM - 14.0),
                      "ASIA PACIFIC UNIVERSITY OF TECHNOLOGY & INNOVATION")
    c.setFillColor(INK_SOFT)
    c.setFont(F, 8.2)
    c.drawRightString((PAGE_W_MM - MARGIN) * mm, yt(PAGE_H_MM - 7.6),
                      "Vision figures measured on an RTX 4050 laptop GPU, FP16  ·  "
                      "methods and provenance in METRICS.md")


def main():
    c = rl_canvas.Canvas(OUT, pagesize=(PAGE_W, PAGE_H))
    c.setTitle("Smart Traffic Management with a Retractable Speed Bump and "
               "Movable Lane Divider")
    c.setAuthor(" / ".join(f"{n} ({t})" for n, t in TEAM))
    c.setSubject("Group Design Project poster — A1 portrait")

    c.setFillColor(GREY_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    draw_header(c)
    draw_team(c)
    draw_kpi(c)
    draw_block_band(c)

    used, detail = [], []
    for ci, sections in enumerate(build_columns()):
        y = BODY_TOP
        for s in sections:
            h = s.height(COL_W)
            s.draw(c, COL_X[ci], y, COL_W)
            detail.append((ci + 1, s.title, h))
            y += h + s.gap_after
        used.append(y - sections[-1].gap_after)

    draw_trouble_band(c)
    draw_gaps_band(c)
    draw_close_band(c)
    draw_footer(c)

    c.showPage()
    c.save()

    print(f"wrote {OUT}")
    print(f"page   A1 portrait  {PAGE_W_MM:.0f} x {PAGE_H_MM:.0f} mm  ·  "
          f"{NCOL} columns of {COL_W:.1f} mm")
    print(f"body   {BODY_TOP:.0f} -> {BODY_BOTTOM:.0f} mm "
          f"({BODY_BOTTOM - BODY_TOP:.0f} mm available)")
    for i, u in enumerate(used):
        flag = (f"OVERFLOW by {u - BODY_BOTTOM:.1f} mm" if u > BODY_BOTTOM
                else f"{BODY_BOTTOM - u:6.1f} mm spare")
        print(f"col {i + 1}  ends at {u:6.1f} mm   {flag}")
    if os.environ.get("POSTER_FIT"):
        for ci, title, h in detail:
            print(f"   col{ci}  {h:6.1f} mm  {title}")
    print(f"cost   RM {COST_TOTAL:,.2f} over {len(COSTS)} lines")
    missing = [p for p in (SHOT_BUMP_FLAT, SHOT_BUMP_UP, SHOT_ROAD, SHOT_LORA,
                           SHOT_SERIAL) if not os.path.exists(p)]
    if missing:
        print("photo slots still empty:")
        for p in missing:
            print("   " + os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()

