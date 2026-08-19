#!/usr/bin/env python3
"""Build the A1 conference poster for the Smart Traffic GDP.

    python make_poster.py            -> GDP_Poster_A1.pdf

Everything is vector except the four photographs, so it prints at any size.
The page is A1 portrait (594 x 841 mm) and the layout is a three-column
academic poster modelled on `poster example.jpeg`: a white header carrying the
logo and the title, a full-width team strip, three columns of section boxes,
and a footer band.

To fill in the group's names, edit TEAM below and nothing else.
To use the real APU logo, drop a file at APU_LOGO_PATH and re-run.

Every figure quoted here comes from METRICS.md or from the session archive
(`traffic_all_sessions.csv`, `traffic_all_vehicles.csv`); the provenance of
each number is noted beside it in the CONTENT section.
"""

from __future__ import annotations

import os
import re
import random
import math

from reportlab.lib.units import mm
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "GDP_Poster_A1.pdf")

#: Drop the real logo here (PNG with transparency preferred) and re-run.
APU_LOGO_PATH = os.path.join(HERE, "apu_logo.png")


# ───────────────────────────── page geometry ────────────────────────────────

PAGE_W_MM, PAGE_H_MM = 594.0, 841.0
PAGE_W, PAGE_H = PAGE_W_MM * mm, PAGE_H_MM * mm

MARGIN = 16.0                      # mm
GUTTER = 7.0                       # mm
COL_W = (PAGE_W_MM - 2 * MARGIN - 2 * GUTTER) / 3.0    # 182.67 mm

HEADER_H = 114.0                   # mm, white band with logo + title
TEAM_TOP = 116.0
TEAM_H = 42.0
BAND_TOP = TEAM_TOP + TEAM_H + 4.5    # the five headline figures
BAND_H = 24.0
BODY_TOP = BAND_TOP + BAND_H + 5.0    # 190.5
FOOTER_TOP = 802.0
LESSONS_H = 46.0                      # full-width band closing the three columns
LESSONS_TOP = FOOTER_TOP - 4.0 - LESSONS_H
BODY_BOTTOM = LESSONS_TOP - 5.0

COL_X = [MARGIN + i * (COL_W + GUTTER) for i in range(3)]


def yt(v_mm: float) -> float:
    """A y measured in mm from the top of the page, in PDF points."""
    return PAGE_H - v_mm * mm


# ─────────────────────────────── palette ────────────────────────────────────

INK        = HexColor("#1A1D21")   # body text
INK_SOFT   = HexColor("#4A5158")   # captions, secondary
RULE       = HexColor("#C9D2D9")   # box borders
BAR        = HexColor("#115B82")   # section header bar (sampled from example)
BAR_DEEP   = HexColor("#0C4460")
NAVY       = HexColor("#2E3468")   # chevron dark
PERI       = HexColor("#6F7DBC")   # chevron light
CORAL      = HexColor("#F5876A")
CORAL_PALE = HexColor("#FBC0A9")
APU_RED    = HexColor("#C8102E")
TEAL       = HexColor("#1B8FA8")
GREEN      = HexColor("#2E8B57")
AMBER      = HexColor("#D98C0B")
RED        = HexColor("#C0392B")
BLUE       = HexColor("#2F6FB5")
PAPER      = HexColor("#FFFFFF")
TINT       = HexColor("#F2F6F9")   # panel fill inside boxes
TINT_WARM  = HexColor("#FDF4EE")

CAT = [BAR, CORAL, TEAL, NAVY, AMBER, GREEN]   # categorical series


# ─────────────────────────────── fonts ──────────────────────────────────────

def _reg(name, filename):
    path = os.path.join(r"C:\Windows\Fonts", filename)
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(name, path))
        return True
    return False


HAVE_SEGOE = all([
    _reg("UI",      "segoeui.ttf"),
    _reg("UI-B",    "segoeuib.ttf"),
    _reg("UI-I",    "segoeuii.ttf"),
    _reg("UI-BI",   "segoeuiz.ttf"),
    _reg("UI-L",    "segoeuil.ttf"),
    _reg("UI-SL",   "segoeuisl.ttf"),
])

if HAVE_SEGOE:
    F, FB, FI, FBI, FL, FSL = "UI", "UI-B", "UI-I", "UI-BI", "UI-L", "UI-SL"
else:                                                   # portable fallback
    F, FB, FI, FBI = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"
    FL = FSL = "Helvetica"

_reg("MONO", "consola.ttf")
_reg("MONO-B", "consolab.ttf")
FM = "MONO" if os.path.exists(r"C:\Windows\Fonts\consola.ttf") else "Courier"
FMB = "MONO-B" if os.path.exists(r"C:\Windows\Fonts\consolab.ttf") else "Courier-Bold"


# ──────────────────────────── rich text engine ──────────────────────────────
#
# Body copy is written with **bold** and `mono` inline. Text is measured, then
# wrapped, then drawn — so a box can report its own height before it is placed.

#: **bold**, `mono`, and *italic* / __italic__. The doubled forms are tried
#: first, so `**x**` is never mistaken for an italic run.
_TOKEN = re.compile(r"\*\*(.+?)\*\*|`(.+?)`|__(.+?)__|\*([^*\n]+?)\*")


def parse_runs(text, base=None, bold=None, mono=None, ital=None):
    base = base or F
    bold = bold or FB
    mono = mono or FM
    if ital is None:
        ital = FBI if base == FB else FI
    runs, pos = [], 0
    for m in _TOKEN.finditer(text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], base))
        if m.group(1) is not None:
            runs.append((m.group(1), bold))
        elif m.group(2) is not None:
            runs.append((m.group(2), mono))
        elif m.group(3) is not None:
            runs.append((m.group(3), ital))
        else:
            runs.append((m.group(4), ital))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], base))
    return runs


def _sz(font, size):
    """Mono reads large next to a humanist sans; shrink it a touch."""
    return size * 0.92 if font in (FM, FMB) else size


def wrap_runs(runs, size, max_w):
    """-> [(cells, width)] where cells are (text, font, width) at `size`."""
    lines, cur, cur_w = [], [], 0.0
    for txt, font in runs:
        for part in re.split(r"(\s+)", txt):
            if not part:
                continue
            w = pdfmetrics.stringWidth(part, font, _sz(font, size))
            if part.isspace():
                if cur:
                    cur.append((part, font, w))
                    cur_w += w
                continue
            if cur_w + w > max_w and cur:
                while cur and cur[-1][0].isspace():
                    cur_w -= cur[-1][2]
                    cur.pop()
                lines.append((cur, cur_w))
                cur, cur_w = [], 0.0
            cur.append((part, font, w))
            cur_w += w
    if cur:
        while cur and cur[-1][0].isspace():
            cur_w -= cur[-1][2]
            cur.pop()
        lines.append((cur, cur_w))
    return lines or [([], 0.0)]


def draw_lines(c, lines, x, y_top, size, leading, colour=INK, align="left", max_w=None):
    """Draw wrapped lines downward from y_top (points). Returns the new y."""
    y = y_top
    for cells, w in lines:
        y -= leading
        if align == "center" and max_w:
            cx = x + (max_w - w) / 2.0
        elif align == "right" and max_w:
            cx = x + (max_w - w)
        else:
            cx = x
        c.setFillColor(colour)
        for txt, font, cw in cells:
            c.setFont(font, _sz(font, size))
            c.drawString(cx, y, txt)
            cx += cw
    return y


def text_height(runs, size, leading, max_w):
    return len(wrap_runs(runs, size, max_w)) * leading


# ───────────────────────────── block model ──────────────────────────────────
#
# A section box is a list of blocks. Every block can measure itself against a
# width and then draw itself; that is the whole layout engine.

BODY_SIZE = 8.6
BODY_LEAD = 11.6


class Para:
    def __init__(self, text, size=BODY_SIZE, lead=BODY_LEAD, colour=INK,
                 space=3.2, font=None, align="left", indent=0.0):
        self.text, self.size, self.lead = text, size, lead
        self.colour, self.space, self.align, self.indent = colour, space, align, indent
        self.font = font or F

    def height(self, w_mm):
        runs = parse_runs(self.text, base=self.font)
        h = text_height(runs, self.size, self.lead, (w_mm - self.indent) * mm)
        return h / mm + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        runs = parse_runs(self.text, base=self.font)
        lines = wrap_runs(runs, self.size, (w_mm - self.indent) * mm)
        draw_lines(c, lines, (x_mm + self.indent) * mm, yt(top_mm), self.size,
                   self.lead, self.colour, self.align, (w_mm - self.indent) * mm)


class Bullet:
    """A bullet whose first sentence is usually bold — the poster's workhorse."""

    GAP = 4.6      # mm from bullet glyph to text

    def __init__(self, text, size=BODY_SIZE, lead=BODY_LEAD, space=2.6,
                 marker="\u2022", colour=INK, mcolour=None):
        self.text, self.size, self.lead, self.space = text, size, lead, space
        self.marker, self.colour = marker, colour
        self.mcolour = mcolour or BAR

    def height(self, w_mm):
        runs = parse_runs(self.text)
        return text_height(runs, self.size, self.lead,
                           (w_mm - self.GAP) * mm) / mm + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        c.setFillColor(self.mcolour)
        c.setFont(FB, self.size)
        c.drawString(x_mm * mm + 0.6 * mm, yt(top_mm) - self.lead, self.marker)
        runs = parse_runs(self.text)
        lines = wrap_runs(runs, self.size, (w_mm - self.GAP) * mm)
        draw_lines(c, lines, (x_mm + self.GAP) * mm, yt(top_mm),
                   self.size, self.lead, self.colour)


class Gap:
    def __init__(self, h):
        self.h = h

    def height(self, w_mm):
        return self.h

    def draw(self, c, x_mm, top_mm, w_mm):
        pass


class Rule:
    def __init__(self, space=3.0, colour=RULE, width=0.5):
        self.space, self.colour, self.width = space, colour, width

    def height(self, w_mm):
        return self.space * 2

    def draw(self, c, x_mm, top_mm, w_mm):
        c.setStrokeColor(self.colour)
        c.setLineWidth(self.width)
        y = yt(top_mm + self.space)
        c.line(x_mm * mm, y, (x_mm + w_mm) * mm, y)


class KeyLine:
    """A tinted strip for the one sentence in a section that must be read."""

    def __init__(self, text, fill=TINT, accent=BAR, size=8.8, lead=11.8, space=4.0):
        self.text, self.fill, self.accent = text, fill, accent
        self.size, self.lead, self.space = size, lead, space
        self.pad = 3.0

    def _lines(self, w_mm):
        runs = parse_runs(self.text)
        return wrap_runs(runs, self.size, (w_mm - 2 * self.pad - 2.0) * mm)

    def height(self, w_mm):
        return len(self._lines(w_mm)) * self.lead / mm + 2 * self.pad + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        h = self.height(w_mm) - self.space
        c.setFillColor(self.fill)
        c.rect(x_mm * mm, yt(top_mm + h), w_mm * mm, h * mm, stroke=0, fill=1)
        c.setFillColor(self.accent)
        c.rect(x_mm * mm, yt(top_mm + h), 1.6 * mm, h * mm, stroke=0, fill=1)
        draw_lines(c, self._lines(w_mm), (x_mm + self.pad + 2.0) * mm,
                   yt(top_mm + self.pad - 0.6), self.size, self.lead, INK)


class Table:
    """A compact data table: header row, then rows. cols = (label, align, frac)."""

    def __init__(self, cols, rows, size=8.2, lead=11.0, space=3.6,
                 head_fill=BAR, zebra=True, foot=None, bold_rows=()):
        self.cols, self.rows = cols, rows
        self.size, self.lead, self.space = size, lead, space
        self.head_fill, self.zebra, self.foot = head_fill, zebra, foot
        self.bold_rows = set(bold_rows)
        self.pad = 1.8

    def _row_h(self):
        return self.lead + 2.0

    def height(self, w_mm):
        h = self._row_h() / mm * (len(self.rows) + 1)
        if self.foot:
            runs = parse_runs(self.foot)
            h += text_height(runs, 7.4, 9.4, w_mm * mm) / mm + 1.4
        return h + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        rh = self._row_h()
        widths = [f * w_mm for (_, _, f) in self.cols]
        xs, acc = [], x_mm
        for wd in widths:
            xs.append(acc)
            acc += wd

        # header
        c.setFillColor(self.head_fill)
        c.rect(x_mm * mm, yt(top_mm) - rh, w_mm * mm, rh, stroke=0, fill=1)
        c.setFont(FB, self.size)
        for (label, al, _), cx, wd in zip(self.cols, xs, widths):
            self._cell(c, label, cx, wd, yt(top_mm) - rh + self.pad * mm + 1.2, al,
                       FB, PAPER)

        y = yt(top_mm) - rh
        for i, row in enumerate(self.rows):
            if self.zebra and i % 2 == 1:
                c.setFillColor(TINT)
                c.rect(x_mm * mm, y - rh, w_mm * mm, rh, stroke=0, fill=1)
            fnt = FB if i in self.bold_rows else F
            for cell, (_, al, _), cx, wd in zip(row, self.cols, xs, widths):
                runs = parse_runs(str(cell), base=fnt)
                self._cell_runs(c, runs, cx, wd, y - rh + self.pad * mm + 1.2, al, INK)
            y -= rh
            c.setStrokeColor(RULE)
            c.setLineWidth(0.35)
            c.line(x_mm * mm, y, (x_mm + w_mm) * mm, y)

        if self.foot:
            runs = parse_runs(self.foot, base=FI)
            lines = wrap_runs(runs, 7.4, w_mm * mm)
            draw_lines(c, lines, x_mm * mm, y - 1.4 * mm, 7.4, 9.4, INK_SOFT)

    def _cell(self, c, txt, cx, wd, y, al, font, colour):
        self._cell_runs(c, parse_runs(txt, base=font), cx, wd, y, al, colour)

    def _cell_runs(self, c, runs, cx, wd, y, al, colour):
        total = sum(pdfmetrics.stringWidth(t, f, _sz(f, self.size)) for t, f in runs)
        if al == "r":
            x = (cx + wd) * mm - self.pad * mm - total
        elif al == "c":
            x = cx * mm + (wd * mm - total) / 2
        else:
            x = cx * mm + self.pad * mm
        c.setFillColor(colour)
        for t, f in runs:
            c.setFont(f, _sz(f, self.size))
            c.drawString(x, y, t)
            x += pdfmetrics.stringWidth(t, f, _sz(f, self.size))


class Figure:
    """A photograph with a caption, fitted to the column."""

    def __init__(self, path, caption, h_mm=None, crop=None, space=4.0,
                 frame=True, w_frac=1.0):
        self.path, self.caption, self.space = path, caption, space
        self.h_mm, self.crop, self.frame, self.w_frac = h_mm, crop, frame, w_frac
        self._img = None

    def _image(self):
        if self._img is None:
            from PIL import Image
            im = Image.open(self.path).convert("RGB")
            if self.crop:
                im = im.crop(self.crop)
            self._img = im
        return self._img

    def _dims(self, w_mm):
        im = self._image()
        w = w_mm * self.w_frac
        h = self.h_mm if self.h_mm else w * im.size[1] / im.size[0]
        if self.h_mm:                       # height-led: recompute width
            w = min(w, self.h_mm * im.size[0] / im.size[1])
        return w, h

    def height(self, w_mm):
        w, h = self._dims(w_mm)
        cap = text_height(parse_runs(self.caption, base=FI), 7.6, 9.6, w_mm * mm) / mm
        return h + cap + 1.8 + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        w, h = self._dims(w_mm)
        x = x_mm + (w_mm - w) / 2.0
        c.drawImage(ImageReader(self._image()), x * mm, yt(top_mm + h),
                    w * mm, h * mm, mask="auto")
        if self.frame:
            c.setStrokeColor(RULE)
            c.setLineWidth(0.6)
            c.rect(x * mm, yt(top_mm + h), w * mm, h * mm, stroke=1, fill=0)
        lines = wrap_runs(parse_runs(self.caption, base=FI), 7.6, w_mm * mm)
        draw_lines(c, lines, x_mm * mm, yt(top_mm + h + 1.8), 7.6, 9.6, INK_SOFT)


class FigureRow:
    """Several photographs side by side, one shared caption."""

    def __init__(self, items, caption, h_mm, space=4.0, gap=2.0):
        self.items, self.caption, self.h_mm = items, caption, h_mm
        self.space, self.gap = space, gap
        self._imgs = None

    def _images(self):
        if self._imgs is None:
            from PIL import Image
            self._imgs = [(Image.open(p).convert("RGB"), lab) for p, lab in self.items]
        return self._imgs

    def height(self, w_mm):
        cap = text_height(parse_runs(self.caption, base=FI), 7.6, 9.6, w_mm * mm) / mm
        return self.h_mm + 4.6 + cap + 1.8 + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        imgs = self._images()
        n = len(imgs)
        cw = (w_mm - self.gap * (n - 1)) / n
        for i, (im, label) in enumerate(imgs):
            x = x_mm + i * (cw + self.gap)
            # centre-crop each photograph to the cell's aspect so the row lines up
            target = cw / self.h_mm
            iw, ih = im.size
            if iw / ih > target:
                new_w = int(ih * target)
                box = ((iw - new_w) // 2, 0, (iw - new_w) // 2 + new_w, ih)
            else:
                new_h = int(iw / target)
                box = (0, max(0, (ih - new_h) // 2), iw, max(0, (ih - new_h) // 2) + new_h)
            c.drawImage(ImageReader(im.crop(box)), x * mm, yt(top_mm + self.h_mm),
                        cw * mm, self.h_mm * mm, mask="auto")
            c.setStrokeColor(RULE)
            c.setLineWidth(0.6)
            c.rect(x * mm, yt(top_mm + self.h_mm), cw * mm, self.h_mm * mm, stroke=1, fill=0)
            c.setFillColor(BAR)
            c.rect(x * mm, yt(top_mm + self.h_mm + 4.2), cw * mm, 4.2 * mm, stroke=0, fill=1)
            c.setFillColor(PAPER)
            c.setFont(FB, 7.4)
            tw = pdfmetrics.stringWidth(label, FB, 7.4)
            c.drawString(x * mm + (cw * mm - tw) / 2, yt(top_mm + self.h_mm + 4.2) + 1.3 * mm, label)
        lines = wrap_runs(parse_runs(self.caption, base=FI), 7.6, w_mm * mm)
        draw_lines(c, lines, x_mm * mm, yt(top_mm + self.h_mm + 4.6 + 1.8), 7.6, 9.6, INK_SOFT)


class Chart:
    """Wraps a drawing function so charts stack like any other block."""

    def __init__(self, fn, h_mm, caption=None, space=4.0):
        self.fn, self.h_mm, self.caption, self.space = fn, h_mm, caption, space

    def height(self, w_mm):
        cap = 0.0
        if self.caption:
            cap = text_height(parse_runs(self.caption, base=FI), 7.6, 9.6,
                              w_mm * mm) / mm + 1.8
        return self.h_mm + cap + self.space

    def draw(self, c, x_mm, top_mm, w_mm):
        self.fn(c, x_mm, top_mm, w_mm, self.h_mm)
        if self.caption:
            lines = wrap_runs(parse_runs(self.caption, base=FI), 7.6, w_mm * mm)
            draw_lines(c, lines, x_mm * mm, yt(top_mm + self.h_mm + 1.8), 7.6, 9.6, INK_SOFT)


class Section:
    """One titled box: a coloured header bar over a bordered white body."""

    BAR_H = 8.6
    PAD_X = 3.6
    PAD_T = 3.0
    PAD_B = 2.6

    def __init__(self, title, blocks, gap_after=5.0, bar=BAR, tag=None):
        self.title, self.blocks, self.gap_after = title, blocks, gap_after
        self.bar, self.tag = bar, tag

    def inner_w(self, w_mm):
        return w_mm - 2 * self.PAD_X

    def height(self, w_mm):
        iw = self.inner_w(w_mm)
        h = sum(b.height(iw) for b in self.blocks)
        return self.BAR_H + self.PAD_T + h + self.PAD_B

    def draw(self, c, x_mm, top_mm, w_mm):
        h = self.height(w_mm)
        # body
        c.setFillColor(PAPER)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.7)
        c.rect(x_mm * mm, yt(top_mm + h), w_mm * mm, (h - self.BAR_H) * mm,
               stroke=1, fill=1)
        # header bar
        c.setFillColor(self.bar)
        c.rect(x_mm * mm, yt(top_mm + self.BAR_H), w_mm * mm, self.BAR_H * mm,
               stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 12.4)
        c.drawString((x_mm + self.PAD_X) * mm, yt(top_mm + self.BAR_H) + 2.3 * mm,
                     self.title)
        if self.tag:
            c.setFont(F, 8.0)
            tw = pdfmetrics.stringWidth(self.tag, F, 8.0)
            c.setFillColor(Color(1, 1, 1, 0.72))
            c.drawString((x_mm + w_mm - self.PAD_X) * mm - tw,
                         yt(top_mm + self.BAR_H) + 2.6 * mm, self.tag)

        y = top_mm + self.BAR_H + self.PAD_T
        iw = self.inner_w(w_mm)
        for b in self.blocks:
            b.draw(c, x_mm + self.PAD_X, y, iw)
            y += b.height(iw)


# ─────────────────────────── decorative motifs ──────────────────────────────

def network_pattern(c, x0, y0, x1, y1, seed=7, n=46, alpha=0.5):
    """The faint connected-node field behind the header and footer."""
    rnd = random.Random(seed)
    pts = [(rnd.uniform(x0, x1), rnd.uniform(y0, y1)) for _ in range(n)]
    cols = [HexColor("#E9A7C6"), HexColor("#A9C6E8"), HexColor("#C6B7E2"),
            HexColor("#F3BBA6"), HexColor("#BFD8E8")]
    c.saveState()
    c.setLineWidth(0.4)
    link = Color(0.72, 0.76, 0.82, alpha * 0.55)
    c.setStrokeColor(link)
    reach = (x1 - x0) * 0.16
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            d = math.hypot(p[0] - q[0], p[1] - q[1])
            if d < reach:
                c.line(p[0], p[1], q[0], q[1])
    for i, (px_, py_) in enumerate(pts):
        base = cols[i % len(cols)]
        c.setFillColor(Color(base.red, base.green, base.blue,
                             alpha * rnd.uniform(0.35, 0.95)))
        r = rnd.uniform(0.7, 2.4) * mm
        c.circle(px_, py_, r, stroke=0, fill=1)
    c.restoreState()


def chevron(c, cx, top, w, h):
    """The two-tone downward V from the example's header."""
    c.setFillColor(PERI)
    p = c.beginPath()
    p.moveTo(cx - w / 2, top)
    p.lineTo(cx, top - h)
    p.lineTo(cx + w / 2, top)
    p.close()
    c.drawPath(p, stroke=0, fill=1)
    c.setFillColor(NAVY)
    p = c.beginPath()
    p.moveTo(cx, top)
    p.lineTo(cx + w / 2, top)
    p.lineTo(cx, top - h)
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def footer_motif(c, cx, base, w, h):
    """The coral + navy triangle pair that closes the example poster."""
    c.setFillColor(CORAL_PALE)
    p = c.beginPath()
    p.moveTo(cx - w / 2, base)
    p.lineTo(cx + w / 2, base)
    p.lineTo(cx + w * 0.10, base + h)
    p.close()
    c.drawPath(p, stroke=0, fill=1)
    c.setFillColor(CORAL)
    p = c.beginPath()
    p.moveTo(cx - w * 0.12, base)
    p.lineTo(cx + w / 2, base)
    p.lineTo(cx + w * 0.10, base + h)
    p.close()
    c.drawPath(p, stroke=0, fill=1)
    c.setFillColor(NAVY)
    p = c.beginPath()
    p.moveTo(cx - w / 2, base)
    p.lineTo(cx - w * 0.10, base)
    p.lineTo(cx - w * 0.30, base + h * 0.62)
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def draw_apu_logo(c, right_x, centre_y, target_h):
    """The real logo if it has been supplied, otherwise a clean wordmark."""
    if os.path.exists(APU_LOGO_PATH):
        from PIL import Image
        im = Image.open(APU_LOGO_PATH)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        w = target_h * im.size[0] / im.size[1]
        c.drawImage(ImageReader(im), right_x - w, centre_y - target_h / 2,
                    w, target_h, mask="auto")
        return

    # Fallback wordmark — swap by dropping apu_logo.png beside this script.
    size = target_h * 0.62
    c.setFont(FB, size)
    word = "APU"
    ww = pdfmetrics.stringWidth(word, FB, size)
    sub1 = "ASIA PACIFIC UNIVERSITY"
    sub2 = "OF TECHNOLOGY & INNOVATION"
    s_sz = target_h * 0.135
    w1 = pdfmetrics.stringWidth(sub1, FB, s_sz)
    w2 = pdfmetrics.stringWidth(sub2, F, s_sz)
    block_w = max(ww, w1, w2)
    left = right_x - block_w

    c.setFillColor(APU_RED)
    c.drawString(left + (block_w - ww) / 2, centre_y - size * 0.12, word)
    c.setStrokeColor(APU_RED)
    c.setLineWidth(1.1)
    c.line(left, centre_y - size * 0.30, left + block_w, centre_y - size * 0.30)
    c.setFillColor(HexColor("#3A3F45"))
    c.setFont(FB, s_sz)
    c.drawString(left + (block_w - w1) / 2, centre_y - size * 0.30 - s_sz * 1.45, sub1)
    c.setFont(F, s_sz)
    c.drawString(left + (block_w - w2) / 2, centre_y - size * 0.30 - s_sz * 2.65, sub2)


# ──────────────────────────────── charts ────────────────────────────────────
#
# Drawn straight onto the canvas so they stay vector and share the poster's
# palette. Each takes (canvas, x_mm, top_mm, w_mm, h_mm).

def _axis_frame(c, x, top, w, h, pad_l=13.0, pad_b=9.0, pad_t=3.0, pad_r=2.0):
    """Returns the plotting rectangle in mm as (px, py_top, pw, ph)."""
    return x + pad_l, top + pad_t, w - pad_l - pad_r, h - pad_t - pad_b


def _label(c, txt, x_mm, y_mm, size=7.0, font=None, colour=INK_SOFT, align="l"):
    font = font or F
    c.setFont(font, size)
    c.setFillColor(colour)
    tw = pdfmetrics.stringWidth(txt, font, size)
    x = x_mm * mm
    if align == "r":
        x -= tw
    elif align == "c":
        x -= tw / 2
    c.drawString(x, yt(y_mm), txt)


# Source: traffic_all_sessions.csv, 12,919 samples across 66 recorded sessions,
# rows with at least one vehicle present (7,406 of them), binned by congestion
# index. Speed is the mean of "Avg Speed (km/h)"; load is mean "Total Vehicles".
CI_BINS = [
    ("0.0-0.2", 42.2, 2.4, 967),
    ("0.2-0.4", 32.6, 5.9, 3220),
    ("0.4-0.6", 24.3, 8.5, 1951),
    ("0.6-0.8", 17.4, 13.7, 1172),
    ("0.8-1.0", 10.0, 16.1, 96),
]


def chart_speed_vs_congestion(c, x, top, w, h):
    px_, py, pw, ph = _axis_frame(c, x, top, w, h, pad_l=13.0, pad_b=11.0, pad_r=13.0)
    smax = 46.0
    vmax = 18.0

    # grid + left axis
    c.setStrokeColor(RULE)
    c.setLineWidth(0.35)
    for i in range(0, 5):
        v = smax * i / 4.0
        gy = py + ph - ph * (v / smax)
        c.line(px_ * mm, yt(gy), (px_ + pw) * mm, yt(gy))
        _label(c, f"{v:.0f}", px_ - 1.4, gy + 1.0, 6.8, align="r")
    _label(c, "km/h", px_ - 1.4, py - 1.2, 6.8, FB, BAR, align="r")
    _label(c, "veh", px_ + pw + 1.4, py - 1.2, 6.8, FB, CORAL)

    n = len(CI_BINS)
    slot = pw / n
    bw = slot * 0.46
    pts = []
    for i, (lab, spd, veh, cnt) in enumerate(CI_BINS):
        cx = px_ + slot * (i + 0.5)
        bh = ph * (spd / smax)
        c.setFillColor(BAR)
        c.rect((cx - bw / 2) * mm, yt(py + ph), bw * mm, bh * mm, stroke=0, fill=1)
        _label(c, f"{spd:.1f}", cx, py + ph - bh - 1.4, 7.0, FB, BAR, align="c")
        _label(c, lab, cx, py + ph + 4.4, 6.8, F, INK_SOFT, align="c")
        pts.append((cx, py + ph - ph * (veh / vmax)))

    # right-hand series: mean vehicles present
    c.setStrokeColor(CORAL)
    c.setLineWidth(1.5)
    c.setLineJoin(1)
    p = c.beginPath()
    p.moveTo(pts[0][0] * mm, yt(pts[0][1]))
    for cx, cy in pts[1:]:
        p.lineTo(cx * mm, yt(cy))
    c.drawPath(p, stroke=1, fill=0)
    for (cx, cy), (_, _, veh, _) in zip(pts, CI_BINS):
        c.setFillColor(PAPER)
        c.circle(cx * mm, yt(cy), 1.5 * mm, stroke=0, fill=1)
        c.setFillColor(CORAL)
        c.circle(cx * mm, yt(cy), 1.05 * mm, stroke=0, fill=1)
        _label(c, f"{veh:.1f}", cx + 3.0, cy + 1.0, 6.6, FB, CORAL)

    c.setStrokeColor(INK_SOFT)
    c.setLineWidth(0.6)
    c.line(px_ * mm, yt(py + ph), (px_ + pw) * mm, yt(py + ph))
    _label(c, "congestion index", px_ + pw / 2, py + ph + 8.4, 7.0, FI, INK_SOFT, align="c")


# Source: METRICS.md §3 — 60 consecutive night frames, FP16, RTX 4050.
MODELS = [
    ("11s@640",  70.6, 18.4, False),
    ("11s@960",  62.5, 20.8, False),
    ("11s@1280", 54.3, 22.8, False),
    ("11m@640",  58.0, 17.0, False),
    ("11m@960",  52.8, 19.3, False),
    ("11m@1280", 36.6, 23.0, True),
    ("11l@640",  44.4, 17.9, False),
    ("11l@960",  41.6, 19.8, False),
    ("11l@1280", 30.5, 22.5, False),
    ("12s@1280", 41.7, 23.6, False),
    ("12m@1280", 25.8, 22.1, False),
]


def chart_model_tradeoff(c, x, top, w, h):
    px_, py, pw, ph = _axis_frame(c, x, top, w, h, pad_l=10.0, pad_b=11.0,
                                  pad_t=6.4, pad_r=4.0)
    xlo, xhi = 22.5, 74.0
    ylo, yhi = 15.6, 25.0

    def X(v):
        return px_ + pw * (v - xlo) / (xhi - xlo)

    def Y(v):
        return py + ph - ph * (v - ylo) / (yhi - ylo)

    c.setStrokeColor(RULE)
    c.setLineWidth(0.35)
    for v in range(16, 25, 2):
        c.line(px_ * mm, yt(Y(v)), (px_ + pw) * mm, yt(Y(v)))
        _label(c, str(v), px_ - 1.4, Y(v) + 1.0, 6.8, align="r")
    for v in (30, 40, 50, 60, 70):
        _label(c, str(v), X(v), py + ph + 4.4, 6.8, F, INK_SOFT, align="c")

    _label(c, "vehicles found / frame", px_ - 8.6, py - 2.4, 7.0, FB, BAR)
    _label(c, "frames per second", px_ + pw / 2, py + ph + 8.4, 7.0, FI, INK_SOFT, align="c")

    # the three YOLO11 families, joined, are the point of the chart
    FAM_COL = {"11s": BLUE, "11m": BAR, "11l": NAVY}
    fam = {k: [] for k in FAM_COL}
    for name, fps, veh, _ in MODELS:
        key = name.split("@")[0]
        if key in fam:
            fam[key].append((X(fps), Y(veh)))
    c.setLineWidth(0.9)
    for key, pts in fam.items():
        pts = sorted(pts)
        base = FAM_COL[key]
        c.setStrokeColor(Color(base.red, base.green, base.blue, 0.34))
        p = c.beginPath()
        p.moveTo(pts[0][0] * mm, yt(pts[0][1]))
        for qx, qy in pts[1:]:
            p.lineTo(qx * mm, yt(qy))
        c.drawPath(p, stroke=1, fill=0)

    # nudge the two labels that would otherwise sit on a neighbour
    ABOVE = {"11l@960", "11s@960"}
    for name, fps, veh, is_def in MODELS:
        cx, cy = X(fps), Y(veh)
        col = CORAL if is_def else FAM_COL.get(name.split("@")[0], TEAL)
        r = 1.9 if is_def else 1.35
        c.setFillColor(col)
        c.circle(cx * mm, yt(cy), r * mm, stroke=0, fill=1)
        if is_def:
            c.setStrokeColor(CORAL)
            c.setLineWidth(0.8)
            c.circle(cx * mm, yt(cy), 3.2 * mm, stroke=1, fill=0)
            _label(c, "default", cx + 4.6, cy - 1.4, 7.2, FB, CORAL)
            _label(c, name, cx + 4.6, cy + 2.6, 6.6, F, INK_SOFT)
        else:
            dy = 3.4 if name in ABOVE else -2.7
            _label(c, name, cx, cy + dy, 6.4, F, INK_SOFT, align="c")

    lx = px_ + 28.0
    for key, col in list(FAM_COL.items()) + [("12s / 12m", TEAL)]:
        c.setFillColor(col)
        c.circle(lx * mm, yt(py - 2.4) + 0.9 * mm, 1.2 * mm, stroke=0, fill=1)
        _label(c, key, lx + 2.2, py - 2.4, 6.6, F, INK_SOFT)
        lx += pdfmetrics.stringWidth(key, F, 6.6) / mm + 8.4

    c.setStrokeColor(INK_SOFT)
    c.setLineWidth(0.6)
    c.line(px_ * mm, yt(py + ph), (px_ + pw) * mm, yt(py + ph))


# Source: METRICS.md §4 — python benchmark.py --stages, yolo11m @ 1280.
STAGES = [
    ("detection + tracking", 29.6, 28.8, BAR),
    ("beacon detection", 9.8, 5.2, CORAL),
    ("livery (CLIP)", 5.4, 1.1, TEAL),
]


def chart_stage_cost(c, x, top, w, h):
    # pad_r reserves a fixed lane for the "44.8 ms -> 22.3 fps" totals, so the
    # longest bar can no longer push its own label off the edge.
    px_, py, pw, ph = _axis_frame(c, x, top, w, h, pad_l=15.0, pad_b=13.5, pad_r=27.0)
    scale = 48.0
    rows = [("night", 1, 44.8, 22.3), ("day", 2, 35.1, 28.5)]
    bar_h = ph * 0.30

    c.setStrokeColor(RULE)
    c.setLineWidth(0.35)
    for v in (0, 10, 20, 30, 40):
        gx = px_ + pw * (v / scale)
        c.line(gx * mm, yt(py), gx * mm, yt(py + ph))
        _label(c, str(v), gx, py + ph + 4.2, 6.8, F, INK_SOFT, align="c")
    _label(c, "milliseconds per frame", px_ + pw / 2, py + ph + 8.6, 7.0, FI, INK_SOFT, align="c")

    for label, idx, total, fps in rows:
        cy = py + ph * (0.30 if idx == 1 else 0.74)
        _label(c, label, px_ - 1.6, cy + bar_h * 0.18, 8.0, FB, INK, align="r")
        cursor = px_
        for name, night, day, col in STAGES:
            v = night if idx == 1 else day
            seg = pw * (v / scale)
            c.setFillColor(col)
            c.rect(cursor * mm, yt(cy + bar_h), seg * mm, bar_h * mm, stroke=0, fill=1)
            if seg > 9:
                c.setFillColor(PAPER)
                c.setFont(FB, 6.8)
                tw = pdfmetrics.stringWidth(f"{v:.1f}", FB, 6.8)
                c.drawString((cursor + seg / 2) * mm - tw / 2,
                             yt(cy + bar_h / 2) - 2.2, f"{v:.1f}")
            cursor += seg
        _label(c, f"{total:.1f} ms", px_ + pw + 2.4, cy + bar_h * 0.06, 7.6, FB, INK)
        _label(c, f"\u2192 {fps:.1f} fps", px_ + pw + 2.4, cy + bar_h * 0.06 + 3.4,
               7.4, FB, CORAL)

    # legend
    lx = px_
    for name, _, _, col in STAGES:
        c.setFillColor(col)
        c.rect(lx * mm, yt(py + ph + 11.6), 3.0 * mm, 3.0 * mm, stroke=0, fill=1)
        _label(c, name, lx + 4.0, py + ph + 11.4, 6.8, F, INK_SOFT)
        lx += pdfmetrics.stringWidth(name, F, 6.8) / mm + 9.0

    c.setStrokeColor(INK_SOFT)
    c.setLineWidth(0.6)
    c.line(px_ * mm, yt(py + ph), px_ * mm, yt(py))


# Source: traffic_all_vehicles.csv — 21,526 tracked vehicles over 69 sessions.
# The bar fractions are set by eye: 21,526 -> 244 -> 26 is far too steep a drop
# to draw proportionally without the last two bars disappearing.
FUNNEL = [
    ("tracked vehicles", 21526, 0.76,
     "every vehicle the tracker held for 3+ frames", BAR),
    ("flagged by livery", 244, 0.38,
     "CLIP read it as ambulance, police or fire \u2014 it looks the part", AMBER),
    ("beacon-confirmed", 26, 0.15,
     "a beacon that actually switched \u2014 priority granted", RED),
]


def chart_funnel(c, x, top, w, h):
    """Bar carries the count; the name and share sit outside it, so a bar can
    be short without its label being clipped."""
    rows = len(FUNNEL)
    gap = 1.4
    note_h = 4.2
    bh = (h - (gap + note_h) * rows) / rows
    top_v = FUNNEL[0][1]
    y = top
    for name, val, frac, note, col in FUNNEL:
        bw = w * frac
        c.setFillColor(col)
        c.rect(x * mm, yt(y + bh), bw * mm, bh * mm, stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 11.6)
        c.drawString(x * mm + 2.8 * mm, yt(y + bh) + bh * mm * 0.30, f"{val:,}")

        lx = x + bw + 2.4
        c.setFillColor(col)
        c.setFont(FB, 8.4)
        c.drawString(lx * mm, yt(y + bh) + bh * mm * 0.32, name)
        if val != top_v:
            off = pdfmetrics.stringWidth(name, FB, 8.4) / mm + 2.6
            _label(c, f"\u00b7  {val / top_v * 100:.2f}% of tracked", lx + off,
                   y + bh * 0.68 + 0.4, 7.4, F, INK_SOFT)
        _label(c, note, x + 0.4, y + bh + note_h - 1.0, 6.9, FI, INK_SOFT)
        y += bh + note_h + gap


# Source: traffic_all_vehicles.csv class counts.
CLASSES = [("Car", 18519, BAR), ("Motorcycle", 1523, TEAL),
           ("Truck", 1233, CORAL), ("Bus", 251, NAVY)]


def chart_class_mix(c, x, top, w, h):
    total = sum(v for _, v, _ in CLASSES)
    bar_h = 7.0
    cursor = x
    for name, val, col in CLASSES:
        seg = w * val / total
        c.setFillColor(col)
        c.rect(cursor * mm, yt(top + bar_h), seg * mm, bar_h * mm, stroke=0, fill=1)
        cursor += seg
    lx = x
    for name, val, col in CLASSES:
        c.setFillColor(col)
        c.rect(lx * mm, yt(top + bar_h + 7.0), 3.0 * mm, 3.0 * mm, stroke=0, fill=1)
        txt = f"{name} {val:,}"
        _label(c, txt, lx + 4.0, top + bar_h + 6.8, 7.0, F, INK)
        lx += pdfmetrics.stringWidth(txt, F, 7.0) / mm + 8.0


def chart_signal_machine(c, x, top, w, h):
    """The signal state machine: R -> Y -> G -> Y -> R with its two rules."""
    r = 9.0
    cy = top + r + 4.0
    xs = [x + w * f for f in (0.13, 0.385, 0.64, 0.895)]
    states = [("RED", RED), ("YEL", AMBER), ("GREEN", GREEN), ("YEL", AMBER)]

    for i, ((label, col), sx) in enumerate(zip(states, xs)):
        if i:
            c.setStrokeColor(INK_SOFT)
            c.setLineWidth(0.9)
            c.line((xs[i - 1] + r * 0.62) * mm, yt(cy), (sx - r * 0.62) * mm, yt(cy))
            p = c.beginPath()
            p.moveTo((sx - r * 0.62) * mm, yt(cy))
            p.lineTo((sx - r * 0.62 - 1.6) * mm, yt(cy - 1.1))
            p.lineTo((sx - r * 0.62 - 1.6) * mm, yt(cy + 1.1))
            p.close()
            c.setFillColor(INK_SOFT)
            c.drawPath(p, stroke=0, fill=1)
        c.setFillColor(col)
        c.circle(sx * mm, yt(cy), r / 2 * mm, stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 6.6)
        tw = pdfmetrics.stringWidth(label, FB, 6.6)
        c.drawString(sx * mm - tw / 2, yt(cy) - 2.2, label)

    _label(c, "5 s, committed", (xs[0] + xs[1]) / 2, cy - r * 0.62, 6.6, FB, AMBER, align="c")
    _label(c, "5 s, committed", (xs[2] + xs[3]) / 2, cy - r * 0.62, 6.6, FB, AMBER, align="c")
    _label(c, "min dwell 3 s", xs[2], cy + r * 0.62 + 4.0, 6.6, F, INK_SOFT, align="c")
    _label(c, "min dwell 3 s", xs[0], cy + r * 0.62 + 4.0, 6.6, F, INK_SOFT, align="c")


def chart_pipeline(c, x, top, w, h):
    """The end-to-end block diagram: source -> CV node -> hardware / dashboard."""
    box_h = 8.4
    lane_gap = 4.4

    def box(bx, by, bw, label, fill, fg=PAPER, size=7.4, sub=None):
        c.setFillColor(fill)
        c.roundRect(bx * mm, yt(by + box_h), bw * mm, box_h * mm, 1.4 * mm,
                    stroke=0, fill=1)
        c.setFillColor(fg)
        c.setFont(FB, size)
        tw = pdfmetrics.stringWidth(label, FB, size)
        off = 1.2 if sub else 0.0
        c.drawString(bx * mm + (bw * mm - tw) / 2, yt(by + box_h / 2) - 2.0 + off * mm, label)
        if sub:
            c.setFont(F, size - 1.4)
            tw = pdfmetrics.stringWidth(sub, F, size - 1.4)
            c.setFillColor(Color(1, 1, 1, 0.92) if fg == PAPER else INK_SOFT)
            c.drawString(bx * mm + (bw * mm - tw) / 2, yt(by + box_h / 2) - 4.6, sub)

    def arrow(x0, y0, x1, y1):
        c.setStrokeColor(INK_SOFT)
        c.setLineWidth(0.8)
        c.line(x0 * mm, yt(y0), x1 * mm, yt(y1))
        ang = math.atan2(-(yt(y1) - yt(y0)), (x1 - x0) * mm)
        L = 1.7 * mm
        p = c.beginPath()
        p.moveTo(x1 * mm, yt(y1))
        p.lineTo(x1 * mm - L * math.cos(ang - 0.42), yt(y1) + L * math.sin(ang - 0.42))
        p.lineTo(x1 * mm - L * math.cos(ang + 0.42), yt(y1) + L * math.sin(ang + 0.42))
        p.close()
        c.setFillColor(INK_SOFT)
        c.drawPath(p, stroke=0, fill=1)

    y = top
    # row 1 — three sources feeding one intake
    src_w = (w - 2 * 3.0) / 3.0
    for i, s in enumerate(("USB camera", "video file", "stream URL")):
        box(x + i * (src_w + 3.0), y, src_w, s, TINT, INK, 7.0)
    y += box_h + lane_gap

    for i in range(3):
        arrow(x + i * (src_w + 3.0) + src_w / 2, y - lane_gap,
              x + w / 2, y - 0.6)

    # row 2 — the vision pipeline
    box(x, y, w, "CV NODE   \u00b7   broadcast_server.py", BAR_DEEP, PAPER, 8.0)
    y += box_h + 2.2
    stage_w = (w - 3 * 2.2) / 4.0
    stages = [("YOLO11-m", "detect"), ("BoT-SORT", "track"),
              ("homography", "km/h"), ("index", "0\u20131")]
    for i, (a, b) in enumerate(stages):
        box(x + i * (stage_w + 2.2), y, stage_w, a, BAR, PAPER, 7.0, sub=b)
    y += box_h + 2.2
    em_w = (w - 2.2) / 2.0
    box(x, y, em_w, "CLIP livery", TEAL, PAPER, 7.0, sub="what it is")
    box(x + em_w + 2.2, y, em_w, "beacon detector", CORAL, PAPER, 7.0, sub="what it's doing")
    y += box_h + lane_gap

    arrow(x + w * 0.27, y - lane_gap, x + w * 0.27, y - 0.6)
    arrow(x + w * 0.73, y - lane_gap, x + w * 0.73, y - 0.6)

    # row 3 — the two consumers
    half = (w - 4.0) / 2.0
    box(x, y, half, "ESP32 hub  \u00b7  COM6", NAVY, PAPER, 7.4)
    box(x + half + 4.0, y, half, "Streamlit dashboard", NAVY, PAPER, 7.4)
    y += box_h + 2.2
    third = (half - 2 * 1.8) / 3.0
    for i, s in enumerate(("signal Uno", "lane Uno", "LED / LCD")):
        box(x + i * (third + 1.8), y, third, s, TINT, INK, 6.4)
    for i, s in enumerate(("9 pages", "charts", "Excel log")):
        box(x + half + 4.0 + i * (third + 1.8), y, third, s, TINT, INK, 6.4)


# ────────────────────────────── the content ─────────────────────────────────

TITLE = ("COMPUTER VISION-BASED SMART TRAFFIC CONGESTION ANALYTICS "
         "WITH ADAPTIVE SIGNAL OPTIMIZATION")
SUBTITLE = ("A camera, a GPU and three microcontrollers that watch real traffic, "
            "measure it in km/h, and change the signals to suit it")
FACULTY = "School of Computing  \u00b7  Asia Pacific University of Technology & Innovation"
PROGRAMME = "Group Design Project  \u00b7  Final Submission  \u00b7  14 weeks"

#: The group. Replace the four bracketed slots with the real names.
TEAM = [
    ("Zeyad Khairy", "Computer Vision & Detection",
     "Detection and tracking, km/h speed from road-plane calibration, "
     "the congestion index."),
    ("[ Member 2 ]", "Dashboard & System Integration",
     "The Streamlit control centre, telemetry logging, the database and the "
     "link to the CV node."),
    ("[ Member 3 ]", "Hardware & Control",
     "The ESP32 hub, the Arduino signal heads, the servo lane changer, the "
     "LED strip and the LCDs."),
    ("[ Member 4 ]", "Emergency Vehicle Priority",
     "Flashing-beacon detection, livery recognition, and the priority "
     "response that clears a lane."),
    ("[ Member 5 ]", "Testing, Calibration & Documentation",
     "Speed calibration, the automated test suites, the measured metrics "
     "and the report."),
]

HEADLINES = [
    ("27.1", "fps end to end", "YOLO11-m @ 1280 px, FP16, RTX 4050"),
    ("\u00b15%", "speed accuracy", "calibrated \u2014 exact vs. ground truth"),
    ("0", "false alarms at night", "down from 29% of all traffic"),
    ("831/833", "automated checks pass", "7 suites, reproducible"),
    # 21,526 vehicles span 69 sessions; the 12,919 one-second samples come from
    # the 66 sessions in traffic_all_sessions.csv. Kept separate on purpose.
    ("21,526", "vehicles logged", "across 69 recorded sessions"),
]

NIGHT_SHOT = os.path.join(HERE, "Screenshot 2026-08-09 014446.png")
EMERG = [
    (os.path.join(HERE, "out_demo", "yt_shots", "s111_id3_ambulance.png"), "AMBULANCE"),
    (os.path.join(HERE, "out_demo", "yt_shots", "s22_id60_police.png"), "POLICE"),
    (os.path.join(HERE, "out_demo", "yt_shots", "s109_id2_fire truck.png"), "FIRE TRUCK"),
]


def build_sections():
    """The three columns, as lists of Sections."""

    # ── column 1 ────────────────────────────────────────────────────────────
    intro = Section("Introduction", [
        Para("Urban signals overwhelmingly run on **fixed timers**. They hold a "
             "green over an empty approach and a red over a queue, because they "
             "cannot see the road they govern. Where sensors do exist they are "
             "usually inductive loops buried in the tarmac \u2014 expensive to "
             "install, blind to what kind of vehicle passed, and unable to say "
             "how fast anything was going."),
        Para("This project replaces that with **a camera and a graphics card**. "
             "One video feed is turned into engineering measurements \u2014 a "
             "vehicle count, a speed in km/h, a congestion index between 0 and "
             "1 \u2014 and those measurements drive real signal hardware, a "
             "movable lane divider and an emergency-vehicle response."),
        Gap(1.0),
        Para("**Three things happen to every frame:**", space=2.0),
        Bullet("**SEE** \u2014 YOLO11-m detects vehicles at 1280 px and BoT-SORT "
               "keeps an identity on each one across frames."),
        Bullet("**MEASURE** \u2014 pixels are converted to metres on the road "
               "plane, giving a true km/h, and a weighted density and speed "
               "deficit are combined into one congestion index."),
        Bullet("**ACT** \u2014 a demand-responsive signal, a servo lane divider "
               "and a \u201cmake space\u201d LED strip are driven over one USB "
               "cable, and every second is written to a spreadsheet."),
        Gap(0.6),
        KeyLine("The system runs on whatever it is pointed at \u2014 the camera "
                "on the desk, an uploaded video, or a public roadside stream "
                "URL \u2014 with the **same pipeline** and no restart."),
    ], tag="why this exists")

    problem = Section("Problem & Motivation", [
        Para("A camera-based traffic system is easy to demonstrate badly. Three "
             "specific failures sit between a demo and a measurement, and "
             "closing them is what this project is actually about."),
        Gap(0.8),
        Bullet("**Pixels are not metres.** A distant car crossing 3 px per frame "
               "and a near car crossing 30 px per frame can be travelling at "
               "exactly the same speed. Any system reporting \u201cpixels per "
               "second\u201d has not measured a speed at all."),
        Bullet("**The computer's clock is not the road's clock.** Replay footage "
               "on a slow machine and the traffic appears slow; replay it on a "
               "fast one and it appears to speed. A reported speed must not be "
               "a property of the hardware watching it."),
        Bullet("**Looking like an ambulance is not being on a call.** An "
               "ambulance parked outside a hospital is still an ambulance. "
               "Appearance alone flagged **88 of 299 vehicles \u2014 29% of all "
               "traffic** \u2014 on our night footage."),
        Gap(0.8),
        Para("**Objective.** To build a working smart-traffic analytics "
             "prototype that measures congestion from live video and adapts "
             "signal timing to it, with every claim backed by a stated method "
             "rather than an impression."),
    ], tag="the three gaps")

    arch = Section("System Architecture", [
        Para("The system is **two programs, deliberately separated**, because "
             "they need different things. The CV node has to be where the "
             "camera and the GPU are. The dashboard only needs a browser \u2014 "
             "so it can run on a laptop in another room or another building."),
        Chart(chart_pipeline, 67.0,
              "Every frame follows one path. The source is chosen while running, "
              "so the detector, the speed estimator, the congestion index, the "
              "signal and both workbooks are identical whatever is being watched.",
              space=3.0),
        KeyLine("**Only the node holds the USB port.** Windows will not let two "
                "programs open the same COM port, so a dashboard button does not "
                "talk to the Arduino \u2014 it asks the node to. That single rule "
                "is what lets the dashboard live on a different computer.",
                fill=TINT_WARM, accent=CORAL),
    ], tag="two programs, one pipeline")

    hardware = Section("Hardware & Control", [
        Para("One USB cable leaves the PC and everything hangs off it. The "
             "**ESP32** is the hub; it relays the seven-field signal line down "
             "two UARTs to a pair of Arduino Unos that own the physical outputs."),
        Table(
            [("Board", "l", 0.28), ("Link", "l", 0.24), ("Drives", "l", 0.48)],
            [["ESP32 hub", "USB \u00b7 COM6", "WS2815 strip (60 px), 2\u00d7 I\u00b2C LCD"],
             ["Signal Uno", "GPIO33 \u2192 D7", "Two ramp signal heads + status LCD"],
             ["Lane Uno", "GPIO25/26", "Servo divider, 18-px strip, LCD"]],
            size=7.8, lead=10.4,
            foot="Grounds are tied together; the Unos need power, not a data cable."),
        Gap(0.4),
        Bullet("**Servo lane divider** \u2014 three positions (50\u00b0 / 90\u00b0 / "
               "130\u00b0) reallocating lanes toward the busier direction."),
        Bullet("**Emergency LED strip on pin 6** \u2014 the \u201cmake space\u201d "
               "output, separate from the signal heads."),
        Bullet("**Delivery is acknowledged.** Every command's reply says whether "
               "it reached the wire, so an unplugged board is visible instead of "
               "a button that pretends it worked."),
    ], tag="one USB cable")

    # ── column 2 ────────────────────────────────────────────────────────────
    speed = Section("Measuring Speed in km/h", [
        Para("Speed is the measurement the whole system rests on, so it is taken "
             "on the **road plane in metres**, never in pixels. Two calibration "
             "modes are offered."),
        Gap(0.6),
        Bullet("**Automatic (\u00b125%, works immediately).** A car is about "
               "1.5 m tall and a bus about 3.2 m, so the box height in pixels "
               "gives the metres-per-pixel scale *at that exact spot*. Every "
               "vehicle carries its own scale, so near and far are both handled."),
        Bullet("**Calibrated (\u00b15%, once per camera position).** Four corners "
               "of a road rectangle are clicked and its real size typed in "
               "\u2014 a lane is about 3.5 m. A homography then maps the whole "
               "road plane properly."),
        Gap(0.8),
        Para("**Three rules turn a position into a speed:**", space=2.0),
        Bullet("The **bottom-centre** of the box is used \u2014 the only part of "
               "a vehicle actually touching the road."),
        Bullet("Displacement is measured over **0.8 s**, not between two frames, "
               "so box jitter is not reported as motion."),
        Bullet("Anything over **200 km/h** is discarded as a tracker identity "
               "swap rather than a vehicle."),
        Gap(0.6),
        Table(
            [("Validation test", "l", 0.50), ("Expected", "r", 0.25), ("Measured", "r", 0.25)],
            [["Homography mode", "18 km/h", "**18.00**"],
             ["Homography mode", "36 km/h", "**36.00**"],
             ["Homography mode", "72 km/h", "**72.00**"],
             ["Height-based (auto)", "18 km/h", "**18.00**"],
             ["Parked vehicle", "0 km/h", "**0.12**"]],
            size=7.8, lead=10.2,
            foot="Constructed ground truth: the vehicle is synthetic, so its speed "
                 "is known rather than estimated."),
        KeyLine("**A video gets its own clock.** A vehicle scripted at 16.2 km/h "
                "was played at real time, 2\u00d7, half speed and unthrottled. All "
                "four runs measured **16.2 km/h**, agreeing to 0.00 \u2014 so the "
                "reported speed is a property of the traffic, not of the computer."),
    ], tag="the load-bearing measurement")

    congestion = Section("The Congestion Index", [
        Para("One number between 0 and 1, from two ingredients: **how slow "
             "everyone is (60%)** and **how full the road is (40%)**. Occupancy "
             "is weighted, because a bus does not take the same room as a "
             "motorbike."),
        Gap(0.4),
        Table(
            [("Motorcycle", "c", 0.25), ("Car", "c", 0.25), ("Bus", "c", 0.25), ("Truck", "c", 0.25)],
            [["0.5", "1.0", "3.0", "3.0"]],
            size=8.2, lead=10.4, zebra=False,
            foot="Passenger-car-equivalent weights used for the occupancy term."),
        Gap(0.6),
        Para("**Two guard rules stop it saying something stupid:**", space=2.0),
        Bullet("**An empty road is never congested.** The first version "
               "remembered the last speed it saw, so an empty road could still "
               "report HEAVY. Zero vehicles now means zero congestion, full stop."),
        Bullet("**One slow car is not a jam** \u2014 it is one slow driver. The "
               "speed term only counts once roughly 3+ vehicles are present to "
               "be slow together."),
        Bullet("**Hysteresis on the label**, so FREE / MODERATE / HEAVY does not "
               "flicker on the boundary \u2014 the LEDs and the servo would "
               "physically chatter."),
    ], tag="0 \u2192 1, and honest at both ends")

    signal = Section("Adaptive Signal Control", [
        Para("The signal is **demand-responsive**. In the default `merge` layout "
             "the heads sit on the side roads feeding two carriageways, so they "
             "hold merging traffic back exactly when the main road is busiest."),
        Table(
            [("Congestion on main road", "l", 0.46), ("Side-road signal", "l", 0.54)],
            [["HEAVY", "**RED** \u2014 close the merge"],
             ["MODERATE", "**GREEN** \u2014 there is room"],
             ["FREE (empty)", "**GREEN**"],
             ["any change", "**YELLOW** for 5 s first"]],
            size=7.8, lead=10.2),
        Chart(chart_signal_machine, 24.0, space=1.5),
        Bullet("**Transitions are committed.** Once yellow starts it runs the "
               "full five seconds and always lands on the phase it was aiming "
               "at \u2014 real signals never abort an amber."),
        Bullet("**Minimum dwell of 3 s** on green and red, or traffic flipping "
               "during the yellow could produce a green lasting one frame."),
        KeyLine("Driven against a **synthetic clock** rather than by sleeping, so "
                "the interval is asserted exactly. With congestion flipping every "
                "0.5 s for two minutes, no phase was ever momentary and the "
                "R\u2192Y\u2192G\u2192Y\u2192R ordering never broke."),
        Gap(0.6),
        Para("**An emergency vehicle does not seize the signal.** Emergency "
             "traffic may proceed through any indication, so taking the light "
             "over would only disrupt everyone else. A dedicated LED strip tells "
             "nearby drivers to leave space and the LCD names the vehicle, while "
             "the signal carries on serving normal traffic.", space=1.0),
    ], tag="demand-responsive")

    live = Section("The System Running", [
        Figure(NIGHT_SHOT,
               "**Night footage, live.** The overlay reports the congestion "
               "index, the level, the signal phase and the servo angle; each "
               "box carries a tracking ID, a class and a speed in km/h. The "
               "cyan polygon is the region of interest \u2014 only vehicles "
               "inside it are counted, so the opposite carriageway does not "
               "inflate the measurement.",
               h_mm=132.0, crop=(23, 10, 529, 700), space=1.0),
    ], tag="night is the hard case")

    # ── column 3 ────────────────────────────────────────────────────────────
    emergency = Section("Emergency Vehicle Priority", [
        Para("YOLO has never been taught the word *ambulance* \u2014 it reports "
             "\u201ctruck\u201d. Two recognisers fill the gap, and keeping them "
             "**separate is the whole design**, because they answer different "
             "questions."),
        Gap(0.6),
        Table(
            [("Recogniser", "l", 0.30), ("Asks", "l", 0.36), ("Role", "l", 0.34)],
            [["**CLIP** ViT-B/16", "*What kind of vehicle is that?*", "Names it"],
             ["**Beacon** detector", "*Is it on a call right now?*", "Triggers priority"]],
            size=7.6, lead=10.4, zebra=False),
        FigureRow(EMERG,
                  "Recognised on unseen internet footage the system was never "
                  "tuned for. CLIP is **zero-shot** \u2014 nothing was trained on "
                  "this project's cameras, which is why it travels between "
                  "countries.", h_mm=30.0),
        Para("**The beacon detector is not an AI at all** \u2014 it is arithmetic "
             "about light and time. It looks only at the roof band (stretched "
             "18% above it, since a lamp glows past its housing, and safely "
             "clear of brake lights), demands a properly saturated red or blue "
             "above 165/255, then starts a stopwatch.", space=2.4),
        Bullet("At least **3 flashes** inside a 3.4 s window."),
        Bullet("A rate of **1\u20136 Hz** \u2014 real beacons are regulated by law."),
        Bullet("It must go properly **out** between flashes, not merely dim."),
        KeyLine("**Switch versus slide.** A lamp is either on or off and is "
                "almost never caught halfway; a red car passing under a street "
                "light spends most of its time halfway. On the night footage the "
                "real beacon scored **0.00** and every false positive "
                "**0.18\u20130.43** \u2014 complete separation.",
                fill=TINT_WARM, accent=CORAL),
        Chart(chart_funnel, 32.0,
              "The two cues in the session archive. 244 vehicles looked the part; "
              "26 were actually working. Recognising the difference is what stops "
              "a parked ambulance holding a junction.", space=2.0),
        Para("**Livery alone was tried as the trigger and rejected.** Making the "
             "beacon the trigger took false alarms on the night footage from "
             "**88 of 299 vehicles (29%) to zero**, while still catching every "
             "planted emergency vehicle.", space=1.0),
    ], tag="two cues, one decision", bar=BAR_DEEP)

    results = Section("Results", [
        Para("**Congestion costs speed, and the archive shows it.** 12,919 "
             "one-second samples across 66 recorded sessions, binned by "
             "congestion index. Mean speed falls monotonically as the index "
             "rises, while the vehicle count more than sixfolds.", space=2.4),
        Chart(chart_speed_vs_congestion, 52.0,
              "Bars: mean speed in km/h. Line: mean vehicles present. The index "
              "was built to be read this way \u2014 it is not a proxy for the "
              "vehicle count alone.", space=3.0),
        Chart(chart_model_tradeoff, 58.0,
              "**Resolution buys detections; model size does not.** Every family "
              "gains ~5 vehicles/frame from 640 \u2192 1280 px, but at fixed "
              "resolution the small model matches the large one \u2014 yolo11-**l** "
              "@ 1280 finds *fewer* than yolo11-**m** for 20% more time. YOLO12 "
              "is the wrong family here: 12m is the slowest point on the chart "
              "and finds fewer.", space=3.0),
        Chart(chart_stage_cost, 40.0,
              "Detection is two-thirds of the budget, so it is the only stage "
              "worth optimising. Both recognisers cost more at night \u2014 CLIP "
              "because dark crops fail the readability gate, the beacon detector "
              "because at night there are beacons to measure.", space=3.0),
        Chart(chart_class_mix, 18.0,
              "21,526 tracked vehicles by class, across the whole archive.",
              space=1.0),
    ], tag="measured, not estimated")

    validation = Section("Validation & Honesty", [
        Table(
            [("Suite", "l", 0.52), ("Checks", "r", 0.14), ("Covers", "l", 0.34)],
            [["`test_traffic_system.py`", "217", "vision pipeline"],
             ["`test_server.py`", "165", "hardware, HTTP, relay"],
             ["`test_video_source.py`", "113", "sources, media clock"],
             ["`test_dashboard_interactive.py`", "109", "widgets, exports"],
             ["`test_analytics.py`", "90", "analytics maths"],
             ["`test_dashboard.py`", "83", "all 9 pages render"],
             ["`test_traffic_light.py`", "56", "signal timing"],
             ["**Total**", "**833**", ""]],
            size=7.4, lead=9.8, bold_rows=(7,)),
        Para("**831 of 833 pass in one run, and 833 is not reachable in a single "
             "run.** The two that do not are the pair that POST to the CV node; "
             "they pass when it is up, but four Live Camera checks written "
             "against the simulated feed then fail. No single run can satisfy "
             "both states. **831 is the honest, reproducible number.**", space=2.6),
        Gap(0.4),
        Para("**What this project does *not* claim:**", space=2.0),
        Bullet("**No mAP is quoted.** There are no hand-drawn boxes for this "
               "footage, so there is nothing to score against. `benchmark.py` "
               "refuses to print an accuracy figure rather than borrow the "
               "model's published COCO score.", mcolour=CORAL),
        Bullet("**6.6% is not an accuracy.** On an 11:43 unseen compilation the "
               "recogniser singled out 166 of 2,511 tracked vehicles. That is a "
               "selectivity figure; confirming a false-positive rate would mean "
               "labelling all 166 by hand.", mcolour=CORAL),
        Bullet("**A known open question, deliberately not acted on.** `11s@1280` "
               "reaches 99% of `11m`'s yield at 48% more throughput \u2014 but "
               "vehicles/frame is a count, not a correctness score, so the "
               "default stays until real labels exist.", mcolour=CORAL),
    ], tag="833 checks \u00b7 7 suites")

    dashboard = Section("Control Centre & Data", [
        Para("The dashboard is a nine-page Streamlit application that never "
             "touches the camera or the serial port itself — it reads the "
             "node's telemetry over HTTP and asks the node to act. That is what "
             "lets it run on a different machine from the one doing the seeing."),
        Table(
            [("Page", "l", 0.40), ("Answers", "l", 0.60)],
            [["Operations", "health, count, density, phase"],
             ["Live Camera", "the annotated feed + detections table"],
             ["Traffic Analytics", "trends across every session"],
             ["Emergency Control", "priority state, manual override"],
             ["Hardware Monitor", "board, LEDs, servo, LCD"],
             ["Logs · Settings · About", "diagnostics and configuration"]],
            size=7.6, lead=10.0),
        Gap(0.4),
        Para("**Two diaries, because they answer two questions.** "
             "`traffic_history` holds the current run and is what a demo shows; "
             "`traffic_all_sessions` is appended to and never reset. CSVs are "
             "written **every second** and are always complete; the Excel "
             "workbooks are built on request, so a download is never stale.",
             space=2.6),
        Bullet("Sheets: **Live Status, Traffic Log, Vehicle Speeds, Session "
               "Summary** — plus All Sessions, All Records and Lifetime "
               "Summary in the archive workbook."),
        Bullet("**One row per vehicle** that passed through, with its class, "
               "dwell time, distance, and average and top speed."),
        Bullet("Every chart carries a *how to read this* note, and a panel turns "
               "the numbers into plain sentences — the busiest hour, whether "
               "jams are constant or occasional, what congestion costs in speed."),
        KeyLine("**Frozen numbers are reported as frozen.** When a feed ends the "
                "dashboard says so rather than showing the last values as though "
                "they were live — the same principle as the acknowledged "
                "hardware command."),
    ], tag="9 pages · 12,919 samples")

    conclusion = Section("Conclusion & Future Work", [
        Para("A single camera and a laptop GPU are enough to measure a road in "
             "real engineering units and drive real signal hardware from that "
             "measurement, at **27.1 fps end to end** with **\u00b15% speed "
             "accuracy** once calibrated."),
        Para("Almost every design decision here exists because something went "
             "wrong first, got measured, and got fixed \u2014 an empty road "
             "reporting HEAVY, a slow computer reporting slow cars, 29% of "
             "traffic reported as police. That trail, and the 833 checks that "
             "hold it down, is the real result.", space=3.0),
        Gap(0.4),
        Para("**Next:**", space=2.0),
        Bullet("**A full four-way junction.** One controller drives one "
               "approach; four plus a coordinator is the extension, and "
               "`request_phase()` already exists for it.", mcolour=TEAL),
        Bullet("**A few hundred labelled frames**, to settle the `11s`-vs-`11m` "
               "question with a real mAP instead of a vehicle count.", mcolour=TEAL),
        Bullet("**Retrain the livery model per camera.** On the camera it was "
               "built for, the trained classifier beats CLIP (71% vs 41\u201353% "
               "recognition) at a seventh of the cost \u2014 CLIP stays the "
               "default only because it travels.", mcolour=TEAL),
    ], tag="what it proves, what is next", bar=BAR_DEEP)

    return (
        [intro, problem, arch, hardware, live],
        [speed, congestion, signal, emergency],
        [results, dashboard, validation, conclusion],
    )


# ────────────────────────────── the drawing ─────────────────────────────────

def draw_header(c):
    c.setFillColor(PAPER)
    c.rect(0, yt(HEADER_H), PAGE_W, HEADER_H * mm, stroke=0, fill=1)
    network_pattern(c, 0, yt(HEADER_H), PAGE_W, PAGE_H, seed=11, n=54, alpha=0.55)
    chevron(c, PAGE_W * 0.5, PAGE_H, 62 * mm, 34 * mm)

    # left: programme lockup
    lx = MARGIN * mm
    cy = yt(30.0)
    c.setFillColor(BAR)
    c.setFont(FB, 33)
    c.drawString(lx, cy, "SMART")
    w1 = pdfmetrics.stringWidth("SMART", FB, 33)
    c.setFillColor(CORAL)
    c.drawString(lx + w1 + 5, cy, "TRAFFIC")
    c.setFillColor(BAR)
    c.setFont(FL if HAVE_SEGOE else F, 12.6)
    c.drawString(lx + 1.2, cy - 15, "C O N G E S T I O N   A N A L Y T I C S")
    c.setStrokeColor(CORAL)
    c.setLineWidth(1.6)
    c.line(lx, cy - 22, lx + 74 * mm, cy - 22)

    # right: institution
    draw_apu_logo(c, PAGE_W - MARGIN * mm, yt(26.0), 22 * mm)

    # title block
    tw = PAGE_W_MM - 2 * MARGIN - 20
    tx = MARGIN + 10
    y = 50.0
    lines = wrap_runs(parse_runs(TITLE, base=FB), 24.5, tw * mm)
    y_pt = draw_lines(c, lines, tx * mm, yt(y), 24.5, 29.0, HexColor("#12181D"),
                      "center", tw * mm)
    y_pt -= 4
    lines = wrap_runs(parse_runs(SUBTITLE, base=FI), 13.4, (tw - 40) * mm)
    y_pt = draw_lines(c, lines, (tx + 20) * mm, y_pt, 13.4, 16.4, BAR,
                      "center", (tw - 40) * mm)
    y_pt -= 5

    names = "  \u00b7  ".join(n for n, _, _ in TEAM)
    lines = wrap_runs(parse_runs(names, base=FB), 11.2, tw * mm)
    y_pt = draw_lines(c, lines, tx * mm, y_pt, 11.2, 13.8, INK, "center", tw * mm)
    lines = wrap_runs(parse_runs(FACULTY, base=F), 10.4, tw * mm)
    y_pt = draw_lines(c, lines, tx * mm, y_pt - 1, 10.4, 13.0, INK_SOFT, "center", tw * mm)
    lines = wrap_runs(parse_runs(PROGRAMME, base=F), 9.2, tw * mm)
    draw_lines(c, lines, tx * mm, y_pt, 9.2, 11.6, BAR, "center", tw * mm)


def draw_team_strip(c):
    x = MARGIN
    w = PAGE_W_MM - 2 * MARGIN
    h = TEAM_H

    c.setFillColor(BAR_DEEP)
    c.rect(x * mm, yt(TEAM_TOP + h), w * mm, h * mm, stroke=0, fill=1)

    # title rail down the left
    rail = 27.0
    c.setFillColor(CORAL)
    c.rect(x * mm, yt(TEAM_TOP + h), rail * mm, h * mm, stroke=0, fill=1)
    c.saveState()
    c.setFillColor(PAPER)
    c.setFont(FB, 12.0)
    c.drawCentredString((x + rail / 2) * mm, yt(TEAM_TOP + h / 2) + 1.4 * mm, "PROJECT")
    c.setFont(FB, 12.0)
    c.drawCentredString((x + rail / 2) * mm, yt(TEAM_TOP + h / 2) - 3.6 * mm, "TEAM")
    c.restoreState()

    cx = x + rail
    cw = (w - rail) / len(TEAM)
    for i, (name, role, contrib) in enumerate(TEAM):
        bx = cx + i * cw
        if i:
            c.setStrokeColor(Color(1, 1, 1, 0.22))
            c.setLineWidth(0.6)
            c.line(bx * mm, yt(TEAM_TOP + h - 4), bx * mm, yt(TEAM_TOP + 4))
        pad = 4.0
        iw = cw - 2 * pad

        # index chip
        c.setFillColor(CORAL)
        c.circle((bx + pad + 2.6) * mm, yt(TEAM_TOP + pad + 2.8), 2.6 * mm,
                 stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FB, 7.6)
        c.drawCentredString((bx + pad + 2.6) * mm, yt(TEAM_TOP + pad + 2.8) - 2.5,
                            str(i + 1))

        c.setFillColor(PAPER)
        nm = name
        while pdfmetrics.stringWidth(nm, FB, 10.4) > (iw - 7.0) * mm and len(nm) > 4:
            nm = nm[:-2]
        c.setFont(FB, 10.4)
        c.drawString((bx + pad + 6.6) * mm, yt(TEAM_TOP + pad + 4.2), nm)

        lines = wrap_runs(parse_runs(role, base=FB), 8.4, iw * mm)
        y_pt = draw_lines(c, lines, (bx + pad) * mm, yt(TEAM_TOP + pad + 6.4),
                          8.4, 10.2, CORAL_PALE)
        lines = wrap_runs(parse_runs(contrib, base=F), 7.4, iw * mm)
        draw_lines(c, lines, (bx + pad) * mm, y_pt - 1.2, 7.4, 9.2,
                   Color(1, 1, 1, 0.86))


def draw_headline_band(c, top, h):
    x = MARGIN
    w = PAGE_W_MM - 2 * MARGIN
    n = len(HEADLINES)
    gap = 3.0
    cw = (w - gap * (n - 1)) / n
    for i, (fig, label, note) in enumerate(HEADLINES):
        bx = x + i * (cw + gap)
        c.setFillColor(PAPER)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.7)
        c.rect(bx * mm, yt(top + h), cw * mm, h * mm, stroke=1, fill=1)
        c.setFillColor(CORAL if i in (1, 2) else BAR)
        c.rect(bx * mm, yt(top + h), cw * mm, 1.6 * mm, stroke=0, fill=1)

        size = 21.0
        while pdfmetrics.stringWidth(fig, FB, size) > (cw - 7) * mm:
            size -= 0.7
        c.setFillColor(BAR_DEEP)
        c.setFont(FB, size)
        c.drawCentredString((bx + cw / 2) * mm, yt(top + 10.6), fig)

        c.setFillColor(INK)
        c.setFont(FB, 8.6)
        c.drawCentredString((bx + cw / 2) * mm, yt(top + 14.8), label)

        lines = wrap_runs(parse_runs(note, base=F), 7.2, (cw - 6) * mm)
        draw_lines(c, lines, (bx + 3) * mm, yt(top + 16.4), 7.2, 8.8,
                   INK_SOFT, "center", (cw - 6) * mm)


#: Symptom -> fix. Straight from the project's own record of what went wrong.
LESSONS = [
    ("Empty road reported as HEAVY traffic",
     "The index remembered the last speed it saw. Zero vehicles now means zero congestion, full stop."),
    ("Slow computer, slow cars",
     "A video was being timed by the wall clock. It now carries its own clock, one tick per frame."),
    ("One slow driver counted as a jam",
     "The speed deficit only contributes once about three vehicles are present to be slow together."),
    ("The signal flickered between phases",
     "Yellow became committed to its full five seconds, plus a three-second minimum dwell."),
    ("29% of night traffic called “police”",
     "An unfair 14-against-15 vote. Made one-on-one: best emergency guess vs. ordinary vehicle."),
    ("A red car under a street light",
     "Lamps switch, reflections slide. Anything caught halfway more than 15% of the time is rejected."),
    ("“The newer model must be better”",
     "yolo12m was measured: 33% slower and it found fewer vehicles. yolo11m was kept."),
]


def draw_lessons_band(c, top, h):
    x = MARGIN
    w = PAGE_W_MM - 2 * MARGIN
    bar_h = 9.0

    c.setFillColor(PAPER)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.7)
    c.rect(x * mm, yt(top + h), w * mm, h * mm, stroke=1, fill=1)
    c.setFillColor(BAR_DEEP)
    c.rect(x * mm, yt(top + bar_h), w * mm, bar_h * mm, stroke=0, fill=1)
    c.setFillColor(PAPER)
    c.setFont(FB, 12.4)
    c.drawString((x + 3.6) * mm, yt(top + bar_h) + 2.4 * mm,
                 "Engineering Decisions")
    c.setFont(F, 9.6)
    c.setFillColor(Color(1, 1, 1, 0.82))
    c.drawString((x + 3.6) * mm + pdfmetrics.stringWidth("Engineering Decisions", FB, 12.4) + 8,
                 yt(top + bar_h) + 2.5 * mm,
                 "—  almost every rule in this system exists because something went wrong first, "
                 "got measured, and got fixed")
    c.setFont(F, 8.4)
    tag = "7 of them"
    c.setFillColor(Color(1, 1, 1, 0.7))
    c.drawString((x + w - 3.6) * mm - pdfmetrics.stringWidth(tag, F, 8.4),
                 yt(top + bar_h) + 2.8 * mm, tag)

    n = len(LESSONS)
    gap = 2.4
    cw = (w - 2 * 3.6 - gap * (n - 1)) / n
    cy = top + bar_h + 3.4
    ch = h - bar_h - 3.4 - 3.0

    for i, (bad, good) in enumerate(LESSONS):
        bx = x + 3.6 + i * (cw + gap)
        c.setFillColor(HexColor("#FCEEEA"))
        c.rect(bx * mm, yt(cy + ch), cw * mm, ch * mm, stroke=0, fill=1)
        c.setFillColor(HexColor("#F7DED6"))
        c.rect(bx * mm, yt(cy + 2.0), cw * mm, 2.0 * mm, stroke=0, fill=1)

        pad = 2.4
        iw = cw - 2 * pad
        lines = wrap_runs(parse_runs(bad, base=FBI), 8.2, iw * mm)
        y_pt = draw_lines(c, lines, (bx + pad) * mm, yt(cy + 2.4), 8.2, 10.0,
                          HexColor("#A5352A"))

        y_mm = (PAGE_H - y_pt) / mm + 1.6
        c.setFillColor(CORAL)
        c.setFont(FB, 9.4)
        c.drawString((bx + pad) * mm, yt(y_mm + 2.8), "↓")

        lines = wrap_runs(parse_runs(good, base=F), 7.9, (iw - 4.0) * mm)
        draw_lines(c, lines, (bx + pad + 4.0) * mm, yt(y_mm - 0.4), 7.9, 9.8, INK)


def draw_footer(c):
    h = PAGE_H_MM - FOOTER_TOP
    c.setFillColor(PAPER)
    c.rect(0, 0, PAGE_W, h * mm, stroke=0, fill=1)
    network_pattern(c, 0, 0, PAGE_W, h * mm, seed=23, n=26, alpha=0.5)
    footer_motif(c, PAGE_W * 0.44, 0, 46 * mm, h * mm * 0.86)

    c.setStrokeColor(BAR)
    c.setLineWidth(1.4)
    c.line(MARGIN * mm, yt(FOOTER_TOP), (PAGE_W_MM - MARGIN) * mm, yt(FOOTER_TOP))

    c.setFillColor(BAR_DEEP)
    c.setFont(FB, 15.0)
    c.drawString(MARGIN * mm, yt(PAGE_H_MM - 22.0),
                 "SMART TRAFFIC CONGESTION ANALYTICS")
    c.setFillColor(INK_SOFT)
    c.setFont(F, 10.0)
    c.drawString(MARGIN * mm, yt(PAGE_H_MM - 14.0),
                 "Group Design Project  \u00b7  School of Computing")

    c.setFillColor(BAR)
    c.setFont(FB, 14.0)
    right = "ASIA PACIFIC UNIVERSITY OF TECHNOLOGY & INNOVATION"
    c.drawRightString((PAGE_W_MM - MARGIN) * mm, yt(PAGE_H_MM - 20.0), right)
    c.setFillColor(INK_SOFT)
    c.setFont(F, 9.4)
    c.drawRightString((PAGE_W_MM - MARGIN) * mm, yt(PAGE_H_MM - 13.0),
                      "All figures measured on an RTX 4050 laptop GPU, FP16  \u00b7  "
                      "methods in METRICS.md")


def main():
    c = rl_canvas.Canvas(OUT, pagesize=(PAGE_W, PAGE_H))
    c.setTitle("Computer Vision-Based Smart Traffic Congestion Analytics")
    c.setAuthor(" / ".join(n for n, _, _ in TEAM))
    c.setSubject("Group Design Project poster \u2014 A1")

    # page ground
    c.setFillColor(HexColor("#EDF1F4"))
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    draw_header(c)
    draw_team_strip(c)

    draw_headline_band(c, BAND_TOP, BAND_H)

    cols = build_sections()
    used = []
    for ci, sections in enumerate(cols):
        y = BODY_TOP
        for s in sections:
            h = s.height(COL_W)
            s.draw(c, COL_X[ci], y, COL_W)
            y += h + s.gap_after
        used.append(y - s.gap_after)

    draw_lessons_band(c, LESSONS_TOP, LESSONS_H)
    draw_footer(c)
    c.showPage()
    c.save()

    print(f"wrote {OUT}")
    print(f"page   A1 portrait  {PAGE_W_MM:.0f} x {PAGE_H_MM:.0f} mm")
    print(f"body   {BODY_TOP:.0f} -> {BODY_BOTTOM:.0f} mm  "
          f"({BODY_BOTTOM - BODY_TOP:.0f} mm available)")
    for i, u in enumerate(used):
        flag = "OVERFLOW" if u > BODY_BOTTOM else f"{BODY_BOTTOM - u:6.1f} mm spare"
        print(f"col {i + 1}  ends at {u:6.1f} mm   {flag}")


if __name__ == "__main__":
    main()
