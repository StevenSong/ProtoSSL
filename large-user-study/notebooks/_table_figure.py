"""Render a DataFrame as a booktabs-style table figure, for sharing tables as PNG/PDF
instead of screenshotting notebook output.

Index levels become the leading (left-aligned) columns and values are drawn as strings,
right-aligned. Repeated upper column levels (e.g. "Split" over train/val/test) are drawn
once as a spanning header. With a MultiIndex, the first index level is shown once per
group, wrapped to `wrap` characters, with alternating group shading and a rule between
groups; otherwise rows are zebra striped.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

FONT_SIZE = 9  # pt
PAD_X = 7  # pt, each side of a cell
ROW_H = FONT_SIZE * 1.9
SHADE = "#efefef"
GROUP_RULE = "#b0b0b0"
DPI = 200  # of the saved PNG


def _spans(values: list) -> list[tuple[int, int, str]]:
    """(start, stop, value) runs of equal consecutive values."""
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((start, i, values[start]))
            start = i
    return out


def table_figure(
    df: pd.DataFrame,
    title: str | None = None,
    note: str | None = None,
    wrap: int | None = 22,
) -> Figure:
    regular = FontProperties(size=FONT_SIZE)
    bold = FontProperties(size=FONT_SIZE, weight="bold")
    small = FontProperties(size=FONT_SIZE - 1)
    renderer = FigureCanvasAgg(Figure(dpi=72)).get_renderer()

    def width(s: str, prop: FontProperties) -> float:
        # in points at dpi=72; unlike glyph extents this counts leading spaces
        return renderer.get_text_width_height_descent(s, prop, ismath=False)[0]

    n_idx = df.index.nlevels
    grouped = n_idx > 1
    idx_vals = [[str(v) for v in df.index.get_level_values(i)] for i in range(n_idx)]
    idx_names = [str(n) if n is not None else "" for n in df.index.names]
    cells = [[str(v) for v in df[c]] for c in df.columns]
    n_rows = len(df)
    n_hdr = df.columns.nlevels
    col_lvls = [[str(v) for v in df.columns.get_level_values(i)] for i in range(n_hdr)]

    # group label text: first index level, once per group, wrapped
    groups = _spans(idx_vals[0]) if grouped else []
    group_text = {
        start: textwrap.fill(v, wrap) if wrap else v for start, _, v in groups
    }

    # column widths from the widest cell, including the bottom header row
    widths = []
    for i in range(n_idx):
        if grouped and i == 0:
            body = max(
                width(line, regular)
                for t in group_text.values()
                for line in t.split("\n")
            )
        else:
            body = max(width(v, regular) for v in idx_vals[i])
        widths.append(max(body, width(idx_names[i], bold)) + 2 * PAD_X)
    for j in range(len(df.columns)):
        widths.append(
            max(max(width(v, regular) for v in cells[j]), width(col_lvls[-1][j], bold))
            + 2 * PAD_X
        )
    # upper header spans must fit their text
    for lvl in col_lvls[:-1]:
        for start, stop, v in _spans(lvl):
            have = sum(widths[n_idx + start : n_idx + stop])
            need = width(v, bold) + 2 * PAD_X
            if need > have:
                for j in range(n_idx + start, n_idx + stop):
                    widths[j] += (need - have) / (stop - start)
    x0 = np.concatenate([[0], np.cumsum(widths)])
    W = x0[-1]

    title_h = ROW_H * 1.3 if title else 0
    note_lines = (
        textwrap.fill(note, max(40, int(W / (FONT_SIZE * 0.5)))).split("\n")
        if note
        else []
    )
    note_h = (len(note_lines) + 0.6) * FONT_SIZE * 1.35 if note else 0
    margin = 6
    H = title_h + (n_hdr + n_rows) * ROW_H + note_h + 2 * margin

    fig = Figure(figsize=((W + 2 * margin) / 72, H / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-margin, W + margin)
    ax.set_ylim(H - margin, -margin)  # y grows downward, in points
    ax.axis("off")

    def text(x, y, s, prop, ha="left", va="center", **kw):
        ax.text(x, y, s, fontproperties=prop, ha=ha, va=va, **kw)

    def rule(y, lw, x_from=0, x_to=None, color="black"):
        ax.plot(
            [x_from, W if x_to is None else x_to],
            [y, y],
            color=color,
            lw=lw,
            solid_capstyle="butt",
        )

    y = 0
    if title:
        text(
            0, y + title_h / 2, title, FontProperties(size=FONT_SIZE + 2, weight="bold")
        )
        y += title_h
    top = y
    rule(top, 1.2)

    # header: upper levels as spanning, centered cells with a rule under each span
    for h, lvl in enumerate(col_lvls):
        yc = top + (h + 0.5) * ROW_H
        if h < n_hdr - 1:
            for start, stop, v in _spans(lvl):
                if not v:
                    continue
                a, b = x0[n_idx + start], x0[n_idx + stop]
                text((a + b) / 2, yc, v, bold, ha="center")
                rule(top + (h + 1) * ROW_H - 2, 0.6, a + PAD_X, b - PAD_X)
        else:
            for i, name in enumerate(idx_names):
                text(x0[i] + PAD_X, yc, name, bold)
            for j, v in enumerate(lvl):
                text(x0[n_idx + j + 1] - PAD_X, yc, v, bold, ha="right")
    body_top = top + n_hdr * ROW_H
    rule(body_top, 0.8)

    # shading and group rules under the text
    if grouped:
        for g, (start, stop, _) in enumerate(groups):
            if g % 2 == 1:
                ax.fill_between(
                    [0, W],
                    body_top + start * ROW_H,
                    body_top + stop * ROW_H,
                    color=SHADE,
                    lw=0,
                )
            if g > 0:
                rule(body_top + start * ROW_H, 0.5, color=GROUP_RULE)
    else:
        for r in range(1, n_rows, 2):
            ax.fill_between(
                [0, W],
                body_top + r * ROW_H,
                body_top + (r + 1) * ROW_H,
                color=SHADE,
                lw=0,
            )

    for r in range(n_rows):
        yc = body_top + (r + 0.5) * ROW_H
        for i in range(n_idx):
            if grouped and i == 0:
                if r in group_text:
                    text(
                        x0[0] + PAD_X,
                        body_top + r * ROW_H + ROW_H * 0.22,
                        group_text[r],
                        regular,
                        va="top",
                        linespacing=1.25,
                    )
                continue
            text(x0[i] + PAD_X, yc, idx_vals[i][r], regular)
        for j in range(len(df.columns)):
            text(x0[n_idx + j + 1] - PAD_X, yc, cells[j][r], regular, ha="right")
    bottom = body_top + n_rows * ROW_H
    rule(bottom, 1.2)

    for k, line in enumerate(note_lines):
        text(0, bottom + (k + 1) * FONT_SIZE * 1.35, line, small, color="#444444")
    return fig


def save_table(df: pd.DataFrame, path, **kwargs):
    """Render `df` with `table_figure` and save it as `path` with .png and .pdf suffixes.

    Returns the saved PNG for inline display, since a Figure made without pyplot only
    has a text repr in notebooks.
    """
    from IPython.display import Image

    fig = table_figure(df, **kwargs)
    path = Path(path)
    for suffix in [".png", ".pdf"]:
        fig.savefig(path.with_suffix(suffix), dpi=DPI)
    print(f"Saved table to {path.with_suffix('.png')} and .pdf")
    # shown at 100 px per inch, not the saved DPI
    return Image(filename=path.with_suffix(".png"), width=int(fig.get_figwidth() * 100))
