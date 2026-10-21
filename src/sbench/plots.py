"""Charts shared by the notebooks: one house style and the recurring chart types.

Colours follow a colour-blind-validated categorical order (assigned in fixed
order, never cycled); diverging maps run red (low) to blue (high) through a
neutral grey, so 0 always reads as "no difference" and positive skill reads as
blue. Every function returns the figure; pass
`path` to also save it as a PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]   # secondary encoding, same order as the colours
NEUTRAL = "#8a8984"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("red_grey_blue", ["#a32c2b", "#ec9a99", "#f0efec", "#86b6ef", "#184f95"])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL), "lines.linewidth": 1.5,
        "legend.frameon": False,
    })


def colors_for(names) -> dict:
    """Stable colour per name, in first-seen order."""
    return {n: CATEGORICAL[i % len(CATEGORICAL)] for i, n in enumerate(dict.fromkeys(names))}


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def series_panels(series, shade: dict | None = None, marks: dict | None = None, ncols: int = 3,
                  title: str | None = None, path=None):
    """Small multiples of price paths.

    shade : {series.key: boolean array} spans to shade (e.g. a turbulent regime)
    marks : {series.key: boolean array} points to mark (e.g. jump days)
    """
    nrows = -(-len(series) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.3 * nrows), squeeze=False)
    for ax, s in zip(axes.flat, series):
        t, y = s.df["t"], s.df["close"]
        ax.plot(t, y, color=CATEGORICAL[0], lw=1.2)
        if shade and s.key in shade:
            ax.fill_between(t, 0, 1, where=np.asarray(shade[s.key], bool), transform=ax.get_xaxis_transform(),
                            color=NEUTRAL, alpha=0.18, lw=0, step="mid")
        if marks and s.key in marks:
            m = np.asarray(marks[s.key], bool)
            ax.scatter(t[m], y[m], s=14, color=CATEGORICAL[7], zorder=3, edgecolor=SURFACE, lw=0.8)
        ax.set_title(s.key)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    for ax in axes.flat[len(series):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def dot_plot(df: pd.DataFrame, y: str, x: str, hue: str | None = None, lo: str | None = None,
             hi: str | None = None, order=None, ref: float | None = 0.0, xlabel: str | None = None,
             title: str | None = None, ax=None, path=None):
    """Dots (with optional interval whiskers) per y category; hue offsets and colours the dots.

    Hues get both a colour and a marker shape, so identity never relies on colour alone.
    At most 3 hues: no 4-colour set stays distinguishable for all pairs, so facet
    or use a heatmap instead.
    """
    order = list(order) if order is not None else list(dict.fromkeys(df[y]))
    hues = list(dict.fromkeys(df[hue])) if hue else [None]
    if len(hues) > 3:
        raise ValueError(f"dot_plot supports at most 3 hues, got {len(hues)}; facet or use heatmap()")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) * max(1, len(hues) ** 0.5) + 1.2))
    fig = ax.figure
    off = np.linspace(-0.25, 0.25, len(hues)) if len(hues) > 1 else [0.0]
    pos = {k: i for i, k in enumerate(order)}
    for j, h in enumerate(hues):
        d = df if h is None else df[df[hue] == h]
        d = d[d[y].isin(pos)]
        yy = d[y].map(pos).to_numpy() + off[j]
        c = CATEGORICAL[j % len(CATEGORICAL)]
        if lo and hi:
            ax.hlines(yy, d[lo], d[hi], color=c, lw=1.2, alpha=0.6)
        ax.scatter(d[x], yy, color=c, marker=MARKERS[j % len(MARKERS)], s=30, zorder=3,
                   edgecolor=SURFACE, lw=0.8, label=h)
    if ref is not None:
        ax.axvline(ref, color=INK_2, lw=0.8)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel or x)
    if hue:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(len(hues), 4), fontsize=8)
    if title:
        ax.set_title(title, pad=24 if hue else 6)
    fig.tight_layout()
    return _save(fig, path)


def heatmap(table: pd.DataFrame, center: float = 0.0, span: float | None = None, fmt: str = "{:+.2f}",
            title: str | None = None, cbar_label: str = "", path=None):
    """Annotated heatmap on a diverging map centred at `center` (NaN cells left blank)."""
    v = table.to_numpy(dtype=float)
    span = span or np.nanmax(np.abs(v - center)) or 1.0
    fig, ax = plt.subplots(figsize=(1.0 * v.shape[1] + 2.8, 0.36 * v.shape[0] + 1.3))
    im = ax.imshow(v, cmap=DIVERGING, vmin=center - span, vmax=center + span, aspect="auto")
    ax.set_xticks(range(v.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(v.shape[0]), table.index)
    ax.grid(False)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if not np.isnan(v[i, j]):
                strong = abs(v[i, j] - center) > 0.6 * span
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=7.5,
                        color=SURFACE if strong else INK)
    lo, hi = np.nanmin(v) < center - span, np.nanmax(v) > center + span   # capped values: arrow-ended colour bar
    extend = "both" if lo and hi else "min" if lo else "max" if hi else "neither"
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=cbar_label, extend=extend)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def reliability_panels(rel: pd.DataFrame, order=None, ncols: int = 4, title: str | None = None, path=None):
    """Forecast vs observed frequency per predictor (output of scoring.reliability); dot area ~ count."""
    order = list(order) if order is not None else list(dict.fromkeys(rel["predictor"]))
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), squeeze=False,
                             sharex=True, sharey=True)
    for ax, p in zip(axes.flat, order):
        r = rel[rel["predictor"] == p]
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls="--")
        ax.plot(r["forecast"], r["observed"], color=CATEGORICAL[0], lw=1.0)
        ax.scatter(r["forecast"], r["observed"], s=8 + 120 * r["n"] / max(rel["n"].max(), 1),
                   color=CATEGORICAL[0], edgecolor=SURFACE, lw=0.6, zorder=3)
        ax.set_title(p, fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("forecast")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def line_panels(df: pd.DataFrame, panel: str, x: str, y: str, hue: str, ref: pd.DataFrame | None = None,
                ref_label: str = "reference", order=None, ncols: int = 3, xlabel: str | None = None,
                ylabel: str | None = None, title: str | None = None, path=None):
    """One panel per `panel` value with a line per `hue` (at most 3, colour plus marker).

    ref : optional frame with columns x, y, lo, hi drawn in every panel as a neutral
          line with its interval band (e.g. what actually happened).
    """
    order = list(order) if order is not None else list(dict.fromkeys(df[panel]))
    hues = list(dict.fromkeys(df[hue]))
    if len(hues) > 3:
        raise ValueError(f"line_panels supports at most 3 hues, got {len(hues)}")
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows + 0.5), squeeze=False,
                             sharex=True, sharey=True)
    for ax, p in zip(axes.flat, order):
        if ref is not None:
            ax.fill_between(ref[x], ref["lo"], ref["hi"], color=NEUTRAL, alpha=0.2, lw=0)
            ax.plot(ref[x], ref[y], color=NEUTRAL, lw=1.5, ls="--", label=ref_label)
        d = df[df[panel] == p]
        for j, h in enumerate(hues):
            g = d[d[hue] == h].sort_values(x)
            if len(g):
                ax.plot(g[x], g[y], color=CATEGORICAL[j], lw=2, marker=MARKERS[j], ms=5,
                        markeredgecolor=SURFACE, label=h)
        ax.set_title(p, fontsize=8.5)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel or x)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel or y)
    handles = {lab: h for ax in axes.flat for h, lab in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="upper left", bbox_to_anchor=(0.01, 0.995 if not title else 0.96),
               ncol=len(handles), fontsize=8)
    if title:
        fig.suptitle(title, x=0.01, y=1.0, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))   # room for title and legend
    return _save(fig, path)


def scatter_panels(df: pd.DataFrame, panel: str, x: str, y: str, order=None, ncols: int = 4,
                   xlabel: str | None = None, ylabel: str | None = None, title: str | None = None, path=None):
    """One scatter per `panel` value with the identity line, e.g. a model's P(up) vs the oracle's."""
    order = list(order) if order is not None else list(dict.fromkeys(df[panel]))
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), squeeze=False,
                             sharex=True, sharey=True)
    for ax, p in zip(axes.flat, order):
        d = df[df[panel] == p]
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls="--")
        ax.scatter(d[x], d[y], s=9, color=CATEGORICAL[0], alpha=0.55, lw=0)
        r = d[[x, y]].corr().iloc[0, 1] if len(d) > 2 else np.nan
        ax.set_title(f"{p}\nr = {r:.2f}", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel or x)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel or y)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
