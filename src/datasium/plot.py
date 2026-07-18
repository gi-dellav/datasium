"""Plot tab: generate Plotly figures from a dataset.

The :func:`build_figure` helper turns a materialised :class:`polars.DataFrame`
plus a :class:`PlotSpec` into the declarative ``dict`` figure that
:meth:`nicegui.ui.plotly` renders fastest (straight to JSON-friendly arrays,
no ``go.Figure`` overhead). It is UI-free so it can be unit-tested.

The :class:`PlotPanel` widget wires the spec selectors and the
data-scope toggle (entire dataset vs. the filtered selection from the Select
tab) to the app's plot callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl

from nicegui import ui

from datasium.calculate import _is_numeric


# (label, key) for the plot-type select.
PLOT_TYPES: list[tuple[str, str]] = [
    ("Scatter", "scatter"),
    ("Line", "line"),
    ("Bar (aggregated)", "bar"),
    ("Histogram", "histogram"),
    ("Box", "box"),
    ("Violin", "violin"),
]

# (label, key) for the aggregation-statistic select (used by the Bar plot).
AGG_STATS: list[tuple[str, str]] = [
    ("raw (no aggregation)", "raw"),
    ("average", "mean"),
    ("sum", "sum"),
    ("minimum", "min"),
    ("maximum", "max"),
    ("median", "median"),
    ("count", "count"),
]

_AGG_OPS = {"mean", "sum", "min", "max", "median", "count"}
_ODD = "—"  # placeholder for empty option lists


@dataclass(frozen=True)
class PlotSpec:
    """Declarative description of a single Plotly figure."""

    plot_type: str = "scatter"
    x: str | None = None
    y: str | None = None
    color: str | None = None   # optional group / color-by column
    agg: str = "raw"           # statistic for the Bar plot
    nbins: int = 30            # bins for the Histogram plot


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def _columns(df: pl.DataFrame) -> set[str]:
    return set(df.columns)


def _require(df: pl.DataFrame, col: str | None, role: str) -> None:
    if col is None or col == _ODD:
        raise ValueError(f"{role} column is required")
    if col not in _columns(df):
        raise ValueError(f"{role} column {col!r} not found")


def _groups(df: pl.DataFrame, color: str | None) -> list[tuple[str | None, pl.DataFrame]]:
    """Yield ``(label, sub-frame)`` per unique value of ``color`` (or one group
    covering the whole frame when ``color`` is unset). First-appearance order."""
    if color is None or color not in _columns(df):
        return [(None, df)]
    uniq = df[color].unique(maintain_order=True).to_list()
    out: list[tuple[str | None, pl.DataFrame]] = []
    for v in uniq:
        if v is None:
            sub = df.filter(pl.col(color).is_null())
            label = "null"
        else:
            sub = df.filter(pl.col(color) == v)
            label = str(v)
        out.append((label, sub))
    return out


def _agg_expr(y: str | None, agg: str):
    """Build the polars aggregation expression, aliased to ``_ds_val``."""
    if agg == "count":
        return pl.len().alias("_ds_val")
    if y is None or y == _ODD:
        raise ValueError(f"{agg} aggregation needs a Y column")
    col = pl.col(y)
    ops = {
        "mean": col.mean,
        "sum": col.sum,
        "min": col.min,
        "max": col.max,
        "median": col.median,
    }
    if agg not in ops:
        raise ValueError(f"unknown statistic {agg!r}")
    return ops[agg]().alias("_ds_val")


def _figure(
    traces: list[dict], spec: PlotSpec, *,
    xaxis: str | None = None, yaxis: str | None = None,
    barmode: str | None = None,
) -> dict:
    layout: dict = {
        "title": {"text": _title(spec)},
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        "legend": {"orientation": "h"},
    }
    if xaxis:
        layout["xaxis"] = {"title": xaxis}
    if yaxis:
        layout["yaxis"] = {"title": yaxis}
    if barmode:
        layout["barmode"] = barmode
    return {"data": traces, "layout": layout}


def _title(spec: PlotSpec) -> str:
    pt = next((lbl for lbl, key in PLOT_TYPES if key == spec.plot_type), spec.plot_type)
    what = spec.y if spec.y and spec.plot_type != "histogram" else (spec.x or "")
    if spec.plot_type == "bar" and spec.agg not in ("raw", _ODD, ""):
        what = f"{spec.agg} of {spec.y or 'rows'}"
    by = f" by {spec.color}" if spec.color else ""
    return f"{pt}{': ' if what else ''}{what}{by}"


# --------------------------------------------------------------------------- #
# per-type builders
# --------------------------------------------------------------------------- #
def _scatter_line(df: pl.DataFrame, spec: PlotSpec, pt: str) -> dict:
    _require(df, spec.x, "X")
    _require(df, spec.y, "Y")
    traces = []
    for label, sub in _groups(df, spec.color):
        traces.append({
            "type": "scatter",
            "mode": "markers" if pt == "scatter" else "lines",
            "name": label or spec.y,
            "x": sub[spec.x].to_numpy(),
            "y": sub[spec.y].to_numpy(),
        })
    return _figure(traces, spec, xaxis=spec.x, yaxis=spec.y)


def _bar(df: pl.DataFrame, spec: PlotSpec) -> dict:
    _require(df, spec.x, "X")
    agg = spec.agg or "raw"
    if agg not in {"raw", *_AGG_OPS}:
        raise ValueError(f"unknown statistic {agg!r}")
    if agg == "raw":
        _require(df, spec.y, "Y")
        traces = []
        for label, sub in _groups(df, spec.color):
            traces.append({
                "type": "bar",
                "name": label or spec.y,
                "x": sub[spec.x].to_numpy(),
                "y": sub[spec.y].to_numpy(),
            })
        return _figure(traces, spec, xaxis=spec.x, yaxis=spec.y, barmode="group")

    # aggregated: one bar per distinct X (per group when color is set)
    # (de-dup: a color column equal to X would yield a DuplicateError)
    group_cols: list[str] = [spec.x]
    if spec.color and spec.color != spec.x:
        group_cols.append(spec.color)
    gdf = df.group_by(group_cols).agg(_agg_expr(spec.y, agg)).sort(spec.x)
    traces = []
    if spec.color:
        for label, sub in _groups(gdf, spec.color):
            traces.append({
                "type": "bar",
                "name": label,
                "x": sub[spec.x].to_numpy(),
                "y": sub["_ds_val"].to_numpy(),
            })
    else:
        traces.append({
            "type": "bar",
            "name": (spec.y if spec.y and agg != "count" else "count"),
            "x": gdf[spec.x].to_numpy(),
            "y": gdf["_ds_val"].to_numpy(),
        })
    y_label = "count" if agg == "count" else f"{agg} of {spec.y}"
    return _figure(traces, spec, xaxis=spec.x, yaxis=y_label, barmode="group")


def _histogram(df: pl.DataFrame, spec: PlotSpec) -> dict:
    _require(df, spec.x, "X")
    nbins = max(1, int(spec.nbins or 30))
    traces = []
    for label, sub in _groups(df, spec.color):
        traces.append({
            "type": "histogram",
            "name": label or spec.x,
            "x": sub[spec.x].to_numpy(),
            "nbinsx": nbins,
            "marker": {"opacity": 0.6} if spec.color else {},
        })
    return _figure(traces, spec, xaxis=spec.x, yaxis="count", barmode="overlay")


def _box_violin(df: pl.DataFrame, spec: PlotSpec, kind: str) -> dict:
    _require(df, spec.y, "Y")
    group_col = spec.color or spec.x
    traces = []
    for label, sub in _groups(df, group_col):
        traces.append({
            "type": kind,
            "name": label or spec.y,
            "y": sub[spec.y].to_numpy(),
        })
    return _figure(traces, spec, yaxis=spec.y)


def build_figure(df: pl.DataFrame, spec: PlotSpec) -> dict:
    """Build the declarative Plotly figure ``dict`` for ``df`` per ``spec``.

    Raises ``ValueError`` with a user-facing message when required columns are
    missing or the frame has no rows.
    """
    if df.height == 0:
        raise ValueError("no rows to plot")
    pt = spec.plot_type
    if pt in ("scatter", "line"):
        return _scatter_line(df, spec, pt)
    if pt == "bar":
        return _bar(df, spec)
    if pt == "histogram":
        return _histogram(df, spec)
    if pt == "box":
        return _box_violin(df, spec, "box")
    if pt == "violin":
        return _box_violin(df, spec, "violin")
    raise ValueError(f"unknown plot type {pt!r}")


# --------------------------------------------------------------------------- #
# UI panel
# --------------------------------------------------------------------------- #
class PlotPanel:
    """Selectors + render slot for a Plotly figure."""

    def __init__(
        self,
        parent,
        columns: list[tuple[str, pl.DataType]],
        on_plot: Callable[[], None],
    ) -> None:
        self._on_plot = on_plot
        names = [n for n, _ in columns]
        opts = {n: n for n in names} or {_ODD: _ODD}
        numeric = [n for n, d in columns if _is_numeric(d)]
        self._numeric = {n: n for n in numeric}

        with parent:
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                self.type_select = ui.select(
                    options={k: lbl for lbl, k in PLOT_TYPES},
                    value="scatter", label="Plot type",
                    on_change=lambda _e: self._refresh_and_plot(),
                ).props("dense outlined").classes("w-44")
                self.scope_select = ui.select(
                    {
                        "selection": "Current selection (filtered rows)",
                        "dataset": "Entire dataset (all rows)",
                    },
                    value="selection", label="Data scope",
                    on_change=lambda _e: on_plot(),
                ).props("dense outlined").classes("w-64")
                self.x_select = ui.select(
                    options=opts, value=None, label="X column",
                    on_change=lambda _e: on_plot(),
                ).props("dense outlined").classes("w-40")
                self.y_select = ui.select(
                    options=opts, value=None, label="Y column",
                    on_change=lambda _e: on_plot(),
                ).props("dense outlined").classes("w-40")
                self.color_select = ui.select(
                    options=opts, value=None, label="Color / group by",
                    clearable=True, on_change=lambda _e: on_plot(),
                ).props("dense outlined").classes("w-40")
                self.agg_select = ui.select(
                    options={k: lbl for lbl, k in AGG_STATS},
                    value="raw", label="Statistic (bar)",
                    on_change=lambda _e: on_plot(),
                ).props("dense outlined").classes("w-44")
                self.bins_input = ui.number(
                    value=30, min=1, max=500, label="Bins (histogram)",
                    on_change=lambda _e: on_plot(),
                ).props("dense outlined").classes("w-28")
                ui.button("Plot", icon="show_chart", on_click=lambda _=None: on_plot()) \
                    .props("unelevated color=primary dense")
            self.meta = ui.label("").classes("text-xs opacity-50")
            self.plot_container = ui.column().classes("w-full")
        self._refresh()

    # ---- internal -----------------------------------------------------------
    def _refresh(self) -> None:
        pt = self.type_select.value
        self.agg_select.set_visibility(pt == "bar")
        self.bins_input.set_visibility(pt == "histogram")

    def _refresh_and_plot(self) -> None:
        self._refresh()
        self._on_plot()

    # ---- read-only spec accessors ------------------------------------------
    @property
    def plot_type(self) -> str:
        return self.type_select.value

    @property
    def scope(self) -> str:
        return self.scope_select.value

    @property
    def x(self) -> str | None:
        v = self.x_select.value
        return None if v in (None, _ODD) else v

    @property
    def y(self) -> str | None:
        v = self.y_select.value
        return None if v in (None, _ODD) else v

    @property
    def color(self) -> str | None:
        v = self.color_select.value
        return None if v in (None, _ODD) else v

    @property
    def agg(self) -> str:
        return self.agg_select.value or "raw"

    @property
    def nbins(self) -> int:
        v = self.bins_input.value
        return int(v) if v else 30

    @property
    def spec(self) -> PlotSpec:
        return PlotSpec(
            plot_type=self.plot_type, x=self.x, y=self.y,
            color=self.color, agg=self.agg, nbins=self.nbins,
        )

    # ---- output slots -------------------------------------------------------
    def set_meta(self, text: str) -> None:
        self.meta.set_text(text)

    def render_plot(self, fig: dict) -> None:
        self.plot_container.clear()
        with self.plot_container:
            ui.plotly(fig).classes("w-full h-96")

    def render_error(self, msg: str) -> None:
        self.plot_container.clear()
        with self.plot_container:
            ui.label(msg).classes("text-sm text-negative")