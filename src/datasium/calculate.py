"""Row-calculation component for the filtered dataset.

Renders a column / operation / (optional) threshold selector and computes a
single statistic over the rows that survive the active :class:`FilterBuilder`
expression (or every row when no filter is set).

The pure :func:`compute_stat` helper is UI-free so it can be unit-tested.
"""

from __future__ import annotations

from typing import Callable

import polars as pl

from nicegui import ui


# (label, key) for the operation select.
_STATS: list[tuple[str, str]] = [
    ("average", "mean"),
    ("maximum", "max"),
    ("minimum", "min"),
    ("sum", "sum"),
    ("median", "median"),
    ("std dev", "std"),
    ("count (non-null)", "count"),
    ("count > X", "count_gt"),
    ("count >= X", "count_ge"),
    ("count < X", "count_lt"),
    ("count <= X", "count_le"),
    ("count == X", "count_eq"),
]

_THRESHOLD_OPS = {"count_gt", "count_ge", "count_lt", "count_le", "count_eq"}


def compute_stat(
    series: pl.Series,
    op: str,
    raw: str | None = None,
) -> float | int | None:
    """Compute a statistic over ``series``.

    Returns a number, or ``None`` when the result is null (e.g. mean of an
    empty series). Raises ``ValueError`` with a user-facing message on bad
    threshold input.
    """
    if op == "mean":
        return series.mean()
    if op == "max":
        return series.max()
    if op == "min":
        return series.min()
    if op == "sum":
        return series.sum()
    if op == "median":
        return series.median()
    if op == "std":
        return series.std()
    if op == "count":
        return int(series.len() - series.null_count())

    if op in _THRESHOLD_OPS:
        if raw is None or str(raw).strip() == "":
            raise ValueError("supply a threshold value")
        try:
            threshold = float(str(raw))
        except ValueError:
            raise ValueError(f"expected a number, got {raw!r}")
        non_null = series.drop_nulls()
        if op == "count_gt":
            return int((non_null > threshold).sum())
        if op == "count_ge":
            return int((non_null >= threshold).sum())
        if op == "count_lt":
            return int((non_null < threshold).sum())
        if op == "count_le":
            return int((non_null <= threshold).sum())
        if op == "count_eq":
            return int((non_null == threshold).sum())

    raise ValueError(f"unknown operation {op!r}")


def _is_numeric(dtype: pl.DataType) -> bool:
    base = dtype.base_type()
    numeric = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    return base in numeric


class Calculator:
    """A single column / operation / threshold calculator."""

    def __init__(
        self,
        parent,
        columns: list[tuple[str, pl.DataType]],
        on_calculate: Callable[[], None],
    ) -> None:
        self._columns = columns
        self._on_calculate = on_calculate
        self.result: float | int | None = None
        self.error: str | None = None

        numeric = [n for n, d in columns if _is_numeric(d)]
        with parent:
            with ui.row().classes("items-center gap-2 w-full"):
                self.col_select = (
                    ui.select(
                        options={n: n for n in numeric} or {"—": "—"},
                        value=numeric[0] if numeric else None,
                        label="Column",
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
                self.op_select = (
                    ui.select(
                        options={k: lbl for lbl, k in _STATS},
                        value="mean",
                        label="Operation",
                        on_change=self._on_op_change,
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
                self.value_input = (
                    ui.input(
                        value="",
                        label="Threshold X",
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )
                ui.button(
                    "Calculate",
                    icon="calculate",
                    on_click=lambda _=None: on_calculate(),
                ).props("dense unelevated color=primary")
            self.result_label = ui.label("").classes("text-base ds-mono opacity-80")
        self._refresh()

    def _on_op_change(self, _e) -> None:
        self._refresh()

    def _refresh(self) -> None:
        self.value_input.set_visibility(self.op_select.value in _THRESHOLD_OPS)

    @property
    def column(self) -> str | None:
        v = self.col_select.value
        return None if v in (None, "—") else v

    @property
    def operation(self) -> str | None:
        return self.op_select.value

    @property
    def threshold(self) -> str:
        return self.value_input.value or ""

    def set_result(self, value: float | int | None) -> None:
        self.result = value
        self.error = None
        if value is None:
            self.result_label.set_text("—")
            self.result_label.classes(replace="opacity-40")
        else:
            if isinstance(value, float):
                text = f"{value:.4g}"
            else:
                text = f"{value:,}"
            self.result_label.set_text(text)
            self.result_label.classes(replace="opacity-80")

    def set_error(self, msg: str) -> None:
        self.error = msg
        self.result = None
        self.result_label.set_text(msg)
        self.result_label.classes(replace="text-negative")
