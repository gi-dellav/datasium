"""Row / column removal component for the active dataset.

Renders a column multi-select plus a row-removal mode toggle with three
strategies (remove rows matching a value, remove null rows, remove the rows
given by the active Select-tab selection) and turns the current state into a
new Polars ``LazyFrame`` via :func:`apply_removal`.

The pure helpers (:func:`remove_columns`, :func:`remove_rows_by_value`,
:func:`remove_nulls`, :func:`apply_removal`) and the :class:`RemovalSpec`
dataclass are UI-free so they can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import polars as pl

from nicegui import ui

from datasium.filter import _OPERATORS, _NULLARY, _dtype_group, build_term

RowMode = Literal["none", "values", "nulls", "selection"]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
@dataclass
class RemovalSpec:
    """Declarative description of a removal operation."""

    row_mode: RowMode = "none"
    columns: list[str] = field(default_factory=list)
    value_column: str | None = None
    value_op: str | None = None
    value_raw: str = ""
    null_subset: list[str] | None = None  # None => every column
    selection_expr: pl.Expr | None = None


def _missing_columns(lf: pl.LazyFrame, cols: list[str]) -> list[str]:
    names = set(lf.collect_schema().names())
    return [c for c in cols if c not in names]


def remove_columns(lf: pl.LazyFrame, cols: list[str]) -> pl.LazyFrame:
    """Drop ``cols`` from ``lf``. No-op when empty; raises on unknown columns."""
    if not cols:
        return lf
    missing = _missing_columns(lf, cols)
    if missing:
        raise ValueError(f"column(s) not found: {', '.join(missing)}")
    return lf.drop(cols)


def remove_rows_by_value(
    lf: pl.LazyFrame, column: str | None, op: str | None, raw: str,
) -> pl.LazyFrame:
    """Remove rows where ``column`` matches the value term (built via filter.build_term)."""
    if not column:
        raise ValueError("select a column to remove rows from")
    schema = dict(lf.collect_schema().items())
    if column not in schema:
        raise ValueError(f"column {column!r} not found")
    expr = build_term(column, op, raw, schema[column])  # rows to remove
    return lf.filter(~expr)


def remove_nulls(
    lf: pl.LazyFrame, subset: list[str] | None = None,
) -> pl.LazyFrame:
    """Drop rows that are null in ``subset`` (or in any column when ``None``/empty)."""
    if not subset:
        return lf.drop_nulls()
    missing = _missing_columns(lf, subset)
    if missing:
        raise ValueError(f"column(s) not found: {', '.join(missing)}")
    return lf.drop_nulls(subset=subset)


def apply_removal(lf: pl.LazyFrame, spec: RemovalSpec) -> pl.LazyFrame:
    """Apply ``spec`` to ``lf`` and return the resulting LazyFrame.

    Row removal runs first so value/selection filters see the full schema;
    column drop runs afterwards. Raises ``ValueError`` with user-facing
    messages on invalid input.
    """
    mode = spec.row_mode
    if mode == "values":
        lf = remove_rows_by_value(lf, spec.value_column, spec.value_op, spec.value_raw)
    elif mode == "nulls":
        lf = remove_nulls(lf, spec.null_subset)
    elif mode == "selection":
        if spec.selection_expr is None:
            raise ValueError("no row selection is active in the Select tab")
        lf = lf.filter(~spec.selection_expr)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"unknown row mode {mode!r}")

    if spec.columns:
        lf = remove_columns(lf, spec.columns)
    return lf


# ---------------------------------------------------------------------------
# UI component
# ---------------------------------------------------------------------------
def _dtype_of(columns: list[tuple[str, pl.DataType]], name: str | None) -> pl.DataType | None:
    for n, d in columns:
        if n == name:
            return d
    return None


class RemovePanel:
    """Column multi-select + row-mode toggle with per-mode inputs."""

    def __init__(
        self,
        parent,
        columns: list[tuple[str, pl.DataType]],
        on_preview: Callable[[], None],
        on_apply: Callable[[], None],
    ) -> None:
        self._columns = columns
        self._on_preview = on_preview
        self._on_apply = on_apply
        self._selection_expr_fn: Callable[[], pl.Expr | None] | None = None

        self.col_names = [n for n, _ in columns]

        with parent:
            # ---- Remove columns ----
            ui.label("Remove columns").classes("text-lg font-medium mt-2")
            ui.label("Pick one or more columns to drop from the dataset.").classes(
                "text-xs opacity-50")
            self.col_select = ui.select(
                options={n: n for n in self.col_names} or {"—": "—"},
                multiple=True,
                value=[],
                clearable=True,
                label="Columns",
            ).props("dense outlined use-chips").classes("w-full")

            ui.separator()

            # ---- Remove rows ----
            ui.label("Remove rows").classes("text-lg font-medium mt-2")
            ui.label(
                "Choose a strategy; rows matching the rule are removed."
            ).classes("text-xs opacity-50")
            self.mode_toggle = ui.toggle(
                {
                    "none": "None",
                    "values": "Certain values",
                    "nulls": "Null values",
                    "selection": "Given selection",
                },
                value="none",
                on_change=lambda _e: self._refresh_mode(),
            ).props("dense")

            self.mode_container = ui.column().classes("w-full gap-2 mt-1")
            self._refresh_mode()

            ui.separator()

            with ui.row().classes("items-center gap-2 mt-2"):
                ui.button(
                    "Preview", icon="visibility",
                    on_click=lambda _=None: on_preview(),
                ).props("dense unelevated color=primary")
                ui.button(
                    "Apply", icon="delete_forever",
                    on_click=lambda _=None: on_apply(),
                ).props("dense unelevated color=negative")

    # ---- mode switching -------------------------------------------------
    def _refresh_mode(self) -> None:
        self.mode_container.clear()
        mode = self.mode_toggle.value or "none"
        with self.mode_container:
            if mode == "none":
                ui.label("No rows will be removed.").classes("text-sm opacity-50")
            elif mode == "values":
                self._build_values_inputs()
            elif mode == "nulls":
                self._build_nulls_inputs()
            elif mode == "selection":
                ui.label(
                    "Removes the rows matched by the filters defined in the "
                    "Select tab (every row when none are set)."
                ).classes("text-sm opacity-50")

    def _build_values_inputs(self) -> None:
        with ui.row().classes("items-center gap-2 w-full"):
            self.val_col = ui.select(
                options={n: n for n in self.col_names} or {"—": "—"},
                value=self.col_names[0] if self.col_names else None,
                label="Column",
                on_change=lambda _e: self._refresh_value_ops(),
            ).props("dense outlined").classes("w-40")
            self.val_op = ui.select(
                options=[], value=None, label="Remove rows where",
                on_change=lambda _e: self._refresh_value_vis(),
            ).props("dense outlined").classes("w-40")
            self.val_raw = ui.input(
                value="", label="Value",
            ).props("dense outlined").classes("w-32")
        self._refresh_value_ops()

    def _refresh_value_ops(self) -> None:
        if not hasattr(self, "val_op"):
            return
        dtype = _dtype_of(self._columns, self.val_col.value)
        group = _dtype_group(dtype) if dtype is not None else "other"
        ops = _OPERATORS[group]
        self.val_op.options = {k: lbl for lbl, k in ops}  # type: ignore[assignment]
        self.val_op.value = ops[0][1]
        self._refresh_value_vis()

    def _refresh_value_vis(self) -> None:
        if not hasattr(self, "val_raw"):
            return
        self.val_raw.set_visibility(self.val_op.value not in _NULLARY)

    def _build_nulls_inputs(self) -> None:
        self.null_all = ui.switch(
            "All columns", value=True,
            on_change=lambda _e: self.null_subset.set_visibility(not self.null_all.value),
        ).props("dense")
        self.null_subset = ui.select(
            options={n: n for n in self.col_names} or {"—": "—"},
            multiple=True, value=[], clearable=True, label="Columns",
        ).props("dense outlined use-chips").classes("w-full")
        self.null_subset.set_visibility(False)

    # ---- public ---------------------------------------------------------
    @property
    def spec(self) -> RemovalSpec:
        mode = self.mode_toggle.value or "none"
        columns = list(self.col_select.value or [])

        value_column = value_op = None
        value_raw = ""
        null_subset: list[str] | None = None
        selection_expr = None

        if mode == "values" and hasattr(self, "val_col"):
            value_column = self.val_col.value
            value_op = self.val_op.value
            value_raw = self.val_raw.value or ""

        if mode == "nulls" and hasattr(self, "null_all"):
            if self.null_all.value:
                null_subset = None
            else:
                null_subset = list(self.null_subset.value or [])

        if mode == "selection":
            selection_expr = self._on_selection_expr()  # provided by App

        return RemovalSpec(
            row_mode=mode,  # type: ignore[arg-type]
            columns=columns,
            value_column=value_column,
            value_op=value_op,
            value_raw=value_raw,
            null_subset=null_subset,
            selection_expr=selection_expr,
        )

    def set_selection_expr_provider(self, fn: Callable[[], pl.Expr | None]) -> None:
        """Register the App-side provider of the active Select-tab filter expr."""
        self._selection_expr_fn = fn

    def _on_selection_expr(self) -> pl.Expr | None:
        if self._selection_expr_fn is not None:
            return self._selection_expr_fn()
        return None