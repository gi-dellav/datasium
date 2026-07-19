"""Edit component for the active dataset.

Covers four mutations on the active dataset's ``LazyFrame``:

* **cast** a column to a different dtype,
* **add a column** (name + dtype + optional fill value),
* **add a row** with typed per-column inputs,
* **edit a cell** — either by 1-based row number or by ``equal_to`` constraints
  on a set of key columns that select exactly one row.

The pure helpers (:func:`cast_column`, :func:`add_column`, :func:`add_row`,
:func:`set_cell_by_index`, :func:`set_cell_by_key`, :func:`count_matches`)
and the :class:`EditPanel` UI are kept self-contained. Helpers raise
``ValueError`` with user-facing messages on invalid input so they can be
unit-tested without a browser.
"""

from __future__ import annotations

from typing import Callable

import polars as pl

from nicegui import ui

from datasium.filter import _coerce


# ---------------------------------------------------------------------------
# Dtype choices offered across the panel.
# ---------------------------------------------------------------------------
DTYPE_CHOICES: list[tuple[str, str, pl.DataType]] = [
    ("Int64 (integer)", "int", pl.Int64),
    ("Float64 (decimal)", "float", pl.Float64),
    ("String (text)", "str", pl.String),
    ("Boolean (true/false)", "bool", pl.Boolean),
    ("Date (YYYY-MM-DD)", "date", pl.Date),
    ("Datetime", "datetime", pl.Datetime),
    ("Time", "time", pl.Time),
]
DTYPE_BY_KEY: dict[str, pl.DataType] = {k: dt for _lbl, k, dt in DTYPE_CHOICES}
DTYPE_LABELS: dict[str, str] = {k: lbl for lbl, k, _dt in DTYPE_CHOICES}


def _coerce_or_none(raw: str | None, dtype: pl.DataType | None) -> object:
    """Coerce ``raw`` to the dtype, returning ``None`` for blank input."""
    if raw is None or raw.strip() == "":
        return None
    return _coerce(raw, dtype)


def _schema(lf: pl.LazyFrame) -> dict[str, pl.DataType]:
    return dict(lf.collect_schema().items())


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def cast_column(lf: pl.LazyFrame, column: str, target: pl.DataType) -> pl.LazyFrame:
    """Cast ``column`` to ``target``. Raises if the column is unknown."""
    names = lf.collect_schema().names()
    if column not in names:
        raise ValueError(f"column {column!r} not found")
    return lf.with_columns(pl.col(column).cast(target))


def add_column(
    lf: pl.LazyFrame,
    name: str,
    target: pl.DataType,
    raw_fill: str,
) -> pl.LazyFrame:
    """Append a new column of ``target`` dtype, filled with ``raw_fill`` (or null)."""
    if not name or not name.strip():
        raise ValueError("enter a column name")
    if name in lf.collect_schema().names():
        raise ValueError(f"column {name!r} already exists")
    fill = _coerce_or_none(raw_fill, target)
    return lf.with_columns(pl.lit(fill).cast(target).alias(name))


def add_row(lf: pl.LazyFrame, values: dict[str, str]) -> pl.LazyFrame:
    """Append a single row built from per-column raw strings.

    Blank inputs become null. Raises if any column is missing from ``values``.
    """
    schema = _schema(lf)
    missing = [c for c in schema if c not in values]
    if missing:
        raise ValueError(f"missing inputs for: {', '.join(missing)}")
    row: dict[str, object] = {}
    for col, dtype in schema.items():
        row[col] = _coerce_or_none(values.get(col, ""), dtype)
    new_row = pl.DataFrame(row, schema=schema)
    return pl.concat([lf, new_row.lazy()], how="vertical")


def set_cell_by_index(
    lf: pl.LazyFrame,
    index: int,
    column: str,
    raw: str,
    dtype: pl.DataType,
) -> pl.LazyFrame:
    """Set ``column`` of the 0-based ``index``-th row to ``raw`` (coerced)."""
    height = lf.select(pl.len()).collect().item()
    if index < 0 or index >= height:
        raise ValueError(f"row number {index + 1} is out of range (1..{height})")
    names = lf.collect_schema().names()
    if column not in names:
        raise ValueError(f"column {column!r} not found")
    value = _coerce_or_none(raw, dtype)
    return lf.with_columns(
        pl.when(pl.int_range(0, pl.len()) == index)
        .then(pl.lit(value).cast(dtype))
        .otherwise(pl.col(column))
        .alias(column)
    )


def count_matches(
    lf: pl.LazyFrame,
    key_cols: list[str],
    key_vals: list[str],
) -> int:
    """Number of rows where every ``key_col`` equals its ``key_val`` (coerced)."""
    if not key_cols:
        raise ValueError("select at least one key column")
    schema = _schema(lf)
    expr = pl.lit(True)
    for k, v in zip(key_cols, key_vals):
        if k not in schema:
            raise ValueError(f"column {k!r} not found")
        kv = _coerce_or_none(v, schema[k])
        expr = expr & (pl.col(k) == kv)
    return int(lf.filter(expr).select(pl.len()).collect().item())


def set_cell_by_key(
    lf: pl.LazyFrame,
    key_cols: list[str],
    key_vals: list[str],
    column: str,
    raw: str,
    dtype: pl.DataType,
) -> pl.LazyFrame:
    """Set ``column`` on the unique row matched by the key constraints.

    Raises when zero or more than one row matches.
    """
    if not key_cols:
        raise ValueError("select at least one key column")
    schema = _schema(lf)
    if column not in schema:
        raise ValueError(f"column {column!r} not found")
    names = lf.collect_schema().names()
    key_expr = pl.lit(True)
    for k, v in zip(key_cols, key_vals):
        if k not in names:
            raise ValueError(f"column {k!r} not found")
        kv = _coerce_or_none(v, schema[k])
        key_expr = key_expr & (pl.col(k) == kv)
    count = int(lf.filter(key_expr).select(pl.len()).collect().item())
    if count == 0:
        raise ValueError("no rows match the given key constraints")
    if count > 1:
        raise ValueError(
            f"{count} rows match — refine the constraints to select exactly one row"
        )
    value = _coerce_or_none(raw, dtype)
    return lf.with_columns(
        pl.when(key_expr)
        .then(pl.lit(value).cast(dtype))
        .otherwise(pl.col(column))
        .alias(column)
    )


def fill_nulls(
    lf: pl.LazyFrame,
    column: str,
    strategy: str = "value",
    fill_value: str = "",
    dtype: pl.DataType | None = None,
) -> pl.LazyFrame:
    """Fill nulls in *column* using *strategy* (value / forward / backward / mean /
    median / mode / zero / min / max).

    Raises ``ValueError`` with a user-facing message on bad input.
    """
    names = lf.collect_schema().names()
    if column not in names:
        raise ValueError(f"column {column!r} not found")
    if dtype is None:
        dtype = dict(lf.collect_schema().items())[column]

    col = pl.col(column)

    if strategy == "value":
        if fill_value is None or str(fill_value).strip() == "":
            return lf  # nothing to fill with — no-op
        val = _coerce_or_none(fill_value, dtype)
        return lf.with_columns(col.fill_null(val))
    if strategy == "forward":
        return lf.with_columns(col.forward_fill())
    if strategy == "backward":
        return lf.with_columns(col.backward_fill())
    if strategy == "zero":
        return lf.with_columns(col.fill_null(0))
    if strategy == "min":
        return lf.with_columns(col.fill_null(col.min()))
    if strategy == "max":
        return lf.with_columns(col.fill_null(col.max()))
    if strategy == "mean":
        return lf.with_columns(col.fill_null(col.mean()))
    if strategy == "median":
        return lf.with_columns(col.fill_null(col.median()))
    if strategy == "mode":
        return lf.with_columns(col.fill_null(col.drop_nulls().mode().first()))

    raise ValueError(f"unknown fill strategy {strategy!r}")


def replace_values(
    lf: pl.LazyFrame,
    column: str,
    old_raw: str,
    new_raw: str,
    dtype: pl.DataType | None = None,
) -> pl.LazyFrame:
    """Replace every occurrence of *old_raw* with *new_raw* in *column*.

    Blank *old_raw* targets null cells.  Raises ``ValueError`` on
    unrecognised columns or unparseable literals.
    """
    names = lf.collect_schema().names()
    if column not in names:
        raise ValueError(f"column {column!r} not found")
    if dtype is None:
        dtype = dict(lf.collect_schema().items())[column]

    col = pl.col(column)
    if old_raw.strip() == "":
        new_val = _coerce_or_none(new_raw, dtype)
        return lf.with_columns(col.fill_null(new_val))

    old_val = _coerce_or_none(old_raw, dtype)
    new_val = _coerce_or_none(new_raw, dtype)
    return lf.with_columns(
        pl.when(col == old_val)
        .then(pl.lit(new_val).cast(dtype) if new_val is not None else pl.lit(None))
        .otherwise(col)
        .alias(column)
    )


# ---------------------------------------------------------------------------
# UI component
# ---------------------------------------------------------------------------
_EDIT_ROW_MODES = {
    "index": "By row number",
    "keys": "By matching columns",
}

_FILL_STRATEGIES = [
    ("Fill with value", "value"),
    ("Forward fill", "forward"),
    ("Backward fill", "backward"),
    ("Fill with zero", "zero"),
    ("Fill with min", "min"),
    ("Fill with max", "max"),
    ("Fill with mean", "mean"),
    ("Fill with median", "median"),
    ("Fill with mode", "mode"),
]


class EditPanel:
    """Four-section panel: cast, add column, add row, edit row."""

    def __init__(
        self,
        parent,
        columns: list[tuple[str, pl.DataType]],
        on_cast: Callable[[str, str], None],
        on_add_column: Callable[[str, str, str], None],
        on_add_row: Callable[[dict[str, str]], None],
        on_edit_row: Callable[[str, object, list[str], list[str], str, str], None],
        on_fill_nulls: Callable[[str, str, str], None],
        on_replace_values: Callable[[str, str, str], None],
    ) -> None:
        self._columns = columns
        self._col_names = [n for n, _ in columns]
        self._on_cast = on_cast
        self._on_add_column = on_add_column
        self._on_add_row = on_add_row
        self._on_edit_row = on_edit_row
        self._on_fill_nulls = on_fill_nulls
        self._on_replace_values = on_replace_values

        with parent:
            self._build_cast_section()
            ui.separator()
            self._build_add_column_section()
            ui.separator()
            self._build_add_row_section()
            ui.separator()
            self._build_edit_row_section()
            ui.separator()
            self._build_fill_nulls_section()
            ui.separator()
            self._build_replace_values_section()

    # ---- 1. cast --------------------------------------------------------
    def _build_cast_section(self) -> None:
        ui.label("Edit column type").classes("text-lg font-medium")
        ui.label("Cast a column to a different dtype.").classes("text-xs opacity-50")
        with ui.row().classes("items-center gap-2 w-full"):
            self.cast_col = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    value=self._col_names[0] if self._col_names else None,
                    label="Column",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.cast_dtype = (
                ui.select(
                    options={k: lbl for lbl, k, _ in DTYPE_CHOICES},
                    value="int",
                    label="Target type",
                )
                .props("dense outlined")
                .classes("w-48")
            )
            ui.button(
                "Cast",
                icon="swap_horiz",
                on_click=lambda _=None: self._on_cast(
                    self.cast_col.value, self.cast_dtype.value
                ),
            ).props("dense unelevated color=primary")

    # ---- 2. add column --------------------------------------------------
    def _build_add_column_section(self) -> None:
        ui.label("Add column").classes("text-lg font-medium mt-2")
        ui.label("Append a new column. Leave the fill value blank for nulls.").classes(
            "text-xs opacity-50"
        )
        with ui.row().classes("items-center gap-2 w-full"):
            self.add_col_name = (
                ui.input(
                    value="",
                    label="New column name",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.add_col_dtype = (
                ui.select(
                    options={k: lbl for lbl, k, _ in DTYPE_CHOICES},
                    value="int",
                    label="Type",
                )
                .props("dense outlined")
                .classes("w-48")
            )
            self.add_col_fill = (
                ui.input(
                    value="",
                    label="Fill value (optional)",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            ui.button(
                "Add column",
                icon="add",
                on_click=lambda _=None: self._on_add_column(
                    self.add_col_name.value or "",
                    self.add_col_dtype.value,
                    self.add_col_fill.value or "",
                ),
            ).props("dense unelevated color=primary")

    # ---- 3. add row ------------------------------------------------------
    def _build_add_row_section(self) -> None:
        ui.label("Add row").classes("text-lg font-medium mt-2")
        ui.label(
            "Fill in a value per column (blank = null) and append a new row."
        ).classes("text-xs opacity-50")
        self.add_row_inputs: dict[str, ui.input] = {}
        with ui.row().classes("items-start gap-2 w-full flex-wrap"):
            for name, dtype in self._columns:
                lbl = f"{name} · {dtype}"
                self.add_row_inputs[name] = (
                    ui.input(
                        value="",
                        label=lbl,
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
        ui.button(
            "Add row",
            icon="add_box",
            on_click=lambda _=None: self._on_add_row(
                {n: (i.value or "") for n, i in self.add_row_inputs.items()}
            ),
        ).props("dense unelevated color=primary").classes("mt-2")

    # ---- 4. edit row -----------------------------------------------------
    def _build_edit_row_section(self) -> None:
        ui.label("Edit a cell").classes("text-lg font-medium mt-2")
        ui.label(
            "Pick the target row either by its 1-based number or by equal-to "
            "constraints that select exactly one row, then set the new value."
        ).classes("text-xs opacity-50")

        with ui.row().classes("items-center gap-2 w-full"):
            ui.label("Select row by").classes("text-sm opacity-70 w-28")
            self.row_mode = ui.toggle(
                _EDIT_ROW_MODES,
                value="index",
                on_change=lambda _e: self._refresh_row_mode(),
            ).props("dense")

        self.row_mode_box = ui.column().classes("w-full gap-2 mt-1")
        self._refresh_row_mode()

        ui.separator().classes("my-2")
        with ui.row().classes("items-center gap-2 w-full"):
            self.edit_col = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    value=self._col_names[0] if self._col_names else None,
                    label="Column to set",
                    on_change=lambda _e: self._refresh_edit_col_hint(),
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.edit_val = (
                ui.input(
                    value="",
                    label="New value",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            ui.button(
                "Apply edit",
                icon="check",
                on_click=lambda _=None: self._submit_edit(),
            ).props("dense unelevated color=primary")
        self.edit_col_hint = ui.label("").classes("text-xs opacity-50")
        self._refresh_edit_col_hint()

    def _refresh_row_mode(self) -> None:
        self.row_mode_box.clear()
        mode = self.row_mode.value or "index"
        with self.row_mode_box:
            if mode == "index":
                self.row_index = (
                    ui.number(
                        value=1,
                        label="Row number (1-based)",
                        min=1,
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
            else:
                self.key_cols = (
                    ui.select(
                        options={n: n for n in self._col_names} or {"—": "—"},
                        multiple=True,
                        value=[],
                        clearable=True,
                        label="Key columns",
                        on_change=lambda _e: self._refresh_key_inputs(),
                    )
                    .props("dense outlined use-chips")
                    .classes("w-full")
                )
                self.key_inputs_box = ui.column().classes("w-full gap-1 mt-1")
                self._refresh_key_inputs()

    def _refresh_key_inputs(self) -> None:
        if not hasattr(self, "key_inputs_box"):
            return
        self.key_inputs_box.clear()
        cols = list(self.key_cols.value or [])
        self.key_value_inputs: dict[str, ui.input] = {}
        if not cols:
            with self.key_inputs_box:
                ui.label("Select one or more key columns.").classes(
                    "text-sm opacity-50"
                )
            return
        with self.key_inputs_box:
            with ui.row().classes("items-start gap-2 flex-wrap"):
                for name in cols:
                    dtype = next((d for n, d in self._columns if n == name), None)
                    lbl = f"{name} == (value)"
                    if dtype is not None:
                        lbl = f"{name} ({dtype})"
                    self.key_value_inputs[name] = (
                        ui.input(
                            value="",
                            label=lbl,
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )

    def _refresh_edit_col_hint(self) -> None:
        name = self.edit_col.value
        dtype = next((d for n, d in self._columns if n == name), None)
        group_hint = ""
        if dtype is not None:
            group_hint = f" · {dtype}"
        self.edit_col_hint.set_text(f"Target dtype:{group_hint}")

    def _submit_edit(self) -> None:
        mode = self.row_mode.value or "index"
        if mode == "index":
            idx = int(self.row_index.value) if self.row_index.value else 1
            self._on_edit_row(
                "index",
                idx - 1,
                [],
                [],
                self.edit_col.value,
                self.edit_val.value or "",
            )
        else:
            cols = list(self.key_cols.value or [])
            if not cols:
                ui.notify(
                    "select at least one key column", type="warning", position="top"
                )
                return
            vals = [self.key_value_inputs.get(c).value or "" for c in cols]  # type: ignore[union-attr]
            self._on_edit_row(
                "keys",
                None,
                cols,
                vals,
                self.edit_col.value,
                self.edit_val.value or "",
            )

    # ---- 5. fill nulls ---------------------------------------------------
    def _build_fill_nulls_section(self) -> None:
        ui.label("Fill nulls").classes("text-lg font-medium mt-2")
        ui.label("Replace null values in a column using a strategy.").classes(
            "text-xs opacity-50"
        )
        with ui.row().classes("items-center gap-2 w-full"):
            self.fill_col = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    value=self._col_names[0] if self._col_names else None,
                    label="Column",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.fill_strategy = (
                ui.select(
                    options={k: lbl for lbl, k in _FILL_STRATEGIES},
                    value="value",
                    label="Strategy",
                    on_change=lambda _e: self._refresh_fill_vis(),
                )
                .props("dense outlined")
                .classes("w-48")
            )
            self.fill_value = (
                ui.input(
                    value="",
                    label="Fill value",
                )
                .props("dense outlined")
                .classes("w-32")
            )
            ui.button(
                "Fill nulls",
                icon="water_drop",
                on_click=lambda _=None: self._on_fill_nulls(
                    self.fill_col.value or "",
                    self.fill_strategy.value or "value",
                    self.fill_value.value or "",
                ),
            ).props("dense unelevated color=primary")
        self._refresh_fill_vis()

    def _refresh_fill_vis(self) -> None:
        strat = self.fill_strategy.value if hasattr(self, "fill_strategy") else "value"
        self.fill_value.set_visibility(strat == "value")

    # ---- 6. replace values ------------------------------------------------
    def _build_replace_values_section(self) -> None:
        ui.label("Replace values").classes("text-lg font-medium mt-2")
        ui.label(
            "Replace every occurrence of a value in a column. "
            "Leave 'Old value' blank to target null cells."
        ).classes("text-xs opacity-50")
        with ui.row().classes("items-center gap-2 w-full"):
            self.repl_col = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    value=self._col_names[0] if self._col_names else None,
                    label="Column",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.repl_old = (
                ui.input(
                    value="",
                    label="Old value (blank for null)",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.repl_new = (
                ui.input(
                    value="",
                    label="New value",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            ui.button(
                "Replace",
                icon="find_replace",
                on_click=lambda _=None: self._on_replace_values(
                    self.repl_col.value or "",
                    self.repl_old.value or "",
                    self.repl_new.value or "",
                ),
            ).props("dense unelevated color=primary")
