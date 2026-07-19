"""Reusable filter-builder component for Polars ``df.filter`` expressions.

Renders a stack of rows (column / operator / value) plus a combinator
(match all = AND, match any = OR) and produces a single ``pl.Expr`` (or
``None`` when no rows are present) ready to hand to ``LazyFrame.filter``.

The component is self-contained: construct it with the active schema and a
parent container, then call :meth:`build_expr` to compile the current state.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Callable

import polars as pl

from nicegui import ui


def _dtype_group(dtype: pl.DataType) -> str:
    """Coarse classification of a Polars dtype for operator selection."""
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
    if base in numeric:
        return "numeric"
    if base == pl.String:
        return "string"
    if base == pl.Boolean:
        return "boolean"
    if base in (pl.Date, pl.Datetime, pl.Time):
        return "temporal"
    return "other"


# (label, key) per group. The key is interpreted by _build_term.
_OPERATORS: dict[str, list[tuple[str, str]]] = {
    "numeric": [
        ("equals", "eq"),
        ("not equals", "ne"),
        (">", "gt"),
        (">=", "ge"),
        ("<", "lt"),
        ("<=", "le"),
        ("between", "between"),
        ("not between", "not_between"),
        ("is in", "is_in"),
        ("is null", "is_null"),
        ("is not null", "is_not_null"),
        ("is NaN", "is_nan"),
        ("is not NaN", "is_not_nan"),
    ],
    "string": [
        ("equals", "eq"),
        ("not equals", "ne"),
        ("contains", "contains"),
        ("not contains", "not_contains"),
        ("starts with", "starts_with"),
        ("ends with", "ends_with"),
        ("matches regex", "regex"),
        ("is in", "is_in"),
        ("length equals", "str_len_eq"),
        ("length >", "str_len_gt"),
        ("length <", "str_len_lt"),
        ("is null", "is_null"),
        ("is not null", "is_not_null"),
    ],
    "boolean": [
        ("is true", "is_true"),
        ("is false", "is_false"),
        ("is null", "is_null"),
        ("is not null", "is_not_null"),
    ],
    "temporal": [
        ("equals", "eq"),
        ("not equals", "ne"),
        (">", "gt"),
        (">=", "ge"),
        ("<", "lt"),
        ("<=", "le"),
        ("between", "between"),
        ("not between", "not_between"),
        ("is null", "is_null"),
        ("is not null", "is_not_null"),
    ],
    "other": [
        ("equals", "eq"),
        ("not equals", "ne"),
        ("is null", "is_null"),
        ("is not null", "is_not_null"),
    ],
}

_NULLARY = {"is_null", "is_not_null", "is_true", "is_false", "is_nan", "is_not_nan"}
_RANGE_OPS = {"between", "not_between"}


class FilterRow:
    """A single column / operator / value row."""

    def __init__(
        self,
        parent,
        columns: list[tuple[str, pl.DataType]],
        on_change: Callable[[], None],
        on_remove: Callable[["FilterRow"], None],
    ) -> None:
        self._columns = columns
        self._on_change = on_change
        self._col_dtype: pl.DataType | None = None
        self._col_name: str | None = None
        self._op_key: str | None = None

        with parent:
            with ui.row().classes("items-center gap-2 w-full"):
                self.col_select = (
                    ui.select(
                        options={n: n for n, _ in columns} or {"—": "—"},
                        value=columns[0][0] if columns else None,
                        on_change=self._on_col_change,
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
                self.op_select = (
                    ui.select(
                        options=[],
                        value=None,
                        on_change=self._on_op_change,
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )
                self.value_input = (
                    ui.input(
                        value="",
                        on_change=lambda _e: on_change(),
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
                ui.button(
                    icon="close", on_click=lambda _=None, r=self: on_remove(r)
                ).props("flat round dense color=negative").tooltip("Remove filter")

        # Initialise operator list for the default column.
        self._refresh_operators()

    # ---- internal -------------------------------------------------------
    def _current_dtype(self) -> pl.DataType | None:
        name = self.col_select.value
        for n, d in self._columns:
            if n == name:
                return d
        return None

    def _refresh_operators(self) -> None:
        dtype = self._current_dtype()
        group = _dtype_group(dtype) if dtype is not None else "other"
        ops = _OPERATORS[group]
        self.op_select.options = {k: lbl for lbl, k in ops}  # type: ignore[assignment]
        self.op_select.value = ops[0][1]
        self._update_value_visibility()

    def _on_col_change(self, _e) -> None:
        self._refresh_operators()
        self._on_change()

    def _on_op_change(self, _e) -> None:
        self._update_value_visibility()
        self._on_change()

    def _update_value_visibility(self) -> None:
        key = self.op_select.value
        self.value_input.set_visibility(key not in _NULLARY)
        if key in _RANGE_OPS:
            self.value_input.props('placeholder="lo, hi"')
        elif key == "is_in":
            self.value_input.props('placeholder="val1, val2, …"')
        else:
            self.value_input.props('placeholder=""')

    # ---- public ---------------------------------------------------------
    def build_term(self) -> pl.Expr:
        """Compile this row into a Polars expression.

        Raises ``ValueError`` with a user-facing message on bad input.
        """
        name = self.col_select.value
        if name is None or name == "—":
            raise ValueError("select a column")
        dtype = self._current_dtype()
        return build_term(name, self.op_select.value, self.value_input.value, dtype)


def build_term(
    name: str,
    op: str | None,
    raw: str,
    dtype: pl.DataType | None,
) -> pl.Expr:
    """Compile a single column/operator/value row into a Polars expression.

    Pure (UI-free) so it can be unit-tested. Raises ``ValueError`` on bad
    input with a user-facing message.
    """
    if not name:
        raise ValueError("select a column")
    if op is None:
        raise ValueError("select an operator")
    col = pl.col(name)
    if op == "is_null":
        return col.is_null()
    if op == "is_not_null":
        return col.is_not_null()
    if op == "is_true":
        return col
    if op == "is_false":
        return col.not_()
    if op == "is_nan":
        return col.is_nan()
    if op == "is_not_nan":
        return col.is_not_nan()

    group = _dtype_group(dtype) if dtype is not None else "other"

    if op in _RANGE_OPS:
        parts = [v.strip() for v in (raw or "").split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"{name}: 'between' needs two comma-separated values, e.g. '10, 50'"
            )
        lo = _coerce(parts[0], dtype)
        hi = _coerce(parts[1], dtype)
        expr = col.is_between(lo, hi, closed="both")
        return expr if op == "between" else ~expr

    if op == "is_in":
        values = [v.strip() for v in (raw or "").split(",") if v.strip()]
        if not values:
            raise ValueError(f"{name}: supply at least one value for 'is in'")
        coerced = [_coerce(v, dtype) for v in values]
        return col.is_in(coerced)

    if op in ("str_len_eq", "str_len_gt", "str_len_lt"):
        try:
            length = int(raw)
        except (ValueError, TypeError):
            raise ValueError(f"{name}: expected an integer length, got {raw!r}")
        str_len = col.str.len_chars()
        if op == "str_len_eq":
            return str_len == length
        if op == "str_len_gt":
            return str_len > length
        return str_len < length

    literal = _coerce(raw, dtype) if group != "other" else (raw or "")
    if op == "eq":
        return col == literal
    if op == "ne":
        return col != literal
    if op == "gt":
        return col > literal
    if op == "ge":
        return col >= literal
    if op == "lt":
        return col < literal
    if op == "le":
        return col <= literal
    if op == "contains":
        return col.str.contains(raw or "", literal=True)
    if op == "not_contains":
        return ~col.str.contains(raw or "", literal=True)
    if op == "starts_with":
        return col.str.starts_with(raw or "")
    if op == "ends_with":
        return col.str.ends_with(raw or "")
    if op == "regex":
        if not raw:
            raise ValueError(f"{name}: supply a regex pattern")
        return col.str.contains(raw, literal=False)
    raise ValueError(f"unknown operator {op!r}")


class FilterBuilder:
    """A stack of :class:`FilterRow` rows combined with AND/OR."""

    def __init__(
        self,
        columns: list[tuple[str, pl.DataType]],
        container: ui.column,
        *,
        combinator: str = "all",
    ) -> None:
        self._columns = columns
        self._combinator = combinator  # "all" -> AND, "any" -> OR
        self._rows: list[FilterRow] = []
        self._container = container
        self._on_change: Callable[[], None] | None = None

        with container:
            with ui.row().classes("items-center gap-2 w-full"):
                self._combinator_select = (
                    ui.toggle(
                        {"all": "Match ALL (and)", "any": "Match ANY (or)"},
                        value=combinator,
                    )
                    .props("dense")
                    .tooltip("How rows are combined")
                )
                ui.space()
                ui.button(
                    "Add filter", icon="add", on_click=lambda _=None: self.add_row()
                ).props("dense unelevated color=primary")
                ui.button(
                    "Clear", icon="delete_sweep", on_click=lambda _=None: self.clear()
                ).props("dense flat color=negative")
            self.rows_box = ui.column().classes("w-full gap-1 mt-1")

        self.add_row()

    # ------------------------------------------------------------------
    def on_change(self, cb: Callable[[], None]) -> None:
        self._on_change = cb

    @property
    def combinator(self) -> str:
        return self._combinator_select.value or "all"

    @property
    def rows(self) -> list[FilterRow]:
        return list(self._rows)

    def add_row(self) -> None:
        row = FilterRow(
            self.rows_box,
            self._columns,
            self._notify,
            self._remove_row,
        )
        self._rows.append(row)

    def _remove_row(self, row: FilterRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            row.col_select.delete()
            row.op_select.delete()
            row.value_input.delete()
            self._notify()

    def clear(self) -> None:
        for row in list(self._rows):
            self._remove_row(row)
        self.add_row()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()

    # ------------------------------------------------------------------
    def build_expr(self) -> pl.Expr | None:
        """Compile all rows into one expression, ``None`` if no active rows.

        Raises ``ValueError`` on invalid value input.
        """
        if not self._rows:
            return None
        # Only rows with a genuine column selection count; a single empty
        # row (e.g. after Clear) yields None.
        terms = [r.build_term() for r in self._rows]
        if not terms:
            return None
        acc = terms[0]
        for t in terms[1:]:
            acc = acc & t if self.combinator == "all" else acc | t
        return acc


# ---------------------------------------------------------------------------
def _coerce(raw: str, dtype: pl.DataType | None) -> object:
    """Coerce a raw string literal to a Python value matching ``dtype``."""
    group = _dtype_group(dtype) if dtype is not None else "other"
    if group == "numeric":
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                raise ValueError(f"expected a number, got {raw!r}")
    if group == "boolean":
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"expected true/false, got {raw!r}")
    if group == "temporal":
        base = dtype.base_type() if dtype is not None else None  # type: ignore[attr-defined]
        if base == pl.Date:
            parser: Callable[[str], object] = date.fromisoformat
        elif base == pl.Time:
            parser = time.fromisoformat
        else:
            parser = datetime.fromisoformat
        try:
            return parser(raw)
        except ValueError:
            raise ValueError(f"could not parse {raw!r} as a date/time")
    return raw
