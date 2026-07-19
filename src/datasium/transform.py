"""Transform component: sort, rename, computed columns, and group-by aggregation.

Pure helpers are UI-free so they can be unit-tested.  The :class:`TransformPanel`
wires them into the NiceGUI workbench.

Computed-column categories
--------------------------
* **arithmetic** – combine two columns or a column and a scalar
  (``add``, ``sub``, ``mul``, ``div``, ``mod``, ``pow``, ``floordiv``).
* **aggregation broadcast** – a single statistic (sum, mean, min, max, median,
  std, count) computed over an entire column and broadcast to every row.
* **cumulative** – running / cumulative versions of the same statistics
  (``cum_sum``, ``cum_mean``, ``cum_min``, ``cum_max``, ``cum_count``).
* **string** – ``upper``, ``lower``, ``strip``, ``str_len``, ``title``,
  ``reverse``.
* **rank / index** – ``rank`` (dense), ``row_index`` (0-based).
* **conditional** – ``when / then / otherwise`` on a simple comparison.
"""

from __future__ import annotations

from typing import Callable

import polars as pl

from nicegui import ui

from datasium.calculate import _is_numeric


# ---------------------------------------------------------------------------
# Operator catalogues
# ---------------------------------------------------------------------------
ARITH_OPS: list[tuple[str, str]] = [
    ("A + B", "add"),
    ("A − B", "sub"),
    ("A × B", "mul"),
    ("A ÷ B", "div"),
    ("A mod B", "mod"),
    ("A ** B", "pow"),
    ("A // B (floor div)", "floordiv"),
]

AGG_OPS: list[tuple[str, str]] = [
    ("sum", "sum"),
    ("mean", "mean"),
    ("min", "min"),
    ("max", "max"),
    ("median", "median"),
    ("std", "std"),
    ("count", "count"),
]

CUM_OPS: list[tuple[str, str]] = [
    ("cumulative sum", "cum_sum"),
    ("cumulative mean", "cum_mean"),
    ("cumulative min", "cum_min"),
    ("cumulative max", "cum_max"),
    ("cumulative count", "cum_count"),
]

STR_OPS: list[tuple[str, str]] = [
    ("UPPERCASE", "upper"),
    ("lowercase", "lower"),
    ("strip whitespace", "strip"),
    ("string length", "str_len"),
    ("Title Case", "title"),
    ("reverse", "reverse"),
]

RANK_OPS: list[tuple[str, str]] = [
    ("rank (dense)", "rank"),
    ("row index (0-based)", "row_index"),
]

COND_OPS: list[tuple[str, str]] = [
    ("when col > X then A else B", "cond_gt"),
    ("when col < X then A else B", "cond_lt"),
    ("when col == X then A else B", "cond_eq"),
    ("when col is null then A else B", "cond_null"),
]

CATEGORY_OPS: dict[str, list[tuple[str, str]]] = {
    "arithmetic": ARITH_OPS,
    "aggregation": AGG_OPS,
    "cumulative": CUM_OPS,
    "string": STR_OPS,
    "rank / index": RANK_OPS,
    "conditional": COND_OPS,
}

CATEGORY_LABELS: dict[str, str] = {
    "arithmetic": "Arithmetic (two columns or column ± scalar)",
    "aggregation": "Aggregation broadcast (statistic → every row)",
    "cumulative": "Cumulative / running statistic",
    "string": "String transform",
    "rank / index": "Rank / row index",
    "conditional": "Conditional (when / then / otherwise)",
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def sort_frame(
    lf: pl.LazyFrame,
    columns: list[str],
    descending: list[bool] | None = None,
) -> pl.LazyFrame:
    """Sort *lf* by one or more *columns*.

    *descending* is a parallel list of booleans (default all ascending).
    Raises ``ValueError`` on unknown columns or empty input.
    """
    if not columns:
        raise ValueError("select at least one column to sort by")
    names = set(lf.collect_schema().names())
    missing = [c for c in columns if c not in names]
    if missing:
        raise ValueError(f"column(s) not found: {', '.join(missing)}")
    if descending is None:
        descending = [False] * len(columns)
    if len(descending) != len(columns):
        raise ValueError("descending list must match columns list")
    return lf.sort(columns, descending=descending)


def rename_column(lf: pl.LazyFrame, old: str, new: str) -> pl.LazyFrame:
    """Rename column *old* to *new*.  Raises on unknown or duplicate names."""
    names = lf.collect_schema().names()
    if old not in names:
        raise ValueError(f"column {old!r} not found")
    if not new or not new.strip():
        raise ValueError("enter a new column name")
    if new in names and new != old:
        raise ValueError(f"column {new!r} already exists")
    return lf.rename({old: new})


def _arith_expr(op: str, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return a / b
    if op == "mod":
        return a % b
    if op == "pow":
        return a.pow(b)
    if op == "floordiv":
        return (a / b).floor()
    raise ValueError(f"unknown arithmetic op {op!r}")


def add_computed_column(
    lf: pl.LazyFrame,
    name: str,
    category: str,
    op: str,
    *,
    col_a: str | None = None,
    col_b: str | None = None,
    scalar: str | None = None,
    then_value: str | None = None,
    else_value: str | None = None,
) -> pl.LazyFrame:
    """Append a new computed column to *lf*.

    Parameters depend on *category*:

    * **arithmetic** – *col_a* (required), *col_b* or *scalar* (one required).
    * **aggregation** – *col_a* (the source column).
    * **cumulative** – *col_a*.
    * **string** – *col_a*.
    * **rank / index** – *col_a* (for rank), unused for row_index.
    * **conditional** – *col_a*, *scalar* (threshold), *then_value*, *else_value*.

    Raises ``ValueError`` with a user-facing message on bad input.
    """
    if not name or not name.strip():
        raise ValueError("enter a name for the new column")
    existing = lf.collect_schema().names()
    if name in existing:
        raise ValueError(f"column {name!r} already exists")

    schema = dict(lf.collect_schema().items())

    def _col(c: str) -> pl.Expr:
        if c not in schema:
            raise ValueError(f"column {c!r} not found")
        return pl.col(c)

    def _num(c: str) -> pl.Expr:
        expr = _col(c)
        if not _is_numeric(schema[c]):
            raise ValueError(f"column {c!r} is not numeric")
        return expr

    def _scalar_f64(raw: str | None) -> float:
        if raw is None or raw.strip() == "":
            raise ValueError("supply a numeric value")
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"expected a number, got {raw!r}")

    # ---- arithmetic ----
    if category == "arithmetic":
        if not col_a:
            raise ValueError("select column A")
        a = _num(col_a)
        if col_b:
            b = _num(col_b)
        elif scalar is not None and scalar.strip():
            b = pl.lit(_scalar_f64(scalar))
        else:
            raise ValueError("select column B or supply a scalar value")
        expr = _arith_expr(op, a, b)

    # ---- aggregation broadcast ----
    elif category == "aggregation":
        if not col_a:
            raise ValueError("select a source column")
        a = _num(col_a)
        agg_map = {
            "sum": a.sum,
            "mean": a.mean,
            "min": a.min,
            "max": a.max,
            "median": a.median,
            "std": a.std,
        }
        if op == "count":
            expr = a.count().cast(pl.UInt32)
        elif op in agg_map:
            expr = agg_map[op]()
        else:
            raise ValueError(f"unknown aggregation {op!r}")

    # ---- cumulative ----
    elif category == "cumulative":
        if not col_a:
            raise ValueError("select a source column")
        a = _num(col_a)
        cum_map = {
            "cum_sum": a.cum_sum,
            "cum_min": a.cum_min,
            "cum_max": a.cum_max,
        }
        if op == "cum_count":
            expr = a.cum_count()
        elif op == "cum_mean":
            expr = a.cum_sum() / a.cum_count()
        elif op in cum_map:
            expr = cum_map[op]()
        else:
            raise ValueError(f"unknown cumulative op {op!r}")

    # ---- string ----
    elif category == "string":
        if not col_a:
            raise ValueError("select a source column")
        a = _col(col_a)
        str_map = {
            "upper": lambda: a.str.to_uppercase(),
            "lower": lambda: a.str.to_lowercase(),
            "strip": lambda: a.str.strip_chars(),
            "str_len": lambda: a.str.len_chars(),
            "title": lambda: a.str.to_titlecase(),
            "reverse": lambda: a.str.reverse(),
        }
        if op not in str_map:
            raise ValueError(f"unknown string op {op!r}")
        expr = str_map[op]()

    # ---- rank / index ----
    elif category == "rank / index":
        if op == "rank":
            if not col_a:
                raise ValueError("select a column to rank by")
            expr = _col(col_a).rank(method="dense")
        elif op == "row_index":
            expr = pl.int_range(0, pl.len())
        else:
            raise ValueError(f"unknown rank op {op!r}")

    # ---- conditional ----
    elif category == "conditional":
        if not col_a:
            raise ValueError("select a column for the condition")
        a = _col(col_a)
        then_v = _parse_literal(then_value)
        else_v = _parse_literal(else_value)
        if op == "cond_null":
            expr = pl.when(a.is_null()).then(pl.lit(then_v)).otherwise(pl.lit(else_v))
        elif op == "cond_eq":
            threshold = _parse_literal(scalar)
            expr = (
                pl.when(a == threshold).then(pl.lit(then_v)).otherwise(pl.lit(else_v))
            )
        else:
            threshold = _scalar_f64(scalar)
            if op == "cond_gt":
                cond = a > threshold
            elif op == "cond_lt":
                cond = a < threshold
            else:
                raise ValueError(f"unknown conditional op {op!r}")
            expr = pl.when(cond).then(pl.lit(then_v)).otherwise(pl.lit(else_v))

    else:
        raise ValueError(f"unknown category {category!r}")

    return lf.with_columns(expr.alias(name))


def _parse_literal(raw: str | None) -> object:
    """Best-effort parse of a user-supplied literal (number or string)."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    if raw.strip() == "":
        return None
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null" or low == "none":
        return None
    return raw


def group_by_agg(
    lf: pl.LazyFrame,
    group_cols: list[str],
    agg_col: str | None,
    agg_op: str,
    output_name: str,
) -> pl.LazyFrame:
    """Group *lf* by *group_cols* and aggregate *agg_col* with *agg_op*.

    Returns a **new** (smaller) LazyFrame with the group columns plus the
    aggregated column named *output_name*.  When *agg_op* is ``"count"``
    the *agg_col* may be ``None`` (counts rows).

    Raises ``ValueError`` on bad input.
    """
    if not group_cols:
        raise ValueError("select at least one group-by column")
    names = set(lf.collect_schema().names())
    missing = [c for c in group_cols if c not in names]
    if missing:
        raise ValueError(f"group column(s) not found: {', '.join(missing)}")
    if not output_name or not output_name.strip():
        raise ValueError("enter a name for the aggregated column")

    if agg_op == "count":
        agg_expr = pl.len().alias(output_name)
    else:
        if not agg_col:
            raise ValueError("select a column to aggregate")
        if agg_col not in names:
            raise ValueError(f"column {agg_col!r} not found")
        col = pl.col(agg_col)
        agg_map = {
            "sum": col.sum,
            "mean": col.mean,
            "min": col.min,
            "max": col.max,
            "median": col.median,
            "std": col.std,
            "first": col.first,
            "last": col.last,
            "n_unique": col.n_unique,
        }
        if agg_op not in agg_map:
            raise ValueError(f"unknown aggregation {agg_op!r}")
        agg_expr = agg_map[agg_op]().alias(output_name)

    return lf.group_by(group_cols).agg(agg_expr)


# ---------------------------------------------------------------------------
# UI panel
# ---------------------------------------------------------------------------
class TransformPanel:
    """Sort / rename / computed-column / group-by panel."""

    def __init__(
        self,
        parent,
        columns: list[tuple[str, pl.DataType]],
        *,
        on_sort: Callable[[list[str], list[bool]], None],
        on_rename: Callable[[str, str], None],
        on_computed: Callable[
            [str, str, str, str | None, str | None, str | None, str | None, str | None],
            None,
        ],
        on_group_by: Callable[[list[str], str | None, str, str], None],
    ) -> None:
        self._columns = columns
        self._col_names = [n for n, _ in columns]
        self._on_sort = on_sort
        self._on_rename = on_rename
        self._on_computed = on_computed
        self._on_group_by = on_group_by

        with parent:
            self._build_sort_section()
            ui.separator()
            self._build_rename_section()
            ui.separator()
            self._build_computed_section()
            ui.separator()
            self._build_group_by_section()

    # ---- 1. sort ----------------------------------------------------------
    def _build_sort_section(self) -> None:
        ui.label("Sort").classes("text-lg font-medium")
        ui.label(
            "Sort rows by one or more columns. Toggle descending per column."
        ).classes("text-xs opacity-50")
        self.sort_cols = (
            ui.select(
                options={n: n for n in self._col_names} or {"—": "—"},
                multiple=True,
                value=[],
                clearable=True,
                label="Sort columns",
                on_change=lambda _e: self._refresh_sort_desc(),
            )
            .props("dense outlined use-chips")
            .classes("w-full")
        )
        self.sort_desc_box = ui.row().classes("items-center gap-2 w-full mt-1")
        self._sort_desc_switches: dict[str, object] = {}

    def _refresh_sort_desc(self) -> None:
        self.sort_desc_box.clear()
        self._sort_desc_switches = {}
        cols = list(self.sort_cols.value or [])
        if not cols:
            return
        with self.sort_desc_box:
            for c in cols:
                self._sort_desc_switches[c] = ui.switch(
                    f"{c} desc",
                    value=False,
                ).props("dense")
        with self.sort_desc_box:
            ui.button(
                "Sort",
                icon="sort",
                on_click=lambda _=None: self._submit_sort(),
            ).props("dense unelevated color=primary")

    def _submit_sort(self) -> None:
        cols = list(self.sort_cols.value or [])
        if not cols:
            ui.notify("select at least one sort column", type="warning", position="top")
            return
        desc = []
        for c in cols:
            sw = self._sort_desc_switches.get(c)
            desc.append(bool(sw.value) if sw else False)
        self._on_sort(cols, desc)

    # ---- 2. rename --------------------------------------------------------
    def _build_rename_section(self) -> None:
        ui.label("Rename column").classes("text-lg font-medium mt-2")
        ui.label("Rename a column in the dataset.").classes("text-xs opacity-50")
        with ui.row().classes("items-center gap-2 w-full"):
            self.rename_old = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    value=self._col_names[0] if self._col_names else None,
                    label="Column",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.rename_new = (
                ui.input(
                    value="",
                    label="New name",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            ui.button(
                "Rename",
                icon="drive_file_rename_outline",
                on_click=lambda _=None: self._on_rename(
                    self.rename_old.value or "", self.rename_new.value or ""
                ),
            ).props("dense unelevated color=primary")

    # ---- 3. computed column -----------------------------------------------
    def _build_computed_section(self) -> None:
        ui.label("Computed column").classes("text-lg font-medium mt-2")
        ui.label(
            "Create a new column from an expression. Choose a category, "
            "then fill in the relevant fields."
        ).classes("text-xs opacity-50")

        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            self.comp_category = (
                ui.select(
                    options={k: v for k, v in CATEGORY_LABELS.items()},
                    value="arithmetic",
                    label="Category",
                    on_change=lambda _e: self._refresh_comp_ops(),
                )
                .props("dense outlined")
                .classes("w-72")
            )
            self.comp_op = (
                ui.select(
                    options={},
                    value=None,
                    label="Operation",
                    on_change=lambda _e: self._refresh_comp_fields(),
                )
                .props("dense outlined")
                .classes("w-56")
            )

        self.comp_fields_box = ui.column().classes("w-full gap-2 mt-1")
        with ui.row().classes("items-center gap-2 w-full"):
            self.comp_name = (
                ui.input(
                    value="",
                    label="New column name",
                )
                .props("dense outlined")
                .classes("w-48")
            )
            ui.button(
                "Add column",
                icon="add_circle",
                on_click=lambda _=None: self._submit_computed(),
            ).props("dense unelevated color=primary")

        self._refresh_comp_ops()

    def _refresh_comp_ops(self) -> None:
        cat = self.comp_category.value or "arithmetic"
        ops = CATEGORY_OPS.get(cat, [])
        self.comp_op.options = {k: lbl for lbl, k in ops}  # type: ignore[assignment]
        self.comp_op.value = ops[0][1] if ops else None
        self._refresh_comp_fields()

    def _refresh_comp_fields(self) -> None:
        self.comp_fields_box.clear()
        cat = self.comp_category.value or "arithmetic"
        op = self.comp_op.value or ""
        numeric = [n for n, d in self._columns if _is_numeric(d)]
        all_names = self._col_names

        with self.comp_fields_box:
            if cat == "arithmetic":
                with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                    self._comp_col_a = (
                        ui.select(
                            options={n: n for n in numeric} or {"—": "—"},
                            value=numeric[0] if numeric else None,
                            label="Column A",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )
                    self._comp_col_b = (
                        ui.select(
                            options={n: n for n in numeric} or {"—": "—"},
                            value=None,
                            clearable=True,
                            label="Column B (optional)",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )
                    self._comp_scalar = (
                        ui.input(
                            value="",
                            label="…or scalar value",
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )

            elif cat in ("aggregation", "cumulative"):
                with ui.row().classes("items-center gap-2 w-full"):
                    self._comp_col_a = (
                        ui.select(
                            options={n: n for n in numeric} or {"—": "—"},
                            value=numeric[0] if numeric else None,
                            label="Source column",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )

            elif cat == "string":
                with ui.row().classes("items-center gap-2 w-full"):
                    self._comp_col_a = (
                        ui.select(
                            options={n: n for n in all_names} or {"—": "—"},
                            value=all_names[0] if all_names else None,
                            label="Source column",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )

            elif cat == "rank / index":
                if op == "rank":
                    with ui.row().classes("items-center gap-2 w-full"):
                        self._comp_col_a = (
                            ui.select(
                                options={n: n for n in all_names} or {"—": "—"},
                                value=all_names[0] if all_names else None,
                                label="Rank by column",
                            )
                            .props("dense outlined")
                            .classes("w-40")
                        )
                else:
                    ui.label("No extra inputs needed.").classes("text-sm opacity-50")

            elif cat == "conditional":
                with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                    self._comp_col_a = (
                        ui.select(
                            options={n: n for n in all_names} or {"—": "—"},
                            value=all_names[0] if all_names else None,
                            label="Condition column",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )
                    if op != "cond_null":
                        self._comp_scalar = (
                            ui.input(
                                value="",
                                label="Threshold X",
                            )
                            .props("dense outlined")
                            .classes("w-32")
                        )
                    self._comp_then = (
                        ui.input(
                            value="",
                            label="Then value",
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )
                    self._comp_else = (
                        ui.input(
                            value="",
                            label="Else value",
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )

    def _submit_computed(self) -> None:
        cat = self.comp_category.value or "arithmetic"
        op = self.comp_op.value or ""
        name = self.comp_name.value or ""
        col_a = getattr(self, "_comp_col_a", None)
        col_b = getattr(self, "_comp_col_b", None)
        scalar = getattr(self, "_comp_scalar", None)
        then_v = getattr(self, "_comp_then", None)
        else_v = getattr(self, "_comp_else", None)
        self._on_computed(
            name,
            cat,
            op,
            col_a.value if col_a else None,
            col_b.value if col_b else None,
            scalar.value if scalar else None,
            then_v.value if then_v else None,
            else_v.value if else_v else None,
        )

    # ---- 4. group-by aggregation ------------------------------------------
    def _build_group_by_section(self) -> None:
        ui.label("Group-by aggregation").classes("text-lg font-medium mt-2")
        ui.label(
            "Group rows and aggregate a column. Produces a new, smaller "
            "dataset (saved to a new parallel dataset)."
        ).classes("text-xs opacity-50")
        numeric = [n for n, d in self._columns if _is_numeric(d)]
        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            self.gb_cols = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    multiple=True,
                    value=[],
                    clearable=True,
                    label="Group-by columns",
                )
                .props("dense outlined use-chips")
                .classes("w-64")
            )
            self.gb_agg_col = (
                ui.select(
                    options={n: n for n in numeric} or {"—": "—"},
                    value=numeric[0] if numeric else None,
                    clearable=True,
                    label="Aggregate column",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.gb_agg_op = (
                ui.select(
                    options={
                        "mean": "mean",
                        "sum": "sum",
                        "min": "min",
                        "max": "max",
                        "median": "median",
                        "std": "std dev",
                        "count": "count (rows)",
                        "first": "first",
                        "last": "last",
                        "n_unique": "n unique",
                    },
                    value="mean",
                    label="Aggregation",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.gb_out_name = (
                ui.input(
                    value="",
                    label="Output column name",
                    placeholder="e.g. avg_score",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            ui.button(
                "Group & aggregate",
                icon="group_work",
                on_click=lambda _=None: self._submit_group_by(),
            ).props("dense unelevated color=primary")

    def _submit_group_by(self) -> None:
        cols = list(self.gb_cols.value or [])
        if not cols:
            ui.notify(
                "select at least one group-by column", type="warning", position="top"
            )
            return
        agg_col = self.gb_agg_col.value
        agg_op = self.gb_agg_op.value or "mean"
        out_name = self.gb_out_name.value or ""
        self._on_group_by(cols, agg_col, agg_op, out_name)
