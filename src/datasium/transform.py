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


def _is_temporal(dtype: pl.DataType) -> bool:
    return dtype.base_type() in (pl.Date, pl.Datetime, pl.Time)


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

DATETIME_OPS: list[tuple[str, str]] = [
    ("year", "dt_year"),
    ("month", "dt_month"),
    ("day", "dt_day"),
    ("day of week (1=Mon)", "dt_weekday"),
    ("hour", "dt_hour"),
    ("minute", "dt_minute"),
    ("quarter", "dt_quarter"),
    ("week of year", "dt_week"),
]

WINDOW_OPS: list[tuple[str, str]] = [
    ("lag (shift down)", "lag"),
    ("lead (shift up)", "lead"),
    ("diff (difference)", "diff"),
    ("rolling mean", "rolling_mean"),
    ("rolling sum", "rolling_sum"),
    ("rolling min", "rolling_min"),
    ("rolling max", "rolling_max"),
]

BINNING_OPS: list[tuple[str, str]] = [
    ("equal-width bins", "bin_width"),
    ("equal-frequency bins", "bin_freq"),
]

STAT_OPS: list[tuple[str, str]] = [
    ("z-score (standardise)", "z_score"),
    ("min-max normalise", "min_max"),
    ("percentile rank", "pct_rank"),
    ("log (natural)", "log"),
    ("log₂", "log2"),
    ("log₁₀", "log10"),
    ("absolute value", "abs"),
    ("sign (−1 / 0 / +1)", "sign"),
    ("clip (clamp to range)", "clip"),
    ("winsorise (clip at percentiles)", "winsorize"),
    ("empirical CDF", "ecdf"),
    ("round", "round"),
    ("square root", "sqrt"),
    ("negate", "negate"),
]

TS_OPS: list[tuple[str, str]] = [
    ("percent change", "pct_change"),
    ("exponential moving avg (EMA)", "ema"),
    ("detrend (subtract linear trend)", "detrend"),
    ("seasonal difference", "seasonal_diff"),
    ("rolling std dev", "rolling_std"),
    ("rolling median", "rolling_median"),
    ("date difference (days)", "date_diff"),
]

CATEGORY_OPS: dict[str, list[tuple[str, str]]] = {
    "arithmetic": ARITH_OPS,
    "aggregation": AGG_OPS,
    "cumulative": CUM_OPS,
    "string": STR_OPS,
    "rank / index": RANK_OPS,
    "conditional": COND_OPS,
    "datetime": DATETIME_OPS,
    "window": WINDOW_OPS,
    "binning": BINNING_OPS,
    "statistical": STAT_OPS,
    "time series": TS_OPS,
}

CATEGORY_LABELS: dict[str, str] = {
    "arithmetic": "Arithmetic (two columns or column ± scalar)",
    "aggregation": "Aggregation broadcast (statistic → every row)",
    "cumulative": "Cumulative / running statistic",
    "string": "String transform",
    "rank / index": "Rank / row index",
    "conditional": "Conditional (when / then / otherwise)",
    "datetime": "Datetime extraction (year, month, day, …)",
    "window": "Window (lag, lead, diff, rolling …)",
    "binning": "Binning (discretise a numeric column)",
    "statistical": "Statistical transform (z-score, normalise, log, …)",
    "time series": "Time series (pct change, EMA, detrend, …)",
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

    # ---- datetime extraction ----
    elif category == "datetime":
        if not col_a:
            raise ValueError("select a source column")
        a = _col(col_a)
        if not _is_temporal(schema[col_a]):
            raise ValueError(f"column {col_a!r} is not a date/time column")
        dt_map = {
            "dt_year": lambda: a.dt.year(),
            "dt_month": lambda: a.dt.month(),
            "dt_day": lambda: a.dt.day(),
            "dt_weekday": lambda: a.dt.weekday(),
            "dt_hour": lambda: a.dt.hour(),
            "dt_minute": lambda: a.dt.minute(),
            "dt_quarter": lambda: a.dt.quarter(),
            "dt_week": lambda: a.dt.week(),
        }
        if op not in dt_map:
            raise ValueError(f"unknown datetime op {op!r}")
        expr = dt_map[op]()

    # ---- window functions ----
    elif category == "window":
        if not col_a:
            raise ValueError("select a source column")
        a = _num(col_a)
        n = int(_scalar_f64(scalar)) if scalar and scalar.strip() else 1
        if op == "lag":
            expr = a.shift(n)
        elif op == "lead":
            expr = a.shift(-n)
        elif op == "diff":
            expr = a.diff(n)
        elif op == "rolling_mean":
            expr = a.rolling_mean(max(1, n))
        elif op == "rolling_sum":
            expr = a.rolling_sum(max(1, n))
        elif op == "rolling_min":
            expr = a.rolling_min(max(1, n))
        elif op == "rolling_max":
            expr = a.rolling_max(max(1, n))
        else:
            raise ValueError(f"unknown window op {op!r}")

    # ---- binning ----
    elif category == "binning":
        if not col_a:
            raise ValueError("select a source column")
        a = _num(col_a)
        n_bins = int(_scalar_f64(scalar)) if scalar and scalar.strip() else 5
        if n_bins < 2:
            raise ValueError("number of bins must be ≥ 2")
        if op == "bin_freq":
            quantiles = [i / n_bins for i in range(1, n_bins)]
            expr = a.qcut(quantiles, left_closed=True, allow_duplicates=True)
        elif op == "bin_width":
            # cut() needs literal breaks — collect min/max (single column, cheap)
            stats = lf.select(a.min().alias("lo"), a.max().alias("hi")).collect()
            lo = stats["lo"][0]
            hi = stats["hi"][0]
            if lo == hi:
                raise ValueError(f"column {col_a!r} has no range (min == max)")
            width = (hi - lo) / n_bins
            breaks = [lo + width * i for i in range(1, n_bins)]
            expr = a.cut(breaks, left_closed=True)
        else:
            raise ValueError(f"unknown binning op {op!r}")

    # ---- statistical transforms ----
    elif category == "statistical":
        if not col_a:
            raise ValueError("select a source column")
        a = _num(col_a)
        if op == "z_score":
            expr = (a - a.mean()) / a.std()
        elif op == "min_max":
            expr = (a - a.min()) / (a.max() - a.min())
        elif op == "pct_rank":
            expr = a.rank() / a.count() * 100
        elif op == "log":
            expr = a.log()
        elif op == "log2":
            expr = a.log(base=2)
        elif op == "log10":
            expr = a.log(base=10)
        elif op == "abs":
            expr = a.abs()
        elif op == "sign":
            expr = a.sign()
        elif op == "clip":
            parts = [v.strip() for v in (scalar or "").split(",")]
            if len(parts) != 2:
                raise ValueError("clip needs two comma-separated values, e.g. '0, 100'")
            try:
                lo_val, hi_val = float(parts[0]), float(parts[1])
            except ValueError:
                raise ValueError(f"could not parse clip range from {scalar!r}")
            expr = a.clip(lo_val, hi_val)
        elif op == "winsorize":
            parts = [v.strip() for v in (scalar or "").split(",")]
            if len(parts) != 2:
                raise ValueError(
                    "winsorise needs two comma-separated percentiles, e.g. '5, 95'"
                )
            try:
                lo_pct, hi_pct = float(parts[0]) / 100, float(parts[1]) / 100
            except ValueError:
                raise ValueError(f"could not parse percentiles from {scalar!r}")
            lo_q = a.quantile(lo_pct)
            hi_q = a.quantile(hi_pct)
            expr = a.clip(lo_q, hi_q)
        elif op == "ecdf":
            expr = a.rank() / a.count()
        elif op == "round":
            n_dec = int(_scalar_f64(scalar)) if scalar and scalar.strip() else 0
            expr = a.round(n_dec)
        elif op == "sqrt":
            expr = a.sqrt()
        elif op == "negate":
            expr = -a
        else:
            raise ValueError(f"unknown statistical op {op!r}")

    # ---- time series transforms ----
    elif category == "time series":
        if op == "date_diff":
            if not col_a or not col_b:
                raise ValueError("select two date/time columns for date difference")
            a_col = _col(col_a)
            b_col = _col(col_b)
            if not _is_temporal(schema[col_a]) or not _is_temporal(schema[col_b]):
                raise ValueError("both columns must be date/time for date difference")
            expr = (b_col - a_col).dt.total_days()
        else:
            if not col_a:
                raise ValueError("select a source column")
            a = _num(col_a)
            n = int(_scalar_f64(scalar)) if scalar and scalar.strip() else 1
            if op == "pct_change":
                expr = a.pct_change()
            elif op == "ema":
                span = max(1, n)
                expr = a.ewm_mean(span=span)
            elif op == "detrend":
                x = pl.int_range(0, pl.len()).cast(pl.Float64)
                n_f = pl.len().cast(pl.Float64)
                sum_x = x.sum()
                sum_y = a.sum()
                sum_xy = (x * a).sum()
                sum_x2 = (x * x).sum()
                slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_x2 - sum_x * sum_x)
                intercept = (sum_y - slope * sum_x) / n_f
                expr = a - (slope * x + intercept)
            elif op == "seasonal_diff":
                period = max(1, n)
                expr = a - a.shift(period)
            elif op == "rolling_std":
                expr = a.rolling_std(max(1, n))
            elif op == "rolling_median":
                expr = a.rolling_median(max(1, n))
            else:
                raise ValueError(f"unknown time series op {op!r}")

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


def one_hot_encode(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """One-hot encode *column*: one boolean column per unique value.

    New columns are named ``{column}_{value}``.  Raises ``ValueError`` on
    unknown columns.
    """
    names = lf.collect_schema().names()
    if column not in names:
        raise ValueError(f"column {column!r} not found")
    uniques = lf.select(pl.col(column).unique()).collect()[column].to_list()
    exprs = [
        pl.col(column).eq(v).alias(f"{column}_{v}")
        for v in uniques
        if v is not None
    ]
    return lf.with_columns(exprs)


def pivot_frame(
    lf: pl.LazyFrame,
    index: list[str],
    columns: str,
    values: str,
    agg: str = "first",
) -> pl.LazyFrame:
    """Pivot *lf* so unique values of *columns* become new column headers.

    *index* columns stay as row identifiers; *values* supplies the cell
    values, aggregated with *agg* when there are duplicates.
    """
    names = set(lf.collect_schema().names())
    for c in [*index, columns, values]:
        if c not in names:
            raise ValueError(f"column {c!r} not found")
    if not index:
        raise ValueError("select at least one index column")
    agg_map = {
        "first": "first",
        "last": "last",
        "sum": "sum",
        "mean": "mean",
        "min": "min",
        "max": "max",
        "count": "count",
        "median": "median",
    }
    if agg not in agg_map:
        raise ValueError(f"unknown aggregation {agg!r}")
    on_cols = lf.select(pl.col(columns).unique()).collect()[columns]
    return lf.pivot(
        on=columns,
        on_columns=on_cols,
        index=index,
        values=values,
        aggregate_function=agg,
    )


def unpivot_frame(
    lf: pl.LazyFrame,
    id_vars: list[str],
    value_vars: list[str] | None = None,
    variable_name: str = "variable",
    value_name: str = "value",
) -> pl.LazyFrame:
    """Unpivot (melt) *lf* from wide to long format.

    *id_vars* are kept as identifier columns.  *value_vars* are the columns
    to unpivot (defaults to all non-id columns).
    """
    names = set(lf.collect_schema().names())
    missing = [c for c in id_vars if c not in names]
    if missing:
        raise ValueError(f"id column(s) not found: {', '.join(missing)}")
    if value_vars:
        missing = [c for c in value_vars if c not in names]
        if missing:
            raise ValueError(f"value column(s) not found: {', '.join(missing)}")
    return lf.unpivot(
        index=id_vars or None,
        on=value_vars or None,
        variable_name=variable_name,
        value_name=value_name,
    )


def join_frames(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    left_on: list[str],
    right_on: list[str],
    how: str = "inner",
) -> pl.LazyFrame:
    """Join *left* and *right* on the given key columns.

    *how* is one of ``inner``, ``left``, ``right``, ``outer``, ``cross``.
    """
    if how not in ("inner", "left", "right", "outer", "cross"):
        raise ValueError(f"unknown join type {how!r}")
    lnames = set(left.collect_schema().names())
    rnames = set(right.collect_schema().names())
    lmiss = [c for c in left_on if c not in lnames]
    rmiss = [c for c in right_on if c not in rnames]
    if lmiss:
        raise ValueError(f"left column(s) not found: {', '.join(lmiss)}")
    if rmiss:
        raise ValueError(f"right column(s) not found: {', '.join(rmiss)}")
    if how == "cross":
        return left.join(right, how="cross")
    if len(left_on) != len(right_on):
        raise ValueError("left and right key lists must have the same length")
    return left.join(right, left_on=left_on, right_on=right_on, how=how)


def resample_frame(
    lf: pl.LazyFrame,
    time_col: str,
    every: str,
    agg_col: str | None,
    agg_op: str,
    output_name: str,
) -> pl.LazyFrame:
    """Resample a time-indexed frame by grouping into fixed-frequency windows.

    *time_col* must be a Date or Datetime column.  *every* is a Polars
    duration string such as ``"1d"``, ``"1w"``, ``"1mo"``, ``"1h"``.
    *agg_col* is aggregated with *agg_op* inside each window; when *agg_op*
    is ``"count"`` the *agg_col* may be ``None``.

    Returns a new (smaller) LazyFrame sorted by *time_col*.
    """
    names = set(lf.collect_schema().names())
    if time_col not in names:
        raise ValueError(f"time column {time_col!r} not found")
    schema = dict(lf.collect_schema().items())
    if not _is_temporal(schema[time_col]):
        raise ValueError(f"column {time_col!r} is not a date/time column")
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
        }
        if agg_op not in agg_map:
            raise ValueError(f"unknown aggregation {agg_op!r}")
        agg_expr = agg_map[agg_op]().alias(output_name)

    return (
        lf.sort(time_col)
        .group_by_dynamic(time_col, every=every)
        .agg(agg_expr)
    )


# ---------------------------------------------------------------------------
# UI panel
# ---------------------------------------------------------------------------
class TransformPanel:
    """Sort / rename / computed-column / group-by / encode / pivot / unpivot / join panel."""

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
        on_one_hot: Callable[[str], None],
        on_pivot: Callable[[list[str], str, str, str], None],
        on_unpivot: Callable[[list[str], list[str], str, str], None],
        on_join: Callable[[str, list[str], list[str], str], None],
        on_resample: Callable[[str, str, str | None, str, str], None] | None = None,
        dataset_names: list[str] | None = None,
    ) -> None:
        self._columns = columns
        self._col_names = [n for n, _ in columns]
        self._on_sort = on_sort
        self._on_rename = on_rename
        self._on_computed = on_computed
        self._on_group_by = on_group_by
        self._on_one_hot = on_one_hot
        self._on_pivot = on_pivot
        self._on_unpivot = on_unpivot
        self._on_join = on_join
        self._on_resample = on_resample
        self._dataset_names = dataset_names or []

        with parent:
            with ui.expansion("Sort", icon="sort").classes("w-full"):
                self._build_sort_section()
            with ui.expansion("Rename column", icon="drive_file_rename_outline").classes("w-full"):
                self._build_rename_section()
            with ui.expansion("Computed column", icon="add_circle").classes("w-full"):
                self._build_computed_section()
            with ui.expansion("Group-by aggregation", icon="group_work").classes("w-full"):
                self._build_group_by_section()
            with ui.expansion("One-hot encoding", icon="grid_on").classes("w-full"):
                self._build_one_hot_section()
            with ui.expansion("Pivot (wide)", icon="pivot_table_chart").classes("w-full"):
                self._build_pivot_section()
            with ui.expansion("Unpivot (melt to long)", icon="unpublished").classes("w-full"):
                self._build_unpivot_section()
            with ui.expansion("Join / merge datasets", icon="merge_type").classes("w-full"):
                self._build_join_section()
            with ui.expansion("Resample (time-based)", icon="schedule").classes("w-full"):
                self._build_resample_section()

    # ---- 1. sort ----------------------------------------------------------
    def _build_sort_section(self) -> None:
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

            elif cat == "datetime":
                temporal = [n for n, d in self._columns if _is_temporal(d)]
                with ui.row().classes("items-center gap-2 w-full"):
                    self._comp_col_a = (
                        ui.select(
                            options={n: n for n in temporal} or {"—": "—"},
                            value=temporal[0] if temporal else None,
                            label="Datetime column",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )

            elif cat == "window":
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
                    is_rolling = op.startswith("rolling_")
                    self._comp_scalar = (
                        ui.input(
                            value="3" if is_rolling else "1",
                            label="Window size" if is_rolling else "Offset (n)",
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )

            elif cat == "binning":
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
                    self._comp_scalar = (
                        ui.input(
                            value="5",
                            label="Number of bins",
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )

            elif cat == "statistical":
                with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                    self._comp_col_a = (
                        ui.select(
                            options={n: n for n in numeric} or {"—": "—"},
                            value=numeric[0] if numeric else None,
                            label="Source column",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )
                    if op in ("clip", "winsorize"):
                        placeholder = (
                            "lo, hi" if op == "clip" else "lo %, hi % (e.g. 5, 95)"
                        )
                        self._comp_scalar = (
                            ui.input(value="", label=placeholder)
                            .props("dense outlined")
                            .classes("w-40")
                        )
                    elif op == "round":
                        self._comp_scalar = (
                            ui.input(value="0", label="Decimal places")
                            .props("dense outlined")
                            .classes("w-32")
                        )

            elif cat == "time series":
                temporal = [n for n, d in self._columns if _is_temporal(d)]
                with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                    if op == "date_diff":
                        self._comp_col_a = (
                            ui.select(
                                options={n: n for n in temporal} or {"—": "—"},
                                value=temporal[0] if temporal else None,
                                label="Start date column",
                            )
                            .props("dense outlined")
                            .classes("w-44")
                        )
                        self._comp_col_b = (
                            ui.select(
                                options={n: n for n in temporal} or {"—": "—"},
                                value=temporal[1] if len(temporal) > 1 else (temporal[0] if temporal else None),
                                label="End date column",
                            )
                            .props("dense outlined")
                            .classes("w-44")
                        )
                    else:
                        self._comp_col_a = (
                            ui.select(
                                options={n: n for n in numeric} or {"—": "—"},
                                value=numeric[0] if numeric else None,
                                label="Source column",
                            )
                            .props("dense outlined")
                            .classes("w-40")
                        )
                        if op in ("ema", "seasonal_diff", "rolling_std", "rolling_median"):
                            default = "3" if op.startswith("rolling") else ("1" if op == "seasonal_diff" else "10")
                            label = (
                                "Window size" if op.startswith("rolling")
                                else ("Period" if op == "seasonal_diff" else "Span")
                            )
                            self._comp_scalar = (
                                ui.input(value=default, label=label)
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

    # ---- 5. one-hot encoding ----------------------------------------------
    def _build_one_hot_section(self) -> None:
        ui.label(
            "Create one boolean column per unique value of a categorical column."
        ).classes("text-xs opacity-50")
        with ui.row().classes("items-center gap-2 w-full"):
            self.oh_col = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    value=self._col_names[0] if self._col_names else None,
                    label="Column",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            ui.button(
                "Encode",
                icon="grid_on",
                on_click=lambda _=None: self._on_one_hot(self.oh_col.value or ""),
            ).props("dense unelevated color=primary")

    # ---- 6. pivot -----------------------------------------------------------
    def _build_pivot_section(self) -> None:
        ui.label(
            "Spread unique values of a column into new column headers."
        ).classes("text-xs opacity-50")
        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            self.pv_index = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    multiple=True,
                    value=[],
                    clearable=True,
                    label="Index columns",
                )
                .props("dense outlined use-chips")
                .classes("w-56")
            )
            self.pv_columns = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    value=self._col_names[0] if self._col_names else None,
                    label="Columns (spread)",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.pv_values = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    value=self._col_names[0] if self._col_names else None,
                    label="Values",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.pv_agg = (
                ui.select(
                    {
                        "first": "first",
                        "last": "last",
                        "sum": "sum",
                        "mean": "mean",
                        "min": "min",
                        "max": "max",
                        "count": "count",
                        "median": "median",
                    },
                    value="first",
                    label="Aggregation",
                )
                .props("dense outlined")
                .classes("w-32")
            )
            ui.button(
                "Pivot",
                icon="pivot_table_chart",
                on_click=lambda _=None: self._submit_pivot(),
            ).props("dense unelevated color=primary")

    def _submit_pivot(self) -> None:
        index = list(self.pv_index.value or [])
        if not index:
            ui.notify("select at least one index column", type="warning", position="top")
            return
        self._on_pivot(
            index,
            self.pv_columns.value or "",
            self.pv_values.value or "",
            self.pv_agg.value or "first",
        )

    # ---- 7. unpivot (melt) --------------------------------------------------
    def _build_unpivot_section(self) -> None:
        ui.label(
            "Reshape from wide to long format. ID columns stay; value "
            "columns are stacked."
        ).classes("text-xs opacity-50")
        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            self.up_id = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    multiple=True,
                    value=[],
                    clearable=True,
                    label="ID columns (keep)",
                )
                .props("dense outlined use-chips")
                .classes("w-56")
            )
            self.up_values = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    multiple=True,
                    value=[],
                    clearable=True,
                    label="Value columns (stack, empty = all)",
                )
                .props("dense outlined use-chips")
                .classes("w-56")
            )
            self.up_var_name = (
                ui.input(
                    value="variable",
                    label="Variable name",
                )
                .props("dense outlined")
                .classes("w-32")
            )
            self.up_val_name = (
                ui.input(
                    value="value",
                    label="Value name",
                )
                .props("dense outlined")
                .classes("w-32")
            )
            ui.button(
                "Unpivot",
                icon="unpublished",
                on_click=lambda _=None: self._submit_unpivot(),
            ).props("dense unelevated color=primary")

    def _submit_unpivot(self) -> None:
        id_vars = list(self.up_id.value or [])
        value_vars = list(self.up_values.value or [])
        self._on_unpivot(
            id_vars,
            value_vars,
            self.up_var_name.value or "variable",
            self.up_val_name.value or "value",
        )

    # ---- 8. join / merge ----------------------------------------------------
    def _build_join_section(self) -> None:
        ui.label(
            "Join the active dataset with another loaded dataset. "
            "The result is saved as a new dataset."
        ).classes("text-xs opacity-50")
        others = [n for n in self._dataset_names]
        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            self.jn_other = (
                ui.select(
                    options={n: n for n in others} or {"—": "—"},
                    value=others[0] if others else None,
                    label="Other dataset",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.jn_left_on = (
                ui.select(
                    options={n: n for n in self._col_names} or {"—": "—"},
                    multiple=True,
                    value=[],
                    clearable=True,
                    label="Left key(s)",
                )
                .props("dense outlined use-chips")
                .classes("w-48")
            )
            self.jn_right_on = (
                ui.input(
                    value="",
                    label="Right key(s), comma-separated",
                    placeholder="e.g. id, name",
                )
                .props("dense outlined")
                .classes("w-48")
            )
            self.jn_how = (
                ui.select(
                    {
                        "inner": "inner",
                        "left": "left",
                        "right": "right",
                        "outer": "outer (full)",
                        "cross": "cross (cartesian)",
                    },
                    value="inner",
                    label="Join type",
                )
                .props("dense outlined")
                .classes("w-36")
            )
            ui.button(
                "Join",
                icon="merge_type",
                on_click=lambda _=None: self._submit_join(),
            ).props("dense unelevated color=primary")

    def _submit_join(self) -> None:
        other = self.jn_other.value
        if not other or other == "—":
            ui.notify("select another dataset", type="warning", position="top")
            return
        left_on = list(self.jn_left_on.value or [])
        right_raw = self.jn_right_on.value or ""
        right_on = [s.strip() for s in right_raw.split(",") if s.strip()]
        how = self.jn_how.value or "inner"
        self._on_join(other, left_on, right_on, how)

    # ---- 9. resample (time-based) -------------------------------------------
    def _build_resample_section(self) -> None:
        ui.label(
            "Aggregate a time-indexed dataset into fixed-frequency windows. "
            "Produces a new dataset."
        ).classes("text-xs opacity-50")
        temporal = [n for n, d in self._columns if _is_temporal(d)]
        numeric = [n for n, d in self._columns if _is_numeric(d)]
        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            self.rs_time_col = (
                ui.select(
                    options={n: n for n in temporal} or {"—": "—"},
                    value=temporal[0] if temporal else None,
                    label="Time column",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.rs_every = (
                ui.select(
                    {
                        "1h": "hourly",
                        "1d": "daily",
                        "1w": "weekly",
                        "1mo": "monthly",
                        "1q": "quarterly",
                        "1y": "yearly",
                    },
                    value="1d",
                    label="Frequency",
                )
                .props("dense outlined")
                .classes("w-36")
            )
            self.rs_agg_col = (
                ui.select(
                    options={n: n for n in numeric} or {"—": "—"},
                    value=numeric[0] if numeric else None,
                    clearable=True,
                    label="Aggregate column",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            self.rs_agg_op = (
                ui.select(
                    {
                        "mean": "mean",
                        "sum": "sum",
                        "min": "min",
                        "max": "max",
                        "median": "median",
                        "std": "std dev",
                        "count": "count (rows)",
                        "first": "first",
                        "last": "last",
                    },
                    value="mean",
                    label="Aggregation",
                )
                .props("dense outlined")
                .classes("w-36")
            )
            self.rs_out_name = (
                ui.input(
                    value="",
                    label="Output column name",
                    placeholder="e.g. avg_value",
                )
                .props("dense outlined")
                .classes("w-40")
            )
            ui.button(
                "Resample",
                icon="schedule",
                on_click=lambda _=None: self._submit_resample(),
            ).props("dense unelevated color=primary")

    def _submit_resample(self) -> None:
        time_col = self.rs_time_col.value
        if not time_col or time_col == "—":
            ui.notify("select a time column", type="warning", position="top")
            return
        every = self.rs_every.value or "1d"
        agg_col = self.rs_agg_col.value
        agg_op = self.rs_agg_op.value or "mean"
        out_name = self.rs_out_name.value or ""
        if self._on_resample is not None:
            self._on_resample(time_col, every, agg_col, agg_op, out_name)
