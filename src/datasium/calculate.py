"""Row-calculation component for the filtered dataset.

Renders a column / operation / (optional) threshold selector and computes a
single statistic over the rows that survive the active :class:`FilterBuilder`
expression (or every row when no filter is set).

Supports three families of operations:

* **Single-column statistics** – descriptive stats computed over one column.
* **Two-column statistics** – association measures between two numeric columns.
* **Hypothesis tests** – return a test statistic and a p-value.

The pure :func:`compute_stat` helper is UI-free so it can be unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import polars as pl

from nicegui import ui


# ---------------------------------------------------------------------------
# Operation catalogues
# ---------------------------------------------------------------------------

# (label, key) for single-column statistics.
SINGLE_COL_STATS: list[tuple[str, str]] = [
    ("average", "mean"),
    ("maximum", "max"),
    ("minimum", "min"),
    ("sum", "sum"),
    ("median", "median"),
    ("std dev", "std"),
    ("variance", "variance"),
    ("skewness", "skewness"),
    ("kurtosis", "kurtosis"),
    ("IQR", "iqr"),
    ("coeff. of variation", "cv"),
    ("standard error", "se"),
    ("range", "range"),
    ("count (non-null)", "count"),
    ("null count", "null_count"),
    ("unique count", "n_unique"),
    ("mode", "mode"),
    ("quantile", "quantile"),
    ("count > X", "count_gt"),
    ("count >= X", "count_ge"),
    ("count < X", "count_lt"),
    ("count <= X", "count_le"),
    ("count == X", "count_eq"),
]

# (label, key) for two-column association statistics.
TWO_COL_STATS: list[tuple[str, str]] = [
    ("Pearson r", "pearson"),
    ("Spearman ρ", "spearman"),
    ("Kendall τ", "kendall"),
    ("covariance", "covariance"),
]

# (label, key) for single-column hypothesis tests.
SINGLE_COL_TESTS: list[tuple[str, str]] = [
    ("Shapiro-Wilk (normality)", "shapiro"),
    ("1-sample t-test (vs X)", "ttest_1samp"),
]

# (label, key) for two-column hypothesis tests.
TWO_COL_TESTS: list[tuple[str, str]] = [
    ("t-test (independent)", "ttest_ind"),
    ("Mann-Whitney U", "mann_whitney"),
    ("Wilcoxon signed-rank", "wilcoxon"),
]

# Combined list used by the UI select.
_STATS: list[tuple[str, str]] = (
    SINGLE_COL_STATS + TWO_COL_STATS + SINGLE_COL_TESTS + TWO_COL_TESTS
)

_THRESHOLD_OPS = {"count_gt", "count_ge", "count_lt", "count_le", "count_eq",
                  "quantile", "ttest_1samp"}
_TWO_COL_OPS = {k for _, k in TWO_COL_STATS} | {k for _, k in TWO_COL_TESTS}
_TEST_OPS = {k for _, k in SINGLE_COL_TESTS} | {k for _, k in TWO_COL_TESTS}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TestResult:
    """Outcome of a hypothesis test."""

    name: str
    statistic: float
    p_value: float

    def __str__(self) -> str:
        return f"{self.name}: statistic = {self.statistic:.6g}, p-value = {self.p_value:.6g}"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def compute_stat(
    series: pl.Series,
    op: str,
    raw: str | None = None,
    series_b: pl.Series | None = None,
) -> float | int | TestResult | None:
    """Compute a statistic over ``series`` (and optionally ``series_b``).

    Returns a number, a :class:`TestResult` for hypothesis tests, or ``None``
    when the result is null (e.g. mean of an empty series).  Raises
    ``ValueError`` with a user-facing message on bad input.
    """
    # ---- hypothesis tests (check before two-col stats) ----
    if op in _TEST_OPS:
        return _compute_test(series, op, raw, series_b)

    # ---- two-column statistics ----
    if op in _TWO_COL_OPS:
        if series_b is None:
            raise ValueError("select a second column for this operation")
        return _compute_two_col(series, series_b, op)

    # ---- single-column statistics ----
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
    if op == "variance":
        return series.var()
    if op == "skewness":
        return series.skew()
    if op == "kurtosis":
        return series.kurtosis()
    if op == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        if q1 is None or q3 is None:
            return None
        return q3 - q1
    if op == "cv":
        m = series.mean()
        s = series.std()
        if m is None or s is None or m == 0:
            return None
        return s / abs(m)
    if op == "se":
        s = series.std()
        n = series.len() - series.null_count()
        if s is None or n == 0:
            return None
        return s / math.sqrt(n)
    if op == "range":
        lo = series.min()
        hi = series.max()
        if lo is None or hi is None:
            return None
        return hi - lo
    if op == "count":
        return int(series.len() - series.null_count())
    if op == "null_count":
        return int(series.null_count())
    if op == "n_unique":
        return int(series.drop_nulls().n_unique())
    if op == "mode":
        modes = series.drop_nulls().mode()
        if modes.len() == 0:
            return None
        return modes[0]
    if op == "quantile":
        if raw is None or str(raw).strip() == "":
            raise ValueError("supply a quantile value (0–1)")
        try:
            q = float(str(raw))
        except ValueError:
            raise ValueError(f"expected a number, got {raw!r}")
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile must be between 0 and 1")
        return series.quantile(q)

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


def _compute_two_col(
    a: pl.Series,
    b: pl.Series,
    op: str,
) -> float | None:
    """Compute a two-column association statistic."""
    from scipy import stats as sp_stats

    a_clean = a.drop_nulls()
    b_clean = b.drop_nulls()
    # align: drop rows where either is null
    mask = a.is_not_null() & b.is_not_null()
    a_vals = a.filter(mask).to_numpy()
    b_vals = b.filter(mask).to_numpy()

    if len(a_vals) < 3:
        raise ValueError("need at least 3 non-null paired observations")

    if op == "pearson":
        r, _ = sp_stats.pearsonr(a_vals, b_vals)
        return float(r)
    if op == "spearman":
        r, _ = sp_stats.spearmanr(a_vals, b_vals)
        return float(r)
    if op == "kendall":
        r, _ = sp_stats.kendalltau(a_vals, b_vals)
        return float(r)
    if op == "covariance":
        import numpy as np
        return float(np.cov(a_vals, b_vals, ddof=1)[0, 1])

    raise ValueError(f"unknown two-column operation {op!r}")


def _compute_test(
    series: pl.Series,
    op: str,
    raw: str | None = None,
    series_b: pl.Series | None = None,
) -> TestResult:
    """Run a hypothesis test and return the result."""
    from scipy import stats as sp_stats

    vals = series.drop_nulls().to_numpy()

    if op == "shapiro":
        if len(vals) < 3:
            raise ValueError("Shapiro-Wilk needs at least 3 non-null values")
        stat, p = sp_stats.shapiro(vals)
        return TestResult("Shapiro-Wilk", float(stat), float(p))

    if op == "ttest_1samp":
        if raw is None or str(raw).strip() == "":
            raise ValueError("supply a test value (population mean)")
        try:
            popmean = float(str(raw))
        except ValueError:
            raise ValueError(f"expected a number, got {raw!r}")
        if len(vals) < 2:
            raise ValueError("1-sample t-test needs at least 2 non-null values")
        stat, p = sp_stats.ttest_1samp(vals, popmean)
        return TestResult("1-sample t-test", float(stat), float(p))

    if op == "ttest_ind":
        if series_b is None:
            raise ValueError("select a second column for the t-test")
        vals_b = series_b.drop_nulls().to_numpy()
        if len(vals) < 2 or len(vals_b) < 2:
            raise ValueError("t-test needs at least 2 non-null values per column")
        stat, p = sp_stats.ttest_ind(vals, vals_b)
        return TestResult("Independent t-test", float(stat), float(p))

    if op == "mann_whitney":
        if series_b is None:
            raise ValueError("select a second column for Mann-Whitney U")
        vals_b = series_b.drop_nulls().to_numpy()
        if len(vals) < 1 or len(vals_b) < 1:
            raise ValueError("Mann-Whitney U needs at least 1 non-null value per column")
        stat, p = sp_stats.mannwhitneyu(vals, vals_b, alternative="two-sided")
        return TestResult("Mann-Whitney U", float(stat), float(p))

    if op == "wilcoxon":
        if series_b is None:
            raise ValueError("select a second column for the Wilcoxon test")
        # Wilcoxon requires paired observations of equal length
        mask = series.is_not_null() & series_b.is_not_null()
        a_vals = series.filter(mask).to_numpy()
        b_vals = series_b.filter(mask).to_numpy()
        if len(a_vals) < 5:
            raise ValueError("Wilcoxon needs at least 5 paired non-null observations")
        stat, p = sp_stats.wilcoxon(a_vals, b_vals)
        return TestResult("Wilcoxon signed-rank", float(stat), float(p))

    raise ValueError(f"unknown test {op!r}")


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


# ---------------------------------------------------------------------------
# UI component
# ---------------------------------------------------------------------------
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
        self.result: float | int | TestResult | None = None
        self.error: str | None = None

        numeric = [n for n, d in columns if _is_numeric(d)]
        with parent:
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                self.col_select = (
                    ui.select(
                        options={n: n for n in numeric} or {"—": "—"},
                        value=numeric[0] if numeric else None,
                        label="Column",
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
                self.col_b_select = (
                    ui.select(
                        options={n: n for n in numeric} or {"—": "—"},
                        value=None,
                        clearable=True,
                        label="Column B",
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
                    .classes("w-52")
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
        op = self.op_select.value
        self.value_input.set_visibility(op in _THRESHOLD_OPS)
        self.col_b_select.set_visibility(op in _TWO_COL_OPS)
        # update threshold label contextually
        if op == "quantile":
            self.value_input.label = "Quantile (0–1)"
        elif op == "ttest_1samp":
            self.value_input.label = "Test value (μ₀)"
        else:
            self.value_input.label = "Threshold X"

    @property
    def column(self) -> str | None:
        v = self.col_select.value
        return None if v in (None, "—") else v

    @property
    def column_b(self) -> str | None:
        v = self.col_b_select.value
        return None if v in (None, "—") else v

    @property
    def operation(self) -> str | None:
        return self.op_select.value

    @property
    def threshold(self) -> str:
        return self.value_input.value or ""

    def set_result(self, value: float | int | TestResult | None) -> None:
        self.result = value
        self.error = None
        if value is None:
            self.result_label.set_text("—")
            self.result_label.classes(replace="opacity-40")
        elif isinstance(value, TestResult):
            self.result_label.set_text(str(value))
            self.result_label.classes(replace="opacity-80")
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
