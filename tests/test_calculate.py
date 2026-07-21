"""Tests for the pure row-calculation helpers."""

from __future__ import annotations

import math

import polars as pl
import pytest

from datasium.calculate import TestResult, _is_numeric, compute_stat


@pytest.fixture
def series() -> pl.Series:
    # 1..5 plus a null, like a filtered numeric column.
    return pl.Series("v", [1, 2, 3, 4, 5, None], dtype=pl.Int64)


@pytest.fixture
def series_b() -> pl.Series:
    return pl.Series("w", [10, 20, 30, 40, 50, None], dtype=pl.Int64)


def test_is_numeric():
    assert _is_numeric(pl.Int64)
    assert _is_numeric(pl.Float64)
    assert not _is_numeric(pl.String)
    assert not _is_numeric(pl.Date)


# ---------------------------------------------------------------------------
# basic single-column stats (existing)
# ---------------------------------------------------------------------------
def test_mean(series):
    assert compute_stat(series, "mean") == pytest.approx(3.0)


def test_max_min_sum(series):
    assert compute_stat(series, "max") == 5
    assert compute_stat(series, "min") == 1
    assert compute_stat(series, "sum") == 15


def test_median(series):
    assert compute_stat(series, "median") == 3.0


def test_count_ignores_nulls(series):
    assert compute_stat(series, "count") == 5


def test_count_gt(series):
    assert compute_stat(series, "count_gt", "2") == 3
    assert compute_stat(series, "count_ge", "3") == 3
    assert compute_stat(series, "count_lt", "3") == 2
    assert compute_stat(series, "count_le", "3") == 3
    assert compute_stat(series, "count_eq", "3") == 1


def test_threshold_required(series):
    with pytest.raises(ValueError, match="supply a threshold"):
        compute_stat(series, "count_gt", "")
    with pytest.raises(ValueError, match="supply a threshold"):
        compute_stat(series, "count_gt", None)


def test_threshold_invalid(series):
    with pytest.raises(ValueError, match="expected a number"):
        compute_stat(series, "count_eq", "abc")


def test_empty_series_returns_none():
    s = pl.Series("v", [], dtype=pl.Int64)
    assert compute_stat(s, "mean") is None
    assert compute_stat(s, "max") is None


def test_unknown_op(series):
    with pytest.raises(ValueError, match="unknown operation"):
        compute_stat(series, "bogus")


# ---------------------------------------------------------------------------
# new single-column stats
# ---------------------------------------------------------------------------
def test_variance(series):
    # var of [1,2,3,4,5] = 2.5 (sample variance)
    assert compute_stat(series, "variance") == pytest.approx(2.5)


def test_skewness():
    s = pl.Series("v", [1.0, 2.0, 3.0, 4.0, 100.0])
    result = compute_stat(s, "skewness")
    assert result is not None
    assert result > 0  # right-skewed


def test_kurtosis():
    s = pl.Series("v", [1.0, 2.0, 3.0, 4.0, 100.0])
    result = compute_stat(s, "kurtosis")
    assert result is not None


def test_iqr(series):
    # Q1=2, Q3=4 → IQR=2
    assert compute_stat(series, "iqr") == pytest.approx(2.0)


def test_cv(series):
    # std / |mean| = sqrt(2.5) / 3
    expected = math.sqrt(2.5) / 3.0
    assert compute_stat(series, "cv") == pytest.approx(expected)


def test_cv_zero_mean():
    s = pl.Series("v", [-1.0, 1.0])
    assert compute_stat(s, "cv") is None


def test_se(series):
    # std / sqrt(n) = sqrt(2.5) / sqrt(5)
    expected = math.sqrt(2.5) / math.sqrt(5)
    assert compute_stat(series, "se") == pytest.approx(expected)


def test_range(series):
    assert compute_stat(series, "range") == 4


def test_null_count(series):
    assert compute_stat(series, "null_count") == 1


def test_n_unique(series):
    assert compute_stat(series, "n_unique") == 5


def test_mode():
    s = pl.Series("v", [1, 2, 2, 3, 3, 3])
    assert compute_stat(s, "mode") == 3


def test_quantile(series):
    assert compute_stat(series, "quantile", "0.5") == pytest.approx(3.0)
    assert compute_stat(series, "quantile", "0.0") == pytest.approx(1.0)
    assert compute_stat(series, "quantile", "1.0") == pytest.approx(5.0)


def test_quantile_invalid(series):
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_stat(series, "quantile", "1.5")
    with pytest.raises(ValueError, match="supply a quantile"):
        compute_stat(series, "quantile", "")


# ---------------------------------------------------------------------------
# two-column statistics
# ---------------------------------------------------------------------------
def test_pearson(series, series_b):
    r = compute_stat(series, "pearson", series_b=series_b)
    assert r == pytest.approx(1.0)  # perfectly correlated


def test_spearman(series, series_b):
    r = compute_stat(series, "spearman", series_b=series_b)
    assert r == pytest.approx(1.0)


def test_kendall(series, series_b):
    r = compute_stat(series, "kendall", series_b=series_b)
    assert r == pytest.approx(1.0)


def test_covariance(series, series_b):
    cov = compute_stat(series, "covariance", series_b=series_b)
    # cov([1,2,3,4,5], [10,20,30,40,50]) = 10 * var([1..5]) = 10 * 2.5 = 25
    assert cov == pytest.approx(25.0)


def test_two_col_requires_series_b(series):
    with pytest.raises(ValueError, match="second column"):
        compute_stat(series, "pearson")


def test_two_col_too_few():
    a = pl.Series("a", [1.0, 2.0])
    b = pl.Series("b", [3.0, 4.0])
    with pytest.raises(ValueError, match="at least 3"):
        compute_stat(a, "pearson", series_b=b)


# ---------------------------------------------------------------------------
# hypothesis tests
# ---------------------------------------------------------------------------
def test_shapiro():
    s = pl.Series("v", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = compute_stat(s, "shapiro")
    assert isinstance(result, TestResult)
    assert result.name == "Shapiro-Wilk"
    assert 0.0 <= result.p_value <= 1.0


def test_shapiro_too_few():
    s = pl.Series("v", [1.0, 2.0])
    with pytest.raises(ValueError, match="at least 3"):
        compute_stat(s, "shapiro")


def test_ttest_1samp():
    s = pl.Series("v", [10.0, 12.0, 11.0, 13.0, 9.0, 11.0])
    result = compute_stat(s, "ttest_1samp", "10")
    assert isinstance(result, TestResult)
    assert result.name == "1-sample t-test"
    assert 0.0 <= result.p_value <= 1.0


def test_ttest_1samp_no_value(series):
    with pytest.raises(ValueError, match="supply a test value"):
        compute_stat(series, "ttest_1samp", "")


def test_ttest_ind(series, series_b):
    result = compute_stat(series, "ttest_ind", series_b=series_b)
    assert isinstance(result, TestResult)
    assert result.name == "Independent t-test"


def test_ttest_ind_requires_b(series):
    with pytest.raises(ValueError, match="second column"):
        compute_stat(series, "ttest_ind")


def test_mann_whitney(series, series_b):
    result = compute_stat(series, "mann_whitney", series_b=series_b)
    assert isinstance(result, TestResult)
    assert result.name == "Mann-Whitney U"


def test_wilcoxon():
    a = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = pl.Series("b", [1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    result = compute_stat(a, "wilcoxon", series_b=b)
    assert isinstance(result, TestResult)
    assert result.name == "Wilcoxon signed-rank"


def test_wilcoxon_too_few():
    a = pl.Series("a", [1.0, 2.0, 3.0])
    b = pl.Series("b", [1.5, 2.5, 3.5])
    with pytest.raises(ValueError, match="at least 5"):
        compute_stat(a, "wilcoxon", series_b=b)


def test_test_result_str():
    r = TestResult("Shapiro-Wilk", 0.987, 0.543)
    s = str(r)
    assert "Shapiro-Wilk" in s
    assert "0.987" in s
    assert "0.543" in s
