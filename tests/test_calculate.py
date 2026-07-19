"""Tests for the pure row-calculation helpers."""

from __future__ import annotations

import polars as pl
import pytest

from datasium.calculate import _is_numeric, compute_stat


@pytest.fixture
def series() -> pl.Series:
    # 1..5 plus a null, like a filtered numeric column.
    return pl.Series("v", [1, 2, 3, 4, 5, None], dtype=pl.Int64)


def test_is_numeric():
    assert _is_numeric(pl.Int64)
    assert _is_numeric(pl.Float64)
    assert not _is_numeric(pl.String)
    assert not _is_numeric(pl.Date)


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
