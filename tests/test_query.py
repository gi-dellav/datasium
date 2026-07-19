"""Tests for the pure SQL query helper."""

from __future__ import annotations

import polars as pl
import pytest

from datasium.query import run_sql


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "name": ["Ada", "Bo", "Cy", "De"],
            "age": [25, 19, 30, 22],
            "city": ["London", "Paris", "London", "Rome"],
        }
    ).lazy()


def test_returns_lazyframe(lf):
    out = run_sql(lf, "SELECT * FROM self")
    assert isinstance(out, pl.LazyFrame)


def test_select_all(lf):
    out = run_sql(lf, "SELECT * FROM self").collect()
    assert out.shape == (4, 3)
    assert out.columns == ["name", "age", "city"]


def test_select_where(lf):
    out = run_sql(lf, "SELECT name FROM self WHERE age > 20").collect()
    assert set(out["name"]) == {"Ada", "Cy", "De"}


def test_order_and_limit(lf):
    out = run_sql(lf, "SELECT name, age FROM self ORDER BY age DESC LIMIT 2").collect()
    assert out["name"].to_list() == ["Cy", "Ada"]


def test_aggregate(lf):
    out = run_sql(lf, "SELECT city, COUNT(*) AS n FROM self GROUP BY city").collect()
    assert out.height == 3


def test_custom_table_name(lf):
    out = run_sql(lf, "SELECT * FROM frame", table_name="frame").collect()
    assert out.height == 4


def test_empty_query_raises(lf):
    with pytest.raises(ValueError, match="enter a SQL query"):
        run_sql(lf, "")
    with pytest.raises(ValueError, match="enter a SQL query"):
        run_sql(lf, "   ")


def test_bad_sql_raises(lf):
    with pytest.raises(Exception):
        run_sql(lf, "SELECT FROM WHERE").collect()
