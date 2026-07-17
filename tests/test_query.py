"""Tests for the pure SQL query helper."""

from __future__ import annotations

import polars as pl
import pytest

from datasium.query import run_sql


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Ada", "Bo", "Cy", "De"],
            "age": [25, 19, 30, 22],
            "city": ["London", "Paris", "London", "Rome"],
        }
    )


def test_select_all(df):
    out = run_sql(df, "SELECT * FROM self")
    assert out.shape == df.shape
    assert out.columns == ["name", "age", "city"]


def test_select_where(df):
    out = run_sql(df, "SELECT name FROM self WHERE age > 20")
    assert set(out["name"]) == {"Ada", "Cy", "De"}


def test_order_and_limit(df):
    out = run_sql(df, "SELECT name, age FROM self ORDER BY age DESC LIMIT 2")
    assert out["name"].to_list() == ["Cy", "Ada"]


def test_aggregate(df):
    out = run_sql(df, "SELECT city, COUNT(*) AS n FROM self GROUP BY city")
    assert out.height == 3


def test_custom_table_name(df):
    out = run_sql(df, "SELECT * FROM frame", table_name="frame")
    assert out.height == df.height


def test_empty_query_raises(df):
    with pytest.raises(ValueError, match="enter a SQL query"):
        run_sql(df, "")
    with pytest.raises(ValueError, match="enter a SQL query"):
        run_sql(df, "   ")


def test_bad_sql_raises(df):
    with pytest.raises(Exception):
        run_sql(df, "SELECT FROM WHERE")