"""Tests for the pure edit helpers in ``datasium.edit``."""

from __future__ import annotations

import polars as pl
import pytest

from datasium.edit import (
    add_column,
    add_row,
    cast_column,
    count_matches,
    set_cell_by_index,
    set_cell_by_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.scan_csv("sample.csv")


@pytest.fixture
def schema() -> dict[str, pl.DataType]:
    return dict(pl.scan_csv("sample.csv").collect_schema().items())


# ---------------------------------------------------------------------------
# cast_column
# ---------------------------------------------------------------------------
def test_cast_column(lf):
    out = cast_column(lf, "age", pl.Float64).collect()
    assert out.schema["age"] == pl.Float64
    assert out["age"].to_list() == [30.0, 12.0, 40.0, 25.0, 55.0, 9.0, 33.0, 28.0]


def test_cast_column_unknown(lf):
    with pytest.raises(ValueError, match="not found"):
        cast_column(lf, "bogus", pl.Float64)


def test_cast_column_bad_literal_fails_at_collect(lf):
    cast_lf = cast_column(lf, "name", pl.Int64)
    with pytest.raises(Exception):
        cast_lf.collect()


# ---------------------------------------------------------------------------
# add_column
# ---------------------------------------------------------------------------
def test_add_column_with_fill(lf):
    out = add_column(lf, "flag", pl.Int64, "1").collect()
    assert "flag" in out.columns
    assert out.schema["flag"] == pl.Int64
    assert out["flag"].to_list() == [1] * 8


def test_add_column_blank_fill_is_null(lf):
    out = add_column(lf, "note", pl.String, "").collect()
    assert out["note"].null_count() == 8


def test_add_column_duplicate_name(lf):
    with pytest.raises(ValueError, match="already exists"):
        add_column(lf, "age", pl.Int64, "0")


def test_add_column_no_name(lf):
    with pytest.raises(ValueError, match="enter a column name"):
        add_column(lf, "   ", pl.Int64, "0")


# ---------------------------------------------------------------------------
# add_row
# ---------------------------------------------------------------------------
def test_add_row_appends(lf):
    out = add_row(lf, {"name": "Zed", "age": "42", "city": "Rome", "score": "1.5"}).collect()
    assert out.height == 9
    last = out.row(8, named=True)
    assert last["name"] == "Zed"
    assert last["age"] == 42
    assert last["score"] == 1.5


def test_add_row_blank_becomes_null(lf):
    out = add_row(lf, {"name": "", "age": "", "city": "", "score": ""}).collect()
    row = out.row(8, named=True)
    assert row["name"] is None
    assert row["score"] is None


def test_add_row_missing_column(lf):
    with pytest.raises(ValueError, match="missing inputs for"):
        add_row(lf, {"name": "Zed", "age": "42"})


def test_add_row_bad_value(lf):
    with pytest.raises(ValueError, match="expected a number"):
        add_row(lf, {"name": "Zed", "age": "abc", "city": "X", "score": "1.0"})


# ---------------------------------------------------------------------------
# set_cell_by_index
# ---------------------------------------------------------------------------
def test_set_cell_by_index_updates_one_row(lf):
    out = set_cell_by_index(lf, 0, "age", "99", pl.Int64).collect()
    assert out["age"].to_list()[0] == 99
    # other rows untouched
    assert out["age"].to_list()[1:] == [12, 40, 25, 55, 9, 33, 28]


def test_set_cell_by_index_blank_is_null(lf):
    out = set_cell_by_index(lf, 2, "city", "", pl.String).collect()
    assert out["city"].to_list()[2] is None


def test_set_cell_by_index_out_of_range(lf):
    with pytest.raises(ValueError, match="out of range"):
        set_cell_by_index(lf, 99, "age", "1", pl.Int64)


def test_set_cell_by_index_negative(lf):
    with pytest.raises(ValueError, match="out of range"):
        set_cell_by_index(lf, -1, "age", "1", pl.Int64)


def test_set_cell_by_index_unknown_column(lf):
    with pytest.raises(ValueError, match="not found"):
        set_cell_by_index(lf, 0, "bogus", "1", pl.Int64)


# ---------------------------------------------------------------------------
# set_cell_by_key / count_matches
# ---------------------------------------------------------------------------
def test_set_cell_by_key_updates_unique_row(lf):
    out = set_cell_by_key(
        lf, ["name"], ["Ada"], "score", "10.0", pl.Float64,
    ).collect()
    ada = out.filter(pl.col("name") == "Ada")
    assert ada["score"].item() == 10.0
    # other Ada-ish rows untouched (Cy, Gio)
    assert out.filter(pl.col("name") == "Cy")["score"].item() == 9.1


def test_set_cell_by_key_multi_column(lf):
    out = set_cell_by_key(
        lf, ["name", "age"], ["Ada", "30"], "city", "Berlin", pl.String,
    ).collect()
    assert out.filter(pl.col("name") == "Ada")["city"].item() == "Berlin"


def test_set_cell_by_key_no_match(lf):
    with pytest.raises(ValueError, match="no rows match"):
        set_cell_by_key(lf, ["name"], ["Nobody"], "age", "1", pl.Int64)


def test_set_cell_by_key_multiple_match(lf):
    # city == London matches Ada, Cy, Gio (3 rows)
    with pytest.raises(ValueError, match="3 rows match"):
        set_cell_by_key(lf, ["city"], ["London"], "score", "1.0", pl.Float64)


def test_set_cell_by_key_no_columns(lf):
    with pytest.raises(ValueError, match="at least one key column"):
        set_cell_by_key(lf, [], [], "age", "1", pl.Int64)


def test_count_matches_unique(lf):
    assert count_matches(lf, ["name"], ["Ada"]) == 1


def test_count_matches_multiple(lf):
    assert count_matches(lf, ["city"], ["London"]) == 3


def test_count_matches_none(lf):
    assert count_matches(lf, ["name"], ["Nobody"]) == 0


# ---------------------------------------------------------------------------
# fill_nulls
# ---------------------------------------------------------------------------
@pytest.fixture
def null_lf() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "a": [1, None, 3, None, 5],
            "b": ["x", None, "z", None, "w"],
        }
    ).lazy()


def test_fill_nulls_value(null_lf):
    from datasium.edit import fill_nulls
    out = fill_nulls(null_lf, "a", "value", "99").collect()
    assert out["a"].to_list() == [1, 99, 3, 99, 5]


def test_fill_nulls_value_blank_is_noop(null_lf):
    from datasium.edit import fill_nulls
    # blank fill_value -> no-op, frame unchanged
    out = fill_nulls(null_lf, "a", "value", "").collect()
    assert out["a"].null_count() == 2  # still null
    assert out.shape == null_lf.collect().shape


def test_fill_nulls_string_value(null_lf):
    from datasium.edit import fill_nulls
    out = fill_nulls(null_lf, "b", "value", "replaced").collect()
    assert out["b"].to_list() == ["x", "replaced", "z", "replaced", "w"]


def test_fill_nulls_forward(null_lf):
    from datasium.edit import fill_nulls
    out = fill_nulls(null_lf, "a", "forward").collect()
    assert out["a"].to_list() == [1, 1, 3, 3, 5]


def test_fill_nulls_backward(null_lf):
    from datasium.edit import fill_nulls
    out = fill_nulls(null_lf, "a", "backward").collect()
    assert out["a"].to_list() == [1, 3, 3, 5, 5]


def test_fill_nulls_zero(null_lf):
    from datasium.edit import fill_nulls
    out = fill_nulls(null_lf, "a", "zero").collect()
    assert out["a"].to_list() == [1, 0, 3, 0, 5]


def test_fill_nulls_min_max(null_lf):
    from datasium.edit import fill_nulls
    out_min = fill_nulls(null_lf, "a", "min").collect()
    assert out_min["a"].to_list() == [1, 1, 3, 1, 5]
    out_max = fill_nulls(null_lf, "a", "max").collect()
    assert out_max["a"].to_list() == [1, 5, 3, 5, 5]


def test_fill_nulls_mean_median(null_lf):
    from datasium.edit import fill_nulls
    # mean: (1+3+5)/3 = 3.0
    out_mean = fill_nulls(null_lf, "a", "mean").collect()
    assert out_mean["a"].to_list() == [1, 3.0, 3, 3.0, 5]
    # median: 3.0
    out_median = fill_nulls(null_lf, "a", "median").collect()
    assert out_median["a"].to_list() == [1, 3.0, 3, 3.0, 5]


def test_fill_nulls_mode():
    from datasium.edit import fill_nulls
    # Dataset with a clear mode: 7 appears twice
    df = pl.DataFrame({"a": [7, 7, 1, None, 3, None]})
    out = fill_nulls(df.lazy(), "a", "mode").collect()
    assert out["a"].null_count() == 0
    # nulls filled with 7 (the mode)
    assert out["a"].to_list() == [7, 7, 1, 7, 3, 7]


def test_fill_nulls_unknown_column(null_lf):
    from datasium.edit import fill_nulls
    with pytest.raises(ValueError, match="not found"):
        fill_nulls(null_lf, "bogus", "value", "1")


def test_fill_nulls_unknown_strategy(null_lf):
    from datasium.edit import fill_nulls
    with pytest.raises(ValueError, match="unknown fill strategy"):
        fill_nulls(null_lf, "a", "bogus")


# ---------------------------------------------------------------------------
# replace_values
# ---------------------------------------------------------------------------
def test_replace_values_basic(lf):
    from datasium.edit import replace_values
    out = replace_values(lf, "city", "London", "Londra").collect()
    assert set(out.filter(pl.col("city") == "Londra")["name"].to_list()) == {
        "Ada", "Cy", "Gio"}


def test_replace_values_numeric(lf):
    from datasium.edit import replace_values
    out = replace_values(lf, "age", "30", "31").collect()
    ada = out.filter(pl.col("name") == "Ada")
    assert ada["age"].item() == 31


def test_replace_values_null_target(lf):
    # replace nulls: blank old_raw targets null cells
    df = pl.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    from datasium.edit import replace_values
    out = replace_values(df.lazy(), "a", "", "99").collect()
    assert out["a"].to_list() == [1, 99, 3]


def test_replace_values_no_match_is_noop(lf):
    from datasium.edit import replace_values
    out = replace_values(lf, "city", "NonExistent", "X").collect()
    assert out.shape == lf.collect().shape


def test_replace_values_unknown_column(lf):
    from datasium.edit import replace_values
    with pytest.raises(ValueError, match="not found"):
        replace_values(lf, "bogus", "x", "y")