"""Tests for the pure removal helpers and the registry replace() method."""

from __future__ import annotations

import polars as pl
import pytest

from datasium.dataset import DatasetRegistry
from datasium.filter import build_term
from datasium.remove import (
    RemovalSpec,
    apply_removal,
    remove_columns,
    remove_nulls,
    remove_rows_by_value,
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


@pytest.fixture
def null_lf() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "a": [1, None, 3, 4],
            "b": ["x", "y", None, "z"],
            "c": [1.0, 2.0, 3.0, None],
        }
    ).lazy()


# ---------------------------------------------------------------------------
# remove_columns
# ---------------------------------------------------------------------------
def test_remove_columns(lf):
    out = remove_columns(lf, ["age", "score"]).collect()
    assert out.shape == (8, 2)
    assert out.columns == ["name", "city"]


def test_remove_columns_empty_is_noop(lf):
    assert remove_columns(lf, []) is lf


def test_remove_columns_unknown(lf):
    with pytest.raises(ValueError, match="not found"):
        remove_columns(lf, ["bogus"])


# ---------------------------------------------------------------------------
# remove_rows_by_value
# ---------------------------------------------------------------------------
def test_remove_rows_by_value_eq(lf):
    # drop city == London (Ada, Cy, Gio)
    out = remove_rows_by_value(lf, "city", "eq", "London").collect()
    assert set(out["name"]) == {"Bo", "De", "Ed", "Finn", "Hal"}


def test_remove_rows_by_value_gt(lf):
    out = remove_rows_by_value(lf, "age", "gt", "20").collect()
    assert set(out["name"]) == {"Bo", "Finn"}


def test_remove_rows_by_value_is_in(lf):
    out = remove_rows_by_value(lf, "city", "is_in", "London, Rome").collect()
    assert set(out["name"]) == {"Bo", "De", "Hal"}


def test_remove_rows_by_value_is_null(lf):
    # sample.csv has no null ages -> nothing removed
    out = remove_rows_by_value(lf, "age", "is_null", "").collect()
    assert out.height == 8


def test_remove_rows_by_value_requires_column(lf):
    with pytest.raises(ValueError, match="select a column"):
        remove_rows_by_value(lf, "", "eq", "x")


def test_remove_rows_by_value_unknown_column(lf):
    with pytest.raises(ValueError, match="not found"):
        remove_rows_by_value(lf, "bogus", "eq", "x")


# ---------------------------------------------------------------------------
# remove_nulls
# ---------------------------------------------------------------------------
def test_remove_nulls_all(null_lf):
    # only row 0 has no nulls
    assert remove_nulls(null_lf).collect().height == 1


def test_remove_nulls_subset(null_lf):
    # rows where a is not null: 0, 2, 3
    assert remove_nulls(null_lf, subset=["a"]).collect().height == 3


def test_remove_nulls_empty_subset_is_all(null_lf):
    assert remove_nulls(null_lf, subset=[]).collect().height == 1


def test_remove_nulls_unknown_subset(null_lf):
    with pytest.raises(ValueError, match="not found"):
        remove_nulls(null_lf, subset=["bogus"])


# ---------------------------------------------------------------------------
# apply_removal
# ---------------------------------------------------------------------------
def test_apply_removal_none_mode(lf):
    out = apply_removal(lf, RemovalSpec(row_mode="none")).collect()
    assert out.shape == (8, 4)


def test_apply_removal_columns_only(lf):
    out = apply_removal(
        lf, RemovalSpec(row_mode="none", columns=["age", "score"])
    ).collect()
    assert out.shape == (8, 2)


def test_apply_removal_values_and_columns(lf, schema):
    spec = RemovalSpec(
        row_mode="values",
        value_column="city", value_op="eq", value_raw="London",
        columns=["score"],
    )
    out = apply_removal(lf, spec).collect()
    # London rows removed (3), score column dropped (3 cols)
    assert out.shape == (5, 3)
    assert "score" not in out.columns
    assert set(out["name"]) == {"Bo", "De", "Ed", "Finn", "Hal"}


def test_apply_removal_selection(lf, schema):
    expr = build_term("city", "eq", "London", schema["city"])
    out = apply_removal(lf, RemovalSpec(row_mode="selection", selection_expr=expr)).collect()
    assert set(out["name"]) == {"Bo", "De", "Ed", "Finn", "Hal"}


def test_apply_removal_selection_with_columns(lf, schema):
    expr = build_term("city", "eq", "London", schema["city"])
    out = apply_removal(
        lf, RemovalSpec(row_mode="selection", selection_expr=expr, columns=["score"])
    ).collect()
    assert out.shape == (5, 3)
    assert "score" not in out.columns


def test_apply_removal_selection_noexpr(lf):
    with pytest.raises(ValueError, match="no row selection"):
        apply_removal(lf, RemovalSpec(row_mode="selection")).collect()


def test_apply_removal_unknown_mode(lf):
    with pytest.raises(ValueError, match="unknown row mode"):
        apply_removal(lf, RemovalSpec(row_mode="bogus")).collect()


def test_apply_removal_nulls_combo(null_lf):
    spec = RemovalSpec(row_mode="nulls", null_subset=["a"], columns=["c"])
    out = apply_removal(null_lf, spec).collect()
    # rows where a not null (0,2,3) then drop c -> 3 rows, 2 cols
    assert out.shape == (3, 2)
    assert "c" not in out.columns


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------
def test_remove_duplicates_all(lf):
    # sample.csv has no duplicate rows -> no change
    from datasium.remove import remove_duplicates
    out = remove_duplicates(lf).collect()
    assert out.shape == (8, 4)


def test_remove_duplicates_subset():
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5], "c": ["x", "x", "y"]})
    from datasium.remove import remove_duplicates
    out = remove_duplicates(df.lazy(), subset=["a"]).collect()
    assert out.shape == (2, 3)
    assert set(out["a"].to_list()) == {1, 2}
    # keep="first" retains first occurrences: a=1,b=3 and a=2,b=5
    assert out.filter(pl.col("a") == 1)["b"].item() == 3
    assert out.filter(pl.col("a") == 2)["b"].item() == 5


def test_remove_duplicates_keep_last():
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    from datasium.remove import remove_duplicates
    out = remove_duplicates(df.lazy(), subset=["a"], keep="last").collect()
    assert out.shape == (2, 2)
    # keep="last" retains last occurrences: a=1,b=4 and a=2,b=5
    assert out.filter(pl.col("a") == 1)["b"].item() == 4
    assert out.filter(pl.col("a") == 2)["b"].item() == 5


def test_remove_duplicates_keep_none():
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    from datasium.remove import remove_duplicates
    out = remove_duplicates(df.lazy(), subset=["a"], keep="none").collect()
    assert out.shape == (1, 2)
    assert out["a"].to_list() == [2]


def test_remove_duplicates_unknown_subset(lf):
    from datasium.remove import remove_duplicates
    with pytest.raises(ValueError, match="not found"):
        remove_duplicates(lf, subset=["bogus"])


def test_remove_duplicates_bad_keep(lf):
    from datasium.remove import remove_duplicates
    with pytest.raises(ValueError, match="keep must be"):
        remove_duplicates(lf, keep="bogus")


# ---------------------------------------------------------------------------
# apply_removal with duplicates mode
# ---------------------------------------------------------------------------
def test_apply_removal_duplicates():
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    from datasium.remove import apply_removal, RemovalSpec
    spec = RemovalSpec(row_mode="duplicates", dup_subset=["a"])
    out = apply_removal(df.lazy(), spec).collect()
    assert out.shape == (2, 2)


def test_apply_removal_duplicates_with_columns():
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    from datasium.remove import apply_removal, RemovalSpec
    spec = RemovalSpec(row_mode="duplicates", dup_subset=["a"], columns=["b"])
    out = apply_removal(df.lazy(), spec).collect()
    assert out.shape == (2, 1)
    assert out.columns == ["a"]


# ---------------------------------------------------------------------------
# DatasetRegistry.replace
# ---------------------------------------------------------------------------
def test_registry_replace():
    reg = DatasetRegistry()
    reg.load("sample.csv", open("sample.csv", "rb").read())
    new_lf = pl.scan_csv("sample.csv").drop("age").lazy()
    updated = reg.replace("sample", new_lf)
    assert updated.source == "sample.csv"
    assert updated.shape == (8, 3)
    assert "age" not in [n for n, _ in updated.columns]
    assert reg.get("sample") is updated


def test_registry_replace_unknown():
    reg = DatasetRegistry()
    with pytest.raises(KeyError):
        reg.replace("missing", pl.scan_csv("sample.csv"))