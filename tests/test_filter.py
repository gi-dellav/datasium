"""Tests for the pure filter expression builder and dataset registry."""

from __future__ import annotations

import io

import polars as pl
import pytest

from datasium.dataset import DatasetRegistry
from datasium.filter import _coerce, _dtype_group, build_term


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
# _dtype_group
# ---------------------------------------------------------------------------
def test_dtype_groups():
    assert _dtype_group(pl.Int64) == "numeric"
    assert _dtype_group(pl.Float32) == "numeric"
    assert _dtype_group(pl.String) == "string"
    assert _dtype_group(pl.Boolean) == "boolean"
    assert _dtype_group(pl.Date) == "temporal"
    assert _dtype_group(pl.Datetime) == "temporal"
    assert _dtype_group(pl.List(pl.Int64)) == "other"


# ---------------------------------------------------------------------------
# _coerce
# ---------------------------------------------------------------------------
def test_coerce_numeric():
    assert _coerce("5", pl.Int64) == 5
    assert isinstance(_coerce("5", pl.Int64), int)
    assert _coerce("5.2", pl.Float64) == 5.2


def test_coerce_numeric_invalid():
    with pytest.raises(ValueError):
        _coerce("abc", pl.Int64)


def test_coerce_boolean():
    assert _coerce("true", pl.Boolean) is True
    assert _coerce("FALSE", pl.Boolean) is False
    assert _coerce("yes", pl.Boolean) is True
    with pytest.raises(ValueError):
        _coerce("maybe", pl.Boolean)


def test_coerce_temporal():
    import datetime

    assert _coerce("2020-01-01", pl.Date) == datetime.date(2020, 1, 1)


def test_coerce_string():
    assert _coerce("hello", pl.String) == "hello"


# ---------------------------------------------------------------------------
# build_term applied to sample.csv
# ---------------------------------------------------------------------------
def _run(lf: pl.LazyFrame, expr: pl.Expr) -> pl.DataFrame:
    return lf.filter(expr).collect()


def test_numeric_eq(lf, schema):
    expr = build_term("city", "eq", "London", schema["city"])
    assert set(_run(lf, expr)["name"]) == {"Ada", "Cy", "Gio"}


def test_numeric_gt(lf, schema):
    expr = build_term("age", "gt", "20", schema["age"])
    assert set(_run(lf, expr)["name"]) == {"Ada", "Cy", "De", "Ed", "Gio", "Hal"}


def test_string_contains(lf, schema):
    expr = build_term("name", "contains", "a", schema["name"])
    assert set(_run(lf, expr)["name"]) == {"Ada", "Hal"}


def test_string_starts_with(lf, schema):
    expr = build_term("name", "starts_with", "F", schema["name"])
    assert _run(lf, expr)["name"].to_list() == ["Finn"]


def test_is_in(lf, schema):
    expr = build_term("city", "is_in", "Paris, Rome", schema["city"])
    assert set(_run(lf, expr)["name"]) == {"Bo", "De", "Ed", "Finn", "Hal"}


def test_is_in_requires_values(lf, schema):
    with pytest.raises(ValueError, match="is in"):
        build_term("city", "is_in", "", schema["city"])


def test_nullary_operators(lf, schema):
    # is_null / is_not_null compile without a value.
    assert _run(lf, build_term("age", "is_not_null", "", schema["age"])).height == 8
    assert _run(lf, build_term("age", "is_null", "", schema["age"])).height == 0


def test_combine_and(lf, schema):
    a = build_term("age", "gt", "20", schema["age"])
    b = build_term("city", "eq", "London", schema["city"])
    assert set(_run(lf, a & b)["name"]) == {"Ada", "Cy", "Gio"}


def test_combine_or(lf, schema):
    a = build_term("city", "eq", "Rome", schema["city"])
    b = build_term("score", "ge", "9", schema["score"])
    assert set(_run(lf, a | b)["name"]) == {"Ed", "Finn", "Cy"}


def test_unknown_operator_raises(schema):
    with pytest.raises(ValueError, match="unknown operator"):
        build_term("age", "bogus", "1", schema["age"])


# ---------------------------------------------------------------------------
# DatasetRegistry
# ---------------------------------------------------------------------------
def test_registry_load_and_get():
    reg = DatasetRegistry()
    raw = open("sample.csv", "rb").read()
    ds = reg.load("sample.csv", raw)
    assert ds.name == "sample"
    assert ds.shape == (8, 4)
    assert [n for n, _ in ds.columns] == ["name", "age", "city", "score"]
    assert reg.get("sample") is ds
    assert reg.names() == ["sample"]


def test_registry_unique_names():
    reg = DatasetRegistry()
    raw = open("sample.csv", "rb").read()
    reg.load("sample.csv", raw)
    second = reg.load("sample.csv", raw)
    assert second.name == "sample_2"
    assert len(reg) == 2


def test_registry_remove():
    reg = DatasetRegistry()
    ds = reg.load("sample.csv", open("sample.csv", "rb").read())
    reg.remove(ds.name)
    assert reg.get("sample") is None
    assert len(reg) == 0


def test_registry_unsupported_format():
    reg = DatasetRegistry()
    with pytest.raises(Exception):
        reg.load("foo.unknown", b"abc")