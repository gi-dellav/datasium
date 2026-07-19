"""Tests for the pure write helpers in ``datasium.write``."""

from __future__ import annotations

import os
import tempfile

import polars as pl
import pytest

from datasium.write import apply_selection, save_frame, copy_to_clipboard, write_to_database, SUPPORTED_FORMATS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]},
    )


@pytest.fixture
def lf(df) -> pl.LazyFrame:
    return df.lazy()


# ---------------------------------------------------------------------------
# save_frame
# ---------------------------------------------------------------------------
def test_save_frame_csv(lf, df):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.csv")
        result = save_frame(lf, path)
        assert result == path
        roundtrip = pl.read_csv(path)
        assert roundtrip.shape == df.shape
        assert roundtrip.columns == df.columns


def test_save_frame_tsv(lf, df):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.tsv")
        save_frame(lf, path)
        roundtrip = pl.read_csv(path, separator="\t")
        assert roundtrip.shape == df.shape


def test_save_frame_parquet(lf, df):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.parquet")
        save_frame(lf, path)
        roundtrip = pl.read_parquet(path)
        assert roundtrip.shape == df.shape


def test_save_frame_json(lf, df):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.ndjson")
        save_frame(lf, path)
        roundtrip = pl.read_ndjson(path)
        assert roundtrip.shape == df.shape


def test_save_frame_unknown_extension(lf):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.xyz")
        with pytest.raises(ValueError, match="no writer"):
            save_frame(lf, path)


# ---------------------------------------------------------------------------
# apply_selection
# ---------------------------------------------------------------------------
def test_apply_selection_noop(lf):
    result = apply_selection(lf)
    assert result.collect().shape == (3, 3)


def test_apply_selection_filter_only(lf):
    expr = pl.col("a") > 1
    result = apply_selection(lf, expr=expr)
    df = result.collect()
    assert df.shape == (2, 3)
    assert df["a"].to_list() == [2, 3]


def test_apply_selection_columns_only(lf):
    result = apply_selection(lf, columns=["a", "b"])
    df = result.collect()
    assert df.columns == ["a", "b"]
    assert df.shape == (3, 2)


def test_apply_selection_both(lf):
    expr = pl.col("a") > 1
    result = apply_selection(lf, expr=expr, columns=["a"])
    df = result.collect()
    assert df.columns == ["a"]
    assert df["a"].to_list() == [2, 3]


def test_apply_selection_empty_columns(lf):
    result = apply_selection(lf, columns=[])
    assert result.collect().shape == (3, 3)


# ---------------------------------------------------------------------------
# SUPPORTED_FORMATS
# ---------------------------------------------------------------------------
def test_supported_formats_include_new_entries():
    for ext in (".avro", ".xlsx", ".feather", ".arrow"):
        assert ext in SUPPORTED_FORMATS


# ---------------------------------------------------------------------------
# save_frame — new formats
# ---------------------------------------------------------------------------
def test_save_frame_avro(lf, df):
    pytest.importorskip("fastavro")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.avro")
        save_frame(lf, path)
        roundtrip = pl.read_avro(path)
        assert roundtrip.shape == df.shape
        assert roundtrip.columns == df.columns


def test_save_frame_xlsx(lf, df):
    pytest.importorskip("xlsxwriter")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.xlsx")
        save_frame(lf, path)
        pytest.importorskip("fastexcel")
        roundtrip = pl.read_excel(path)
        assert roundtrip.shape == df.shape
        assert roundtrip.columns == df.columns


def test_save_frame_feather(lf, df):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.feather")
        save_frame(lf, path)
        roundtrip = pl.read_ipc(path)
        assert roundtrip.shape == df.shape
        assert roundtrip.columns == df.columns


def test_save_frame_arrow(lf, df):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.arrow")
        save_frame(lf, path)
        roundtrip = pl.read_ipc(path)
        assert roundtrip.shape == df.shape
        assert roundtrip.columns == df.columns


# ---------------------------------------------------------------------------
# copy_to_clipboard
# ---------------------------------------------------------------------------
def test_copy_to_clipboard(lf, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        pl.DataFrame, "write_clipboard", lambda self: captured.update(data=self)
    )
    copy_to_clipboard(lf)
    assert captured["data"].shape == (3, 3)


# ---------------------------------------------------------------------------
# write_to_database
# ---------------------------------------------------------------------------
def test_write_to_database(lf, monkeypatch):
    calls = []
    monkeypatch.setattr(
        pl.DataFrame,
        "write_database",
        lambda self, table, conn: calls.append((table, conn)),
    )
    write_to_database(lf, "my_table", "sqlite://test.db")
    assert calls == [("my_table", "sqlite://test.db")]
