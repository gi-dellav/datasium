"""Tests for dataset readers and registry loading methods."""

from __future__ import annotations

import io
import os
import tempfile

import polars as pl
import pytest

from datasium.dataset import (
    DatasetRegistry,
    UnsupportedFormatError,
    _read_frame,
    SUPPORTED_READ_FORMATS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]},
    )


@pytest.fixture
def registry() -> DatasetRegistry:
    return DatasetRegistry()


# ---------------------------------------------------------------------------
# SUPPORTED_READ_FORMATS
# ---------------------------------------------------------------------------
def test_supported_read_formats_include_new_entries():
    for ext in (".avro", ".xlsx", ".xls", ".ods"):
        assert ext in SUPPORTED_READ_FORMATS


# ---------------------------------------------------------------------------
# _read_frame — Avro
# ---------------------------------------------------------------------------
def test_read_frame_avro(df):
    pytest.importorskip("fastavro")
    buf = io.BytesIO()
    df.write_avro(buf)
    raw = buf.getvalue()
    lf = _read_frame("data.avro", raw)
    result = lf.collect()
    assert result.shape == df.shape
    assert result.columns == df.columns


# ---------------------------------------------------------------------------
# _read_frame — Excel
# ---------------------------------------------------------------------------
def test_read_frame_xlsx(df):
    pytest.importorskip("fastexcel")
    buf = io.BytesIO()
    df.write_excel(buf)
    raw = buf.getvalue()
    lf = _read_frame("data.xlsx", raw)
    result = lf.collect()
    assert result.shape == df.shape
    assert result.columns == df.columns


# ---------------------------------------------------------------------------
# _read_frame — unsupported extension
# ---------------------------------------------------------------------------
def test_read_frame_unsupported():
    with pytest.raises(UnsupportedFormatError, match="No Polars reader"):
        _read_frame("data.xyz", b"")


# ---------------------------------------------------------------------------
# Registry.load round-trips for new formats
# ---------------------------------------------------------------------------
def test_registry_load_avro(df, registry):
    pytest.importorskip("fastavro")
    buf = io.BytesIO()
    df.write_avro(buf)
    ds = registry.load("test.avro", buf.getvalue())
    assert ds.name == "test"
    assert ds.lazyframe.collect().shape == df.shape


def test_registry_load_xlsx(df, registry):
    pytest.importorskip("fastexcel")
    buf = io.BytesIO()
    df.write_excel(buf)
    ds = registry.load("test.xlsx", buf.getvalue())
    assert ds.name == "test"
    assert ds.lazyframe.collect().shape == df.shape


# ---------------------------------------------------------------------------
# Registry.load_clipboard
# ---------------------------------------------------------------------------
def test_registry_load_clipboard(df, registry, monkeypatch):
    tsv = df.write_csv(separator="\t")
    monkeypatch.setattr(pl, "read_clipboard", lambda: pl.read_csv(
        io.BytesIO(tsv.encode()), separator="\t"
    ))
    ds = registry.load_clipboard(name="clip")
    assert ds.name == "clip"
    assert ds.source == "clipboard"
    assert ds.lazyframe.collect().shape == df.shape


# ---------------------------------------------------------------------------
# Registry.load_database
# ---------------------------------------------------------------------------
def test_registry_load_database(df, registry, monkeypatch):
    monkeypatch.setattr(pl, "read_database", lambda q, c: df)
    ds = registry.load_database("SELECT 1", "sqlite://test.db", name="db")
    assert ds.name == "db"
    assert ds.source == "db:sqlite://test.db"
    assert ds.lazyframe.collect().shape == df.shape


# ---------------------------------------------------------------------------
# Registry.load_iceberg
# ---------------------------------------------------------------------------
def test_registry_load_iceberg(df, registry, monkeypatch):
    lf = df.lazy()
    monkeypatch.setattr(pl, "scan_iceberg", lambda source, **kw: lf)

    class _FakeCatalog:
        def load_table(self, identifier):
            return identifier

    import types
    fake_pyiceberg = types.ModuleType("pyiceberg")
    fake_catalog_mod = types.ModuleType("pyiceberg.catalog")
    fake_catalog_mod.load_catalog = lambda name: _FakeCatalog()
    fake_pyiceberg.catalog = fake_catalog_mod
    monkeypatch.setitem(__import__("sys").modules, "pyiceberg", fake_pyiceberg)
    monkeypatch.setitem(__import__("sys").modules, "pyiceberg.catalog", fake_catalog_mod)

    ds = registry.load_iceberg("ns.my_table", name="ice")
    assert ds.name == "ice"
    assert ds.source == "iceberg:ns.my_table"
    assert ds.lazyframe.collect().shape == df.shape
