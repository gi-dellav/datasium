"""Loaded-dataset registry.

Holds named Polars ``LazyFrame`` sources so UI panels and (future) pipeline
actions can reference a dataset by name without re-reading the file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import polars as pl


class UnsupportedFormatError(ValueError):
    """Raised when a file extension is not mapped to a Polars reader."""


@dataclass(frozen=True)
class Dataset:
    name: str
    source: str  # original file name / label
    lazyframe: pl.LazyFrame

    @property
    def columns(self) -> list[tuple[str, pl.DataType]]:
        """Return ``[(name, dtype), ...]`` without materialising data."""
        return list(self.lazyframe.collect_schema().items())

    @property
    def shape(self) -> tuple[int, int]:
        # The height is unknown until collect for some sources (e.g. CSV),
        # so fall back to a cheap count.
        height = self.lazyframe.select(pl.len()).collect().item()
        width = len(self.lazyframe.collect_schema())
        return int(height), int(width)


_READERS = {
    ".csv": pl.read_csv,
    ".tsv": lambda buf, **kw: pl.read_csv(buf, separator="\t", **kw),
    ".psv": lambda buf, **kw: pl.read_csv(buf, separator="|", **kw),
    ".parquet": pl.read_parquet,
    ".json": pl.read_ndjson,
    ".ndjson": pl.read_ndjson,
    ".ipc": pl.read_ipc,
    ".arrow": pl.read_ipc,
    ".feather": pl.read_ipc,
    ".avro": pl.read_avro,
    ".xlsx": pl.read_excel,
    ".xls": pl.read_excel,
    ".ods": pl.read_excel,
}

SUPPORTED_READ_FORMATS = sorted(_READERS.keys())


def _read_frame(filename: str, raw: bytes) -> pl.LazyFrame:
    ext = Path(filename).suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise UnsupportedFormatError(
            f"No Polars reader registered for {ext or 'unknown'} files. "
            f"Supported: {', '.join(sorted(_READERS))}"
        )
    df = reader(io.BytesIO(raw))
    return df.lazy()


class DatasetRegistry:
    """Ordered, name-keyed collection of loaded datasets."""

    def __init__(self) -> None:
        self._items: dict[str, Dataset] = {}

    def load(self, filename: str, raw: bytes, *, name: str | None = None) -> Dataset:
        label = name or Path(filename).stem
        label = self._unique(label)
        dataset = Dataset(
            name=label, source=filename, lazyframe=_read_frame(filename, raw)
        )
        self._items[label] = dataset
        return dataset

    def get(self, name: str) -> Dataset | None:
        return self._items.get(name)

    def names(self) -> list[str]:
        return list(self._items)

    def remove(self, name: str) -> None:
        self._items.pop(name, None)

    def replace(self, name: str, lazyframe: pl.LazyFrame) -> Dataset:
        """Swap a dataset's LazyFrame in place, keeping its name and source."""
        existing = self._items.get(name)
        if existing is None:
            raise KeyError(f"no dataset named {name!r}")
        updated = Dataset(name=name, source=existing.source, lazyframe=lazyframe)
        self._items[name] = updated
        return updated

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def load_clipboard(self, *, name: str | None = None) -> Dataset:
        df = pl.read_clipboard()
        label = self._unique(name or "clipboard")
        dataset = Dataset(name=label, source="clipboard", lazyframe=df.lazy())
        self._items[label] = dataset
        return dataset

    def load_database(
        self, query: str, connection: str, *, name: str | None = None
    ) -> Dataset:
        df = pl.read_database(query, connection)
        label = self._unique(name or "database")
        dataset = Dataset(name=label, source=f"db:{connection}", lazyframe=df.lazy())
        self._items[label] = dataset
        return dataset

    def load_iceberg(
        self, table: str, *, catalog: str = "default", name: str | None = None
    ) -> Dataset:
        from pyiceberg.catalog import load_catalog

        cat = load_catalog(catalog)
        lf = pl.scan_iceberg(table, catalog=cat)
        label = self._unique(name or table.rsplit(".", 1)[-1])
        dataset = Dataset(
            name=label, source=f"iceberg:{table}", lazyframe=lf
        )
        self._items[label] = dataset
        return dataset

    def _unique(self, label: str) -> str:
        if label not in self._items:
            return label
        i = 2
        while f"{label}_{i}" in self._items:
            i += 1
        return f"{label}_{i}"
