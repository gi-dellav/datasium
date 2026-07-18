"""Write / export module.

Pure helpers to persist a Polars ``LazyFrame`` to the filesystem and a
``WritePanel`` UI that wires four operations:

* **Save edits** — overwrite the active dataset's source file with its
  current LazyFrame.
* **Save selection** — apply the Select-tab filters and column selection to
  overwrite the active dataset in the registry.
* **Export dataset** — write the active dataset's full LazyFrame to a new
  file path and optionally register it under a new name.
* **Export selection** — write the filtered + projected LazyFrame to a new
  file path and optionally register it under a new name.

Pure helpers are UI-free so they can be unit-tested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl

from nicegui import ui


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
_WRITERS: dict[str, Callable[[pl.DataFrame, str], None]] = {
    ".csv": lambda df, p: df.write_csv(p),
    ".tsv": lambda df, p: df.write_csv(p, separator="\t"),
    ".parquet": lambda df, p: df.write_parquet(p),
    ".json": lambda df, p: df.write_ndjson(p),
    ".ndjson": lambda df, p: df.write_ndjson(p),
    ".ipc": lambda df, p: df.write_ipc(p),
}

SUPPORTED_FORMATS = sorted(_WRITERS.keys())


def save_frame(lf: pl.LazyFrame, path: str) -> str:
    """Collect ``lf`` and persist it to *path*.  Returns the resolved path.

    The file extension determines the format (see SUPPORTED_FORMATS).
    Raises ``ValueError`` if the extension is not recognised.
    """
    path = os.path.abspath(path)
    ext = Path(path).suffix.lower()
    writer = _WRITERS.get(ext)
    if writer is None:
        raise ValueError(
            f"no writer registered for {ext or 'unknown'} files. "
            f"Supported: {', '.join(sorted(_WRITERS))}"
        )
    df = lf.collect()
    writer(df, path)
    return path


def apply_selection(
    lf: pl.LazyFrame,
    expr: pl.Expr | None = None,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Return a new LazyFrame with *expr* filter and *columns* projection applied.

    A ``None`` filter or empty column list leaves that dimension unchanged.
    """
    if expr is not None:
        lf = lf.filter(expr)
    if columns:
        lf = lf.select(columns)
    return lf


# ---------------------------------------------------------------------------
# UI panel
# ---------------------------------------------------------------------------
@dataclass
class _WriteRow:
    label: str
    desc: str
    btn_label: str
    btn_icon: str
    action: Callable[[], None]


class WritePanel:
    """Four-action panel for persisting the active dataset."""

    def __init__(
        self,
        parent,
        *,
        on_save_edits: Callable[[], None],
        on_save_selection: Callable[[], None],
        on_export_dataset: Callable[[str, str], None],   # path, name-or-""
        on_export_selection: Callable[[str, str], None],  # path, name-or-""
    ) -> None:
        self._on_save_edits = on_save_edits
        self._on_save_selection = on_save_selection
        self._on_export_dataset = on_export_dataset
        self._on_export_selection = on_export_selection

        with parent:
            # --- 1. Save edits (overwrite source file) ---
            ui.label("Save edits").classes("text-lg font-medium mt-2")
            ui.label(
                "Overwrite the original source file with the current state of the dataset."
            ).classes("text-xs opacity-50")
            ui.button(
                "Save edits", icon="save",
                on_click=lambda _=None: self._on_save_edits(),
            ).props("dense unelevated color=primary")

            ui.separator()

            # --- 2. Save selection into the current dataset ---
            ui.label("Save selection into dataset").classes("text-lg font-medium mt-2")
            ui.label(
                "Replace the active dataset with the rows and columns defined "
                "by the Select tab filters."
            ).classes("text-xs opacity-50")
            ui.button(
                "Save selection into dataset", icon="content_copy",
                on_click=lambda _=None: self._on_save_selection(),
            ).props("dense unelevated color=primary")

            ui.separator()

            # --- 3. Export dataset to a new file ---
            ui.label("Export dataset").classes("text-lg font-medium mt-2")
            ui.label(
                "Write the full active dataset to a new file. "
                "Optionally name it for the registry."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full"):
                self.export_ds_path = ui.input(
                    value="", label="File path",
                    placeholder="e.g. /tmp/out.csv",
                ).props("dense outlined").classes("w-64")
                self.export_ds_name = ui.input(
                    value="", label="Registry name (optional)",
                ).props("dense outlined").classes("w-40")
                ui.button(
                    "Export dataset", icon="file_download",
                    on_click=lambda _=None: self._on_export_dataset(
                        self.export_ds_path.value or "",
                        self.export_ds_name.value or "",
                    ),
                ).props("dense unelevated color=primary")

            ui.separator()

            # --- 4. Export selection to a new file / dataset ---
            ui.label("Export selection").classes("text-lg font-medium mt-2")
            ui.label(
                "Write the current selection (filters + columns from the "
                "Select tab) to a new file. Optionally name it for the registry."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full"):
                self.export_sel_path = ui.input(
                    value="", label="File path",
                    placeholder="e.g. /tmp/selection.csv",
                ).props("dense outlined").classes("w-64")
                self.export_sel_name = ui.input(
                    value="", label="Registry name (optional)",
                ).props("dense outlined").classes("w-40")
                ui.button(
                    "Export selection", icon="file_save",
                    on_click=lambda _=None: self._on_export_selection(
                        self.export_sel_path.value or "",
                        self.export_sel_name.value or "",
                    ),
                ).props("dense unelevated color=primary")
