"""datasium application UI (NiceGUI).

A polished, lightweight workbench for loading Polars datasets and inspecting
their schema. Designed so operation panels can be layered on top of the
``DatasetRegistry`` shared with the UI.
"""

from __future__ import annotations

from pathlib import Path

import argparse

from nicegui import events, ui

from datasium.calculate import (
    Calculator,
    _is_numeric as _is_numeric_dtype,
    compute_stat,
)
from datasium.dataset import Dataset, DatasetRegistry, UnsupportedFormatError, SUPPORTED_READ_FORMATS
from datasium.filter import FilterBuilder
from datasium.query import QueryEntry, QueryPanel, run_sql
from datasium.edit import (
    DTYPE_BY_KEY,
    EditPanel,
    add_column as _edit_add_column,
    add_row as _edit_add_row,
    cast_column as _edit_cast_column,
    fill_nulls as _edit_fill_nulls,
    replace_values as _edit_replace_values,
    set_cell_by_index as _edit_set_cell_by_index,
    set_cell_by_key as _edit_set_cell_by_key,
)
from datasium.remove import RemovePanel, apply_removal
from datasium.plot import PlotPanel, build_figure
from datasium.transform import (
    TransformPanel,
    add_computed_column,
    group_by_agg,
    join_frames,
    one_hot_encode,
    pivot_frame,
    rename_column,
    sort_frame,
    unpivot_frame,
)
from datasium.write import WritePanel, save_frame, apply_selection, copy_to_clipboard, write_to_database, SUPPORTED_FORMATS

import polars as pl

_APP_TITLE = "datasium"
_APP_TAGLINE = "a visual data-workbench · Polars"


def _human_dtype(dtype) -> str:
    return str(dtype)


def _format_rows(n: int) -> str:
    return f"{n:,}"


class App:
    """Owns the registry and renders the workbench."""

    def __init__(self) -> None:
        self.registry = DatasetRegistry()
        self.active_name: str | None = None
        self.selected_columns: list[str] | None = None
        self.filter_builder: FilterBuilder | None = None
        self.preview_mode = "selected"
        self.calculator: Calculator | None = None
        self.query_history: list[QueryEntry] = []
        self.query_panel: QueryPanel | None = None
        self.remove_panel: RemovePanel | None = None
        self.remove_meta = None
        self.remove_preview_container = None
        self._remove_preview = None
        self.edit_panel: EditPanel | None = None
        self.write_panel: WritePanel | None = None
        self.plot_panel: PlotPanel | None = None
        self.transform_panel: TransformPanel | None = None
        self.preview_limit_mode = "all"  # "all" | "first" | "random"
        self.preview_limit_n = 100

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        ui.add_head_html(
            "<style>"
            "body { background: var(--nicegui-default-background-color); }"
            ".ds-card { border-radius: 14px; }"
            ".ds-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }"
            "</style>"
        )

        with ui.header().classes("items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("table_chart").classes("text-2xl")
                with ui.column().classes("gap-0"):
                    ui.label(_APP_TITLE).classes("text-xl font-semibold leading-tight")
                    ui.label(_APP_TAGLINE).classes("text-xs opacity-60")
            ui.button(
                icon="dark_mode",
                on_click=lambda: ui.run_javascript(
                    "document.body.classList.toggle('dark')"
                ),
            ).props("flat round").tooltip("Toggle theme")

        with ui.tabs() as self.tabs:
            ui.tab("Load", icon="upload_file")
            ui.tab("Select", icon="filter_alt")
            ui.tab("View", icon="visibility")
            ui.tab("Edit", icon="edit")
            ui.tab("Query", icon="query_stats")
            ui.tab("Remove", icon="delete_sweep")
            ui.tab("Transform", icon="transform")
            ui.tab("Calculate", icon="functions")
            ui.tab("Plot", icon="show_chart")
            ui.tab("Write", icon="save")

        with ui.tab_panels(self.tabs, value="Load").classes("w-full"):
            with ui.tab_panel("Load"):
                self.load_container = ui.column().classes("w-full p-4")
                self._render_load_tab()
            with ui.tab_panel("Select"):
                self.select_container = ui.column().classes("w-full p-4")
                self._render_select_tab()
            with ui.tab_panel("View"):
                self.view_container = ui.column().classes("w-full p-4")
                self._render_view_tab()
            with ui.tab_panel("Edit"):
                self.edit_container = ui.column().classes("w-full p-4")
                self._render_edit_tab()
            with ui.tab_panel("Query"):
                self.query_container = ui.column().classes("w-full p-4")
                self._render_query_tab()
            with ui.tab_panel("Remove"):
                self.remove_container = ui.column().classes("w-full p-4")
                self._render_remove_tab()
            with ui.tab_panel("Transform"):
                self.transform_container = ui.column().classes("w-full p-4")
                self._render_transform_tab()
            with ui.tab_panel("Calculate"):
                self.calc_container = ui.column().classes("w-full p-4")
                self._render_calculate_tab()
            with ui.tab_panel("Plot"):
                self.plot_root = ui.column().classes("w-full p-4")
                self._render_plot_tab()
            with ui.tab_panel("Write"):
                self.write_container = ui.column().classes("w-full p-4")
                self._render_write_tab()

    # ---------------------------------------------------------------- load tab
    def _render_load_tab(self) -> None:
        self.load_container.clear()
        with self.load_container:
            ui.label("Load dataset").classes("text-lg font-medium")
            ui.label(
                f"Supported file formats: {', '.join(SUPPORTED_READ_FORMATS)}"
            ).classes("text-xs opacity-50")
            ui.separator()
            ui.upload(
                label="Choose a file",
                multiple=False,
                auto_upload=True,
                on_upload=self._on_upload,
                on_rejected=lambda: ui.notify(
                    "File rejected",
                    type="negative",
                    position="top",
                ),
            ).props(
                'accept=".csv,.tsv,.psv,.parquet,.json,.ndjson,.ipc,.arrow,'
                '.feather,.avro,.xlsx,.xls,.ods" '
                'color="primary"'
            ).classes(
                "w-full"
            )

            ui.separator()

            # --- Clipboard import ---
            ui.label("Paste from clipboard").classes("text-lg font-medium mt-2")
            ui.label(
                "Read tab-separated data from the system clipboard."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2"):
                self.clipboard_name = (
                    ui.input(value="", label="Dataset name (optional)")
                    .props("dense outlined")
                    .classes("w-40")
                )
                ui.button(
                    "Paste",
                    icon="content_paste",
                    on_click=self._on_load_clipboard,
                ).props("dense unelevated color=primary")

            ui.separator()

            # --- Database import ---
            ui.label("Load from database").classes("text-lg font-medium mt-2")
            ui.label(
                "Run a SQL query against a database via a connection URI "
                "(requires connectorx)."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full"):
                self.db_uri = (
                    ui.input(
                        value="",
                        label="Connection URI",
                        placeholder="e.g. postgresql://user:pass@host/db",
                    )
                    .props("dense outlined")
                    .classes("w-64")
                )
                self.db_query = (
                    ui.input(
                        value="",
                        label="SQL query",
                        placeholder="e.g. SELECT * FROM my_table",
                    )
                    .props("dense outlined")
                    .classes("w-64")
                )
                self.db_load_name = (
                    ui.input(value="", label="Dataset name (optional)")
                    .props("dense outlined")
                    .classes("w-40")
                )
                ui.button(
                    "Load",
                    icon="storage",
                    on_click=self._on_load_database,
                ).props("dense unelevated color=primary")

            ui.separator()

            # --- Iceberg import ---
            ui.label("Load from Iceberg").classes("text-lg font-medium mt-2")
            ui.label(
                "Read an Apache Iceberg table (requires pyiceberg)."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full"):
                self.iceberg_catalog = (
                    ui.input(
                        value="default",
                        label="Catalog name",
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
                self.iceberg_table = (
                    ui.input(
                        value="",
                        label="Table identifier",
                        placeholder="e.g. namespace.table_name",
                    )
                    .props("dense outlined")
                    .classes("w-64")
                )
                self.iceberg_name = (
                    ui.input(value="", label="Dataset name (optional)")
                    .props("dense outlined")
                    .classes("w-40")
                )
                ui.button(
                    "Load",
                    icon="ice_skating",
                    on_click=self._on_load_iceberg,
                ).props("dense unelevated color=primary")

            ui.separator()
            self.list_container = ui.column().classes("w-full gap-1")
            self._render_list()

    def _render_list(self) -> None:
        self.list_container.clear()
        if len(self.registry) == 0:
            with self.list_container:
                ui.label("No datasets loaded yet.").classes("text-sm opacity-50")
            return
        with self.list_container:
            for ds in self.registry:
                active = ds.name == self.active_name
                with (
                    ui.button(on_click=lambda _, n=ds.name: self._select(n))
                    .props(f"flat align=left {''if not active else 'outline'}")
                    .classes("w-full justify-start")
                ):
                    ui.icon("description" if not active else "description_outlined")
                    with ui.column().classes("gap-0 items-start"):
                        ui.label(ds.name).classes("font-medium")
                        ui.label(Path(ds.source).name).classes("text-xs opacity-60")
            if self.active_name:
                ui.separator()
                ui.button(
                    "Remove dataset",
                    icon="delete_outline",
                    on_click=self._on_remove,
                ).props("flat color=negative").classes("w-full")

    # -------------------------------------------------------------- select tab
    def _render_select_tab(self) -> None:
        self.select_container.clear()
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.select_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("inbox", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        rows, cols = ds.shape
        with self.select_container:
            with ui.row().classes("items-center justify-between w-full"):
                with ui.column().classes("gap-0"):
                    ui.label(ds.name).classes("text-xl font-semibold")
                    ui.label(ds.source).classes("text-sm opacity-60 ds-mono")
            with ui.row().classes("gap-4"):
                self._stat("Columns", str(cols), "view_column")
                self._stat("Rows", _format_rows(rows), "table_rows")
                self._stat("Source", Path(ds.source).suffix or "—", "save")

            ui.separator()
            ui.label("Columns").classes("text-lg font-medium mt-2")
            schema_cols = [
                {
                    "name": "idx",
                    "label": "#",
                    "field": "idx",
                    "align": "left",
                    "sortable": True,
                },
                {
                    "name": "name",
                    "label": "Name",
                    "field": "name",
                    "align": "left",
                    "sortable": True,
                },
                {
                    "name": "dtype",
                    "label": "Type",
                    "field": "dtype",
                    "align": "left",
                    "sortable": True,
                },
            ]
            schema_rows = [
                {"idx": i + 1, "name": name, "dtype": _human_dtype(dtype)}
                for i, (name, dtype) in enumerate(ds.columns)
            ]
            ui.table(columns=schema_cols, rows=schema_rows, row_key="idx").props(
                "flat dense rows-per-page-options=[0]"
            ).classes("w-full")

            self._build_filter_panel(ds)

    # ---------------------------------------------------------------- view tab
    def _render_view_tab(self) -> None:
        self.view_container.clear()
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.view_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("visibility", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.view_container:
            self._build_select_panel(ds)
            self._build_preview_panel(ds)

    # -------------------------------------------------------------- remove tab
    def _render_remove_tab(self) -> None:
        self.remove_container.clear()
        self.remove_panel = None
        self.remove_meta = None
        self.remove_preview_container = None
        self._remove_preview = None
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.remove_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("delete_sweep", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.remove_container:
            with ui.row().classes("items-center justify-between w-full"):
                with ui.column().classes("gap-0"):
                    ui.label("Remove").classes("text-lg font-medium")
                    ui.label(
                        "Drop columns and/or rows from the active dataset."
                    ).classes("text-xs opacity-50")
            self.remove_panel = RemovePanel(
                self.remove_container,
                ds.columns,
                self._run_remove_preview,
                self._apply_remove,
            )
            self.remove_panel.set_selection_expr_provider(
                lambda: (
                    self.filter_builder.build_expr() if self.filter_builder else None
                ),
            )
            self.remove_meta = ui.label("").classes("text-xs opacity-50")
            self.remove_preview_container = ui.column().classes("w-full")

    def _run_remove_preview(self) -> None:
        if self.remove_preview_container is None or self.remove_panel is None:
            return
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            return
        before_rows, before_cols = ds.shape
        try:
            new_lf = apply_removal(ds.lazyframe, self.remove_panel.spec)
            df = new_lf.collect()
        except ValueError as err:
            self._remove_preview = None
            ui.notify(str(err), type="warning", position="top")
            if self.remove_meta is not None:
                self.remove_meta.set_text("Removal not applied.")
            self.remove_preview_container.clear()
            return
        except Exception as err:
            self._remove_preview = None
            ui.notify(
                f"Could not build removal: {err}", type="negative", position="top"
            )
            if self.remove_meta is not None:
                self.remove_meta.set_text("Error.")
            self.remove_preview_container.clear()
            return

        self._remove_preview = df
        after_rows, after_cols = df.shape
        d_rows = before_rows - after_rows
        d_cols = before_cols - after_cols
        parts = []
        if d_rows:
            parts.append(f"{d_rows:,} row(s)")
        if d_cols:
            parts.append(f"{d_cols:,} column(s)")
        summary = (
            " and ".join(parts) + " will be removed" if parts else "nothing to remove"
        )
        if self.remove_meta is not None:
            self.remove_meta.set_text(
                f"Result: {after_rows:,} row(s) · {after_cols:,} column(s) — {summary}."
            )

        self.remove_preview_container.clear()
        with self.remove_preview_container:
            if df.width == 0:
                ui.label("No columns remain.").classes("text-sm opacity-50")
                return
            columns = [
                {
                    "name": c,
                    "label": f"{c}\n{_human_dtype(df.schema[c])}",
                    "field": c,
                    "align": "left",
                    "sortable": True,
                }
                for c in df.columns
            ]
            rows = df.rows(named=True)
            ui.table(columns=columns, rows=rows, row_key=df.columns[0]).props(
                "flat dense"
            ).classes("w-full")

    def _apply_remove(self) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None or self._remove_preview is None:
            ui.notify("Preview the removal first", type="warning", position="top")
            return
        try:
            new_lf = apply_removal(ds.lazyframe, self.remove_panel.spec)  # type: ignore[union-attr]
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self.registry.replace(ds.name, new_lf)
        ds_after = self.registry.get(self.active_name)
        ar, ac = ds_after.shape if ds_after is not None else (0, 0)
        self.selected_columns = None
        self._refresh_all_tabs()
        ui.notify(
            f"Removed · now {ar:,} row(s) × {ac} column(s)",
            type="positive",
            position="top",
        )

    # ------------------------------------------------------------ transform tab
    def _render_transform_tab(self) -> None:
        self.transform_container.clear()
        self.transform_panel = None
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.transform_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("transform", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.transform_container:
            with ui.row().classes("items-center justify-between w-full"):
                with ui.column().classes("gap-0"):
                    ui.label("Transform").classes("text-lg font-medium")
                    ui.label(
                        "Sort, rename, create computed columns, and run "
                        "group-by aggregations."
                    ).classes("text-xs opacity-50")
            self.transform_panel = TransformPanel(
                self.transform_container,
                ds.columns,
                on_sort=self._on_transform_sort,
                on_rename=self._on_transform_rename,
                on_computed=self._on_transform_computed,
                on_group_by=self._on_transform_group_by,
                on_one_hot=self._on_transform_one_hot,
                on_pivot=self._on_transform_pivot,
                on_unpivot=self._on_transform_unpivot,
                on_join=self._on_transform_join,
                dataset_names=[
                    n for n in self.registry.names() if n != self.active_name
                ],
            )

    def _on_transform_sort(self, columns: list[str], descending: list[bool]) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            new_lf = sort_frame(ds.lazyframe, columns, descending)
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_transform(new_lf, "Sorted")

    def _on_transform_rename(self, old: str, new: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            new_lf = rename_column(ds.lazyframe, old, new)
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_transform(new_lf, f"Renamed {old} → {new}")

    def _on_transform_computed(
        self,
        name: str,
        category: str,
        op: str,
        col_a: str | None,
        col_b: str | None,
        scalar: str | None,
        then_value: str | None,
        else_value: str | None,
    ) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            new_lf = add_computed_column(
                ds.lazyframe,
                name,
                category,
                op,
                col_a=col_a,
                col_b=col_b,
                scalar=scalar,
                then_value=then_value,
                else_value=else_value,
            )
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_transform(new_lf, f"Added computed column {name!r}")

    def _on_transform_group_by(
        self,
        group_cols: list[str],
        agg_col: str | None,
        agg_op: str,
        out_name: str,
    ) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            new_lf = group_by_agg(ds.lazyframe, group_cols, agg_col, agg_op, out_name)
            df = new_lf.collect()
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        except Exception as err:
            ui.notify(f"Could not aggregate: {err}", type="negative", position="top")
            return
        base = self.active_name or "dataset"
        label = f"{base}_grouped"
        from datasium.dataset import Dataset

        unique = self.registry._unique(label)
        new_ds = Dataset(name=unique, source=f"group_by({base})", lazyframe=df.lazy())
        self.registry._items[unique] = new_ds
        self.active_name = unique
        self._refresh_all_tabs()
        ar, ac = df.shape
        ui.notify(
            f"Group-by → new dataset {unique!r} · {ar:,} row(s) × {ac} column(s)",
            type="positive",
            position="top",
        )

    def _on_transform_one_hot(self, column: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            new_lf = one_hot_encode(ds.lazyframe, column)
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_transform(new_lf, f"One-hot encoded {column!r}")

    def _on_transform_pivot(
        self, index: list[str], columns: str, values: str, agg: str
    ) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            new_lf = pivot_frame(ds.lazyframe, index, columns, values, agg)
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_transform(new_lf, "Pivoted")

    def _on_transform_unpivot(
        self, id_vars: list[str], value_vars: list[str], var_name: str, val_name: str
    ) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            new_lf = unpivot_frame(
                ds.lazyframe, id_vars, value_vars or None, var_name, val_name
            )
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_transform(new_lf, "Unpivoted")

    def _on_transform_join(
        self, other_name: str, left_on: list[str], right_on: list[str], how: str
    ) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        other = self.registry.get(other_name)
        if ds is None or other is None:
            ui.notify("Select both datasets", type="warning", position="top")
            return
        try:
            new_lf = join_frames(ds.lazyframe, other.lazyframe, left_on, right_on, how)
            df = new_lf.collect()
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        except Exception as err:
            ui.notify(f"Could not join: {err}", type="negative", position="top")
            return
        base = self.active_name or "dataset"
        label = f"{base}_joined_{other_name}"
        unique = self.registry._unique(label)
        new_ds = Dataset(
            name=unique, source=f"join({base}, {other_name})", lazyframe=df.lazy()
        )
        self.registry._items[unique] = new_ds
        self.active_name = unique
        self._refresh_all_tabs()
        ar, ac = df.shape
        ui.notify(
            f"Join → new dataset {unique!r} · {ar:,} row(s) × {ac} column(s)",
            type="positive",
            position="top",
        )

    def _apply_transform(self, new_lf: pl.LazyFrame, label: str) -> bool:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return False
        try:
            new_lf.collect_schema()
            df = new_lf.collect()
        except Exception as err:
            ui.notify(
                f"Could not apply transform: {err}", type="negative", position="top"
            )
            return False
        self.registry.replace(ds.name, df.lazy())
        self.selected_columns = None
        self._refresh_all_tabs()
        ar, ac = df.shape
        ui.notify(
            f"{label} · now {ar:,} row(s) × {ac} column(s)",
            type="positive",
            position="top",
        )
        return True

    def _refresh_all_tabs(self) -> None:
        self._render_list()
        self._render_select_tab()
        self._render_view_tab()
        self._render_remove_tab()
        self._render_edit_tab()
        self._render_transform_tab()
        self._render_calculate_tab()
        self._render_query_tab()
        self._render_plot_tab()
        self._render_write_tab()

    # ---------------------------------------------------------------- edit tab
    def _render_edit_tab(self) -> None:
        self.edit_container.clear()
        self.edit_panel = None
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.edit_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("edit", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.edit_container:
            with ui.row().classes("items-center justify-between w-full"):
                with ui.column().classes("gap-0"):
                    ui.label("Edit").classes("text-lg font-medium")
                    ui.label(
                        "Cast column types, add columns/rows, and edit cells."
                    ).classes("text-xs opacity-50")
            self.edit_panel = EditPanel(
                self.edit_container,
                ds.columns,
                self._on_edit_cast,
                self._on_edit_add_column,
                self._on_edit_add_row,
                self._on_edit_edit_row,
                self._on_edit_fill_nulls,
                self._on_edit_replace_values,
            )

    # -- edit handlers ----------------------------------------------------
    def _apply_edit(self, new_lf: pl.LazyFrame, label: str) -> bool:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return False
        try:
            new_lf.collect_schema()
            df = new_lf.collect()
        except Exception as err:  # dtype cast failures, bad literals, ...
            ui.notify(f"Could not apply edit: {err}", type="negative", position="top")
            return False
        self.registry.replace(ds.name, df.lazy())
        self.selected_columns = None
        self._refresh_all_tabs()
        ar, ac = df.shape
        ui.notify(
            f"{label} · now {ar:,} row(s) × {ac} column(s)",
            type="positive",
            position="top",
        )
        return True

    def _on_edit_cast(self, column: str, dtype_key: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None or column in (None, "—"):
            ui.notify("select a column", type="warning", position="top")
            return
        if dtype_key not in DTYPE_BY_KEY:
            ui.notify("select a target dtype", type="warning", position="top")
            return
        try:
            new_lf = _edit_cast_column(ds.lazyframe, column, DTYPE_BY_KEY[dtype_key])
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_edit(new_lf, f"Cast {column} → {dtype_key}")

    def _on_edit_add_column(self, name: str, dtype_key: str, fill: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        if dtype_key not in DTYPE_BY_KEY:
            ui.notify("select a column type", type="warning", position="top")
            return
        try:
            new_lf = _edit_add_column(ds.lazyframe, name, DTYPE_BY_KEY[dtype_key], fill)
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_edit(new_lf, f"Added column {name!r}")

    def _on_edit_add_row(self, values: dict[str, str]) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            new_lf = _edit_add_row(ds.lazyframe, values)
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_edit(new_lf, "Added row")

    def _on_edit_edit_row(
        self,
        mode: str,
        index: object,
        key_cols: list[str],
        key_vals: list[str],
        column: str,
        new_raw: str,
    ) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None or column in (None, "—"):
            ui.notify("select a target column", type="warning", position="top")
            return
        schema = dict(ds.lazyframe.collect_schema().items())
        dtype = schema.get(column)
        if dtype is None:
            ui.notify(f"column {column!r} not found", type="warning", position="top")
            return
        try:
            if mode == "index":
                new_lf = _edit_set_cell_by_index(
                    ds.lazyframe, int(index), column, new_raw, dtype
                )
            else:
                new_lf = _edit_set_cell_by_key(
                    ds.lazyframe, key_cols, key_vals, column, new_raw, dtype
                )
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_edit(new_lf, f"Edited {column}")

    def _on_edit_fill_nulls(self, column: str, strategy: str, fill_value: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None or not column:
            ui.notify("select a column", type="warning", position="top")
            return
        try:
            new_lf = _edit_fill_nulls(ds.lazyframe, column, strategy, fill_value)
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_edit(new_lf, f"Filled nulls in {column}")

    def _on_edit_replace_values(self, column: str, old_raw: str, new_raw: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None or not column:
            ui.notify("select a column", type="warning", position="top")
            return
        try:
            new_lf = _edit_replace_values(ds.lazyframe, column, old_raw, new_raw)
        except ValueError as err:
            ui.notify(str(err), type="warning", position="top")
            return
        self._apply_edit(new_lf, f"Replaced values in {column}")

    # ------------------------------------------------------------- calculate tab
    def _render_calculate_tab(self) -> None:
        self.calc_container.clear()
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.calc_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("functions", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.calc_container:
            with ui.row().classes("items-center justify-between w-full"):
                with ui.column().classes("gap-0"):
                    ui.label("Calculate").classes("text-lg font-medium")
                    desc = "Run a statistic over the rows that pass the filters "
                    desc += "defined in the Select tab (every row when none set)."
                    ui.label(desc).classes("text-xs opacity-50")
            numeric = [(n, d) for n, d in ds.columns if _is_numeric_dtype(d)]
            if not numeric:
                ui.label("This dataset has no numeric columns.").classes(
                    "text-sm opacity-50 mt-2"
                )
                return
            self.calc_panel = ui.column().classes("w-full gap-1 mt-2")
            self.calculator = Calculator(
                self.calc_panel,
                ds.columns,
                self._run_calculate,
            )

    def _run_calculate(self) -> None:
        if self.calculator is None:
            return
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            return
        col = self.calculator.column
        op = self.calculator.operation
        if col is None or op is None:
            self.calculator.set_error("select a column and operation")
            return
        try:
            lf = ds.lazyframe
            expr = self.filter_builder.build_expr() if self.filter_builder else None
            if expr is not None:
                lf = lf.filter(expr)
            df = lf.select(col).collect()
            series = df[col]
            value = compute_stat(series, op, self.calculator.threshold)
        except ValueError as err:
            self.calculator.set_error(str(err))
            ui.notify(str(err), type="warning", position="top")
            return
        except Exception as err:
            msg = f"Could not compute: {err}"
            self.calculator.set_error(msg)
            ui.notify(msg, type="negative", position="top")
            return
        self.calculator.set_result(value)

    # --------------------------------------------------------------- plot tab
    def _render_plot_tab(self) -> None:
        self.plot_root.clear()
        self.plot_panel = None
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.plot_root:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("show_chart", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.plot_root:
            with ui.row().classes("items-center justify-between w-full"):
                with ui.column().classes("gap-0"):
                    ui.label("Plot").classes("text-lg font-medium")
                    ui.label(
                        "Build a Plotly figure from every row or just the rows that pass "
                        "the Select-tab filters. Pick a type, columns, and (for bar) a "
                        "statistic."
                    ).classes("text-xs opacity-50")
            self.plot_panel = PlotPanel(self.plot_root, ds.columns, self._run_plot)

    def _run_plot(self) -> None:
        if self.plot_panel is None:
            return
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            return
        spec = self.plot_panel.spec
        scope = self.plot_panel.scope
        try:
            lf = ds.lazyframe
            if scope == "selection":
                expr = self.filter_builder.build_expr() if self.filter_builder else None
                if expr is not None:
                    lf = lf.filter(expr)
            df = lf.collect()
            fig = build_figure(df, spec)
        except ValueError as err:
            self.plot_panel.set_meta(str(err))
            self.plot_panel.render_error(str(err))
            ui.notify(str(err), type="warning", position="top")
            return
        except Exception as err:  # unhandled polars / numpy / plotly errors
            msg = f"Could not build plot: {err}"
            self.plot_panel.set_meta(msg)
            self.plot_panel.render_error(msg)
            ui.notify(msg, type="negative", position="top")
            return
        filt = (
            " with filters"
            if (
                scope == "selection"
                and self.filter_builder
                and self.filter_builder.rows
            )
            else ""
        )
        self.plot_panel.set_meta(
            f"{spec.plot_type} · {df.height:,} row(s) · "
            f"{'current selection' if scope == 'selection' else 'entire dataset'}{filt}"
        )
        self.plot_panel.render_plot(fig)

    # --------------------------------------------------------------- write tab
    def _render_write_tab(self) -> None:
        self.write_container.clear()
        self.write_panel = None
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.write_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("save", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.write_container:
            with ui.row().classes("items-center justify-between w-full"):
                with ui.column().classes("gap-0"):
                    ui.label("Write").classes("text-lg font-medium")
                    ui.label(
                        "Persist the dataset — overwrite the source file, "
                        "save the selection into the current dataset, or export "
                        "as a new file / dataset."
                    ).classes("text-xs opacity-50")
            ui.label(
                f"Supported export formats: {', '.join(SUPPORTED_FORMATS)}"
            ).classes("text-xs opacity-50")
            self.write_panel = WritePanel(
                self.write_container,
                on_save_edits=self._on_write_save_edits,
                on_save_selection=self._on_write_save_selection,
                on_export_dataset=self._on_write_export_dataset,
                on_export_selection=self._on_write_export_selection,
                on_copy_clipboard=self._on_write_copy_clipboard,
                on_export_database=self._on_write_export_database,
            )

    def _on_write_save_edits(self) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            path = save_frame(ds.lazyframe, ds.source)
            ui.notify(f"Saved to {path}", type="positive", position="top")
        except Exception as err:
            ui.notify(f"Could not save: {err}", type="negative", position="top")

    def _on_write_save_selection(self) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        expr = self.filter_builder.build_expr() if self.filter_builder else None
        new_lf = apply_selection(ds.lazyframe, expr, self.selected_columns)
        try:
            df = new_lf.collect()
        except Exception as err:
            ui.notify(
                f"Could not apply selection: {err}", type="negative", position="top"
            )
            return
        self.registry.replace(ds.name, df.lazy())
        self.selected_columns = None
        self._refresh_all_tabs()
        ar, ac = df.shape
        ui.notify(
            f"Selection saved · now {ar:,} row(s) × {ac} column(s)",
            type="positive",
            position="top",
        )

    def _on_write_export_dataset(self, path: str, name: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        if not path.strip():
            ui.notify("Enter a file path", type="warning", position="top")
            return
        try:
            saved = save_frame(ds.lazyframe, path)
            ui.notify(f"Exported to {saved}", type="positive", position="top")
        except Exception as err:
            ui.notify(f"Could not export: {err}", type="negative", position="top")
            return
        if name.strip():
            new_ds = self.registry.load(
                path, Path(path).read_bytes(), name=name.strip()
            )
            self.active_name = new_ds.name
            self._render_list()

    def _on_write_export_selection(self, path: str, name: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        if not path.strip():
            ui.notify("Enter a file path", type="warning", position="top")
            return
        expr = self.filter_builder.build_expr() if self.filter_builder else None
        new_lf = apply_selection(ds.lazyframe, expr, self.selected_columns)
        try:
            saved = save_frame(new_lf, path)
            ui.notify(f"Exported selection to {saved}", type="positive", position="top")
        except Exception as err:
            ui.notify(f"Could not export: {err}", type="negative", position="top")
            return
        if name.strip():
            new_ds = self.registry.load(
                path, Path(path).read_bytes(), name=name.strip()
            )
            self.active_name = new_ds.name
            self._render_list()

    def _on_write_copy_clipboard(self) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            copy_to_clipboard(ds.lazyframe)
            ui.notify("Copied to clipboard", type="positive", position="top")
        except Exception as err:
            ui.notify(f"Could not copy to clipboard: {err}", type="negative", position="top")

    def _on_write_export_database(self, connection: str, table: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        if not connection.strip() or not table.strip():
            ui.notify("Enter a connection URI and table name", type="warning", position="top")
            return
        try:
            write_to_database(ds.lazyframe, table.strip(), connection.strip())
            ui.notify(f"Exported to {table.strip()}", type="positive", position="top")
        except Exception as err:
            ui.notify(f"Could not export to database: {err}", type="negative", position="top")

    # ------------------------------------------------------------- query tab
    def _render_query_tab(self) -> None:
        self.query_container.clear()
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.query_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-16 gap-2"
                ):
                    ui.icon("query_stats", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.query_container:
            self.query_panel = QueryPanel(
                self.query_container,
                self.query_history,
                self._run_query,
            )

    def _run_query(self, query: str) -> None:
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            ui.notify("No active dataset", type="warning", position="top")
            return
        try:
            result = run_sql(ds.lazyframe, query).collect()
        except ValueError as err:
            self.query_history.append(QueryEntry(query=query, error=str(err)))
            ui.notify(str(err), type="warning", position="top")
        except Exception as err:
            msg = f"{type(err).__name__}: {err}"
            self.query_history.append(QueryEntry(query=query, error=msg))
            ui.notify(msg, type="negative", position="top")
        else:
            self.query_history.append(QueryEntry(query=query, result=result))
            ui.notify("Query complete", type="positive", position="top")
        if self.query_panel is not None:
            self.query_panel._render_history()

    @staticmethod
    def _stat(label: str, value: str, icon: str) -> None:
        with ui.column().classes("gap-1 items-start"):
            with ui.row().classes("items-center gap-1 opacity-70"):
                ui.icon(icon, size="16px")
                ui.label(label).classes("text-xs")
            ui.label(value).classes("text-lg font-medium ds-mono")

    # ---------------------------------------------------------- selection panel
    def _build_select_panel(self, ds: Dataset) -> None:
        ui.separator()
        ui.label("Columns to show").classes("text-lg font-medium mt-2")
        ui.label("Leave empty to use all columns.").classes("text-xs opacity-50")
        names = [n for n, _ in ds.columns]
        self.col_select = (
            ui.select(
                options={n: n for n in names},
                multiple=True,
                value=list(self.selected_columns) if self.selected_columns else [],
                clearable=True,
                label="Columns",
                on_change=lambda e: self._on_columns_change(e.value),
            )
            .props("dense outlined use-chips")
            .classes("w-full")
        )

    def _on_columns_change(self, value) -> None:
        self.selected_columns = list(value) if value else None

    # -------------------------------------------------------------- filter panel
    def _build_filter_panel(self, ds: Dataset) -> None:
        ui.separator()
        ui.label("Row filters").classes("text-lg font-medium mt-2")
        ui.label(
            "Boolean expressions applied with df.filter(). "
            "Combined with AND (all) or OR (any)."
        ).classes("text-xs opacity-50")
        fb_container = ui.column().classes("w-full")
        self.filter_builder = FilterBuilder(
            ds.columns,
            fb_container,
            combinator="all",
        )

    # -------------------------------------------------------------- preview panel
    def _build_preview_panel(self, ds: Dataset) -> None:
        ui.separator()
        with ui.row().classes("w-full items-center justify-between mt-2"):
            ui.label("Result preview").classes("text-lg font-medium")
            with ui.row().classes("items-center gap-2"):
                self.mode_toggle = ui.toggle(
                    {
                        "selected": "Selected columns × rows",
                        "rows-only": "All columns × rows",
                    },
                    value=self.preview_mode,
                    on_change=lambda e: self._on_mode_change(e.value),
                ).props("dense")
                ui.button(
                    "Preview",
                    icon="play_arrow",
                    on_click=lambda _=None: self._run_preview(),
                ).props("unelevated color=primary dense")
        with ui.row().classes("items-center gap-2 mt-1"):
            ui.label("Row limit:").classes("text-sm opacity-70")
            self.limit_toggle = ui.toggle(
                {
                    "all": "All rows",
                    "first": "First N",
                    "random": "Random N",
                },
                value=self.preview_limit_mode,
                on_change=lambda e: self._on_limit_change(e.value),
            ).props("dense")
            self.limit_n = (
                ui.number(
                    value=self.preview_limit_n,
                    min=1,
                    max=100_000,
                    label="N",
                )
                .props("dense outlined")
                .classes("w-24")
            )
        self.preview_meta = ui.label("").classes("text-xs opacity-50")
        self.preview_container = ui.column().classes("w-full")

    def _on_mode_change(self, value) -> None:
        self.preview_mode = value or "selected"

    def _on_limit_change(self, value) -> None:
        self.preview_limit_mode = value or "all"
        self.limit_n.set_visibility(value != "all")

    def _run_preview(self) -> None:
        if self.preview_container is None:
            return
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            return
        try:
            lf = ds.lazyframe
            expr = self.filter_builder.build_expr() if self.filter_builder else None
            if hasattr(self, "mode_toggle"):
                self.mode_toggle.set_visibility(self.selected_columns is not None)
            if expr is not None:
                lf = lf.filter(expr)
            if self.preview_mode == "selected" and self.selected_columns:
                lf = lf.select(self.selected_columns)
            n = int(self.limit_n.value) if hasattr(self, "limit_n") and self.limit_n.value else 100
            if self.preview_limit_mode == "first":
                lf = lf.head(n)
            elif self.preview_limit_mode == "random":
                lf = lf.sample(n=n, seed=42)
            df = lf.collect()
        except ValueError as err:
            ui.notify(f"Filter error: {err}", type="warning", position="top")
            self.preview_meta.set_text("Filter not applied.")
            return
        except Exception as err:
            ui.notify(f"Could not build result: {err}", type="negative", position="top")
            self.preview_meta.set_text("Error.")
            return

        n_rows, n_cols = df.shape
        total_rows = ds.shape[0]
        col_desc = (
            f"{n_cols} column{'s' if n_cols != 1 else ''}"
            if self.preview_mode == "selected" and self.selected_columns
            else f"all {n_cols} column{'s' if n_cols != 1 else ''}"
        )
        filt = (
            " with filters"
            if (self.filter_builder and self.filter_builder.rows)
            else ""
        )
        limit_note = ""
        if self.preview_limit_mode != "all" and n_rows < total_rows:
            limit_note = f" (showing {self.preview_limit_mode} {n_rows:,} of {total_rows:,})"
        self.preview_meta.set_text(f"{n_rows:,} row(s) · {col_desc}{filt}{limit_note}")

        self.preview_container.clear()
        if df.width == 0:
            with self.preview_container:
                ui.label("No columns selected.").classes("text-sm opacity-50")
            return
        with self.preview_container:
            columns = [
                {
                    "name": c,
                    "label": f"{c}\n{_human_dtype(df.schema[c])}",
                    "field": c,
                    "align": "left",
                    "sortable": True,
                }
                for c in df.columns
            ]
            rows = df.rows(named=True)
            ui.table(columns=columns, rows=rows, row_key=df.columns[0]).props(
                "flat dense"
            ).classes("w-full")

    # ----------------------------------------------------------------- handlers
    async def _on_upload(self, e: events.UploadEventArguments) -> None:
        try:
            raw = await e.file.read()
            ds = self.registry.load(e.file.name, raw)
            self.active_name = ds.name
            self._refresh_all_tabs()
            self.tabs.set_value("Select")
            ui.notify(f"Loaded {ds.name}", type="positive", position="top")
        except UnsupportedFormatError as err:
            ui.notify(str(err), type="warning", position="top")
        except Exception as err:  # polars read errors, malformed files, ...
            ui.notify(f"Could not read dataset: {err}", type="negative", position="top")

    def _activate_loaded(self, ds: Dataset) -> None:
        self.active_name = ds.name
        self._refresh_all_tabs()
        self.tabs.set_value("Select")
        ui.notify(f"Loaded {ds.name}", type="positive", position="top")

    def _on_load_clipboard(self) -> None:
        try:
            name = self.clipboard_name.value or None
            ds = self.registry.load_clipboard(name=name)
            self._activate_loaded(ds)
        except Exception as err:
            ui.notify(f"Could not read clipboard: {err}", type="negative", position="top")

    def _on_load_database(self) -> None:
        uri = (self.db_uri.value or "").strip()
        query = (self.db_query.value or "").strip()
        if not uri or not query:
            ui.notify("Enter a connection URI and SQL query", type="warning", position="top")
            return
        try:
            name = self.db_load_name.value or None
            ds = self.registry.load_database(query, uri, name=name)
            self._activate_loaded(ds)
        except Exception as err:
            ui.notify(f"Could not load from database: {err}", type="negative", position="top")

    def _on_load_iceberg(self) -> None:
        table = (self.iceberg_table.value or "").strip()
        if not table:
            ui.notify("Enter a table identifier", type="warning", position="top")
            return
        try:
            catalog = (self.iceberg_catalog.value or "default").strip()
            name = self.iceberg_name.value or None
            ds = self.registry.load_iceberg(table, catalog=catalog, name=name)
            self._activate_loaded(ds)
        except Exception as err:
            ui.notify(f"Could not load Iceberg table: {err}", type="negative", position="top")

    def _on_remove(self) -> None:
        self.registry.remove(self.active_name)  # type: ignore[arg-type]
        self.active_name = next(iter(self.registry.names()), None)
        self._refresh_all_tabs()

    def _select(self, name: str) -> None:
        self.active_name = name
        self.selected_columns = None
        self.preview_mode = "selected"
        self._refresh_all_tabs()
        self.tabs.set_value("Select")


@ui.page("/")
def _page() -> None:
    ui.page_title(_APP_TITLE)
    App().build()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native",
        action="store_true",
        help="Run as a native desktop window instead of a web server",
    )
    args = parser.parse_args()

    ui.run(
        native=args.native,
        reload=False,
        title=_APP_TITLE,
        port=8080,
    )
