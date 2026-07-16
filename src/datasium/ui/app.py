"""datasium application UI (NiceGUI).

A polished, lightweight workbench for loading Polars datasets and inspecting
their schema. Designed so operation panels can be layered on top of the
``DatasetRegistry`` shared with the UI.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import events, ui

from datasium.dataset import Dataset, DatasetRegistry, UnsupportedFormatError
from datasium.filter import FilterBuilder

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
            ui.button(icon="dark_mode", on_click=lambda: ui.run_javascript(
                "document.body.classList.toggle('dark')"
            )).props("flat round").tooltip("Toggle theme")

        with ui.tabs() as self.tabs:
            ui.tab("Load", icon="upload_file")
            ui.tab("Select", icon="filter_alt")
            ui.tab("View", icon="visibility")

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

    # ---------------------------------------------------------------- load tab
    def _render_load_tab(self) -> None:
        self.load_container.clear()
        with self.load_container:
            ui.label("Load dataset").classes("text-lg font-medium")
            ui.separator()
            ui.upload(
                label="Choose a file",
                multiple=False,
                auto_upload=True,
                on_upload=self._on_upload,
                on_rejected=lambda: ui.notify(
                    "File rejected", type="negative", position="top",
                ),
            ).props('accept=".csv,.tsv,.psv,.parquet,.json,.ndjson,.ipc,.arrow,.feather" '
                     'color="primary"').classes("w-full")
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
                with ui.button(on_click=lambda _, n=ds.name: self._select(n)) \
                        .props(f"flat align=left {''if not active else 'outline'}") \
                        .classes("w-full justify-start"):
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
                with ui.column().classes("w-full items-center justify-center py-16 gap-2"):
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
                {"name": "idx", "label": "#", "field": "idx",
                 "align": "left", "sortable": True},
                {"name": "name", "label": "Name", "field": "name",
                 "align": "left", "sortable": True},
                {"name": "dtype", "label": "Type", "field": "dtype",
                 "align": "left", "sortable": True},
            ]
            schema_rows = [
                {"idx": i + 1, "name": name, "dtype": _human_dtype(dtype)}
                for i, (name, dtype) in enumerate(ds.columns)
            ]
            ui.table(columns=schema_cols, rows=schema_rows, row_key="idx") \
                .props("flat dense rows-per-page-options=[0]") \
                .classes("w-full")

            self._build_filter_panel(ds)

    # ---------------------------------------------------------------- view tab
    def _render_view_tab(self) -> None:
        self.view_container.clear()
        ds = self.registry.get(self.active_name) if self.active_name else None
        if ds is None:
            with self.view_container:
                with ui.column().classes("w-full items-center justify-center py-16 gap-2"):
                    ui.icon("visibility", size="48px").classes("opacity-30")
                    ui.label("Load a dataset first").classes("opacity-60")
            return

        with self.view_container:
            self._build_select_panel(ds)
            self._build_preview_panel(ds)

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
        self.col_select = ui.select(
            options={n: n for n in names},
            multiple=True,
            value=list(self.selected_columns) if self.selected_columns else [],
            clearable=True,
            label="Columns",
            on_change=lambda e: self._on_columns_change(e.value),
        ).props("dense outlined use-chips").classes("w-full")

    def _on_columns_change(self, value) -> None:
        self.selected_columns = list(value) if value else None
        self._run_preview()

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
            ds.columns, fb_container, combinator="all",
        )
        self.filter_builder.on_change(self._run_preview)

    # -------------------------------------------------------------- preview panel
    def _build_preview_panel(self, ds: Dataset) -> None:
        ui.separator()
        with ui.row().classes("w-full items-center justify-between mt-2"):
            ui.label("Result preview").classes("text-lg font-medium")
            with ui.row().classes("items-center gap-2"):
                self.mode_toggle = ui.toggle(
                    {"selected": "Selected columns × rows",
                     "rows-only": "All columns × rows"},
                    value=self.preview_mode,
                    on_change=lambda e: self._on_mode_change(e.value),
                ).props("dense")
                ui.button("Preview", icon="play_arrow", on_click=lambda _=None: self._run_preview()) \
                    .props("unelevated color=primary dense")
        self.preview_meta = ui.label("").classes("text-xs opacity-50")
        self.preview_container = ui.column().classes("w-full")
        self._run_preview()

    def _on_mode_change(self, value) -> None:
        self.preview_mode = value or "selected"
        self._run_preview()

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
        col_desc = (
            f"{n_cols} column{'s' if n_cols != 1 else ''}"
            if self.preview_mode == "selected" and self.selected_columns
            else f"all {n_cols} column{'s' if n_cols != 1 else ''}"
        )
        filt = " with filters" if (self.filter_builder and self.filter_builder.rows) else ""
        self.preview_meta.set_text(f"{n_rows:,} row(s) · {col_desc}{filt}")

        self.preview_container.clear()
        if df.width == 0:
            with self.preview_container:
                ui.label("No columns selected.").classes("text-sm opacity-50")
            return
        with self.preview_container:
            columns = [
                {"name": c, "label": f"{c}\n{_human_dtype(df.schema[c])}", "field": c,
                 "align": "left", "sortable": True}
                for c in df.columns
            ]
            rows = df.rows(named=True)
            ui.table(columns=columns, rows=rows, row_key=df.columns[0]) \
                .props("flat dense") \
                .classes("w-full")

    # ----------------------------------------------------------------- handlers
    def _on_upload(self, e: events.UploadEventArguments) -> None:
        try:
            raw = e.content.read()
            ds = self.registry.load(e.name, raw)
            self.active_name = ds.name
            self._render_list()
            self._render_select_tab()
            self._render_view_tab()
            self.tabs.set_value("Select")
            ui.notify(f"Loaded {ds.name}", type="positive", position="top")
        except UnsupportedFormatError as err:
            ui.notify(str(err), type="warning", position="top")
        except Exception as err:  # polars read errors, malformed files, ...
            ui.notify(f"Could not read dataset: {err}", type="negative", position="top")

    def _on_remove(self) -> None:
        self.registry.remove(self.active_name)  # type: ignore[arg-type]
        self.active_name = next(iter(self.registry.names()), None)
        self._render_list()
        self._render_select_tab()
        self._render_view_tab()

    def _select(self, name: str) -> None:
        self.active_name = name
        self.selected_columns = None
        self.preview_mode = "selected"
        self._render_list()
        self._render_select_tab()
        self._render_view_tab()
        self.tabs.set_value("Select")


@ui.page("/")
def _page() -> None:
    ui.page_title(_APP_TITLE)
    App().build()


def main() -> None:
    ui.run(
        reload=False,
        title=_APP_TITLE,
        port=8080,
    )