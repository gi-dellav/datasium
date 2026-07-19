"""SQL query component for the active dataset.

Renders a SQL input and runs the query against the active ``LazyFrame``
using :meth:`polars.LazyFrame.sql`, returning a lazy result that the caller
collects only when it is ready to display it. The calling frame is registered
under the table name ``self`` (the method default), so queries look like
``SELECT * FROM self WHERE age > 20``.

A run history is kept and rendered after the input UI so past queries stay
visible.

The pure :func:`run_sql` helper is UI-free so it can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nicegui import ui


@dataclass
class QueryEntry:
    """A single executed query and its outcome."""

    query: str
    error: str | None = None
    result: pl.DataFrame | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_sql(lf: pl.LazyFrame, query: str, *, table_name: str = "self") -> pl.LazyFrame:
    """Execute ``query`` against ``lf`` via :meth:`polars.LazyFrame.sql`.

    Returns a lazy result; the caller decides when to ``.collect()`` it for
    display. Pure (UI-free) so it can be unit-tested. Raises ``ValueError``
    with a user-facing message on empty input; Polars surfaces SQL parse /
    compute errors as its own exceptions.
    """
    if query is None or not query.strip():
        raise ValueError("enter a SQL query")
    return lf.sql(query, table_name=table_name)


class QueryPanel:
    """SQL input + result + run history for one active dataset."""

    def __init__(
        self,
        parent,
        history: list[QueryEntry],
        on_run: "callable",  # type: ignore[name-defined]
    ) -> None:
        self._history = history
        self._on_run = on_run

        with parent:
            with ui.row().classes("items-center justify-between w-full"):
                with ui.column().classes("gap-0"):
                    ui.label("Query").classes("text-lg font-medium")
                    ui.label(
                        "Run a SQL query against the active DataFrame "
                        "(table name: self)."
                    ).classes("text-xs opacity-50")
            self.sql_input = (
                ui.textarea(
                    value="",
                    placeholder="SELECT * FROM self WHERE age > 20",
                    label="SQL",
                )
                .props("dense outlined autogrow")
                .classes("w-full ds-mono")
            )
            self.sql_input.on("keydown.enter.ctrl", lambda _e: self._submit())
            with ui.row().classes("items-center gap-2"):
                ui.button(
                    "Run",
                    icon="play_arrow",
                    on_click=lambda _=None: self._submit(),
                ).props("dense unelevated color=primary")
                ui.button(
                    "Clear",
                    icon="delete_sweep",
                    on_click=lambda _=None: self._clear_input(),
                ).props("dense flat color=negative")

            ui.separator()
            ui.label("History").classes("text-lg font-medium mt-2")
            self.history_container = ui.column().classes("w-full gap-2")
            self._render_history()

    # ------------------------------------------------------------------ input
    def _submit(self) -> None:
        text = self.sql_input.value or ""
        if not text.strip():
            ui.notify("Enter a SQL query first", type="warning", position="top")
            return
        self._on_run(text)

    def _clear_input(self) -> None:
        self.sql_input.value = ""

    # ---------------------------------------------------------------- history
    def _render_history(self) -> None:
        self.history_container.clear()
        if not self._history:
            with self.history_container:
                ui.label("No queries run yet.").classes("text-sm opacity-50")
            return
        with self.history_container:
            # Newest first.
            for idx, entry in enumerate(reversed(self._history), start=1):
                self._render_entry(entry, idx)

    def _render_entry(self, entry: QueryEntry, idx: int) -> None:
        with ui.card().classes("w-full ds-card"):
            with ui.row().classes("items-center gap-2 w-full"):
                ui.badge(f"#{idx}").props("color=primary")
                ui.label("ok" if entry.ok else "error").classes(
                    "text-xs " + ("text-positive" if entry.ok else "text-negative")
                )
                ui.space()
                ui.button(
                    icon="replay",
                    on_click=lambda _=None, q=entry.query: self._rerun(q),
                ).props("flat round dense").tooltip("Re-run this query")
            with ui.row().classes("w-full"):
                ui.code(entry.query).classes("w-full ds-mono text-xs")
            if entry.ok:
                df = entry.result
                n_rows, n_cols = df.shape
                ui.label(f"{n_rows:,} row(s) · {n_cols} column(s)").classes(
                    "text-xs opacity-50"
                )
                self._render_result_table(df)
            else:
                ui.label(entry.error or "Unknown error").classes(
                    "text-sm text-negative ds-mono"
                )

    def _render_result_table(self, df: pl.DataFrame) -> None:
        if df.width == 0:
            ui.label("No columns in result.").classes("text-sm opacity-50")
            return
        columns = [
            {
                "name": c,
                "label": f"{c}\n{df.schema[c]}",
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

    def _rerun(self, query: str) -> None:
        self.sql_input.value = query
        self._on_run(query)
