"""Minimal NiceGUI app: load a dataset via Polars and show its columns."""

from __future__ import annotations

import io

import polars as pl
from nicegui import ui


@ui.page("/")
def _page() -> None:
    ui.label("Datasium").classes("text-2xl font-bold")
    status = ui.label()
    columns_list = ui.column().classes("w-full gap-1")

    async def handle_upload(e) -> None:
        status.text = f"Loading: {e.name}"
        data = e.content.read()
        try:
            df = _read_bytes(e.name, data)
        except Exception as ex:
            status.text = f"Error: {ex}"
            columns_list.clear()
            return
        status.text = f"{e.name}  —  {df.height} rows × {df.width} cols"
        columns_list.clear()
        with columns_list:
            for col_name, dtype in zip(df.columns, df.dtypes):
                ui.label(f"{col_name}  ({dtype})")
        e.content.close()

    ui.upload(
        on_upload=handle_upload,
        auto_upload=True,
        multiple=False,
        label="Load Dataset",
    ).props("accept=.csv,.parquet,.ndjson,.json,.ipc,.arrow")


def _read_bytes(name: str, data: bytes) -> pl.DataFrame:
    lower = name.lower()
    source = io.BytesIO(data)
    if lower.endswith(".csv"):
        return pl.read_csv(source)
    if lower.endswith(".parquet"):
        return pl.read_parquet(source)
    if lower.endswith(".ndjson") or lower.endswith(".json"):
        return pl.read_ndjson(source)
    if lower.endswith(".ipc") or lower.endswith(".arrow"):
        return pl.read_ipc(source)
    raise ValueError(f"Unsupported file format: {name}")


def main() -> None:
    ui.run(host="127.0.0.1", port=8080, title="Datasium", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()