# datasium

The powerhouse for intuitive data processing — a visual data-workbench built on
**Polars** (lazy pipelines) and **NiceGUI** (browser/desktop UI).

## Run

```bash
uv run datasium
```

Open <http://localhost:8080> (or the native window if run with `--native`),
load a dataset (try `sample.csv`) via the upload, and inspect its columns,
types, and shape in the detail panel. Multiple datasets can be loaded and
switched between in the sidebar.

Per active dataset you can:

- **Select columns** to project (leave empty = all columns).
- **Define row filters** via a reusable filter builder: pick a column, an
  operator (eq, >, contains, is in, is null, …) and a value; combine rows
  with **Match ALL** (`and`) or **Match ANY** (`or`). Filters compile to a
  Polars expression used with `df.filter`.
- **Preview the result** as either *Selected columns × selected rows* or
  *All columns × selected rows*, with a live row/column count.

## Architecture

| Module | Responsibility |
|---|---|
| `datasium.dataset` | `Dataset` / `DatasetRegistry` of named `LazyFrame` sources, format readers |
| `datasium.filter` | Reusable `FilterBuilder` component producing Polars `df.filter` expressions |
| `datasium.ui.app` | NiceGUI workbench: loader, schema view, column select, row filters, result preview |

## Test

```bash
uv run pytest
```
