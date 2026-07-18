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
- **Run SQL** against the active DataFrame via the Query tab (table name
  `self`, e.g. `SELECT * FROM self WHERE age > 20`). Executed queries are
  kept in a history list shown below the input, each with its result (or
  error) and a re-run button.
- **Plot** the active dataset with **Plotly**: choose a plot type
  (scatter, line, bar, histogram, box, violin), X / Y / color columns, and
  (for bar) an aggregation statistic (mean / sum / min / max / median /
  count). Plot from the **entire dataset** or just the **current selection**
  (the rows that pass the Select-tab filters); the figure updates live when
  filters change.

## Architecture

| Module | Responsibility |
|---|---|
| `datasium.dataset` | `Dataset` / `DatasetRegistry` of named `LazyFrame` sources, format readers |
| `datasium.filter` | Reusable `FilterBuilder` component producing Polars `df.filter` expressions |
| `datasium.query` | SQL query component running `polars.DataFrame.sql` with a run history |
| `datasium.plot` | Plotly figure builder (`build_figure` + `PlotSpec`) and `PlotPanel` UI |
| `datasium.ui.app` | NiceGUI workbench: loader, schema view, column select, row filters, result preview, SQL query, plots |

## Test

```bash
uv run pytest
```
