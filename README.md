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
  Polars expression used with `lf.filter`.
- **Preview the result** as either *Selected columns × selected rows* or
  *All columns × selected rows*, with a row/column count. The preview is
  computed on demand — click **Preview** to run it.
- **Run SQL** against the active LazyFrame via the Query tab (table name
  `self`, e.g. `SELECT * FROM self WHERE age > 20`). The query runs lazily
  and is only collected when you click **Run**. Executed queries are kept in
  a history list shown below the input, each with its result (or error) and a
  re-run button.
- **Plot** the active dataset with **Plotly**: choose a plot type
  (scatter, line, bar, histogram, box, violin), X / Y / color columns, and
  (for bar) an aggregation statistic (mean / sum / min / max / median /
  count). Plot from the **entire dataset** or just the **current selection**
  (the rows that pass the Select-tab filters); click **Plot** to render the
  figure.

Datasets are held as Polars **LazyFrames** end to end, so nothing is
materialised until you ask for it. Expensive work — rendering a result table,
running a statistic, building a plot, or executing a SQL query — only happens
when you click the corresponding button (**Preview**, **Calculate**,
**Plot**, **Run**). Editing filters or column selections just stages the
change; it never triggers a recomputation on its own.

## Architecture

| Module | Responsibility |
|---|---|
| `datasium.dataset` | `Dataset` / `DatasetRegistry` of named `LazyFrame` sources, format readers |
| `datasium.filter` | Reusable `FilterBuilder` component producing Polars `df.filter` expressions |
| `datasium.query` | SQL query component running `polars.LazyFrame.sql` with a run history |
| `datasium.plot` | Plotly figure builder (`build_figure` + `PlotSpec`) and `PlotPanel` UI |
| `datasium.ui.app` | NiceGUI workbench: loader, schema view, column select, row filters, result preview, SQL query, plots |

## Test

```bash
uv run pytest
```
