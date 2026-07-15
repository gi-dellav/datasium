# datasium

The powerhouse for intuitive data processing — a visual data-workbench built on
**Polars** (lazy pipelines) and **Flet** (cross-platform UI). Implements
**Phase 1 & 2** of `PLAN.md`.

## Run

```bash
uv run datasium                 # desktop app
```

Load a CSV (try `sample.csv`) via the file-open button, pick actions from the
left palette to build a pipeline, edit parameters in the right panel, and
Run / Preview in the bottom tabs. Undo/redo via the app bar.

## Layout

- **Left panel** — searchable action palette grouped by category + dataset list.
- **Center** — horizontal pipeline editor (source / action node cards / output)
  with per-node move/duplicate/delete, plus preview tabs (Selected node, Final
  output, Schema, Query plan, Console).
- **Right panel** — dynamic parameter form generated from the action's schema
  and a live validation list.

## Architecture

| Module | Responsibility |
|---|---|
| `datasium.expression` | AST-whitelisted Polars expression evaluator |
| `datasium.actions` | `Action` / `ActionParam` / `ParamType` framework + built-in catalog |
| `datasium.dataset` | `DatasetRegistry` of named `LazyFrame` sources |
| `datasium.pipeline` | `Pipeline`, `PipelineNode`, executor, `SnapshotStack` undo/redo |
| `datasium.ui.*` | Flet panels + `AppController` |

## Test

```bash
uv run pytest
```
