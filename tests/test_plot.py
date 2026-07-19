"""Tests for the pure plot-figure builder."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from datasium.plot import PlotSpec, build_figure


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "c", "c"],
            "x": [0, 1, 2, 3, 4, 5],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _trace(fig):
    return fig["data"][0]


def test_scatter_needs_xy(df):
    with pytest.raises(ValueError, match="X column is required"):
        build_figure(df, PlotSpec(plot_type="scatter"))
    with pytest.raises(ValueError, match="Y column is required"):
        build_figure(df, PlotSpec(plot_type="scatter", x="cat"))


def test_scatter_builds_trace(df):
    fig = build_figure(df, PlotSpec(plot_type="scatter", x="x", y="y"))
    assert _trace(fig)["type"] == "scatter"
    assert _trace(fig)["mode"] == "markers"
    np.testing.assert_array_equal(_trace(fig)["x"], df["x"].to_numpy())
    np.testing.assert_array_equal(_trace(fig)["y"], df["y"].to_numpy())
    assert fig["layout"]["xaxis"]["title"] == "x"
    assert fig["layout"]["yaxis"]["title"] == "y"


def test_line_mode():
    df = pl.DataFrame({"x": [0, 1], "y": [0, 1]})
    fig = build_figure(df, PlotSpec(plot_type="line", x="x", y="y"))
    assert _trace(fig)["mode"] == "lines"


def test_color_grouping_makes_one_trace_per_group(df):
    fig = build_figure(df, PlotSpec(plot_type="scatter", x="x", y="y", color="cat"))
    labels = [t["name"] for t in fig["data"]]
    assert labels == ["a", "b", "c"]
    first = [t["x"].tolist() for t in fig["data"]]  # type: ignore[union-attr]
    assert first == [[0, 1], [2, 3], [4, 5]]


def _tr(fig):
    return fig["data"][0]


def test_bar_aggregated_mean(df):
    fig = build_figure(df, PlotSpec(plot_type="bar", x="cat", y="y", agg="mean"))
    t = fig["data"][0]
    assert t["type"] == "bar"
    assert t["x"].tolist() == ["a", "b", "c"]
    assert t["y"].tolist() == [1.5, 3.5, 5.5]  # mean of [1,2],[3,4],[5,6]
    assert fig["layout"]["yaxis"]["title"] == "mean of y"


def test_bar_aggregated_color_equals_x(df):
    # color column equal to X must not raise DuplicateError
    fig = build_figure(
        df, PlotSpec(plot_type="bar", x="cat", y="y", color="cat", agg="mean")
    )
    assert [t["name"] for t in fig["data"]] == ["a", "b", "c"]


def test_bar_aggregated_with_color(df):
    # color splits the aggregation into one trace per group
    df2 = df.with_columns(bin=pl.Series(["p", "q", "p", "q", "p", "q"]))
    fig = build_figure(
        df2, PlotSpec(plot_type="bar", x="cat", y="y", color="bin", agg="sum")
    )
    by_name = {t["name"]: t["x"].tolist() for t in fig["data"]}
    assert set(by_name) == {"p", "q"}


def test_bar_count_without_y(df):
    fig = build_figure(df, PlotSpec(plot_type="bar", x="cat", agg="count"))
    t = fig["data"][0]
    assert t["x"].tolist() == ["a", "b", "c"]
    assert t["y"].tolist() == [2, 2, 2]
    assert fig["layout"]["yaxis"]["title"] == "count"


def test_bar_raw(df):
    fig = build_figure(df, PlotSpec(plot_type="bar", x="cat", y="y", agg="raw"))
    # raw keeps every row; first trace grouped when no color -> one trace
    assert len(fig["data"]) == 1


def test_bar_raw_needs_y(df):
    with pytest.raises(ValueError, match="Y column is required"):
        build_figure(df, PlotSpec(plot_type="bar", x="cat", agg="raw"))


def test_histogram(df):
    fig = build_figure(df, PlotSpec(plot_type="histogram", x="y", nbins=5))
    assert _tr(fig)["type"] == "histogram"
    assert _tr(fig)["nbinsx"] == 5
    np.testing.assert_array_equal(_tr(fig)["x"], df["y"].to_numpy())


def test_box(df):
    fig = build_figure(df, PlotSpec(plot_type="box", y="y", color="cat"))
    assert [t["name"] for t in fig["data"]] == ["a", "b", "c"]
    assert all(t["type"] == "box" for t in fig["data"])


def test_violin(df):
    fig = build_figure(df, PlotSpec(plot_type="violin", y="y"))
    assert _tr(fig)["type"] == "violin"


def test_empty_frame_raises():
    df = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64})
    with pytest.raises(ValueError, match="no rows to plot"):
        build_figure(df, PlotSpec(plot_type="scatter", x="x", y="y"))


def test_unknown_plot_type(df):
    with pytest.raises(ValueError, match="unknown plot type"):
        build_figure(df, PlotSpec(plot_type="nope", x="x", y="y"))


def test_missing_column(df):
    with pytest.raises(ValueError, match="not found"):
        build_figure(df, PlotSpec(plot_type="scatter", x="nope", y="y"))


def test_title_reflects_agg(df):
    fig = build_figure(df, PlotSpec(plot_type="bar", x="cat", y="y", agg="sum"))
    assert "sum of y" in fig["layout"]["title"]["text"]
