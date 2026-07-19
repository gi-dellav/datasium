"""Tests for the pure clustering helpers."""

from __future__ import annotations

import polars as pl
import pytest

from datasium.cluster import ClusterSpec, run_clustering, build_cluster_figure, ALGORITHMS


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
            "y": [2.0, 2.1, 2.2, 8.0, 8.1, 8.2],
            "cat": ["a", "b", "c", "d", "e", "f"],
        }
    )


# --------------------------------------------------------------------------- #
# run_clustering – error paths
# --------------------------------------------------------------------------- #
def test_no_columns_raises(df):
    with pytest.raises(ValueError, match="select at least one numeric column"):
        run_clustering(df, ClusterSpec(columns=[]))


def test_missing_column_raises(df):
    with pytest.raises(ValueError, match="column\\(s\\) not found: z"):
        run_clustering(df, ClusterSpec(columns=["z"]))


def test_non_numeric_column_raises(df):
    with pytest.raises(ValueError, match="column\\(s\\) not numeric: cat"):
        run_clustering(df, ClusterSpec(columns=["x", "cat"]))


def test_empty_frame_raises():
    empty = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="no rows to cluster"):
        run_clustering(empty, ClusterSpec(columns=["x"]))


def test_existing_output_column_raises(df):
    with pytest.raises(ValueError, match="already exists"):
        run_clustering(df, ClusterSpec(columns=["x"], output_column="x"))


def test_all_nulls_raises():
    df = pl.DataFrame(
        {
            "x": pl.Series([None, None], dtype=pl.Float64),
            "y": pl.Series([None, None], dtype=pl.Float64),
        }
    )
    with pytest.raises(ValueError, match="all rows contain nulls"):
        run_clustering(df, ClusterSpec(columns=["x", "y"]))


def test_unknown_algorithm_raises(df):
    with pytest.raises(ValueError, match="unknown algorithm"):
        run_clustering(df, ClusterSpec(columns=["x"], algorithm="bogus"))


# --------------------------------------------------------------------------- #
# run_clustering – basic behaviour
# --------------------------------------------------------------------------- #
def test_kmeans_adds_cluster_column(df):
    spec = ClusterSpec(columns=["x", "y"], algorithm="kmeans", n_clusters=2)
    result = run_clustering(df, spec)
    assert "cluster" in result.columns
    assert result.height == df.height
    assert result["cluster"].dtype == pl.Int32
    assert result["cluster"].n_unique() == 2


def test_kmeans_two_blobs(df):
    spec = ClusterSpec(columns=["x", "y"], algorithm="kmeans", n_clusters=2, scale=False)
    result = run_clustering(df, spec)
    labels = result["cluster"].to_list()
    # first 3 rows should share a label, last 3 should share a label
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_custom_output_column(df):
    spec = ClusterSpec(columns=["x"], algorithm="kmeans", n_clusters=2, output_column="grp")
    result = run_clustering(df, spec)
    assert "grp" in result.columns
    assert "cluster" not in result.columns


def test_null_rows_get_null_label():
    df = pl.DataFrame({"x": [1.0, None, 5.0, 5.1], "y": [2.0, 3.0, 8.0, 8.1]})
    spec = ClusterSpec(columns=["x", "y"], algorithm="kmeans", n_clusters=2)
    result = run_clustering(df, spec)
    assert result.height == 4
    assert result["cluster"][1] is None
    assert result["cluster"][0] is not None


def test_scale_option(df):
    spec_scaled = ClusterSpec(columns=["x", "y"], algorithm="kmeans", n_clusters=2, scale=True)
    spec_raw = ClusterSpec(columns=["x", "y"], algorithm="kmeans", n_clusters=2, scale=False)
    r1 = run_clustering(df, spec_scaled)
    r2 = run_clustering(df, spec_raw)
    assert r1.height == r2.height
    assert "cluster" in r1.columns


# --------------------------------------------------------------------------- #
# run_clustering – every algorithm produces labels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "algo,extra",
    [
        ("kmeans", {"n_clusters": 2}),
        ("minibatch_kmeans", {"n_clusters": 2}),
        ("affinity_propagation", {}),
        ("mean_shift", {}),
        ("spectral", {"n_clusters": 2}),
        ("ward", {"n_clusters": 2}),
        ("agglomerative", {"n_clusters": 2, "linkage": "complete"}),
        ("dbscan", {"eps": 2.0, "min_samples": 2}),
        ("hdbscan", {"min_cluster_size": 2, "min_samples": 2}),
        ("optics", {"min_samples": 2, "eps": 2.0}),
        ("birch", {"n_clusters": 2}),
        ("gmm", {"n_clusters": 2}),
    ],
)
def test_algorithm_produces_labels(df, algo, extra):
    spec = ClusterSpec(columns=["x", "y"], algorithm=algo, **extra)
    result = run_clustering(df, spec)
    assert "cluster" in result.columns
    assert result.height == df.height
    assert result["cluster"].null_count() == 0


# --------------------------------------------------------------------------- #
# run_clustering – density-based noise label
# --------------------------------------------------------------------------- #
def test_dbscan_noise_label():
    df = pl.DataFrame(
        {
            "x": [1.0, 1.1, 1.2, 100.0],
            "y": [1.0, 1.1, 1.2, 100.0],
        }
    )
    spec = ClusterSpec(
        columns=["x", "y"], algorithm="dbscan", eps=1.0, min_samples=2, scale=False
    )
    result = run_clustering(df, spec)
    labels = result["cluster"].to_list()
    assert labels[3] == -1  # outlier is noise


# --------------------------------------------------------------------------- #
# build_cluster_figure
# --------------------------------------------------------------------------- #
def test_figure_scatter_two_columns(df):
    spec = ClusterSpec(columns=["x", "y"], algorithm="kmeans", n_clusters=2)
    result = run_clustering(df, spec)
    fig = build_cluster_figure(result, spec)
    assert fig["data"]
    assert fig["data"][0]["type"] == "scatter"
    assert fig["layout"]["xaxis"]["title"] == "x"
    assert fig["layout"]["yaxis"]["title"] == "y"


def test_figure_histogram_single_column(df):
    spec = ClusterSpec(columns=["x"], algorithm="kmeans", n_clusters=2)
    result = run_clustering(df, spec)
    fig = build_cluster_figure(result, spec)
    assert fig["data"]
    assert fig["data"][0]["type"] == "histogram"


def test_figure_missing_cluster_column_raises(df):
    spec = ClusterSpec(columns=["x", "y"])
    with pytest.raises(ValueError, match="run clustering first"):
        build_cluster_figure(df, spec)


def test_figure_all_null_labels_raises():
    df = pl.DataFrame({"x": [1.0, 2.0], "cluster": [None, None]})
    spec = ClusterSpec(columns=["x"])
    with pytest.raises(ValueError, match="no clustered rows"):
        build_cluster_figure(df, spec)


# --------------------------------------------------------------------------- #
# ALGORITHMS registry
# --------------------------------------------------------------------------- #
def test_algorithms_registry_covers_all():
    keys = {k for _, k in ALGORITHMS}
    assert "kmeans" in keys
    assert "dbscan" in keys
    assert "hdbscan" in keys
    assert "gmm" in keys
    assert len(keys) == 12
